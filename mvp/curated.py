import pandas as pd
import numpy as np

from math import sqrt, pi
from functools import partial
from multiprocessing import Pool
from statsmodels.tsa.stattools import adfuller
from numba import njit, prange, int32, float64

from mvp.rawdata import RawData


@njit(int32(float64, float64, float64[:], int32, int32))
def numba_weights(d, tolerance, w_array, w_size, last_index):
    """
    Compiled function to compute weights of frac_diff method efficiently
    This function must be called until a positive number is returned,
    which indicate that convergence was achieve according to `tolerance`

    Parameters
    ----------
    `d` : ``float``
        order of fractional differentiation. Usually between 0 and 1
    `tolerance` : ``float``
        minimum value for weights
    `w_array` : ``numpy.array``
        values of all weights computed so far
    `w_size` : ``int``
        current size of `w_array`
    `last_index` : ``int``
        index of last weight set in `w_array` and from which must continue

    Modified
    --------
    `w_array`
        with new weights starting from `last_index`

    Return
    ------
    ``int``
        If positive, convergence was achieved and the value is the number
        of weights computed. If negative, weights are above the `tolerance`
        provided, the weights array must be resized adding empty entries,
        and this function must be called again from the last weight set

    """
    for k in prange(last_index + 1, w_size):
        w_array[k] = -(w_array[k - 1] / k) * (d - k + 1)
        if abs(w_array[k]) < tolerance:
            return k + 1
    return -1


def mean_quadratic_freq(data, time_spacing=1):
    displace_data = data - data.mean()
    fft_weights = np.fft.fft(displace_data)
    freq = np.fft.fftfreq(displace_data.size, time_spacing)
    weights_norm = np.abs(fft_weights).sum()
    return sqrt((freq * freq * np.abs(fft_weights)).sum() / weights_norm)


class RefinedData(RawData):
    """
    Integrate to the raw data with open-high-low-close values
    some simple statistical features which provide more tools
    to analyze the data and support primary models

    Parameters
    ----------
    `raw_data` : ``rawdata.RawData class``
    `requested_features : `` dict ``
        Dictionary with features as strings in keys and the
        evaluation feature paramter as values or list of values
        The (keys)strings corresponding to features must be:
        "MA" = Moving Average -> Value = window size
        "DEV" = standart DEViation -> Value = window_size
        "RSI" = RSI indicator -> Value = window_size
        "FRAC_DIFF" = Fractional Diff. -> Value = float in [0,1]
    `daily` : `` bool `` (optional)
        Automatically convert 1-minute raw data to daily data

    """

    def __init__(
        self,
        symbol,
        db_path,
        requested_features={},
        start=None,
        stop=None,
        time_step=1,
    ):
        RawData.__init__(self, symbol, db_path)
        self.__attr = {
            "MA": "get_simple_MA",
            "DEV": "get_deviation",
            "RSI": "get_RSI",
            "FRAC_DIFF": "frac_diff",
            "AUTOCORRELATION": "moving_autocorr",
            "AUTOCORRELATION_PERIOD": "autocorr_period",
        }
        self.__cached_features = {}
        for feature in requested_features.keys():
            if feature not in self.__attr.keys():
                continue
            if isinstance(requested_features[feature], list):
                parameters_list = requested_features[feature]
                for parameter in parameters_list:
                    try:
                        if isinstance(parameter, tuple):
                            self.__getattribute__(self.__attr[feature])(
                                *parameter, start, stop, time_step, True
                            )
                        else:
                            self.__getattribute__(self.__attr[feature])(
                                parameter, start, stop, time_step, True
                            )
                    except Exception as err:
                        print(err, ": param {} given".format(parameter))
            else:
                parameter = requested_features[feature]
                try:
                    if isinstance(parameter, tuple):
                        self.__getattribute__(self.__attr[feature])(
                            *parameter, start, stop, time_step, True
                        )
                    else:
                        self.__getattribute__(self.__attr[feature])(
                            parameter, start, stop, time_step, True
                        )
                except ValueError as err:
                    print(err, ": param {} given".format(parameter))

    def __code_formatter(self, name, start, stop, time_step, extra_par):
        fmt_start = start.strftime("%Y%m%d")
        fmt_stop = stop.strftime("%Y%m%d")
        str_code = "{}_{}_{}_{}_{}".format(
            name, fmt_start, fmt_stop, time_step, extra_par
        )
        return str_code

    def cache_keys(self):
        return list(self.__cached_features.keys())

    def cache_size(self):
        return self.__cached_features.__sizeof__()

    def volume_density(self, start=None, stop=None, time_step=1, append=False):
        start, stop = self.assert_window(start, stop)
        str_code = self.__code_formatter("VOLDEN", start, stop, time_step, 1)
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code]
        df_slice = self.change_sample_interval(start, stop, time_step)
        vol_den = df_slice["Volume"] / df_slice["TickVol"]
        vol_den.replace([-np.inf, np.inf], np.nan).dropna(inplace=True)
        if append:
            self.__cached_features[str_code] = regular_vol_den.astype(int)
        return regular_vol_den.astype(int)

    def get_simple_MA(
        self, window, start=None, stop=None, time_step=1, append=False
    ):
        start, stop = self.assert_window(start, stop)
        str_code = self.__code_formatter("MA", start, stop, time_step, window)
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code]
        df_slice = self.change_sample_interval(start, stop, time_step)
        moving_avg = df_slice["Close"].rolling(window=window).mean()
        if append:
            self.__cached_features[str_code] = moving_avg.dropna()
        return moving_avg.dropna()

    def get_deviation(
        self, window, start=None, stop=None, time_step=1, append=False
    ):
        start, stop = self.assert_window(start, stop)
        str_code = self.__code_formatter("STD", start, stop, time_step, window)
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code]
        df_slice = self.change_sample_interval(start, stop, time_step)
        moving_std = df_slice["Close"].rolling(window=window).std()
        if append:
            self.__cached_features[str_code] = moving_std.dropna()
        return moving_std.dropna()

    def get_RSI(
        self, window, start=None, stop=None, time_step=1, append=False
    ):
        start, stop = self.assert_window(start, stop)
        str_code = self.__code_formatter("RSI", start, stop, time_step, window)
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code]
        df_slice = self.change_sample_interval(start, stop, time_step)
        next_df = df_slice["Close"].shift(periods=1)
        rsi_df = pd.DataFrame(
            columns=[
                "Delta",
                "Gain",
                "Loss",
                "AvgGain",
                "AvgLoss",
                "RS",
                "RSI" + str(window),
            ]
        )
        rsi_df["Delta"] = df_slice["Close"] - next_df
        rsi_df["Gain"] = rsi_df["Delta"].apply(lambda x: 0 if x < 0 else x)
        rsi_df["Loss"] = rsi_df["Delta"].apply(lambda x: 0 if x > 0 else -x)
        rsi_df["AvgGain"] = (
            rsi_df["Gain"].rolling(window=window).mean(skipna=True)
        )
        rsi_df["AvgLoss"] = (
            rsi_df["Loss"].rolling(window=window).mean(skipna=True)
        )
        rsi_df["RS"] = rsi_df["AvgGain"].div(rsi_df["AvgLoss"])
        rsi_df["RSI" + str(window)] = rsi_df["RS"].apply(
            lambda x: 100 - 100 / (1 + x)
        )
        if append:
            self.__cached_features[str_code] = rsi_df[
                "RSI" + str(window)
            ].dropna()
        return rsi_df["RSI" + str(window)].dropna()

    def autocorr_period(
        self, shift, start=None, end=None, time_step=1, append=False
    ):
        """
        Compute auto-correlation function in
        a time interval which fix the window

        Parameters
        ----------
        `start` : ``pandas.Timestamp``
            first date/minute of the period
        `end` : ``pandas.Timestamp``
            end of the period
        `shift` : ``int``
            displacement to separate the two data samples in minutes

        """
        start, stop = self.assert_window(start, stop)
        str_code = self.__code_formatter(
            "AUTOCORR", start, stop, time_step, shift
        )
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code]
        df_slice = self.change_sample_interval(start, stop, time_step)
        df_close = df_slice["Close"]
        if shift > df_close.shape[0] - 1:
            raise ValueError(
                "Period enclosed from {} to {} provided {} "
                "data points, while {} shift was required".format(
                    start, end, df_close.shape[0], shift
                )
            )
        autocorr = df_close.autocorr(lag=shift)
        if append:
            self.__cached_features[str_code] = autocorr.dropna()
        return autocorr.dropna()

    def autocorr_period_matrix(self, starts, ends, shift, append=False):
        """
        Compute auto-correlation function in many time intervals
        for various values of start and end

        Parameters
        ----------
        `start` : ``list pandas.Timestamp``
            list of first date/minute of the period
        `end` : ``list of pandas.Timestamp``
            list of end of the period
        `shift` : ``int``
            displacement to separate the two data samples in minutes

        """
        n_starts = len(starts)
        n_ends = len(ends)
        autocorr = np.empty([n_starts, n_ends])
        invalid_values = False
        for i in range(n_starts):
            for j in range(n_ends):
                try:
                    autocorr[i, j] = self.autocorr_period(
                        starts[i], ends[j], shift, append
                    )
                except:
                    invalid_values = True
                    autocorr[i, j] = np.nan
        if invalid_values:
            print("Some invalid periods (end > start) occurred.")
        autocorr_df = pd.DataFrame(autocorr, columns=ends, index=starts)
        autocorr_df.index.name = "start_dates"
        return autocorr_df

    def moving_autocorr(
        self, window, shift, start=None, stop=None, time_step=1, append=False
    ):
        """
        Compute auto-correlation in a moving window along the dataframe

        Parameters
        ----------
        `window` : ``int``
            size of moving window
        `shift` : ``int``
            displacement to separate the two data samples in moving window
        `append` : ``bool`` (default False)
            Whether to append resulting series in self.df_curated
        `start` :``pd.Timestamp`` or ``int`` (optional)
            First index/date. Default is the beginning of dataframe
        `end` :``pd.Timestamp`` or ``int`` (optional)
            Last index/date. Default is the end of dataframe

        """
        start, stop = self.assert_window(start, stop)
        str_code = self.__code_formatter(
            "MOV_AUTOCORR",
            start,
            stop,
            time_step,
            "{}_{}".format(window, shift),
        )
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code]
        df_slice = self.change_sample_interval(start, stop, time_step)
        close_series = df_slice["Close"]
        moving_autocorr = close_series.rolling(window=window).apply(
            lambda x: x.autocorr(lag=shift), raw=False
        )
        if append:
            self.__cached_features[str_code] = moving_autocorr.dropna()
        return moving_autocorr.dropna()

    def __frac_diff_weights(self, d, tolerance, max_weights=1e8):
        """
        Compute weights of frac_diff binomial-expansion formula

        Parameters
        ----------
        `d` : ``float``
            order of fractional differentiation. Usually between 0 and 1
        `tolerance` : ``float``
            minumum accepted value for weights to compute in series
        `max_weights` : ``int``
            max number of weights (to avoid excessive memory consumption)

        Return
        ------
        ``numpy.array``
            wegiths/coefficients of binomial series expansion

        """
        w_array = np.empty(100)
        w_array[0] = 1.0
        flag = -1
        last_i = 0
        while flag < 0:
            flag = numba_weights(d, tolerance, w_array, w_array.size, last_i)
            if w_array.size > max_weights:
                print(
                    "WARNING : could not achieved required weights "
                    "accuracy in frac_diff. Last weight = {}".format(
                        w_array[-1]
                    )
                )
                return w_array
            if flag < 0:
                last_i = w_array.size - 1
                w_array = np.concatenate([w_array, np.empty(10 * last_i)])
        return w_array[:flag]

    def __apply_weights(self, weights, x_vector):
        return np.dot(weights[::-1], x_vector)

    def frac_diff(
        self,
        d,
        start=None,
        stop=None,
        time_step=1,
        append=False,
        weights_tol=1e-5,
    ):
        """
        Compute fractional differentiation of a series with the binomial
        expansion formula for an arbitrary derivative order. Uses the
        close value of the dataframe.

        Parameters
        ----------
        `d` : ``float``
            derivative order (d = 1 implies daily returns = lose all memory)
        `weights_tol` : `` float `` (default 10^-5)
            minimum value for a weight in the binomial series expansion
            to apply a cutoff
        `append` : `` bool `` (default False)
            To append or not in self.df_curated data-frame

        """
        start, stop = self.assert_window(start, stop)
        str_code = self.__code_formatter(
            "FRAC_DIFF", start, stop, time_step, d
        )
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code]
        df_slice = self.change_sample_interval(start, stop, time_step)
        w = self.__frac_diff_weights(d, weights_tol)
        close_series = df_slice["Close"]
        fracdiff_series = close_series.rolling(window=w.size).apply(
            lambda x: self.__apply_weights(w, x), raw=True
        )
        if append:
            self.__cached_features[str_code] = fracdiff_series.dropna()
        return fracdiff_series.dropna()

    def adf_test(self, frac_diff):
        adf = adfuller(frac_diff, maxlag=1, regression="c", autolag=None)
        return adf

    def vola_freq(
        self, window, start=None, stop=None, time_step=1, append=False
    ):
        """
        Introduce/compute a measure of volatility based on how rapidly
        the stock prices are oscillating around the average or if the
        series presents a narrow peak. Use the Fourier space representation
        of the time series to compute the mean quadratic frequency of
        the spectrum. Note that in case the mean frequency is zero this
        is just the frequency standard deviation.

        Parameters
        ----------
        `window` : `` int ``
            number of data points to consider in FFT

        Return
        ------
        `` pandas.Series ``
            Each Timestamp index contains the mean quadratic frequency.

        """
        start, stop = self.assert_window(start, stop)
        str_code = self.__code_formatter(
            "VOLA_FREQ", start, stop, time_step, window
        )
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code]
        df_slice = self.change_sample_interval(start, stop, time_step)
        money_exch = df_slice["Close"] * df_slice["Volume"]
        vol_series = money_exch.rolling(window=window).apply(
            lambda x: mean_quadratic_freq(x)
        )
        if append:
            self.__cached_features[str_code] = vol_series.dropna()
        return vol_series.dropna()

    def vola_amplitude(
        self, window, start=None, stop=None, time_step=1, append=False
    ):
        """
        Compute the best possible trading in a interval of the
        previous `window` data bars. Use the difference of the
        max and min values of the stock price divided by the
        moving average.

        Parameters
        ----------
        `window` : `` int ``
            number of data points to consider

        Return
        ------
        `` pandas.Series ``
            Each Timestamp index contains the max gain ratio of
            the previous `window` data bars.

        """
        start, stop = self.assert_window(start, stop)
        str_code = self.__code_formatter(
            "VOLA_AMPL", start, stop, time_step, window
        )
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code]
        df_slice = self.change_sample_interval(start, stop, time_step)
        max_price = df_slice["High"].rolling(window=window).max()
        min_price = df_slice["Low"].rolling(window=window).min()
        ma = self.get_simple_MA(window)
        vol_series = (max_price - min_price) / ma
        if append:
            self.__cached_features[str_code] = vol_series.dropna()
        return vol_series.dropna()

from math import sqrt

import numpy as np
import pandas as pd
from numba import float64, int32, njit, prange
from statsmodels.tsa.stattools import adfuller

from mvp import numba_stats
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
    """
    Use `data` Fourier transform to compute mean quadratic freq of the spectrum

    Parameters
    ----------
    `data` : ``numpy.array``
        data series in time basis
    `time_spacing` : ``float``
        sample time spacing usually in minutes

    Return
    ------
    ``float``

    """
    displace_data = data - data.mean()
    fft_weights = np.fft.fft(displace_data)
    freq = np.fft.fftfreq(displace_data.size, time_spacing)
    weights_norm = np.abs(fft_weights).sum()
    return sqrt((freq * freq * np.abs(fft_weights)).sum() / weights_norm)


class RefinedData(RawData):
    """
    Integrate to the raw data with open-high-low-close values
    some simple statistical features which provide more tools
    to analyze data and support models

    Inherit
    -------
    ``mvp.rawdata.RawData``
        class to load data from database and sample it in different formats

    """

    def __init__(
        self,
        symbol,
        db_path,
        preload={"time": [5, 10, 15, 30, 60, "day"]},
        requested_features={},
        start=None,
        stop=None,
        time_step=1,
    ):
        """
        Initialize the class reading data from database. If only `symbol`
        and `db_path` are given simply initialize the RawData class from
        inheritance. Some common features may be computed in initialization
        passing some of the optional parametersi. These features computed
        in initialization are NECESSARILY set in cache memory

        Parameters
        ----------
        `symbol` : ``str``
            symbol code of the company to load data
        `db_path` : ``str``
            full path to database file
        `preload` : ``dict``
            dictionary to inform dataframes set in cache memory
            {
                "time": list[``int`` / "day"]   (new time interval of bars)
                "tick": list[``int``]       (bars in amount of deals occurred)
                "volume": list[``int``]     (bars in amount of volume)
                "money": list[``int``]      (bars in amount of money)
            }
        `requested_features` : ``dict``
            Dictionary codifying some features to compute in initialization
            {
                "MA": ``int``         (window size)
                "DEV": ``int``        (window size)
                "RSI": ``int``        (window size)
                "FRACDIFF": ``float`` (differentiation order between 0 and 1)
            }
        `start` : ``pd.Timestamp`` or ``int``
            index of Dataframe to start in computation of requested features
        `stop` : ``pd.Timestamp`` or ``int``
            index of Dataframe to stop in computation of requested features
        `time_step` : ``int`` or "day"
            define the sample time interval to compute requested features

        """
        RawData.__init__(self, symbol, db_path, preload)
        self.__attr = {
            "MA": "get_simple_MA",
            "DEV": "get_deviation",
            "RSI": "get_RSI",
            "FRACDIFF": "frac_diff",
        }
        self.__cached_features = {}
        self.volume_density(append=True)
        for feature in requested_features.keys():
            if feature not in self.__attr.keys():
                continue
            if isinstance(requested_features[feature], list):
                parameters_list = requested_features[feature]
                for parameter in parameters_list:
                    try:
                        self.__getattribute__(self.__attr[feature])(
                            parameter, start, stop, time_step, True
                        )
                    except Exception as err:
                        print(err, ": param {} given".format(parameter))
            else:
                parameter = requested_features[feature]
                try:
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

    def cache_features_keys(self):
        """ internal keys used to cached features. Information purposes """
        return list(self.__cached_features.keys())

    def cache_features_size(self):
        """ Total size of cached features in bytes """
        full_size = 0
        for feature_data in self.__cached_features.values():
            full_size = full_size + feature_data.__sizeof__()
        return full_size

    def cache_clean(self):
        """ delete all cached memory """
        del self.__cached_features
        self.__cached_features = {}

    def volume_density(self, start=None, stop=None, time_step=1, append=False):
        """
        Return the average volume exchange per tick/deal
        """
        start, stop = self.assert_window(start, stop)
        str_code = self.__code_formatter("VOLDEN", start, stop, time_step, 1)
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code].copy()
        df_slice = self.change_sample_interval(start, stop, time_step)
        vol_den = df_slice["Volume"] / df_slice["TickVol"]
        clean_data = vol_den.replace([-np.inf, np.inf], np.nan).copy()
        clean_data.dropna(inplace=True)
        clean_data.name = "DEAL_DEN"
        if append:
            self.__cached_features[str_code] = clean_data.astype(int)
        return clean_data.astype(int)

    def get_simple_MA(
        self, window, start=None, stop=None, time_step=1, append=False
    ):
        """
        Return simple moving average time series of close price
        """
        start, stop = self.assert_window(start, stop)
        str_code = self.__code_formatter("MA", start, stop, time_step, window)
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code].copy()
        df_slice = self.change_sample_interval(start, stop, time_step)
        moving_avg = df_slice["Close"].rolling(window=window).mean()
        moving_avg.name = "MA"
        if append:
            self.__cached_features[str_code] = moving_avg.dropna()
        return moving_avg.dropna()

    def get_deviation(
        self, window, start=None, stop=None, time_step=1, append=False
    ):
        """
        Return moving standard deviation time series of close price
        """
        start, stop = self.assert_window(start, stop)
        str_code = self.__code_formatter("DEV", start, stop, time_step, window)
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code].copy()
        df_slice = self.change_sample_interval(start, stop, time_step)
        moving_std = df_slice["Close"].rolling(window=window).std()
        moving_std.name = "DEV"
        if append:
            self.__cached_features[str_code] = moving_std.dropna()
        return moving_std.dropna()

    def get_returns(
        self, window=1, start=None, stop=None, time_step=1, append=False
    ):
        """
        Returns the return series time series of close price
        """
        start, stop = self.assert_window(start, stop)
        str_code = self.__code_formatter("RET", start, stop, time_step, window)
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code].copy()
        df_slice = self.change_sample_interval(start, stop, time_step)
        return_df = (
            df_slice["Close"] - df_slice["Close"].shift(window)
        ) / df_slice["Close"].shift(window)
        return_df.name = "RET"
        if append:
            self.__cached_features[str_code] = return_df.dropna()
        return return_df.dropna()

    def get_RSI(
        self, window, start=None, stop=None, time_step=1, append=False
    ):
        """
        Return moving Relative Strength Index time series of close price

        Warning
        ---
        In order to return the same size convention of other moving
        quantities, the first value is set to zero in returns series

        """
        start, stop = self.assert_window(start, stop)
        str_code = self.__code_formatter("RSI", start, stop, time_step, window)
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code].copy()
        df_slice = self.change_sample_interval(start, stop, time_step)
        return_series = df_slice.Close - df_slice.Close.shift(periods=1)
        gain_or_zero = return_series.apply(lambda x: 0 if x < 0 else x)
        loss_or_zero = return_series.apply(lambda x: 0 if x > 0 else -x)
        ratio = (
            gain_or_zero.rolling(window=window).mean().dropna()
            / loss_or_zero.rolling(window=window).mean().dropna()
        )
        rsi_series = ratio.apply(lambda x: 100 - 100 / (1 + x))
        rsi_series.name = "RSI"
        if append:
            self.__cached_features[str_code] = rsi_series
        return rsi_series

    def autocorr_period(
        self, shift, start=None, stop=None, time_step=1, append=False
    ):
        """
        Compute auto-correlation function in a time period
        According to pandas the autocorrelation is computed as follows:
        Given a set {y1, y2, ... , yN} divide it in {y1, y2, ..., yN-s}
        and {ys, ys+1, ..., yN}, compute the average for each set and
        than compute the covariance between the two sets.

        Parameters
        ----------
        `shift` : ``int``
            displacement to separate the two data samples
        `start` : ``pandas.Timestamp`` or ``int``
            first dataframe index of the period
        `stop` : ``pandas.Timestamp`` or ``int``
            last dataframe index of the period
        `time_step` : ``int``
            available ones in `RefinedData.available_time_steps`
        `append` : ``bool``
            whether to include in class cache memory each entry

        Return
        ------
        ``float``

        """
        start, stop = self.assert_window(start, stop)
        str_code = self.__code_formatter(
            "AUTOCORR", start, stop, time_step, shift
        )
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code]
        df_slice = self.change_sample_interval(start, stop, time_step)
        df_close = df_slice["Close"]
        if shift >= df_close.size:
            raise ValueError(
                "Period enclosed from {} to {} provided {} "
                "data points, while {} shift was required".format(
                    start, stop, df_close.size, shift
                )
            )
        autocorr = df_close.autocorr(lag=shift)
        if append:
            self.__cached_features[str_code] = autocorr
        return autocorr

    def autocorr_period_matrix(
        self, shift, starts, stops, time_step=1, append=False
    ):
        """
        Compute auto-correlation function for multiple time intervals
        For each start and stop call `RefinedData.autocorr_period`

        Parameters
        ----------
        `shift` : ``int``
            displacement to separate the two data samples
        `starts` : ``list`` of ``pandas.Timestamp``
            list of first date/minute of the period
        `stops` : ``list`` of ``pandas.Timestamp``
            list of end of the period
        `time_step` : ``int`` or "day"
            values in RefinedData.available_time_steps
        `append` : ``bool``
            whether to append in class cache memory for every window formed

        Return
        ------
        ``pandas.DataFrame``
            matrix of autocorrelation indexed by `starts` and `stops`

        """
        n_starts = len(starts)
        n_stops = len(stops)
        autocorr = np.empty([n_starts, n_stops])
        for i in range(n_starts):
            for j in range(n_stops):
                try:
                    autocorr[i, j] = self.autocorr_period(
                        shift, starts[i], stops[j], time_step, append
                    )
                except Exception:
                    autocorr[i, j] = np.nan
        autocorr_df = pd.DataFrame(autocorr, columns=stops, index=starts)
        autocorr_df.index.name = "start_dates"
        return autocorr_df

    def moving_autocorr(
        self, window, shift, start=None, stop=None, time_step=1, append=False
    ):
        """
        Compute auto-correlation in a moving `window` using close price

        Parameters
        ----------
        `window` : ``int``
            size of moving window. Must be larger than `shift`
        `shift` : ``int``
            displacement to separate the two data samples in moving window
        `start` : ``pd.Timestamp`` or ``int``
            First index/date. Default is the beginning of dataframe
        `stop` : ``pd.Timestamp`` or ``int``
            Last index/date. Default is the end of dataframe
        `append` : ``bool``
            whether to append in class cache or not

        Return
        ------
        ``pandas.Series``

        """
        if shift >= window:
            raise ValueError(
                "The shift between two data sets {} must be smaller "
                "than the moving window {}".format(shift, window)
            )
        start, stop = self.assert_window(start, stop)
        str_code = self.__code_formatter(
            "MOV_AUTOCORR",
            start,
            stop,
            time_step,
            "{}_{}".format(window, shift),
        )
        if str_code in self.__cached_features.keys():
            return self.__cached_features[str_code].copy()
        df_slice = self.change_sample_interval(start, stop, time_step)
        close_prices = df_slice["Close"].values
        if close_prices.size < window:
            raise ValueError(
                "The number of data points between {} and {} "
                "is {} while window of size {} was required".format(
                    start, stop, close_prices.size, window
                )
            )
        pts = close_prices.size
        corr_window = window - shift
        lag_series = close_prices[: pts - shift]
        adv_series = close_prices[shift:]
        mov_autocorr = np.zeros(pts - shift)
        numba_stats.moving_correlation(
            corr_window, lag_series, adv_series, mov_autocorr
        )
        core_data = mov_autocorr[corr_window - 1 :]
        mov_autocorr_ser = pd.Series(
            core_data, index=df_slice.index[window - 1 :]
        )
        mov_autocorr_ser.name = "MOV_AUTOCORR"
        if append:
            self.__cached_features[str_code] = mov_autocorr_ser
        return mov_autocorr_ser

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
        expansion formula for an arbitrary derivative order.

        Parameters
        ----------
        `d` : ``float``
            derivative order (d = 1 implies daily returns) lying in (0,1]
        `start` : ``pandas.Timestamp`` or ``int``
            first dataframe index of the period
        `stop` : ``pandas.Timestamp`` or ``int``
            last dataframe index of the period
        `time_step` : ``int``
            available ones in `RefinedData.available_time_steps`
        `append` : ``bool`` (default False)
            To append or not in cache memory
        `weights_tol` : ``float``
            minimum value for a weight in the binomial series expansion

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
        series presents a narrow peak. Use the Fourier representation
        of the time series to compute the mean quadratic frequency of
        the spectrum. Note that in case the mean frequency is zero this
        is just the frequency standard deviation.

        Parameters
        ----------
        `window` : `` int ``
            number of data points to consider in FFT
        `start` : ``pandas.Timestamp`` or ``int``
            first dataframe index of the period
        `stop` : ``pandas.Timestamp`` or ``int``
            last dataframe index of the period
        `time_step` : ``int``
            available ones in `RefinedData.available_time_steps`
        `append` : ``bool`` (default False)
            To append or not in cache memory

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
            lambda x: mean_quadratic_freq(x, time_step)
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
        ma = self.get_simple_MA(window, start, stop, time_step)
        vol_series = (max_price - min_price) / ma
        if append:
            self.__cached_features[str_code] = vol_series.dropna()
        return vol_series.dropna()

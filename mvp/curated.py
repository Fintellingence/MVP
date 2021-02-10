from functools import partial
from multiprocessing import Pool

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from numba import njit, prange, int32, float64


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


class CuratedData:
    """
    Integrate to the raw data with open-high-low-close values
    some simple statistical features which provide more tools
    to analyze the data and support primary models

    Parameters
    ----------
    `raw_data` : `` rawdata.RawData class``
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

    def __init__(self, raw_data, requested_features={}, daily=False):
        self.symbol = raw_data.symbol
        if daily:
            self.df_curated = raw_data.daily_bars()
        else:
            self.df_curated = raw_data.df.copy()
        self.available_dates = raw_data.available_dates
        self.__volume_density()
        self.features_attr = {
            "MA": "get_simple_MA",
            "DEV": "get_deviation",
            "RSI": "get_RSI",
            "FRAC_DIFF": "frac_diff",
        }
        self.parameters = {}
        for feature in requested_features.keys():
            if feature not in self.features_attr.keys():
                continue
            self.parameters[feature] = []
            if type(requested_features[feature]) is list:
                feature_parameters = requested_features[feature]
                for parameter in feature_parameters:
                    try:
                        self.__getattribute__(self.features_attr[feature])(
                            parameter, append=True
                        )
                    except ValueError as err:
                        print(err, ": {} given".format(parameter))
            else:
                parameter = requested_features[feature]
                try:
                    self.__getattribute__(self.features_attr[feature])(
                        parameter, append=True
                    )
                except ValueError as err:
                    print(err, "{} given".format(parameter))

    def __volume_density(self):
        vol_den = self.df_curated["Volume"] / self.df_curated["TickVol"]
        self.df_curated["VolDen"] = vol_den.dropna().astype(int)

    def get_simple_MA(self, window, append=False):
        moving_avg = self.df_curated["Close"].rolling(window=window).mean()
        if not append:
            return moving_avg.dropna()
        if "MA" not in self.parameters.keys():
            self.parameters["MA"] = []
        if window not in self.parameters["MA"]:
            self.df_curated["MA_{}".format(window)] = moving_avg
            self.parameters["MA"].append(window)

    def get_deviation(self, window, append=False):
        moving_std = self.df_curated["Close"].rolling(window=window).std()
        if not append:
            return moving_std.dropna()
        if "DEV" not in self.parameters.keys():
            self.parameters["DEV"] = []
        if window not in self.parameters["DEV"]:
            self.df_curated["DEV_{}".format(window)] = moving_std
            self.parameters["DEV"].append(window)

    def get_RSI(self, param_RSI, append=False):
        next_df = self.df_curated["Close"].shift(periods=1)
        rsi_df = pd.DataFrame(
            columns=[
                "Delta",
                "Gain",
                "Loss",
                "AvgGain",
                "AvgLoss",
                "RS",
                "RSI" + str(param_RSI),
            ]
        )
        rsi_df["Delta"] = self.df_curated["Close"] - next_df
        rsi_df["Gain"] = rsi_df["Delta"].apply(lambda x: 0 if x < 0 else x)
        rsi_df["Loss"] = rsi_df["Delta"].apply(lambda x: 0 if x > 0 else -x)
        rsi_df["AvgGain"] = (
            rsi_df["Gain"].rolling(window=param_RSI).mean(skipna=True)
        )
        rsi_df["AvgLoss"] = (
            rsi_df["Loss"].rolling(window=param_RSI).mean(skipna=True)
        )
        rsi_df["RS"] = rsi_df["AvgGain"].div(rsi_df["AvgLoss"])
        rsi_df["RSI" + str(param_RSI)] = rsi_df["RS"].apply(
            lambda x: 100 - 100 / (1 + x)
        )
        if not append:
            return rsi_df["RSI" + str(param_RSI)].dropna()
        if "RSI" not in self.parameters.keys():
            self.parameters["RSI"] = []
        if param_RSI not in self.parameters["RSI"]:
            self.df_curated["RSI_{}".format(param_RSI)] = rsi_df[
                "RSI" + str(param_RSI)
            ]
            self.parameters["RSI"].append(param_RSI)

    def autocorr_period(self, start, end, shift):
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
        df_close_chunk = self.df_curated["Close"].loc[start:end]
        if shift > df_close_chunk.shape[0] - 1:
            raise ValueError(
                "Period enclosed from {} to {} provided {} "
                "data points, while {} shift was required".format(
                    start, end, df_close_chunk.shape[0], shift
                )
            )
        return df_close_chunk.autocorr(lag=shift)

    def autocorr_period_matrix(self, starts, ends, shift):
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
                        starts[i], ends[j], shift
                    )
                except:
                    invalid_values = True
                    autocorr[i, j] = np.nan
        if invalid_values:
            print("Some invalid periods (end > start) occurred.")
        autocorr_df = pd.DataFrame(autocorr, columns=ends, index=starts)
        autocorr_df.index.name = "start_dates"
        return autocorr_df

    def autocorr_tail(self, window, shift):
        """
        Compute auto-correlation function using the last(recent) data points

        Parameters
        ----------
        `window` : ``int``
            How many data points to take from bottom of dataframe
        `shift` : ``int``
            displacement to separate the two data samples

        """
        if shift > window - 1:
            raise ValueError("shift must be greater than window")
        df_close_chunk = self.df_curated["Close"].tail(window)
        return df_close_chunk.autocorr(lag=shift)

    def autocorr_tail_matrix(self, windows, shifts):
        """
        Compute auto-correlation function using the last(recent) data points
        for several `windows` and `shifts` provided in lists

        Parameters
        ----------
        `windows` : ``list``
            Various number of data points to take from bottom of dataframe
        `shifts` : ``list``
            Various displacement to separate the two data samples

        Return
        ------
        ``pandas.dataframe``
            windows in indexes and shifts in columns

        """
        n_windows = len(windows)
        n_shifts = len(shifts)
        autocorr = np.empty([n_windows, n_shifts])
        invalid_values = False
        for i in range(n_windows):
            for j in range(n_shifts):
                try:
                    autocorr[i, j] = self.autocorr_tail(windows[i], shifts[j])
                except:
                    invalid_values = True
                    autocorr[i, j] = np.nan
        if invalid_values:
            print("Some invalid entries for window and shift occurred")
        autocorr_df = pd.DataFrame(autocorr, columns=shifts, index=windows)
        autocorr_df.index.name = "window"
        return autocorr_df

    def moving_autocorr(self, window, shift, append=False, start=0, end=-1):
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
        if isinstance(start, int) and isinstance(end, int):
            close_series = self.df_curated["Close"].iloc[start:end]
        elif isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
            close_series = self.df_curated["Close"].loc[start:end]
        else:
            raise ValueError(
                "start/end index type {}/{} not valid".format(
                    type(start), type(end)
                )
            )
        if end == -1:
            end = self.df_curated.shape[0]

        moving_corr = close_series.rolling(window=window).apply(
            lambda x: x.autocorr(lag=shift), raw=False
        )
        if not append:
            return moving_corr.dropna()
        new_feature_name = "AUTOCORR_({},{},{},{})".format(
            start, end, window, shift
        )
        if "AUTOCORR" not in self.parameters.keys():
            self.parameters["AUTOCORR"] = []
        if (start, end, window, shift) not in self.parameters["AUTOCORR"]:
            self.df_curated[new_feature_name] = moving_corr
            self.parameters["AUTOCORR"].append((start, end, window, shift))

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

    def frac_diff(self, d, weights_tol=1e-5, append=False):
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
        w = self.__frac_diff_weights(d, weights_tol)
        close_series = self.df_curated["Close"]
        fracdiff_series = close_series.rolling(window=w.size).apply(
            lambda x: self.__apply_weights(w, x), raw=True
        )
        if not append:
            return fracdiff_series.dropna()
        if "FRAC_DIFF" not in self.parameters.keys():
            self.parameters["FRAC_DIFF"] = []
        if d not in self.parameters["FRAC_DIFF"]:
            self.df_curated["fracdiff_{}".format(d)] = fracdiff_series
            self.parameters["FRAC_DIFF"].append(d)

    def adf_test(self, frac_diff):
        adf = adfuller(frac_diff, maxlag=1, regression="c", autolag=None)
        return adf

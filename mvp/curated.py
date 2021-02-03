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
    which indicate that convergence was achieve accordint to `tolerance`

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
        with new weights

    Return
    ------
    ``int``
        If positive, convergence was achieved and the value is the number
        of weights computed. If negative, weights are above the `tolerance`
        provided, the weights array must be resized adding empty entries,
        and this function must be called again

    """
    for k in prange(last_index + 1, w_size):
        w_array[k] = -(w_array[k - 1] / k) * (d - k + 1)
        # k += 1
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
        "MA" = Moving Average
        "DEV" = standart DEViation
        "RSI" = RSI indicator
    `daily` : `` bool `` (optional)
        Automatically convert 1-minute raw data to daily data

    """

    def __init__(self, raw_data, requested_features, daily=False):
        self.symbol = raw_data.symbol
        if daily:
            self.df_curated = raw_data.daily_bars()
        else:
            self.df_curated = raw_data.df.copy()
        self.initial_features = {
            "MA": "get_simple_MA",
            "DEV": "get_deviation",
            "RSI": "get_RSI",
        }
        self.parameters = {}

        for feature in requested_features.keys():
            self.parameters[feature] = []
            if feature not in self.initial_features.keys():
                continue
            if type(requested_features[feature]) is list:
                feature_parameters = requested_features[feature]
                for parameter in feature_parameters:
                    try:
                        self.__getattribute__(self.initial_features[feature])(
                            parameter, append=True
                        )
                    except ValueError as err:
                        print(err, ": {} given".format(parameter))
            else:
                parameter = requested_features[feature]
                try:
                    self.__getattribute__(self.initial_features[feature])(
                        parameter, append=True
                    )
                except ValueError as err:
                    print(err, "{} given".format(parameter))

    def get_simple_MA(self, window, append=False):
        moving_avg = self.df_curated["Close"].rolling(window=window).mean()
        if not append:
            return moving_avg
        if "MA" not in self.parameters.keys():
            self.parameters["MA"] = []
        if window not in self.parameters["MA"]:
            self.df_curated["MA_{}".format(window)] = moving_avg
            self.parameters["MA"].append(window)

    def get_deviation(self, window, append=False):
        moving_std = self.df_curated["Close"].rolling(window=window).std()
        if not append:
            return moving_std
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
            return rsi_df["RSI" + str(param_RSI)]
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
            displacement to separate the two data samples

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

    def moving_autocorr(self, window, shift, append=False, start=None):
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
        `start` : wither ``pd.Timestamp`` or ``int``
            start in a location different from the first index

        """
        if start != None:
            if isinstance(start, pd.Timestamp):
                close_series = self.df_curated["Close"].loc[start:]
            elif isinstance(start, pd.Timestamp):
                close_series = self.df_curated["Close"].iloc[start:]
            else:
                raise ValueError(
                    "start index type {} not valid".format(type(start))
                )
        else:
            close_series = self.df_curated["Close"]

        moving_corr = close_series.rolling(window=window).apply(
            lambda x: x.autocorr(lag=shift), raw=False
        )
        if not append:
            return moving_corr
        new_feature_name = "AUTOCORR_{}_{}".format(window, shift)
        if "AUTOCORR" not in self.parameters.keys():
            self.parameters["AUTOCORR"] = []
        if [window, shift] not in self.parameters["AUTOCORR"]:
            self.df_curated[new_feature_name] = moving_corr
            self.parameters["AUTOCORR"].append([window, shift])

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
        l_star = w.size
        fracdiff_series = self.df_curated["Close"]
        fracdiff_series = fracdiff_series.rolling(window=l_star).apply(
            lambda x: self.__apply_weights(w, x), raw=True
        )
        if not append:
            return fracdiff_series.dropna()
        if "FRAC_DIFF" not in self.parameters.keys():
            self.parameters["FRAC_DIFF"] = []
        if d not in self.parameters["FRAC_DIFF"]:
            self.df_curated["fracdiff_{}".format(d)] = fracdiff_series.dropna()
            self.parameters["FRAC_DIFF"].append(d)

    def adf_test(self, frac_diff):
        adf = adfuller(frac_diff, maxlag=1, regression="c", autolag=None)
        return adf

    def __parallel_map_df(self, func, data, num_of_threads, chunk_size, **kwargs):
        """
        Apply `func` in linear distributed chunks of the data in `t_data[1]`
        using parallel processing.

        Parameters
        ----------
        `func` : ``Called``
            A function to be applied in along the data chunks. It must return a
            ``DataFrame``.
        `data` : ``[DataFrame, Series]``
            The data that will be divided in different chunks.
        ``num_of_threads : ``int``
            The number of threads that will process the chunks.
        ``chunk_size : ``int``
            The size of the chunk.
        ``**kwargs : ``dict``
            Addicional arguments that will be passed to the `func`.

        Return
        ------
        `df_out` : ``DataFrame``
            The ``DataFrame`` generated by the application of `func` in
            `t_data[1]` with the arguments in `**kwargs`.
        """
        def slice_data(chunk, data):
            chunk_idx = np.ceil(np.linspace(0, len(data), chunk)).astype(int)
            for i in range(1, chunk_idx.size):
                yield data[s]

        slicer = slice_data(chunk, data)
        partial_func = partial(func, **kwargs)
        with Pool(num_of_threads) as pool:
            output = [out for out in pool.imap_unordered(partial_func, slicer)]
        return output

    def interval_count_occurrence(self, bar_index, horizon, interval):
        """
        Determine the number of occurrences of horizons in `interval`.

        Parameters
        ----------
        `bar_index` : ``Index``
            The timestamp of the maximum value of the bars
        `horizon` : ``DataFrame``
            The start and end of each horizon
        `interval` : ``list``
            The timestamps that compose the interval od interest

        Return
        ------
        count : ``Series``
            The number of occurrence of the `interval` in all horizons
        """
        horizon = horizon.copy()
        horizon = horizon.loc[
            (horizon["start"] <= interval[-1])
            & (horizon["end"] >= interval[0])
        ]
        idx_of_interest = bar_index.searchsorted(
            [horizon["start"].min(), horizon["end"].max()]
        )
        count = pd.Series(
            0, index=bar_index[idx_of_interest[0] : idx_of_interest[1] + 1]
        )
        horizon_np = horizon.values
        for s, e in horizon_np:
            count.loc[s:e] += 1
        return count.iloc[insterval[0] : interval[-1]]

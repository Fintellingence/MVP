import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from numba import njit, prange, int32, float64


@njit(int32(float64, float64, float64[:], int32, int32))
def numba_weights(d, tolerance, w_array, w_size, last_index):
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

    def autocorr_calculations(self, window_autocorr, shift):
        slice_time_series_df = self.df_curated["Close"].tail(window_autocorr)
        autocorr_value = slice_time_series_df.autocorr(lag=shift)
        return autocorr_value

    def get_autocorr(self):
        try:
            self.parameters["AC_WINDOW"]
            self.parameters["AC_SHIFT_MAX"]

            self.autocorr_values = pd.concat(
                [
                    pd.DataFrame(
                        ["SHIFT_" + str(param_AC_SHIFT_MAX)],
                        columns=["SHIFTS"],
                    )
                    for param_AC_SHIFT_MAX in self.parameters["AC_SHIFT_MAX"]
                ],
                ignore_index=True,
            )

            for param_AC_WINDOW in self.parameters["AC_WINDOW"]:
                temp = []
                for param_AC_SHIFT_MAX in self.parameters["AC_SHIFT_MAX"]:
                    temp.append(
                        self.autocorr_calculations(
                            param_AC_WINDOW, param_AC_SHIFT_MAX
                        )
                    )
                temp_df = pd.DataFrame(
                    temp, columns=["WINDOW_" + str(param_AC_WINDOW)]
                )
                self.autocorr_values[
                    "WINDOW_" + str(param_AC_WINDOW)
                ] = temp_df
            return self.autocorr_values.drop(columns=["SHIFTS"])

        except:
            return None

    def __frac_diff_weights(self, d, tolerance, max_weights=1e8):
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
            derivative order (d = 1 implies daily returns)
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

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from numba import njit, prange, int32, float64


@njit(int32(float64, float64, float64[:], int32, int32))
def numba_weights(d, thresh, w_array, w_size, last_index):
    for k in prange(last_index + 1, w_size):
        w_array[k] = -(w_array[k - 1] / k) * (d - k + 1)
        # k += 1
        if abs(w_array[k]) < thresh:
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
            self.df_curated = raw_data.df
        self.initial_features = {
            "MA": "get_simple_MA",
            "DEV": "get_deviation",
            "RSI": "get_RSI",
        }
        self.parameters = requested_features

        for feature in requested_features.keys():
            if feature not in self.initial_features.keys():
                continue
            if type(requested_features[feature]) is list:
                feature_parameters = requested_features[feature]
                for parameter in feature_parameters:
                    try:
                        self.df_curated[
                            "{}_{}".format(feature, parameter)
                        ] = self.__getattribute__(
                            self.initial_features[feature]
                        )(
                            parameter
                        )
                    except ValueError as err:
                        print(err, ": {} given".format(parameter))
            else:
                parameter = requested_features[feature]
                try:
                    self.df_curated[
                        "{}_{}".format(feature, parameter)
                    ] = self.__getattribute__(self.initial_features[feature])(
                        parameter
                    )
                except ValueError as err:
                    print(err, "{} given".format(paramter))

    def get_simple_MA(self, param_MA):
        return self.df_curated["Close"].rolling(window=param_MA).mean()

    def get_deviation(self, param_DEV):
        return self.df_curated["Close"].rolling(window=param_DEV).std()

    def get_RSI(self, param_RSI):
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
        return rsi_df["RSI" + str(param_RSI)]

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

    def new_get_weights(self, d, thresh, max_weights=1e7):
        w_array = np.empty(100)
        w_array[0] = 1.0
        flag = -1
        last_i = 0
        while flag < 0:
            flag = numba_weights(d, thresh, w_array, w_array.size, last_i)
            if flag < 0:
                last_i = w_array.size - 1
                w_array = np.concatenate([w_array, np.empty(10 * last_i)])
        return w_array[:flag]

    def get_weights(self, d, thresh):
        w = [1.0]
        k = 1
        while abs(w[-1]) > thresh:
            w_ = -(w[-1] / k) * (d - k + 1)
            w.append(w_)
            k += 1
        return w

    def apply_weights(self, weights, x_vector):
        return np.dot(weights[::-1], x_vector)

    def frac_diff(self, d, thresh, improve=False):
        if improve:
            w = self.new_get_weights(d, thresh)
        else:
            w = self.get_weights(d, thresh)
        l_star = len(w)
        fracdiff_series = self.df_curated["Close"]
        fracdiff_series = fracdiff_series.rolling(window=l_star).apply(
            lambda x: self.apply_weights(w, x), raw=True
        )
        return fracdiff_series.dropna()

    def adf_test(self, frac_diff):
        adf = adfuller(frac_diff, maxlag=1, regression="c", autolag=None)
        return adf

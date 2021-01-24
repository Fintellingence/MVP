import pandas as pd
import numpy as np


class CuratedData:
    # In this model, data is provided as a class.
    def __init__(self, data, parameters, daily=False):
        self.symbol = data.symbol
        if daily:
            self.df_curated = data.daily_bars()
        if not daily:
            self.df_curated = data.df

        self.parameters = parameters
        # =========================================
        # Statistics #
        # =========================================
        try:
            self.parameters["MA"]
            for param_MA in self.parameters["MA"]:
                self.df_curated["MA" + str(param_MA)] = self.get_simple_MA(
                    param_MA
                )
        except:
            pass

        try:
            self.parameters["DEV"]
            for param_dev in self.parameters["DEV"]:
                self.df_curated["DEV" + str(param_dev)] = self.get_deviation(
                    param_dev
                )
        except:
            pass

        try:
            self.parameters["RSI"]
            for param_RSI in self.parameters["RSI"]:
                self.df_curated["RSI" + str(param_RSI)] = self.get_RSI(
                    param_RSI
                )
        except:
            pass

        try:
            self.parameters["AC_WINDOW"]
            self.parameters["AC_SHIFT"]

            self.autocorr_values = pd.concat(
                [
                    pd.DataFrame(
                        ["SHIFT_" + str(param_AC_SHIFT)], columns=["SHIFTS"]
                    )
                    for param_AC_SHIFT in self.parameters["AC_SHIFT"]
                ],
                ignore_index=True,
            )

            for param_AC_WINDOW in parameters["AC_WINDOW"]:
                temp = []
                for param_AC_SHIFT in parameters["AC_SHIFT"]:
                    temp.append(
                        self.get_autocorr(param_AC_WINDOW, param_AC_SHIFT)
                    )
                temp_df = pd.DataFrame(
                    temp, columns=["WINDOW_" + str(param_AC_WINDOW)]
                )
                self.autocorr_values[
                    "WINDOW_" + str(param_AC_WINDOW)
                ] = temp_df

        except:
            pass

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

    def get_autocorr(self, window_autocorr, shift):
        slice_time_series_df = self.df_curated["Close"].tail(window_autocorr)
        autocorr_value = slice_time_series_df.autocorr(lag=shift)
        return autocorr_value

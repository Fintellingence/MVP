import pandas as pd


class CuratedData:
    # In this model, data is provided as a class.
    def __init__(self, data, parameters):
        self.ticker = data.ticker
        self.df_curated = data.df
        self.parameters = parameters
        # =========================================
        # Statistics #
        # =========================================
        for param_MA in self.parameters["MA"]:
            self.df_curated["MA" + str(param_MA)] = self.get_simple_MA(param_MA)

        for param_dev in self.parameters["DEV"]:
            self.df_curated["DEV" + str(param_dev)] = self.get_deviation(param_dev)

        for param_RSI in self.parameters["RSI"]:
            self.df_curated["RSI" + str(param_RSI)] = self.get_RSI(param_RSI)

    def get_simple_MA(self, param_MA):
        return self.df_curated["Close"].rolling(window=param_MA).mean()

    def get_deviation(self, param_DEV):
        return self.df_curated["Close"].rolling(window=param_DEV).std()

    def get_RSI(self, param_RSI):
        next_df = self.df_curated["Close"].shift(periods=1)
        delta_df = self.df_curated["Close"] - next_df
        return delta_df

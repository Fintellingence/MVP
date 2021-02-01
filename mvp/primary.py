import pandas as pd
import mvp
import numpy as np


class PrimaryModel:
    def __init__(self, raw_data, model_type, parameters):

        self.model_type = model_type
        self.model_parameters = parameters["ModelParameters"]
        self.operation_parameters = parameters["OperationParameters"]
        self.feature_data = mvp.curated.CuratedData(
            raw_data, parameters["ModelParameters"]
        )

    def sign(self, x):
        if x >= 0:
            return 1
        else:
            return -1

    def events_crossing_MA(self):
        MA_params = list(set(self.model_parameters["MA"]))
        if len(MA_params) > 2:
            print(
                "[+] Aborting strategy: Number of parameters exceeded maximum of two:"
                + print(MA_params)
            )
            return None
        delta_MA = pd.DataFrame()
        delta_MA["Delta"] = (
            self.feature_data.df_curated["MA_" + str(MA_params[0])]
            - self.feature_data.df_curated["MA_" + str(MA_params[1])]
        ).dropna()
        events_MA = delta_MA["Delta"].apply(self.sign)
        events_MA = (
            events_MA.rolling(window=2)
            .apply(
                lambda x: -1
                if x[0] > x[1]
                else (1 if x[0] < x[1] else np.nan),
                raw=True,
            )
            .dropna()
        )
        return events_MA

    def events_classical_filter(self):
        pass

    def events_bollinger(self):
        pass

    def get_labels_standard(self, events):
        stop_loss = self.operation_parameters["SL"]
        take_profit = self.operation_parameters["TP"]
        horizon = self.operation_parameters["IH"]

        event_date_time = events.keys()
        close_data = self.feature_data.df_curated["Close"].reset_index()
        for event in event_date_time[:2]:
            signal_position = close_data.index[
                close_data["DateTime"] == event
            ][0]
            horizon_data = (
                close_data.iloc[signal_position : signal_position + horizon]
                .reset_index()
                .drop(columns="index")
            )
            profit_touch = pd.DataFrame()
            loss_touch = pd.DataFrame()
            if event == 1:
                profit_touch = horizon_data[
                    horizon_data["Close"].gt(
                        horizon_data["Close"][0] + take_profit
                    )
                ]
                print(horizon_data["Close"][0])
                loss_touch = horizon_data[
                    horizon_data["Close"].lt(
                        horizon_data["Close"][0] + stop_loss
                    )
                ]
            print(profit_touch)
            print(loss_touch)
        pass

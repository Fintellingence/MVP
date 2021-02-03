import pandas as pd
import mvp
import numpy as np
import datetime as dt


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

    def horizon_data(self, event_datetime):
        horizon = self.operation_parameters["IH"]
        close_data = self.feature_data.df_curated["Close"].reset_index()
        signal_position = close_data[
            close_data["DateTime"] == event_datetime
        ].index[0]
        horizon_data = (
            close_data.iloc[signal_position : signal_position + horizon]
            .reset_index()
            .drop(columns="index")
        )
        return horizon_data

    def touches(self, horizon_df):
        stop_loss = self.operation_parameters["SL"]
        take_profit = self.operation_parameters["TP"]
        profit_touch = None
        loss_touch = None
        entry_position = horizon_df.iloc[0]["Close"]
        profits = horizon_df[
            horizon_df["Close"].gt(entry_position + take_profit)
        ]
        losses = horizon_df[horizon_df["Close"].lt(entry_position - stop_loss)]
        if not profits.empty:
            profit_touch = profits.iloc[0]["DateTime"]
        if not losses.empty:
            loss_touch = losses.iloc[0]["DateTime"]
        touches = list(
            map(
                lambda x: dt.datetime(3000, 1, 1, 0, 0, 0) if x == None else x,
                [profit_touch, loss_touch],
            ),
        )
        return touches

    def labels(self, touches, event_trigger):
        if touches[0] == touches[1]:
            return 0
        else:
            if touches[0] < touches[1]:
                return event_trigger
            else:
                return -event_trigger

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
        return pd.DataFrame(events_MA).reset_index()

    def events_classical_filter(self):
        pass

    def events_bollinger(self):
        pass

    def event_labels(self, events_df):
        labels = []
        for event in events_df.values:
            event_datetime = event[0]
            event_trigger = event[1]
            horizon_df = self.horizon_data(event_datetime)
            labels.append(self.labels(self.touches(horizon_df), event_trigger))
        return labels
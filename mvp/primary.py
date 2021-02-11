import pandas as pd
import mvp
import numpy as np
import datetime as dt


class PrimaryModel:
    """
    Define a dataframe containing events of buy/sell recommentadions, also provide a way of labelling these events.
    This class uses curated data to process two models:

    - Crossing Averages Model
    - Bollinger Bands Model

    Parameters
    ----------
    raw_data : ``RawData Object``
        A RawData object from the file rawdata.py
    model_type : ``str``
        Three available types: ``crossing-MA``, `bollinger-bands``, ``classical-filter``
    parameters : ``dict```
        A dict containing two keys: `ModelParameters` and `OperationParameters`
        Inside `ModelParameters` key we have to provide another dict in one of three available options:
            For Crossing Averages Model it should be like:
                {'MA':[500,1000]}
            For Bollinger Bands Model it should be like:
                {'MA':[500],'DEV':[20],'K_value':2}
            For Classical Model it should be like
            {'threshold': 0.01}
        Inside `OperationParameters` key we have to provide another dict containing three values:
            - StopLoss (SL)
            - TakeProfit (TP)
            - InvestmentHorizon (IH)
        These values should be provided like the following:
            {'SL': 0.01, 'TP': 0.01, 'IH': 1000}}

    """

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

    def states_condition(self, x):
        x = x.tolist()
        close = x[0]
        plusK = x[1]
        minusK = x[2]

        if close < plusK and close > minusK:
            return 0
        if close >= plusK:
            return 1
        if close <= minusK:
            return -1

        return np.nan

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
        return (
            pd.DataFrame(events_MA)
            .rename(columns={"Delta": "Trigger"})
            .reset_index()
        )

    def events_classical_filter(self, threshold=0.1):
        close_data = self.feature_data.df_curated[["Close"]]
        close_data["Min"] = close_data[
            (close_data["Close"].shift(1) > close_data["Close"])
            & (close_data["Close"].shift(-1) > close_data["Close"])
        ]["Close"]
        close_data["Max"] = close_data[
            (close_data["Close"].shift(1) < close_data["Close"])
            & (close_data["Close"].shift(-1) < close_data["Close"])
        ]["Close"]
        saddle_events = (
            (close_data[close_data["Max"] >= 0].index)
            .append(close_data[close_data["Min"] >= 0].index)
            .sort_values()
        )
        if close_data.loc[saddle_events[0]]["Max"] != np.nan:
            side = -1
        else:
            side = 1
        events_CF = []
        for start, end in list(zip(saddle_events, saddle_events[1:])):
            inter_saddle_df = close_data[start:end]
            inter_saddle_df["Returns"] = (
                inter_saddle_df["Close"] / inter_saddle_df["Close"].shift(1)
                - 1
            )
            inter_saddle_df["Cusum"] = 0
            if side == 1:
                for i in range(1, len(inter_saddle_df["Cusum"].values)):
                    inter_saddle_df["Cusum"].iloc[i] = max(
                        (
                            inter_saddle_df["Cusum"].iloc[i - 1]
                            + inter_saddle_df["Returns"].iloc[i]
                        ),
                        0,
                    )
                occurances = inter_saddle_df[
                    inter_saddle_df["Cusum"] > threshold
                ]
            if side == -1:
                for i in range(1, len(inter_saddle_df["Cusum"].values)):
                    inter_saddle_df["Cusum"].iloc[i] = -max(
                        -(
                            inter_saddle_df["Cusum"].iloc[i - 1]
                            + inter_saddle_df["Returns"].iloc[i]
                        ),
                        0,
                    )
                occurances = inter_saddle_df[
                    inter_saddle_df["Cusum"] < -threshold
                ]
            if not occurances.empty:
                event = [occurances.index[0], side]
                events_CF.append(event)
            side = -side

        return pd.DataFrame(
            events_CF, columns=["DateTime", "Trigger"]
        ).set_index(["DateTime"])

    def events_bollinger(self):
        MA_param = self.model_parameters["MA"]
        if len(MA_param) > 1:
            raise IOError(
                "[+] Aborting strategy: Number of Moving Average parameters exceeded maximum of one: {}".format(
                    MA_param
                )
            )
        DEV_param = self.model_parameters["DEV"]
        if len(DEV_param) > 1:
            raise IOError(
                "[+] Aborting strategy: Number of Standard Deviation parameters exceeded maximum of one: {}".format(
                    DEV_param
                )
            )
        K_value = self.model_parameters["K_value"]
        if type(K_value) != int:
            raise IOError(
                "[+] Aborting strategy: K value parameter needs to be an integer"
            )
        temp = pd.DataFrame()
        temp["Close"] = self.feature_data.df_curated["Close"]
        temp["plusK"] = (
            self.feature_data.df_curated["MA_" + str(MA_param[0])]
            + K_value
            * self.feature_data.df_curated["DEV_" + str(DEV_param[0])]
        ).dropna()
        temp["minusK"] = (
            self.feature_data.df_curated["MA_" + str(MA_param[0])]
            - K_value
            * self.feature_data.df_curated["DEV_" + str(DEV_param[0])]
        ).dropna()

        temp["Trigger"] = temp.apply(self.states_condition, raw=True, axis=1)

        events_bol = (
            temp["Trigger"]
            .rolling(window=2)
            .apply(
                lambda x: -1
                if x[0] == 0 and x[1] == 1
                else (1 if x[0] == 0 and x[1] == -1 else np.nan),
                raw=True,
            )
        ).dropna()

        return pd.DataFrame(events_bol).reset_index()

    def event_labels(self, events_df):
        labels = []
        for event in events_df.values:
            event_datetime = event[0]
            event_trigger = event[1]
            horizon_df = self.horizon_data(event_datetime)
            labels.append(self.labels(self.touches(horizon_df), event_trigger))
        return labels

    def events(self):
        if self.model_type == "crossing-MA":
            return self.events_crossing_MA()
        elif self.model_type == "bollinger-bands":
            return self.events_bollinger()
        elif self.model_type == "classical-filter":
            return self.events_classical_filter(
                self.model_parameters["threshold"]
            )

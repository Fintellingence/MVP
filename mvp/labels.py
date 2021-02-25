import pandas as pd
import mvp
import numpy as np
import datetime as dt


class Labels:
    """"""

    def __init__(self, events_df, operation_parameters, mode):
        self.events_df = events_df
        self.operation_parameters = operation_parameters
        if mode == "suggestion":
            self.labeled_df = (
                self.get_labels()
                .drop(columns=["Close"])
                .rename(columns={"Trigger": "Suggestion"})
            )
        if mode == "static":
            labeled_df = self.get_labels().drop(columns=["Close"]).copy()
            labeled_df["Label"] = labeled_df["Trigger"] * labeled_df["Label"]
            self.labeled_df = labeled_df.drop(columns=["Trigger"])

    def horizon_data(self, event_datetime):
        horizon = self.operation_parameters["IH"]
        close_data = self.events_df["Close"].reset_index()
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

    def event_labels(self):
        """
        Given a dataframe events_df, containing an index column 'DateTime' and a 'Trigger' column
        containing the event side (Buy = 1 or Sell = -1) this method returns a list of the triple label:
            1 if take profit was achieved
            0 if investment horizon was achieved
           -1 if stop loss was achieved
        for each event in events_df.

        Parameters
        ----------
        `events_df`: ``pd.DataFrame()``
            columns = ['DateTime','Trigger']
            'DateTime' is the pd.index column, containing pd.Timestamp() values for the calculated events.
            'Trigger' is a coulmn containing the side of the stratgey: Buy = 1, Sell = -1.

        Modified
        ----------
        None

        Return
        ----------

        ``list``
            A list containing a label (1, 0, or -1) for each of the triggers in events_df.
        """
        events_df = self.events_df[["Trigger"]].dropna().reset_index()
        labels = []
        endDateTime = []
        for event in events_df.values:
            event_datetime = event[0]
            event_trigger = event[1]
            horizon_df = self.horizon_data(event_datetime)
            touches = self.touches(horizon_df)
            endDateTime.append(min(touches))
            labels.append(self.labels(touches, event_trigger))
        return labels, endDateTime

    def get_labels(self):
        labeled_df = self.events_df.dropna().copy()
        labeled_df["Label"] = self.event_labels()[0]
        labeled_df["EndDateTime"] = self.event_labels()[1]
        return labeled_df

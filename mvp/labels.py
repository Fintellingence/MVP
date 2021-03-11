import pandas as pd
import mvp
import numpy as np
import datetime as dt


class Labels:
    """
    This class is defined to label trading events based on the Triple-Barrier Method,
    in order to be able to label the event triggers (`events`) the user should also
    provide the operation parameters. Given the labels, one could feed it to the meta-
    learning model with an enhanced feature space by freely utilizing the methods of
    the `CuratedData` object in the `CuratedData.feature_data` attribute.

    Parameters
    ----------
    `events` : ``pandas.DataFrame()``
        A pandas Dataframe containing three columns:
        - 'DateTime' is the index
        - 'Side' is the Side of the operation (Buy = 1, Sell = -1)
    `close_data`: ``pandas.DataFrame()``
        A pandas Dataframe containing a single column:
        - 'DateTime' is the index
        - 'Close' is the Close price for the raw series
    `operation_parameters`: ``dict``
        Inside `OperationParameters` key we have to provide another dict containing three values:
            - StopLoss (SL)
            - TakeProfit (TP)
            - InvestmentHorizon (IH)
            - MarginMode (margin_mode)
        These values should be provided like the following:
            {'SL': 0.01, 'TP': 0.01, 'IH': 1000,'margin_mode':'percent'}}
    Usage
    -----
    The class should be initialized with an event dataframe from
    `primary.PrimaryModel.events`, a dict containing operation parameters (SL,TP,IH,MarginMode),
    and a mode of labeling. The main output of this class is stored in the `labels.Labels.label_data`
    attribute.
    """

    def __init__(
        self,
        events,
        close_data,
        operation_parameters,
    ):
        self.events = events.reset_index()
        self.close_data = close_data.reset_index()
        self.operation_parameters = operation_parameters
        self.label_data = self.get_label_data()

    def horizon_reduced_dataframe(self, close_df, event_datetime, IH):
        entry_point = close_df[close_df["DateTime"] == event_datetime].index[0]
        reduced_df = close_df.iloc[
            entry_point : entry_point + IH
        ].reset_index()
        return reduced_df.drop(columns=["index"]).copy()

    def barriers(self, event_side, entry_value, SL, TP, margin_mode):
        if margin_mode == "pips":
            if event_side == 1:
                return entry_value + TP, entry_value - SL
            if event_side == -1:
                return entry_value + SL, entry_value - TP
        if margin_mode == "percent":
            if event_side == 1:
                return entry_value * (1 + TP / 100), entry_value * (
                    1 - SL / 100
                )
            if event_side == -1:
                return entry_value * (1 + SL / 100), entry_value * (
                    1 - TP / 100
                )

    def barrier_break(self, horizon_data, upBarrier, downBarrier):
        upBreak, downBreak = None, None
        if not horizon_data[horizon_data["Close"] >= upBarrier].empty:
            upBreak = horizon_data[horizon_data["Close"] >= upBarrier].iloc[0]
        if not horizon_data[horizon_data["Close"] <= downBarrier].empty:
            downBreak = horizon_data[
                horizon_data["Close"] <= downBarrier
            ].iloc[0]
        if upBreak is None and downBreak is None:
            return horizon_data.iloc[-1], "no_break"
        if upBreak is None and downBreak is not None:
            return downBreak, "down_break"
        if upBreak is not None and downBreak is None:
            return upBreak, "up_break"
        if upBreak.DateTime < downBreak.DateTime:
            return upBreak, "up_break"
        if downBreak.DateTime < upBreak.DateTime:
            return downBreak, "down_break"

    def label(self, break_direction, event_side):
        if break_direction == "no_break":
            return 0
        if event_side == 1:
            if break_direction == "up_break":
                return 1
            if break_direction == "down_break":
                return -1
        if event_side == -1:
            if break_direction == "up_break":
                return -1
            if break_direction == "down_break":
                return 1

    def get_label_data(self):
        labeled_events = [["DateTime", "Side", "Label", "PositionEnd"]]
        for i in range(0, len(self.events.index)):
            event = self.events.iloc[i]
            horizon_data = self.horizon_reduced_dataframe(
                self.close_data,
                event.DateTime,
                self.operation_parameters["IH"],
            )
            upBarrier, downBarrier = self.barriers(
                event.Side,
                horizon_data.iloc[0]["Close"],
                self.operation_parameters["SL"],
                self.operation_parameters["TP"],
                self.operation_parameters["margin_mode"],
            )
            break_event, break_direction = self.barrier_break(
                horizon_data, upBarrier, downBarrier
            )
            event_label = self.label(break_direction, event.Side)
            if break_event is not None:
                labeled_events.append(
                    [
                        event.DateTime,
                        event.Side,
                        event_label,
                        break_event.DateTime,
                    ]
                )
        headers = labeled_events.pop(0)
        return pd.DataFrame(labeled_events, columns=headers).set_index(
            "DateTime"
        )

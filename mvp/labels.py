import pandas as pd

def horizon_reduced_dataframe(close_df, event_datetime, IH):
    entry_point = close_df[close_df["DateTime"] == event_datetime].index[0]
    reduced_df = close_df.iloc[
        entry_point : entry_point + IH
    ].reset_index()
    return reduced_df.drop(columns=["index"]).copy()

def barriers(event_side, entry_value, SL, TP, margin_mode):
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

def barrier_break(horizon_data, upBarrier, downBarrier):
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

def label(break_direction, event_side):
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

def get_labels(events, close_data, SL, TP, IH, margin_mode):
    events = pd.DataFrame(events)
    events.columns = ['Side']
    events = events.reset_index()
    close_data = close_data.reset_index()
    labeled_events = [["DateTime", "Side", "Label", "PositionEnd"]]
    for i in range(0, len(events.index)):
        event = events.iloc[i]
        horizon_data = horizon_reduced_dataframe(
            close_data,
            event.DateTime,
            IH,
        )
        upBarrier, downBarrier = barriers(
            event.Side,
            horizon_data.iloc[0]["Close"],
            SL,
            TP,
            margin_mode
        )
        break_event, break_direction = barrier_break(
            horizon_data, upBarrier, downBarrier
        )
        event_label = label(break_direction, event.Side)
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

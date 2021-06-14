""" Result labeling of primary strategies

This is a complement module to mvp.primary which consume information
from trade triggers to label the result of the operation considering
thresholds to constrain the profit, loss and maximum period

Functions
---------

``horizon_trading_range(
    minute1_data -> pandas.DataFrame,
    ih-> object,
    ih_type -> str
)``

``event_label(
    side -> int,
    minute1_data -> pandas.DataFrame,
    sl -> float,
    tp -> float,
    ih -> object,
    ih_type -> str
)``

``continuous_event_label(
    side -> int,
    minute1_data -> pandas.DataFrame,
    sl -> float,
    tp -> float,
    ih -> object,
    ih_type -> str
)``

``event_label_series(
    side_series -> pandas.Series,
    minute1_data -> pandas.DataFrame,
    sl -> float,
    tp -> float,
    ih -> object,
    ih_type -> str,
    label_type -> str
)``

"""
import pandas as pd
from mvp.utils import smallest_cusum_i


def horizon_trading_range(minute1_data, ih, ih_type):
    """
    Compute and return dataframe chunk corresponding to investiment horizon
    Accept some different types for investiment horizon definition as below
    The investiment horizon returned is ALWAYS with respect to the input
    dataframe starting point, note that this function does not require a
    starting datetime or index to slice `minute1_data`, as such consider
    the starting point as the same of `minute1_data`

    Parameters
    ----------
    `minute1_data` : ``pandas.Dataframe``
        dataframe with Timestamp indexes and full bar data
        with open-high-low-close prices and volume as well
        See ``RawData.df`` attribute to consult columns
        `minute1_data.index[0]` is the start of horizon
    `ih` : ``pandas.Timedelta/pandas.Timestamp/int``
        investiment horizon
    `ih_type` : ``str``
        string code of type of investment horizon. Accept the following
        1. "bars"       - the number of sequential bars
        2. "Volume"     - IH in volume traded
        3. "TickVol"    - IH in deals occurred
        4. "money"      - IH in money traded
        This value is ignored if `ih` is not integer

    Return
    ------
    ``pandas.DataFrame``
        chunk of the initial dataframe corresponding to the invest horizon

    """
    init = minute1_data.index[0]
    if ih is None:
        ih_index = minute1_data.index.size
    elif isinstance(ih, pd.Timestamp):
        if ih <= minute1_data.index[0]:
            raise ValueError(
                "Invalid datetime {} for invest horizon".format(ih)
            )
        if ih > minute1_data.index[-1]:
            ih_index = minute1_data.index.size
        else:
            ih_index = 1 + minute1_data.index.get_loc(ih, method="backfill")
    elif isinstance(ih, pd.Timedelta):
        ih_index = 1 + minute1_data.index.get_loc(init + ih, method="backfill")
    elif ih_type == "bars":
        ih_index = ih
    else:
        if ih_type == "Volume" or ih_type == "TickVol":
            horizon_vals = minute1_data[ih_type].values
        elif ih_type.lower() == "money":
            horizon_vals = (minute1_data.Volume * minute1_data.Close).values
        ih_index = smallest_cusum_i(
            horizon_vals.size, horizon_vals.astype("int32"), ih
        )
    return minute1_data.iloc[:ih_index]


def event_label(side, minute1_data, sl, tp, ih, ih_type="bars"):
    """
    Label of trade positioning given the respective operation side
    To label trades, thresholds to abort the initial operation are
    required, also known as barriers of take profit(tp), stop loss
    (sl) and investiment horizon(ih)

    Parameters
    ----------
    `side` : ``int``
        binary info with buy(+1) or sell(-1) information
    `minute1_data` : ``pandas.DataFrame``
        A chunk of dataframe from `RefinedData.df` or `RawData.df`
        attribute which must start at specific datetime the `side`
        trigger occurred, that is, `minute1_data.index[0]` must be
        the trigger instant
    `sl` : ``float``
        stop loss of the positioning in percent value (4 means 4%)
    `tp` : ``float``
        take profit to close the positioning in percent value
    `ih` : ``pandas.Timedelta/pandas.Timestamp/int``
        investiment horizon
    `ih_type` : ``str``
        string code of type of investment horizon. Accept the following
        1. "bars"       - the number of sequential bars
        2. "Volume"     - IH in volume traded
        3. "TickVol"    - IH in deals occurred
        4. "money"      - IH in money traded
        This value is ignored if `ih` is not integer

    Return
    ------
    ``tuple (pandas.Timestamp, int)``
        tuple with the smallest datetime prices touch one of the barriers
        and the result (which barrier was hit first)
        +1 profit
         0 investment horizon hit before stop loss or take profit
        -1 loss
    """
    hor_data = horizon_trading_range(minute1_data, ih, ih_type)
    start_price = hor_data.Close[0]
    percent_var = side * (hor_data.Close / start_price - 1) * 100
    stop_event = hor_data.index[-1]
    result = 0
    if sl is not None:
        loss_region = percent_var.index[percent_var <= -sl]
        if not loss_region.empty:
            result = -1
            stop_event = loss_region[0]
    if tp is not None:
        profit_region = percent_var.index[percent_var >= tp]
        if not profit_region.empty and profit_region[0] < stop_event:
            result = 1
            stop_event = profit_region[0]
    return stop_event, result


def continuous_event_label(side, minute1_data, sl, tp, ih, ih_type="bars"):
    """
    A continuous version of the label method of `event_label` function
    The difference is only when hit investiment horizon barrier and in
    which case return the relative percent price variation from the
    initial value normalized by `tp` or `sl` if some profit or loss
    occurred respectively
    """
    stop_event, result = event_label(side, minute1_data, sl, tp, ih, ih_type)
    if result != 0:
        return stop_event, result
    percent_var = (
        side
        * (minute1_data.Close[stop_event] / minute1_data.Close[0] - 1)
        * 100
    )
    if percent_var < 0:
        result = percent_var / sl
    else:
        result = percent_var / tp
    return stop_event, result


def event_label_series(
    side_series,
    minute1_data,
    sl,
    tp,
    ih,
    ih_type="bars",
    label_type="discrete",
):
    """
    Label a series of requested side-trading events from primary models

    Parameters
    ----------
    `side_series` : ``pandas.Series``
        Series indexed by timestamps with the advisable positioning side
        with +1 indicating buy and -1 sell
    `minute1_data` : ``pandas.DataFrame``
        The core minute 1 time frame market data either from `RefinedData.df`
        or `RawData.df` attribute. A 1-minute data is always preferable since
        provide better accuracy to label the result as the prices touch some
        barrier from stopp loss or take profit
    `sl` : ``float``
        stop loss of the positioning in percent value (4 means 4%)
    `tp` : ``float``
        take profit to close the positioning in percent value
    `ih` : ``pandas.Timedelta/int``
        investiment horizon. If integer check `ih_type`
    `ih_type` : ``str``
        string code of type of investment horizon. Accept the following
        1. "bars"       - the number of sequential bars
        2. "Volume"     - IH in volume traded
        3. "TickVol"    - IH in deals occurred
        4. "money"      - IH in money traded
        This value is ignored if `ih` is not integer
    `label_type` : ``str``
        Accept only two values with the labeling type
        1. "continuous" - if hit the invest horizon barrier return float
        2. "discrete"   - usual {+1,0,-1} for {tp,ih,sl} events

    Return
    ------
    ``pandas.DataFrame``
        A dataframe with the same indexes of `side_series` and the columns
        `["Side", "Label", "PositionEnd"]`
    """
    triggers = side_series.index
    sides = side_series.values
    labels_df = pd.DataFrame(
        index=triggers, columns=["Side", "Label", "PositionEnd"]
    )
    if label_type == "continuous":
        label_fun = continuous_event_label
    else:
        label_fun = event_label
    for i, (dt, side) in enumerate(zip(triggers, sides)):
        stop_event, label = label_fun(
            side, minute1_data.loc[dt:], sl, tp, ih, ih_type
        )
        labels_df.Side[i] = side
        labels_df.Label[i] = label
        labels_df.PositionEnd[i] = stop_event
    labels_df.PositionEnd = pd.to_datetime(labels_df.PositionEnd)
    return labels_df

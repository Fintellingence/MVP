""" Primary strategies selection module

    In this module, based on basic statistical features provided by
    refined data, simple strategies are build. These strategies are
    based on combining features to generate events with some advice
    about what should be done. The advices are binary info, as such
    are represented here by +1 for buy and -1 for sell, also known
    as `side` of the suggested operation

    All functions in this module implement a primary strategy hence
    always consume a ``RefinedData`` obj and return a pandas series
    with +1 and -1 values

"""

import pandas as pd
import numpy as np

from mvp import utils


def crossing_ma(refined_obj, window1, window2, kwargs={}, draw=False):
    """
    Use fast and slow moving averages crossing as event definition
    The primary strategy (advice) depends on cross relation

    Paramters
    ---------
    `refined_obj` : ``mvp.RefinedData``
        object with simple statistical features
    `window1` : ``int``
        size of one of the moving averages window
    `window2` : ``int``
        size of one of the moving averages window
    `kwargs` : ``dict``
        optional arguments of the `refined_obj` method involved
        In this case `refined_obj.get_sma`. Check its documentation

    Return
    ------
    ``pandas.Series``
        time series with +1 (buy side) and -1(sell side)

    """
    sshift = 0
    if kwargs.get("step") == "day":
        target = kwargs.get("target")
        if not target or target.lower().find("close") > 0:
            sshift = 1
    slow_window = min(window1, window2)
    fast_window = max(window1, window2)
    slow_ma = refined_obj.get_sma(slow_window, **kwargs)
    fast_ma = refined_obj.get_sma(fast_window, **kwargs)
    diff = (fast_ma - slow_ma).shift(sshift).dropna()
    cross_time = diff.index[diff[1:] * diff.shift(1) < 0]
    if draw:
        return (
            pd.Series(np.sign(diff[cross_time].values), cross_time, np.int32),
            slow_ma,
            fast_ma,
        )
    return pd.Series(np.sign(diff[cross_time].values), cross_time, np.int32)


def trend(refined_obj, threshold, window=1, kwargs={}):
    """
    Consider the return series over (generally) very short moving average
    to slice in sequences of positive and negative values. For each slice
    perform a cummulative sum of the returns, and in case it each exceeds
    a threshold, mark the index as the beggining of a trend in the prices
    The return series here is the relative variation of the prices

    Paramters
    ---------
    `refined_obj` : ``mvp.RefinedData``
        object with simple statistical features
    `threshold` : ``float``
        tolerance to define a new trend if cummulative returns exceed it
    `window` : ``int``
        size of the moving average window. May be use to smooth data
    `kwargs` : ``dict``
        optional arguments of moving average

    Return
    ------
    ``pandas.Series``
        time series with +1 (buy side) and -1(sell side)

    """
    sshift = 0
    if kwargs.get("step") == "day":
        target = kwargs.get("target")
        if not target or target.lower().find("close") > 0:
            sshift = 1
    sma = refined_obj.get_sma(window, **kwargs)
    diff = (sma - sma.shift(1)).dropna()
    returns = (sma / sma.shift(1) - 1).dropna().values
    n = returns.size
    events_ind = np.empty(n, dtype=np.int32)
    events_sides = np.empty(n, dtype=np.int32)
    nevents = utils.sign_mark_cusum(
        n, returns, events_ind, events_sides, threshold
    )
    events_ind = events_ind - 1 + sshift
    events_time = diff[events_ind[:nevents]].index
    return pd.Series(events_sides[:nevents], events_time, np.int32)


def cummulative_returns(refined_obj, threshold, window=1, kwargs={}):
    """
    Consider the return series over (generally) very short moving average
    to perform cummulative sum and mark as events when its absolute value
    exceed a threshold, then reset and start again.
    This is different from `trend` model which requires the cumsum in the
    period to have exclusive positive or negative returns, variations are
    not allowed as in this case

    Paramters
    ---------
    `refined_obj` : ``mvp.RefinedData``
        object with simple statistical features
    `threshold` : ``float``
        tolerance to define a new trend if cummulative returns exceed it
    `window` : ``int``
        size of the moving average window. May be use to smooth data
    `kwargs` : ``dict``
        optional arguments of moving average

    Return
    ------
    ``pandas.Series``
        time series with +1 (buy side) and -1(sell side)

    """
    sshift = 0
    if kwargs.get("step") == "day":
        target = kwargs.get("target")
        if not target or target.lower().find("close") > 0:
            sshift = 1
    sma = refined_obj.get_sma(window, **kwargs)
    returns = (sma / sma.shift(1) - 1).dropna()
    n = returns.size
    events_ind = np.empty(n, np.int32)
    events_sides = np.empty(n, np.int32)
    nevents = utils.indexing_cusum_abs(
        n, returns.values, events_ind, events_sides, threshold
    )
    events_ind = events_ind - 1 + sshift
    events_time = returns[events_ind[1:nevents]].index
    return pd.Series(events_sides[1:nevents], events_time, np.int32)


def bollinger_bands(
    refined_obj, dev_window, ma_window, mult, kwargs={}, draw=False
):
    """
    Use two bands of standard deviation around the moving average which
    launch an event every time the prices touch one of the bands

    Paramters
    ---------
    `refined_obj` : ``mvp.RefinedData``
        object with simple statistical features
    `dev_window` : ``int``
        window size to compute moving standard deviation
    `ma_window` : ``int``
        window size to compute moving average
    `mult` : ``float``
        multiple of standard deviation to compute upper and lower bands
    `kwargs` : ``dict``
        dictionary with optional arguments of the features involved

    Return
    ------
    ``pandas.Series``
        time series with +1 (buy side) and -1(sell side)

    """
    sshift = 0
    if kwargs.get("step") == "day":
        target = kwargs.get("target")
        if not target or target.lower().find("close") > 0:
            sshift = 1
    data_series = refined_obj.get_sma(1, **kwargs)
    dev = refined_obj.get_dev(dev_window, **kwargs)
    sma = refined_obj.get_sma(ma_window, **kwargs)
    upper = sma + mult * dev
    lower = sma - mult * dev
    upper_diff = (upper - data_series).shift(sshift).dropna()
    lower_diff = (data_series - lower).shift(sshift).dropna()
    sell_time = upper_diff[upper_diff.shift(1) * upper_diff < 0][
        upper_diff < 0
    ].index
    buy_time = lower_diff[lower_diff.shift(1) * lower_diff < 0][
        lower_diff < 0
    ].index
    buy_series = pd.Series(np.ones(buy_time.size, np.int32), buy_time)
    sell_series = pd.Series(-np.ones(sell_time.size, np.int32), sell_time)
    if draw:
        return (
            buy_series.append(sell_series, verify_integrity=True).sort_index(),
            upper,
            lower,
        )
    return buy_series.append(sell_series, verify_integrity=True).sort_index()


def day_openning_gaps(refined_obj, relative_gap_threshold=0.01):
    """
    Generate buy triggers if the openning value is larger than previous
    day close and it was a 'green' candle. Conversely, sell triggers if
    openning value is smaller than previous day close and it was a red
    candle. Green and red means positive and negative price variations
    in intraday trades, respectively.

    Parameters
    ----------
    `refined_obj` : ``mvp.RefinedData``
        object with simple statistical features
    `relative_gap_threshold` : ``float``
        relative price variation between adjacent days threshold

    Return
    ------
    ``pandas.Series``
        Series with side information either buy(+1) or sell(-1)

    """
    daily_bars = refined_obj.daily_bars()
    full_df_index = refined_obj.df.index
    daily_open = daily_bars.Open
    daily_close = daily_bars.Close
    shift_daily_result = (daily_close - daily_open).shift(1).dropna()
    relative_gap = (daily_open / daily_close.shift(1) - 1).dropna()
    day_buy_trig = relative_gap.index[
        (relative_gap > relative_gap_threshold) & (shift_daily_result > 0)
    ]
    day_sell_trig = relative_gap.index[
        (relative_gap < -relative_gap_threshold) & (shift_daily_result < 0)
    ]
    nbuys = day_buy_trig.size
    nsells = day_sell_trig.size
    int_buy_ind = np.empty(nbuys, dtype=np.int32)
    int_sell_ind = np.empty(nsells, dtype=np.int32)
    for i, dt in enumerate(day_buy_trig):
        int_buy_ind[i] = full_df_index.get_loc(dt, method="backfill")
    for i, dt in enumerate(day_sell_trig):
        int_sell_ind[i] = full_df_index.get_loc(dt, method="backfill")
    buy_sides = pd.Series(
        np.ones(nbuys, dtype=np.int32), full_df_index[int_buy_ind]
    )
    sell_sides = pd.Series(
        -np.ones(nsells, dtype=np.int32), full_df_index[int_sell_ind]
    )
    return buy_sides.append(sell_sides, verify_integrity=True).sort_index()


def overlap_strategies(
    refined_obj,
    time_gap,
    agree_threshold,
    strategies_list,
    args_list,
    kwargs_list,
):
    """
    Wihtin events from a main primary strategy, consider a time interval
    for other events from different strategies to happen. In case other
    strategies provides events in the respective interval, consider all
    recomendations to evaluate an overlap region and define new events

    Parameters
    ----------
    `refined_obj` : ``mvp.RefinedData``
        object with simple statistical features
    `time_gap` : ``pandas.Timedelta``
        tolered time to match events from all strategies
    `agree_threshold` : ``float`` in (0, 1]
        Minimum required for average of sides that match the same period
    `strategies_list` : ``list[mvp.primary functions]``
        list with functions available from primary module
    `args_list` : `list[tuple]``
        list of tuples to use as positional arguments
    `kwargs_list` : ``list[dict]``
        list of dictionaries to use as optional arguments

    Return
    ------
    ``pandas.Series``

    """
    if agree_threshold < 0 or agree_threshold > 1:
        raise ValueError(
            "Threshold of agreement among event must lies in (0, 1]"
            " interval. {} was given".format(agree_threshold)
        )
    events_list = [
        fun(refined_obj, *args_list[i], **kwargs_list[i])
        for i, fun in enumerate(strategies_list)
    ]
    if len(events_list) == 1:
        return events_list[0]
    market_freq = pd.Timedelta(minutes=1)
    master_events = events_list.pop(0)
    for t, master_advice in zip(master_events.index, master_events.values):
        t_end = t + time_gap
        t_range = pd.period_range(t, t_end, freq=market_freq).to_timestamp()
        intersection_indexes = [
            t_range.intersection(event.index) for event in events_list
        ]
        if all([index_set.empty for index_set in intersection_indexes]):
            master_events[t] = np.NaN
            continue
        combined_advices = 0
        total_advices = 0
        for index_set, events in zip(intersection_indexes, events_list):
            combined_advices += events[index_set].sum()
            total_advices += events[index_set].size
        overall_advice = combined_advices / total_advices
        if master_advice * overall_advice < agree_threshold:
            master_events[t] = np.NaN
    intersect_events = master_events.dropna()
    intersect_events.index = intersect_events.index + time_gap
    return intersect_events.astype(np.int32)

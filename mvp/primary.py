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


def crossing_ma(refined_obj, window1, window2, kwargs={}):
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
    slow_window = min(window1, window2)
    fast_window = max(window1, window2)
    slow_ma = refined_obj.get_sma(slow_window, **kwargs)
    fast_ma = refined_obj.get_sma(fast_window, **kwargs)
    diff = (fast_ma - slow_ma).dropna()
    cross_time = diff.index[diff[1:] * diff.shift(1) < 0]
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
    sma = refined_obj.get_sma(window, **kwargs)
    diff = (sma - sma.shift(1)).dropna()
    returns = (sma / sma.shift(1) - 1).dropna().values
    n = returns.size
    events_ind = np.empty(n, dtype=np.int32)
    nevents = utils.sign_mark_cusum(n, returns, events_ind, threshold)
    events_time = diff[events_ind[:nevents]].index
    return pd.Series(np.sign(diff[events_time].values), events_time, np.int32)


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
    sma = refined_obj.get_sma(window, **kwargs)
    returns = (sma / sma.shift(1) - 1).dropna()
    n = returns.size
    events_ind = np.empty(n, np.int32)
    events_sign = np.empty(n, np.int32)
    nevents = utils.indexing_cusum_abs(
        n, returns.values, events_ind, events_sign, threshold
    )
    events_time = returns[events_ind[1:nevents]].index
    return pd.Series(events_sign[1:nevents], events_time, np.int32)


def bollinger_bands(refined_obj, dev_window, ma_window, mult, kwargs={}):
    """
    Use two bands of standard deviation around the moving average which
    launch an event every time the prices touch one of the bands

    Paramters
    ---------
    `refined_obj` : ``mvp.Refineddata``
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
    try:
        target = kwargs["target"]
        bar_type, data_name = target.split(":")
    except KeyError:
        bar_type = "time"
        data_name = "close"
    try:
        step = kwargs["step"]
    except KeyError:
        step = 1
    get_data_kwargs = {"bar_type": bar_type, "step": step}
    get_data_method = refined_obj.__getattribute__("get_" + data_name)
    data_series = get_data_method(**get_data_kwargs)
    dev = refined_obj.get_dev(dev_window, **kwargs)
    sma = refined_obj.get_sma(ma_window, **kwargs)
    upper = sma + mult * dev
    lower = sma - mult * dev
    upper_diff = (upper - data_series).dropna()
    lower_diff = (data_series - lower).dropna()
    sell_time = upper_diff[upper_diff.shift(1) * upper_diff < 0][
        upper_diff < 0
    ].index
    buy_time = lower_diff[lower_diff.shift(1) * lower_diff < 0][
        lower_diff < 0
    ].index
    buy_series = pd.Series(np.ones(buy_time.size, np.int32), buy_time)
    sell_series = pd.Series(-np.ones(sell_time.size, np.int32), sell_time)
    return buy_series.append(sell_series, verify_integrity=True).sort_index()


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
    `refined_obj` : ``mvp.Refineddata``
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

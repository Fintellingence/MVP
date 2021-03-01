import datetime as dt

import pandas as pd
from pytest import fixture, mark
from pandas._testing import assert_frame_equal, assert_series_equal

from mvp import selection


@fixture
def closed_prices():
    delta = dt.timedelta(minutes=1)
    format = "%Y-%m-%d %H:%M:%S"
    initial_time_str = "2018-11-12 09:15:32"
    prices = [-9.4, -0.3, 0.5, -5.3, 7.1, 2.8, 9.8, 3.5, 13.3, -9.2]
    initial_time = dt.datetime.strptime(initial_time_str, format)
    index = [initial_time + (delta * i) for i in range(len(prices))]
    return pd.DataFrame(prices, index=index)


@fixture
def horizons(closed_prices):
    events = closed_prices.iloc[[0, 3, 5, 7, 8]].index.values
    vertical_barriers = closed_prices.iloc[[2, 5, 7, 9, 9]].index.values
    return pd.Series(vertical_barriers, index=events)


@fixture
def expected_indicator(closed_prices):
    return pd.DataFrame(
        [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
        ],
        index=closed_prices.index,
    )

    
@fixture
def occurrences(expected_indicator):
    return expected_indicator.sum(axis=1)


def test_indicator(closed_prices, horizons, expected_indicator):
    assert_frame_equal(
        selection.indicator(closed_prices.index, horizons), expected_indicator
    )


@mark.parametrize(
    "idx_interval, occurrences",
    [([0, 3], [1, 1, 1, 1, 1, 1]), ([5, 7, 8], [2, 1, 2, 2, 2])],
)
def test_interval_count_occurrences(
    closed_prices, horizons, idx_interval, occurrences
):
    interval = closed_prices.index[idx_interval]
    expected_occurrences = pd.Series(
        occurrences,
        index=closed_prices.loc[interval[0] : horizons[interval].max()].index,
    )
    assert_series_equal(
        selection.interval_count_occurrences(
            closed_prices.index, horizons, interval
        ),
        expected_occurrences,
    )


@mark.parametrize(
    "idx_interval, raw_avg_uniqueness",
    [([0, 3], [1, 5 / 6]), ([5, 7, 8], [2 / 3, 1 / 2, 1 / 2])],
)
def test_interval_avg_uniqueness(
    closed_prices, horizons, occurrences, idx_interval, raw_avg_uniqueness
):
    interval = closed_prices.index[idx_interval]
    expected_avg_uniqueness = pd.Series(raw_avg_uniqueness, index=interval)
    assert_series_equal(
        selection.interval_avg_uniqueness(horizons, occurrences, interval),
        expected_avg_uniqueness,
    )

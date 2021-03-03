import datetime as dt

import numpy as np
import pandas as pd
from pytest import fixture, mark
from pandas._testing import assert_frame_equal, assert_series_equal

from mvp import selection


@fixture
def large_closed_prices():
    seed = 12345
    rng = np.random.default_rng(seed)
    delta = dt.timedelta(minutes=1)
    format = "%Y-%m-%d %H:%M:%S"
    initial_time_str = "2018-11-12 09:15:32"
    prices = rng.random(10000) * 5
    initial_time = dt.datetime.strptime(initial_time_str, format)
    index = [initial_time + (delta * i) for i in range(len(prices))]
    return pd.Series(prices, index=index)


@fixture
def large_horizons(large_closed_prices):
    seed = 12345
    closed_size = large_closed_prices.size
    rng = np.random.default_rng(seed)
    start_idx = np.unique(
        rng.choice(np.arange(closed_size), int(0.8 * closed_size))
    )
    end_idx = start_idx + rng.choice(np.arange(10, 50), start_idx.size)
    end_idx[end_idx >= closed_size] = closed_size - 1
    events = large_closed_prices.iloc[start_idx].index.values
    vertical_barriers = large_closed_prices.iloc[end_idx].index.values
    return pd.Series(vertical_barriers, index=events)


@fixture
def large_occurrences(large_closed_prices, large_horizons):
    occ = selection.chunk_count_occurrences(
        large_horizons.index, large_closed_prices.index, large_horizons
    )
    return occ.reindex(large_closed_prices.index).fillna(0)


@fixture
def large_avg_uniqueness(large_horizons, large_occurrences):
    avg_uniqueness = selection.chunk_avg_uniqueness(
        large_horizons.index, large_horizons, large_occurrences
    )
    return avg_uniqueness


@fixture
def large_sample_weights(
    large_occurrences, large_horizons, large_closed_prices
):
    sample_weights = selection.chunk_sample_weights(
        large_horizons.index,
        large_occurrences,
        large_horizons,
        large_closed_prices,
    )
    return sample_weights


@fixture
def closed_prices():
    delta = dt.timedelta(minutes=1)
    format = "%Y-%m-%d %H:%M:%S"
    initial_time_str = "2018-11-12 09:15:32"
    prices = [9.4, 0.3, 0.5, 5.3, 7.1, 2.8, 9.8, 3.5, 13.3, 9.2]
    initial_time = dt.datetime.strptime(initial_time_str, format)
    index = [initial_time + (delta * i) for i in range(len(prices))]
    return pd.Series(prices, index=index)


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
    "idx_chunk_events, occurrences",
    [([0, 3], [1, 1, 1, 1, 1, 1]), ([5, 7, 8], [2, 1, 2, 2, 2])],
)
def test_chunk_count_occurrences(
    closed_prices, horizons, idx_chunk_events, occurrences
):
    event_chunks = closed_prices.index[idx_chunk_events]
    expected_occurrences = pd.Series(
        occurrences,
        index=closed_prices.loc[
            event_chunks[0] : horizons[event_chunks].max()
        ].index,
    )
    assert_series_equal(
        selection.chunk_count_occurrences(
            event_chunks, closed_prices.index, horizons
        ),
        expected_occurrences,
    )


@mark.parametrize(
    "idx_chunk_events, raw_avg_uniqueness",
    [([0, 3], [1, 5 / 6]), ([5, 7, 8], [2 / 3, 1 / 2, 1 / 2])],
)
def test_chunk_avg_uniqueness(
    closed_prices, horizons, occurrences, idx_chunk_events, raw_avg_uniqueness
):
    event_chunks = closed_prices.index[idx_chunk_events]
    expected_avg_uniqueness = pd.Series(raw_avg_uniqueness, index=event_chunks)
    assert_series_equal(
        selection.chunk_avg_uniqueness(event_chunks, horizons, occurrences),
        expected_avg_uniqueness,
    )


@mark.parametrize(
    "idx_chunk_events, raw_sample_weights",
    [
        ([0, 3], [2.9338568698359038, 2.188004281174159]),
        (
            [5, 7, 8],
            [0.2727155764717333, 0.03158945081076592, 0.48322025777981326],
        ),
    ],
)
def test_chunk_sample_weights(
    closed_prices, horizons, occurrences, idx_chunk_events, raw_sample_weights
):

    event_chunks = closed_prices.index[idx_chunk_events]
    expected_sample_wweights = pd.Series(
        raw_sample_weights, index=event_chunks
    )
    assert_series_equal(
        selection.chunk_sample_weights(
            event_chunks, occurrences, horizons, closed_prices
        ),
        expected_sample_wweights,
    )


@mark.parametrize(
   "num_of_threads, num_of_chunks", [(1, 1), (2, 100), (6, 203), (8, 8)]
)
def test_count_occurrences(
   large_closed_prices,
   large_horizons,
   large_occurrences,
   num_of_threads,
   num_of_chunks,
):
   assert_series_equal(
       selection.count_occurrences(
           large_closed_prices.index,
           large_horizons,
           num_of_threads,
           num_of_chunks,
       ),
       large_occurrences,
   )


@mark.parametrize(
    "num_of_threads, num_of_chunks", [(1, 1), (2, 100), (6, 203), (8, 8)]
)
def test_avg_uniqueness(
    large_horizons,
    large_occurrences,
    large_avg_uniqueness,
    num_of_threads,
    num_of_chunks,
):
    assert_series_equal(
        selection.avg_uniqueness(
            large_occurrences,
            large_horizons,
            num_of_threads,
            num_of_chunks,
        ),
        large_avg_uniqueness,
    )


@mark.parametrize(
    "num_of_threads, num_of_chunks", [(1, 1), (2, 100), (6, 203), (8, 8)]
)
def test_sample_weights(
    large_closed_prices,
    large_horizons,
    large_occurrences,
    large_sample_weights,
    num_of_threads,
    num_of_chunks,
):
    assert_series_equal(
        selection.sample_weights(
            large_occurrences,
            large_horizons,
            large_closed_prices,
            num_of_threads,
            num_of_chunks,
        ),
        large_sample_weights,
    )

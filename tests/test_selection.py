import datetime as dt

import numpy as np
import pandas as pd
from pytest import fixture, mark
from numpy.testing import assert_array_equal
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


@fixture
def choices(expected_indicator):
    def avguni(v):
        shape = (-1, v.shape[1])
        out = np.zeros(v.shape[1])
        sum_v = v.sum(axis=1).reshape(-1, 1)
        mask = np.squeeze(sum_v > 0)
        v = v[mask] / sum_v[mask]
        v = v.reshape(shape)
        for i in range(v.shape[1]):
            l = v[:, i]
            out[i] = l[l > 0].mean()
        return out

    selected_events = []
    rng = np.random.default_rng(12345)
    indicator = expected_indicator.values
    n_events = indicator.shape[1]
    for j in range(20):
        new_avg = np.zeros(n_events)
        for i in range(n_events):
            ind_ = indicator[:, selected_events + [i]]
            new_avg[i] = avguni(ind_)[-1]
        prob = new_avg / new_avg.sum()
        selected_events += [rng.choice(range(n_events), p=prob)]
    return np.array(selected_events)


@fixture
def avg_uniqueness():
    raw = [
        1.52332925e00,
        2.07490941e01,
        7.07305060e01,
        3.95508574e00,
        3.77155931e00,
        4.18225373e00,
        7.12743518e01,
        1.97034712e01,
        6.50536597e-01,
        3.55649958e01,
        1.28177811e00,
        1.39668986e00,
        3.39996398e01,
        8.80981784e-01,
        7.47815640e01,
        2.30370230e00,
        1.27443619e-01,
        4.88830856e01,
        2.85005316e00,
        3.73878119e01,
        9.85180189e01,
        8.98921587e01,
        1.52720809e00,
        2.80830795e01,
        1.22431055e00,
        2.90880013e01,
        1.25179112e-01,
        8.57172401e01,
        1.82543190e-01,
        1.34396364e01,
        8.86800454e01,
        1.06050673e00,
        2.05883161e00,
        1.39948028e00,
        4.57262050e00,
        5.57145991e-01,
        3.18167644e00,
        1.75147206e01,
        5.97681937e-02,
        8.14168781e00,
    ]
    delta = dt.timedelta(minutes=1)
    format = "%Y-%m-%d %H:%M:%S"
    initial_time_str = "2017-11-12 09:15:32"
    initial_time = dt.datetime.strptime(initial_time_str, format)
    index = [initial_time + (delta * i) for i in range(len(raw))]
    return pd.Series(raw, index=index)


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
    expected_sample_weights = pd.Series(raw_sample_weights, index=event_chunks)
    assert_series_equal(
        selection.chunk_sample_weights(
            event_chunks, occurrences, horizons, closed_prices
        ),
        expected_sample_weights,
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


def test_indicator_avg_uniqueness(
    large_closed_prices, large_horizons, large_avg_uniqueness
):
    indicator = selection.indicator(large_closed_prices.index, large_horizons)
    avg_uniqueness = pd.Series(
        selection.indicator_avg_uniqueness(indicator).values,
        index=large_horizons.index,
    )
    assert_series_equal(
        avg_uniqueness,
        large_avg_uniqueness,
    )


def test_bootstrap_selection(choices, expected_indicator):
    assert_array_equal(
        selection.bootstrap_selection(expected_indicator, 20),
        choices,
    )


@mark.parametrize(
    "p, raw_expected_time_weights",
    [
        (
            1.0,
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
        ),
        (
            0.5,
            [
                0.50083606,
                0.51222387,
                0.5510432,
                0.55321388,
                0.55528384,
                0.55757921,
                0.59669702,
                0.60751096,
                0.60786799,
                0.62738728,
                0.62809077,
                0.62885732,
                0.64751749,
                0.648001,
                0.68904369,
                0.69030804,
                0.69037798,
                0.7172067,
                0.7187709,
                0.73929062,
                0.79336068,
                0.84269658,
                0.84353477,
                0.85894772,
                0.85961967,
                0.87558416,
                0.87565286,
                0.92269742,
                0.92279761,
                0.93017374,
                0.97884439,
                0.97942644,
                0.98055639,
                0.98132448,
                0.98383409,
                0.98413987,
                0.98588608,
                0.99549876,
                0.99553156,
                1.0,
            ],
        ),
        (
            0.25,
            [
                0.25125408,
                0.2683358,
                0.3265648,
                0.32982083,
                0.33292577,
                0.33636881,
                0.39504553,
                0.41126644,
                0.41180199,
                0.44108093,
                0.44213615,
                0.44328598,
                0.47127623,
                0.4720015,
                0.53356553,
                0.53546206,
                0.53556697,
                0.57581005,
                0.57815636,
                0.60893593,
                0.69004103,
                0.76404487,
                0.76530215,
                0.78842159,
                0.7894295,
                0.81337624,
                0.81347929,
                0.88404614,
                0.88419641,
                0.89526061,
                0.96826659,
                0.96913965,
                0.97083459,
                0.97198671,
                0.97575113,
                0.9762098,
                0.97882912,
                0.99324814,
                0.99329734,
                1.0,
            ],
        ),
    ],
)
def test_time_weights(avg_uniqueness, p, raw_expected_time_weights):
    expected_time_weights = pd.Series(
        raw_expected_time_weights, index=avg_uniqueness.index
    )
    assert_series_equal(
        selection.time_weights(avg_uniqueness, p=p),
        expected_time_weights,
    )

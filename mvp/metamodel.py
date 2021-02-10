from multiprocessing import Pool

import numpy as np
import pandas as pd


def parallel_map_df(func, data, num_of_threads, chunk_size, **kwargs):
    """
    Apply `func` in linear distributed chunks of the data in `data`
    using parallel processing.

    Parameters
    ----------
    `func` : ``Called``
        A function to be applied in along the data chunks. It must return a
        ``DataFrame``.
    `data` : ``[DataFrame, Series]``
        The data that will be divided in different chunks.
    ``num_of_threads : ``int``
        The number of threads that will process the chunks.
    ``chunk_size : ``int``
        The size of the chunk.
    ``**kwargs : ``dict``
        Addicional arguments that will be passed to the `func`.

    Return
    ------
    `df_out` : ``DataFrame``
        The ``DataFrame`` generated by the application of `func` in
        `data` with the arguments in `**kwargs`.
    """
    def slice_data(chunk, data):
        chunk_idx = np.ceil(np.linspace(0, len(data), chunk)).astype(int)
        for i in range(1, chunk_idx.size):
            yield data[s]

    slicer = slice_data(chunk, data)
    partial_func = partial(func, **kwargs)
    with Pool(num_of_threads) as pool:
        output = [out for out in pool.imap_unordered(partial_func, slicer)]
    df_out = pd.concat(output, axis=0).sort_index()
    if np.any(df.index.duplicated()):
        raise RuntimeError("Duplicated index.")
    return df_out


class DataSelector():
    def __init__(self):
        os.makedirs("logs/", exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            "logs/dataselector.log", maxBytes=200 * 1024 * 1024, backupCount=1
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        self._logger = logging.getLogger("Logger")
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(handler)


    def interval_count_occurrences(self, closed_index, horizon, interval):
        """
        Determine the number of occurrences of horizons in `interval`.

        Parameters
        ----------
        `closed_index` : ``Index``
            The sorted timestamps of the closed prices
        `horizon` : ``DataFrame``
            The start and end of each horizon
        `interval` : ``[list, Series, Index]``
            The timestamps that compose the interval of interest

        Return
        ------
        count : ``Series``
            The number of occurrence of the `interval` in all horizons
        """
        horizon = horizon.loc[
            (horizon["start"] <= interval[-1])
            & (horizon["end"] >= interval[0])
        ]
        idx_of_interest = closed_index.searchsorted(
            [horizon["start"].min(), horizon["end"].max()]
        )
        count = pd.Series(
            0, index=closed_index[idx_of_interest[0] : idx_of_interest[1] + 1]
        )
        horizon_np = horizon.values
        for s, e in horizon_np:
            count.loc[s:e] += 1
        return count.loc[interval[0] : interval[-1]]

    def interval_average_uniqueness(self, horizon, occurrences, interval):
        """
        Determine the average uniqueness of `horizon`, i.e., the average
        uniqueness of labels, in `interval`.

        Parameters
        ----------
        `horizon` : ``DataFrame``
            The start and end of each horizon
        `occurrences` : ``Series``
            The number of occurrence of all horizons in all events
            (see ``interval_count_occurrences``)
        `interval` : ``[list, Series, Index]``
            The timestamps that compose the interval of interest

        Return
        ------
        avg_uniqueness : ``Series``
            Average uniquess associated with `horizon` in `interval`
        """
        avg_uniqueness = pd.Series(index=interval)
        horizon = horizon.loc[
            (horizon["start"] >= interval[0])
            & (horizon["end"] <= interval[-1])
        ]
        horizon_np = horizon.values
        for s, e in horizon_np:
            avg_uniqueness.loc[s] = (1. / occurrences.loc[s:e]).mean()
        return avg_uniquess

    def count_occurences(
        self, closed_index, horizon, num_of_threads, chunk_size
    ):
        """
        Compute all occurrences into the event space.
        """
        events = horizon["start"]
        occurances = parallel_map_df(
            interval_count_occurrences,
            events,
            num_of_threads,
            chunk_size,
            horizon=horizon,
            closed_index=closed_index,
        )
        occurrences = occurrences.reindex(closed_index).fillna(0)
        return occurrences

    def average_uniqueness(
        self, occurrences, horizon, num_of_threads, chunk_size
    ):
        """
        Compute all average uniqueness into the event space.
        """
        events = horizon["start"]
        avg_uniqueness = parallel_map_df(
            interval_average_uniqueness,
            events,
            num_of_threads,
            chunk_size,
            horizon=horizon,
            ocurrences=ocurrences,
        )
        return avg_uniqueness

    def bootstrap_selection(self):
        pass

    def sample_weights(self):
        pass

    def cross_validation(self):
        pass


class MetaModel():
    def __init__(self):
        os.makedirs("logs/", exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            "logs/metamodel.log", maxBytes=200 * 1024 * 1024, backupCount=1
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        self._logger = logging.getLogger("Logger")
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(handler)

    def bagging_model(self):
        pass

    def random_forest(self):
        pass

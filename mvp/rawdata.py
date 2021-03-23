import os
import numpy as np
import pandas as pd
import datetime as dt
import sqlite3 as sql3
import pandas_datareader as pdr
from numba import njit, prange, int32, float64

__all__ = ["RawData", "DailyDataYahoo"]


@njit(int32(int32, float64[:], int32[:], float64))
def indexing_cusum(n, values, accum_ind, threshold):
    """
    Mark all new indexes before which the accumulated
    sum of `values` exceeded a `threshold`. Used to
    mark indexes of new candles in a dataframe.

    Parameters
    ----------
    `n` : ``int``
        size of `values` array
    `values` : ``numpy.array(numpy.float64)``
        values to compute accumulated sum
    `accum_ind` : ``numpy.array(int32)``
        store indexes strides in which cusum exceeds the threshold. Size `n`
    `threshold` : ``float``

    """
    cusum = 0.0
    accum_ind[0] = 0
    j = 1
    for i in prange(n):
        cusum = cusum + values[i]
        if cusum >= threshold:
            accum_ind[j] = i + 1
            j = j + 1
            cusum = 0.0
    return j


@njit(int32(int32, int32[:], int32[:]))
def indexing_new_days(n, days, new_days_ind):
    """
    Mark all indexes in which a new day begins

    Parameters
    ----------
    `n` : ``int``
        size of `days` array
    `days` : ``numpy.array(numpy.int32)``
        days correponding to datetime dataframe index. Have `n` elements
    `new_days_ind` : ``numpy.array(numpy.int32)``
        store indexes in which a new day begins. Must have size of `n`

    """
    new_days_ind[0] = 0
    j = 1
    for i in prange(n - 1):
        if days[i + 1] != days[i]:
            new_days_ind[j] = i + 1
            j = j + 1
    return j


def get_db_symbols(db_path):
    """
    Get all symbols from sqlite using the file in `db_path`.

    """
    conn = sql3.connect(db_path)
    cursor = conn.cursor()
    table_names = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    db_symbols = [name[0] for name in table_names]
    conn.close()
    return db_symbols


class RawData:
    """
    Class to read symbol from minute-1 database and set as dataframe. Provide
    methods to sample the dataframe in other formats than 1-minute time
    spaced bar, which may provide enhanced statistical properties.

    Instance Variables
    ------------------
    `df` : ``pandas.DataFrame``
        dataframe containing the fundamental 1-minute stock prices
        open-high-low-close-volume values.
    `symbol` : ``string``
        company symbol code in stock market
    `db_path` : ``string``
        absolute path to database file
    `available_dates` : list[``pandas.Timestamp``]
        dates that stock market was openned (exclude holidays)
    `available_time_steps` : list[1, 5, 10, 15, 30 ,60, "day"]
        values accepted to change the sample interval from 1-minute

    """

    def __init__(self, symbol, db_path, preload={}):
        """
        Initialize a class with 1-minute time frame stock prices data
        and optionally other types of candle bars if informed in `preload`

        Parameters
        ----------
        `symbol` : ``str``
            symbol code of the company to load data
        `db_path` : ``str``
            full path to 1-minute sample database file
        `preload` : ``dict``
            {
                "time": list[``int`` , "day"]   (new time interval of bars)
                "tick": list[``int``]       (bars in amount of deals occurred)
                "volume": list[``int``]     (bars in amount of volume)
                "money": list[``int``]      (bars in amount of money)
            }
            A single integer value works as well instead of list of integers
            For daily bars use the string "day" in "time" key. Specifically
            in time intervals, the set of accepted values are 1, 5, 10, 15,
            30, 60 and as mentioned "day".

        """
        self.symbol = symbol
        if not os.path.isfile(db_path):
            raise IOError("Database file {} not found".format(db_path))
        self.db_path = db_path
        self.db_version = self.db_path.split("_")[-1].split(".")[0]
        try:
            self.df = self.__get_data_from_db(self.db_path)
        except:
            raise ValueError(
                "symbol {} not found in database {}".format(symbol, db_path)
            )
        self.available_dates = self.df.index.normalize()
        self.available_time_steps = [1, 5, 10, 15, 30, 60, "day"]
        self.__bar_attr = {
            "time": "change_sample_interval",
            "tick": "tick_bars",
            "volume": "volume_bars",
            "money": "money_bars",
        }
        self.__cached_dataframes = {}
        self.__cached_dataframes["time_1"] = self.df
        for df_type in preload.keys():
            if df_type not in self.__bar_attr.keys():
                print(
                    "bar {} requested not in availabe ones : {}".format(
                        df_type, list(self.__bar_attr.keys())
                    )
                )
                continue
            if isinstance(preload[df_type], list):
                for step in preload[df_type]:
                    self.__cache_insert_dataframe(df_type, step)
            else:
                self.__cache_insert_dataframe(df_type, preload[df_type])

    def __get_data_from_db(self, db_path):
        conn = sql3.connect(db_path)
        df = pd.read_sql("SELECT * FROM {}".format(self.symbol), conn)
        conn.close()
        df.index = pd.to_datetime(df["DateTime"])
        df.drop(["DateTime"], axis=1, inplace=True)
        return df

    def __cache_insert_dataframe(self, bar_type, step):
        if isinstance(step, float):
            step = int(step)
        key_format = "{}_{}".format(bar_type, step)
        if key_format in self.__cached_dataframes.keys():
            return
        if bar_type == "time":
            if step not in self.available_time_steps:
                print("Time step requested {} not available".format(step))
                return
            base_dir = os.path.dirname(self.db_path)
            db_filename = "minute{}_database_{}.db".format(
                step, self.db_version
            )
            if step == "day":
                db_filename = "daily_database_{}.db".format(self.db_version)
            db_full_path = os.path.join(base_dir, db_filename)
            if os.path.isfile(db_full_path):
                try:
                    self.__cached_dataframes[
                        key_format
                    ] = self.__get_data_from_db(db_full_path)
                    print("Interval {} loaded from database".format(step))
                    return
                except:
                    pass
        self.__cached_dataframes[key_format] = self.__getattribute__(
            self.__bar_attr[bar_type]
        )(step=step)

    def __reassemble_df(self, df, strides):
        """
        Group intervals of the dataframe `df` in new open-high-low-close bars
        between strides of indexes `strides[i - 1]` to `strides[i]` for i > 0

        Return
        ------
        ``pandas.Dataframe``

        """
        nbars = strides.size
        bar_opening_time = df.index[strides[: nbars - 1]]
        bar_final_time = df.index[strides[1:nbars] - 1]
        df_matrix = np.empty([nbars - 1, 7])
        for i in range(1, nbars):
            df_slice = df.iloc[strides[i - 1] : strides[i]]
            bar_opn = df_slice.iloc[0]["Open"]
            bar_cls = df_slice.iloc[-1]["Close"]
            bar_max = df_slice["High"].max()
            bar_min = df_slice["Low"].min()
            bar_vol = df_slice["Volume"].sum()
            bar_tks = df_slice["TickVol"].sum()
            bar_money = (
                df_slice["Volume"] * (df_slice["Open"] + df_slice["Close"]) / 2
            ).sum()
            df_matrix[i - 1, 0] = bar_opn
            df_matrix[i - 1, 1] = bar_max
            df_matrix[i - 1, 2] = bar_min
            df_matrix[i - 1, 3] = bar_cls
            df_matrix[i - 1, 4] = bar_tks
            df_matrix[i - 1, 5] = bar_vol
            df_matrix[i - 1, 6] = bar_money
        new_df = pd.DataFrame(
            df_matrix,
            columns=[
                "Open",
                "High",
                "Low",
                "Close",
                "TickVol",
                "Volume",
                "Money",
            ],
            index=bar_final_time,
        )
        new_df.index.name = "DateTime"
        new_df.insert(0, "OpenTime", bar_opening_time)
        return new_df.astype({"TickVol": "int32", "Volume": "int32"})

    def cache_dataframes(self):
        return list(self.__cached_dataframes)

    def cache_dataframes_size(self):
        full_size = 0
        for df in self.__cached_dataframes.values():
            full_size = full_size + df.__sizeof__()
        return full_size + self.df.__sizeof__()

    def cache_dataframes_clean(self):
        del self.__cached_dataframes
        self.__cached_dataframes = {}

    def assert_window(self, start, stop):
        """
        Ensure that two variables can be used to slice a dataframe window
        either by time or index location. In case `None` is given the
        first and last point of dataframe is considered

        Parameters
        ----------
        `start` : ``None`` or ``pandas.Timestamp`` or ``int``
        `stop` : ``None`` or ``pandas.Timestamp`` or ``int``

        Return
        ------
        ``tuple(pandas.Timestamp, pandas.Timestamp)``
            `start`, `stop` values as time indexing

        """
        if start == stop == None:
            return self.df.index[0], self.df.index[-1]
        if start == None:
            start = self.df.index[0]
        if stop == None:
            stop = self.df.index[-1]
        if not isinstance(start, int) and not isinstance(start, pd.Timestamp):
            raise ValueError("{} is not a valid starting point".format(start))
        if not isinstance(stop, int) and not isinstance(stop, pd.Timestamp):
            raise ValueError("{} is not a valid stopping point".format(stop))
        if isinstance(start, int):
            start = self.df.index[start]
        if isinstance(stop, int):
            stop = self.df.index[stop]
        if stop <= start:
            raise ValueError("{} is not greater than {}".format(stop, start))
        return start, stop

    def tick_bars(self, start=None, stop=None, step=10000):
        """
        Convert 1-minute time spaced dataframe to
        (approximately) `step` ticks spaced

        Parameter
        ---------
        `start` : ``pandas.Timestamp`` or ``int``
            first index to use
        `stop` : ``pandas.Timestamp`` or ``int``
            last index to use
        `step` : ``int``
            number of ticks/deals to form a new bar (dataframe row)

        Return
        ``pandas.DataFrame``
            Dataframe sampled in ticks/deals

        """
        start, stop = self.assert_window(start, stop)
        if not isinstance(step, int):
            step = int(step)
        cache_key = "tick_{}".format(step)
        if cache_key in self.__cached_dataframes.keys():
            print("Taken from cache")
            return self.__cached_dataframes[cache_key].loc[start:stop].copy()
        df_window = self.df.loc[start:stop]
        nlines = df_window.shape[0]
        strides = np.empty(nlines + 1, dtype=np.int32)
        ticks_parser = df_window["TickVol"].astype(float).to_numpy()
        nbars = indexing_cusum(nlines, ticks_parser, strides, step)
        return self.__reassemble_df(df_window, strides[:nbars])

    def volume_bars(self, start=None, stop=None, step=1e7):
        """
        Convert 1-minute time spaced dataframe to
        (approximately) `step` volume spaced

        Parameter
        ---------
        `start` : ``pandas.Timestamp`` or ``int``
            first index to use
        `stop` : ``pandas.Timestamp`` or ``int``
            last index to use
        `step` : ``int``
            volume required to form a new bar (dataframe row)

        Return
        ``pandas.DataFrame``
            dataframe sampled in volume

        """
        start, stop = self.assert_window(start, stop)
        if not isinstance(step, int):
            step = int(step)
        cache_key = "volume_{}".format(step)
        if cache_key in self.__cached_dataframes.keys():
            print("Taken from cache")
            return self.__cached_dataframes[cache_key].loc[start:stop].copy()
        df_window = self.df.loc[start:stop]
        nlines = df_window.shape[0]
        strides = np.empty(nlines + 1, dtype=np.int32)
        vol_parser = df_window["Volume"].astype(float).to_numpy()
        nbars = indexing_cusum(nlines, vol_parser, strides, step)
        return self.__reassemble_df(df_window, strides[:nbars])

    def money_bars(self, start=None, stop=None, step=1e8):
        """
        Convert 1-minute time spaced dataframe to
        (approximately) `step` money spaced

        Parameter
        ---------
        `start` : ``pandas.Timestamp`` or ``int``
            first index to use
        `stop` : ``pandas.Timestamp`` or ``int``
            last index to use
        `step` : ``float``
            money volume required to form a new bar (dataframe row)

        Return
        ``pandas.DataFrame``
            dataframe sampled according to `step` money exchanged

        """
        start, stop = self.assert_window(start, stop)
        if not isinstance(step, int):
            step = int(step)
        cache_key = "money_{}".format(step)
        if cache_key in self.__cached_dataframes.keys():
            print("Taken from cache")
            return self.__cached_dataframes[cache_key].loc[start:stop].copy()
        df_window = self.df.loc[start:stop]
        nlines = df_window.shape[0]
        strides = np.empty(nlines + 1, dtype=np.int32)
        money_parser = (
            df_window["Volume"] * (df_window["Close"] + df_window["Open"]) / 2
        ).to_numpy()
        nbars = indexing_cusum(nlines, money_parser, strides, step)
        return self.__reassemble_df(df_window, strides[:nbars])

    def daily_bars(self, start=None, stop=None):
        """
        Convert 1-minute bars dataframe to daily bars

        Parameter
        ---------
        `start` : ``pandas.Timestamp`` or ``int``
            first index to use
        `end` : ``pandas.Timestamp`` or ``int``
            last index to use

        """
        start, stop = self.assert_window(start, stop)
        cache_key = "time_day"
        if cache_key in self.__cached_dataframes.keys():
            print("Taken from cache")
            return self.__cached_dataframes[cache_key].loc[start:stop].copy()
        df_window = self.df.loc[start:stop]
        nlines = df_window.shape[0]
        strides = np.empty(nlines + 1, dtype=np.int32)
        days_parser = df_window.index.day.to_numpy().astype(np.int32)
        nbars = indexing_new_days(nlines, days_parser, strides)
        daily_df = self.__reassemble_df(df_window, strides[:nbars])
        date_index = pd.to_datetime(daily_df.index.date)
        daily_df.index = date_index
        daily_df.index.name = "DateTime"
        return daily_df.drop("OpenTime", axis=1)

    def change_sample_interval(self, start=None, stop=None, step=60):
        """
        Return a dataframe resizing the sample time interval
        IGNORING the market opening and closing periods.

        Parameters
        ----------
        `start` : ``datetime.datetime``
            Initial time instant (pandas.Timestamp).
        `stop` : ``datetime.datetime``
            Final time instant (pandas.Timestamp).
        `step` : ``int``
            In minutes. Available values are [1,5,10,15,30,60] (default 60)

        Return
        ------
        ``pandas.DataFrame``
            new dataframe with bars sampled in the `step` interval given

        WARNING
        -------
        In case step is 5, 10, 15, 30, 60 this function is highly
        demanding to compute for the entire 1-minute dataframe of
        about 10^4 data bars. Tipically for the entire dataframe,
        it takes several minutes.

        """
        start, stop = self.assert_window(start, stop)
        if step not in self.available_time_steps:
            raise ValueError(
                "Time step requested not in ", self.available_time_steps
            )
        cache_key = "time_{}".format(step)
        if cache_key in self.__cached_dataframes.keys():
            print("Taken from cache")
            return self.__cached_dataframes[cache_key].loc[start:stop].copy()
        if step == "day":
            return self.daily_bars(start, stop)
        work_df = self.df.loc[start:stop]
        if step == 1:
            return work_df.copy()
        nlines = work_df.shape[0]
        strides = np.empty(nlines + 1, dtype=np.int32)
        days_parser = work_df.index.day.to_numpy().astype(np.int32)
        ndays = indexing_new_days(nlines, days_parser, strides)
        day_strides = strides[:ndays]
        bar_list = []
        for i in range(ndays - 1):
            opening_min = work_df.index[day_strides[i]].minute
            first_bar_minutes = step + 1 - opening_min % step
            t1 = work_df.index[day_strides[i]] + pd.Timedelta(
                minutes=first_bar_minutes
            )
            t2 = t1 + pd.Timedelta(minutes=step)
            while t2 < work_df.index[day_strides[i + 1] - 1]:
                window_df = work_df.loc[t1:t2]
                if window_df.empty:
                    t1 = t2
                    t2 = t2 + pd.Timedelta(minutes=step)
                    continue
                bar_vol = window_df["Volume"].sum()
                bar_tck = window_df["TickVol"].sum()
                bar_max = window_df["High"].max()
                bar_min = window_df["Low"].min()
                bar_opn = window_df.iloc[0]["Open"]
                bar_cls = window_df.iloc[-1]["Close"]
                bar_list.append(
                    [
                        t2 - pd.Timedelta(minutes=1),
                        bar_opn,
                        bar_max,
                        bar_min,
                        bar_cls,
                        bar_tck,
                        bar_vol,
                    ]
                )
                t1 = t2
                t2 = t2 + pd.Timedelta(minutes=step)
        new_df = pd.DataFrame(
            bar_list,
            columns=[
                "DateTime",
                "Open",
                "High",
                "Low",
                "Close",
                "TickVol",
                "Volume",
            ],
        )
        new_df.set_index("DateTime", inplace=True)
        return new_df


class DailyDataYahoo:
    """
    Class to read symbol from day-1 Yahoo database and set as data-frame
    """

    def __init__(self, symbol, db_path):
        self.symbol = symbol
        if not os.path.isfile(db_path):
            raise IOError("Database file {} not found".format(db_path))
        self.db_path = db_path
        self.df = self.__get_data_from_db()

    def __get_data_from_db(self):
        conn = sql3.connect(self.db_path)
        try:
            df = pd.read_sql("SELECT * FROM {}".format(self.symbol), conn)
            df.index = pd.to_datetime(df["Date"])
            df.drop(["Date"], axis=1, inplace=True)
        except Exception as e:
            init_day = dt.date(2010, 1, 2)
            final_day = dt.date.today()
            print(
                e,
                "Trying to download with pandas-datareader"
                " from {} to {}".format(init_day, final_day),
            )
            df = pdr.DataReader(
                self.symbol + ".SA", "yahoo", init_day, final_day
            )
            df.rename(columns={"Adj Close": "AdjClose"}, inplace=True)
        conn.close()
        return df

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
    Mark all indexes in which the accumulated sum of values exceed a threshold

    Parameters
    ----------
    `n` : ``int``
        size of `values` array
    `values` : ``numpy.array(numpy.float64)``
        values to calculate accumulated sum
    `accum_ind` : ``numpy.array(int32)``
        store indexes in which cusum exceeds the threshold. Size `n`
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
    Get all symbols for the connected Sqlite database.

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
    methods to sample the dataframe in other formats than linearly time
    spaced bar, which may provide enhanced statistical properties.
    """

    def __init__(self, symbol, db_path):
        """
        Initialize a class with a simple 1-minute time frame stock prices data

        Parameters
        ----------
        `symbol` : ``str``
            symbol code of the company
        `db_path` : ``str``
            full path to database file

        """
        self.symbol = symbol
        if not os.path.isfile(db_path):
            raise IOError("Database file {} not found".format(db_path))
        self.db_path = db_path
        try:
            self.df = self.__get_data_from_db()
        except:
            raise ValueError(
                "symbol {} not found in database {}".format(symbol, db_path)
            )
        self.available_dates = self.df.index.normalize()

    def __get_data_from_db(self):
        conn = sql3.connect(self.db_path)
        df = pd.read_sql("SELECT * FROM {}".format(self.symbol), conn)
        conn.close()
        df.index = pd.to_datetime(df["DateTime"])
        df.drop(["DateTime"], axis=1, inplace=True)
        return df

    def __assert_window(self, start, stop):
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
        if start == None:
            start = self.df.index[0]
        if stop == None:
            stop = self.df.index[-1]
        if not isinstance(start, int):
            if not isinstance(start, pd.Timestamp):
                raise ValueError(
                    "{} is not a valid starting point".format(start)
                )
        if not isinstance(stop, int):
            if not isinstance(stop, pd.Timestamp):
                raise ValueError(
                    "{} is not a valid stopping point".format(stop)
                )
        if isinstance(start, int):
            start = self.df.index[start]
        if isinstance(stop, int):
            stop = self.df.index[stop]
        if stop <= start:
            raise ValueError("{} is not greater than {}".format(stop, start))
        return start, stop

    def __reassemble_df(self, df, strides):
        """
        Group intervals of a dataframe in new open-high-low-close bars
        between strides of indexes `strides[i - 1]` to `strides[i]`

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
            index=bar_opening_time,
        )
        new_df.index.name = "OpenTime"
        new_df.insert(0, "CloseTime", bar_final_time)
        return new_df.astype({"TickVol": "int32", "Volume": "int32"})

    def tick_bars(self, start=None, stop=None, bar_size_th=10000):
        """
        Convert 1-minute time spaced dataframe to
        (approximately) `bar_size_th` ticks spaced

        Parameter
        ---------
        `start` : ``pandas.Timestamp`` or ``int``
            first index to use
        `stop` : ``pandas.Timestamp`` or ``int``
            last index to use
        `bar_size_th` : ``int``
            number of ticks/deals to form a new bar (dataframe row)

        Return
        ``pandas.DataFrame``
            Dataframe sampled in ticks/deals

        """
        start, stop = self.__assert_window(start, stop)
        if not isinstance(bar_size_th, int):
            bar_size_th = int(bar_size_th)
        df_window = self.df.loc[start:stop]
        nlines = df_window.shape[0]
        strides = np.empty(nlines + 1, dtype=np.int32)
        ticks_parser = df_window["TickVol"].astype(float).to_numpy()
        nbars = indexing_cusum(nlines, ticks_parser, strides, bar_size_th)
        return self.__reassemble_df(df_window, strides[:nbars])

    def volume_bars(self, start=None, stop=None, bar_size_th=1e7):
        """
        Convert 1-minute time spaced dataframe to
        (approximately) `bar_size_th` volume spaced

        Parameter
        ---------
        `start` : ``pandas.Timestamp`` or ``int``
            first index to use
        `stop` : ``pandas.Timestamp`` or ``int``
            last index to use
        `bar_size_th` : ``int``
            volume required to form a new candle-stick (data-frame row)

        Return
        ``pandas.DataFrame``
            dataframe sampled in volume

        """
        start, stop = self.__assert_window(start, stop)
        if not isinstance(bar_size_th, int):
            bar_size_th = int(bar_size_th)
        df_window = self.df.loc[start:stop]
        nlines = df_window.shape[0]
        strides = np.empty(nlines + 1, dtype=np.int32)
        vol_parser = df_window["Volume"].astype(float).to_numpy()
        nbars = indexing_cusum(nlines, vol_parser, strides, bar_size_th)
        return self.__reassemble_df(df_window, strides[:nbars])

    def money_bars(self, start=None, stop=None, bar_size_th=1e8):
        """
        Convert 1-minute time spaced dataframe to
        (approximately) `bar_size_th` money spaced

        Parameter
        ---------
        `start` : ``pandas.Timestamp`` or ``int``
            first index to use
        `stop` : ``pandas.Timestamp`` or ``int``
            last index to use
        `bar_size_th` : ``float``
            money volume required to form a new candle-stick (dataframe row)

        Return
        ``pandas.DataFrame``
            dataframe sampled according to `bar_size_th` money exchanged

        """
        start, stop = self.__assert_window(start, stop)
        if not isinstance(bar_size_th, int):
            bar_size_th = int(bar_size_th)
        df_window = self.df.loc[start:stop]
        nlines = df_window.shape[0]
        strides = np.empty(nlines + 1, dtype=np.int32)
        money_parser = (
            df_window["Volume"] * (df_window["Close"] + df_window["Open"]) / 2
        ).to_numpy()
        nbars = indexing_cusum(nlines, money_parser, strides, bar_size_th)
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
        start, stop = self.__assert_window(start, stop)
        df_window = self.df.loc[start:stop]
        nlines = df_window.shape[0]
        strides = np.empty(nlines + 1, dtype=np.int32)
        days_parser = df_window.index.day.to_numpy().astype(np.int32)
        nbars = indexing_new_days(nlines, days_parser, strides)
        daily_df = self.__reassemble_df(df_window, strides[:nbars])
        date_index = pd.to_datetime(daily_df.index.date)
        daily_df.index = date_index
        return daily_df.drop("BarCloseTime", axis=1)

    def change_sample_interval(self, start=None, stop=None, time_step=60):
        """
        Return a dataframe resizing the sample time interval
        ignoring the market opening and closing periods.

        Parameters
        ----------
        start : ``datetime.datetime``
            Initial time instant (pandas.Timestamp).
        stop : ``datetime.datetime``
            Final time instant (pandas.Timestamp).
        time_step : ``int``
            In minutes. Available values are [1,5,10,15,30,60] (default 60)

        """
        available_steps = [1, 5, 10, 15, 30, 60]
        if time_step not in available_steps:
            raise ValueError("Time step requested not in ", available_steps)
        start, stop = self.__assert_window(start, stop)
        work_df = self.df.loc[start:stop]
        if time_step == 1:
            return work_df.copy()
        nlines = work_df.shape[0]
        strides = np.empty(nlines + 1, dtype=np.int32)
        days_parser = work_df.index.day.to_numpy().astype(np.int32)
        ndays = indexing_new_days(nlines, days_parser, strides)
        day_strides = strides[:ndays]
        bar_list = []
        for i in range(ndays - 1):
            opening_min = work_df.index[day_strides[i]].minute
            first_bar_minutes = time_step + 1 - opening_min % time_step
            t1 = work_df.index[day_strides[i]] + pd.Timedelta(
                minutes=first_bar_minutes
            )
            t2 = t1 + pd.Timedelta(minutes=time_step)
            while t2 < work_df.index[day_strides[i + 1] - 1]:
                window_df = work_df.loc[t1:t2]
                if window_df.empty:
                    t1 = t2
                    t2 = t2 + pd.Timedelta(minutes=time_step)
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
                t2 = t2 + pd.Timedelta(minutes=time_step)
        new_df = pd.DataFrame(
            bar_list,
            columns=[
                "CloseTime",
                "Open",
                "High",
                "Low",
                "Close",
                "TickVol",
                "Volume",
            ],
        )
        new_df.set_index("CloseTime", inplace=True)
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

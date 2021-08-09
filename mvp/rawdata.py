""" Raw data processing from databases

    This module provides tools to to extract data from databases and set
    in suitable data structure to provide a systematic workflow for data
    analysis. The database must have a specific format that must provide
    data-tables with the following columns:
        DateTime : YYYY-MM-DD HH:MM:SS
        Open : float value
        High : float value
        Low : float value
        Close : float Value
        TickVol : integer value
        Volume : integer value
    These value must correspond to packaged data bar in 1-minute of stock
    market trades. The column names provided above are case sensitive and
    `DateTime` must be specifically given as text entry in that format

    Main class in this module:

    ``RawData``
        Provide attributes to access data in ``pandas.DataFrame`` format.
        The `DateTime` database column is set as index and the others are
        kept as dataframe columns in `RawData.df` attribute. The 1-minute
        data are consider as the most fundamental, from which all methods
        act on to transform in other formats.
        Despite it is usual to organize stock market data in linear spaced
        time intervals, from 1-minute packed data, it is possible to build
        other type of packed data using different thresholds to form bars.
        Instead of packaging data in minute of market trades occurred this
        class provide ways to pack trades in amount of deals/ticks, volume
        of shares exchanged or even money. For these features see methods
        that has the suffix "_bars"

"""

import os
import sqlite3

import numpy as np
import pandas as pd

from mvp import utils


def get_db_symbols(db_path):
    """ Get list of symbols available in database located at `db_path` """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    table_names = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    db_symbols = [name[0] for name in table_names]
    conn.close()
    return db_symbols


def assert_bar_type(bar_type_name):
    """
    If `bar_type_name` is not valid raise ``AttributeError``
    Get all available bar types in function `available_bars`
    """
    if not hasattr(RawData, bar_type_name + "_bars"):
        raise AttributeError("Invalid bar type '{}'".format(bar_type_name))


def assert_data_field(data_field_name):
    """
    If `data_field_name` is not valid raise ``AttributeError``
    Get valid data fields in function `available_data_fields`
    """
    if not hasattr(RawData, "get_" + data_field_name):
        raise AttributeError("Invalid data field '{}'".format(data_field_name))


def available_bars():
    """
    Return list of bar names accessible from ``RawData`` class
    These names corresponding to ``RawData`` methods without a
    "_bars" suffix. These methods describe ways to pack stocks
    trade prices in a certain intervals
    """
    methods_list = list(RawData.__dict__.keys())
    return [
        method_name.split("_")[0]
        for method_name in methods_list
        if method_name.split("_")[1] == "bars"
    ]


def available_data_fields():
    """
    Return list of stock market data types accessible from ``RawData``
    class. These names corresponding to ``RawData`` methods without a
    "get_" prefix. These methods describe which data fields are present
    in bars/candlesticks packed data
    """
    methods_list = list(RawData.__dict__.keys())
    return [
        method_name.split("_")[1]
        for method_name in methods_list
        if method_name.split("_")[0] == "get"
    ]


class RawData:
    """
    Class to read databases and manipulate dataframes for
    a stock market company. Provide methods to reset data
    in different types of bars/candlesticks besides time

    Also provide methods to access basic candlestick data
    such as open, high, low and close values

    Some general parameters are required to access data:

    `start` : ``pandas.Timestamp``(preferable) or ``int``(acceptable)
        Initial date/time to start the data time series
        If integer use it as index from 1-minute dataframe series
    `stop` :  ``pandas.Timestamp``(preferable) or ``int``(acceptable)
        Final date/time to stop the data time series
        If integer use it as index from 1-minute dataframe series
    `step` : ``int``
        Value required to accumulate in bars packaging
        In `time_bars` is the time interval in minutes
        and in `tick_bars` is how many trades occurred

    Warnings
    --------
    Mind that if integer is given for `start` / `stop` the
    time index considered is taken from 1-minute dataframe
    For non-time bars, suitable steps must be chosen since
    if a very small value is given the uncertainty becomes
    appreciable due to the 1-minute bars fluctuations, and
    the result will be very similar of those with few minute
    bars, such as 15-minute or 60-minute

    Attributes
    ----------
    `df` : ``pandas.DataFrame``
        Fundamental dataframe containing the 1-minute stock prices
        open-high-low-close-volume values.
    `symbol` : ``str``
        company symbol code in stock market
    `available_dates` : list[``pandas.Timestamp``]
        dates that stock market was openned (exclude holidays and weekends)
    `available_time_steps` : list[1, 5, 10, 15, 30 ,60, "day"]
        values accepted to change the sample interval from 1-minute

    """

    def __init__(self, symbol, db_path, preload={}):
        """
        Initialize a class with 1-minute time frame stock prices data
        and optionally other types of bars if informed in `preload`

        Parameters
        ----------
        `symbol` : ``str``
            symbol code of the company to load data
        `db_path` : ``str``
            full path to 1-minute database
        `preload` : ``dict``
            Initialize dataframes packaging data in different intervals
            of 1-minute bars, not necessarily linearly time spaced. The
            interval is understood as a series of trades in stock market
            that produce some threshold in accumulated value. Examples:
            {
                "time": ``int`` or "day"    (new time interval of bars)
                "tick": ``int``             (bars in amount of trades)
                "volume": ``int``           (bars in amount of volume)
                "money": ``int``            (bars in amount of money)
            }
            For the dict values, list of the types above are also accepted
            For daily bars, use the string "day" in "time" key. Especially
            in time intervals, the set of accepted values are 1, 5, 10, 15,
            30, 60 and "day".

        """
        self.symbol = symbol
        if not os.path.isfile(db_path):
            raise IOError("Database file {} not found".format(db_path))
        self.db_path = db_path
        self.db_version = self.db_path.split("_")[-1].split(".")[0]
        try:
            self.df = self.__get_data_from_db(self.db_path)
        except Exception:
            raise ValueError(
                "symbol {} not found in database {}".format(symbol, db_path)
            )
        self.available_dates = self.df.index.normalize()
        self.available_time_steps = [1, 5, 10, 15, 30, 60, "day"]
        self.__cached_dataframes = {}
        self.__cached_dataframes["time_1"] = self.df
        for bar_type, params in preload.items():
            if not hasattr(self, bar_type + "_bars"):
                print("\nBar '{}' requested not available".format(bar_type))
                continue
            if not isinstance(params, list):
                params = [params]
            for param in params:
                self.__cache_insert_dataframe(bar_type, param)

    def __get_data_from_db(self, db_path):
        """ Query in database `self.symbol` and set dataframe """
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT * FROM {}".format(self.symbol), conn)
        conn.close()
        df.index = pd.to_datetime(df["DateTime"])
        df.drop(["DateTime"], axis=1, inplace=True)
        return df

    def __format_new_interval_db_name(self, time_step):
        """ Format name of database according to time interval convention """
        base_dir = os.path.dirname(self.db_path)
        db_filename = "minute{}_database_{}.db".format(
            time_step, self.db_version
        )
        if time_step == "day":
            db_filename = "daily_database_{}.db".format(self.db_version)
        return os.path.join(base_dir, db_filename)

    def __cache_insert_dataframe(self, bar_type, step):
        """ Insert dataframe in inner memory cache dictionary """
        try:
            step = int(step)
        except Exception:
            pass
        if step is None:
            step = 1
        if bar_type != "time" and step == 1:
            t_init, t_end = self.df.index[0], self.df.index[-1]
            step = self.__appropriate_step(t_init, t_end, bar_type)
        key_format = "{}_{}".format(bar_type, step)
        if key_format in self.__cached_dataframes.keys():
            return
        if bar_type == "time":
            if step not in self.available_time_steps:
                print("Time step requested {} not available".format(step))
                return
            db_full_path = self.__format_new_interval_db_name(step)
            try:
                self.__cached_dataframes[key_format] = self.__get_data_from_db(
                    db_full_path
                )
                return
            except Exception:
                pass
        bar_method = self.__getattribute__(bar_type + "_bars")
        self.__cached_dataframes[key_format] = bar_method(step=step)

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
        """ Return list of dataframes-names set in cache frame """
        return list(self.__cached_dataframes.keys())

    def cache_dataframes_size(self):
        """ Return memory required for cache in bytes """
        full_size = 0
        for df in self.__cached_dataframes.values():
            full_size = full_size + df.__sizeof__()
        return full_size

    def cache_dataframes_clean(self):
        """ Remove all dataframes currently in cache """
        minute1_df_recovery = self.df.copy()
        del self.__cached_dataframes
        del self.df
        self.df = minute1_df_recovery
        self.__cached_dataframes = {}
        self.__cached_dataframes["time_1"] = self.df

    def assert_window(self, start=None, stop=None):
        """
        Ensure that two variables can be used to slice a dataframe window
        If ``None`` types are given return the constraining dates of data
        Integers are considered as index of minute-1 dataframe

        Parameters
        ----------
        `start` : ``None`` or ``pandas.Timestamp`` or ``int``
        `stop` : ``None`` or ``pandas.Timestamp`` or ``int``

        Return
        ------
        ``tuple(pandas.Timestamp, pandas.Timestamp)``
            `start`, `stop` values as time indexing

        """
        if start is None and stop is None:
            return self.df.index[0], self.df.index[-1]
        if start is None:
            start = self.df.index[0]
        if stop is None:
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

    def __appropriate_step(self, start, stop, bar_type):
        """ Return a suitable step for `bar_type` not being time """
        daily_df = self.daily_bars(start, stop)
        if bar_type == "tick":
            float_step = daily_df.TickVol.mean() / 2
        elif bar_type == "volume":
            float_step = daily_df.Volume.mean() / 2
        else:
            float_step = (daily_df.Volume * daily_df.Close).mean() / 2
        return int(float_step)

    def get_close(self, start=None, stop=None, step=1, bar_type="time"):
        """ Get close price time series """
        bar_method = self.__getattribute__(bar_type + "_bars")
        if bar_type != "time" and step == 1:
            step = self.__appropriate_step(start, stop, bar_type)
        return bar_method(start, stop, step).Close

    def get_open(self, start=None, stop=None, step=1, bar_type="time"):
        """ Get open price time series """
        bar_method = self.__getattribute__(bar_type + "_bars")
        if bar_type != "time" and step == 1:
            step = self.__appropriate_step(start, stop, bar_type)
        df_bars = bar_method(start, stop, step)
        try:
            df_bars.set_index("OpenTime", inplace=True)
        except KeyError:
            if not isinstance(step, str) and step > 1:
                df_bars.set_index(df_bars.index - pd.Timedelta(step))
        df_bars.index.name = "DateTime"
        return df_bars.Open

    def get_high(self, start=None, stop=None, step=1, bar_type="time"):
        """ Get high price time series """
        bar_method = self.__getattribute__(bar_type + "_bars")
        if bar_type != "time" and step == 1:
            step = self.__appropriate_step(start, stop, bar_type)
        return bar_method(start, stop, step).High

    def get_low(self, start=None, stop=None, step=1, bar_type="time"):
        """ Get low price time series """
        bar_method = self.__getattribute__(bar_type + "_bars")
        if bar_type != "time" and step == 1:
            step = self.__appropriate_step(start, stop, bar_type)
        return bar_method(start, stop, step).Low

    def get_volume(self, start=None, stop=None, step=1, bar_type="time"):
        """ Get volume time series """
        bar_method = self.__getattribute__(bar_type + "_bars")
        if bar_type != "time" and step == 1:
            step = self.__appropriate_step(start, stop, bar_type)
        return bar_method(start, stop, step).Volume

    def tick_bars(self, start=None, stop=None, step=None):
        """
        Set dataframe bars for every `step`-deals/ticks traded in stock market

        Parameter
        ---
        `start` : ``pandas.Timestamp`` or ``int``
            Initial time instant or index of 1-minute dataframe
        `stop` : ``pandas.Timestamp`` or ``int``
            Final time instant or index of 1-minute dataframe
        `step` : ``int``
            number of ticks/deals to form a new bar (dataframe row)
            Hint : To use some reasonable value compute the mean of
            `TickVol` column in daily intervals
            By default use half of day-average

        Return
        ---
        ``pandas.DataFrame``
            Dataframe sampled in ticks/deals intervals

        Warning
        ---
        Avoid using typically small values for `step` according to the symbol
        since it induces small time intervals increasing fluctuations. These
        fluctuations are due to number of trades variations in 1-minute bars
        that is the fundamental data used for any other bar type

        """
        start, stop = self.assert_window(start, stop)
        try:
            step = int(step)
        except Exception:
            pass
        if step is None or step == 1:
            step = self.__appropriate_step(start, stop, "tick")
        cache_key = "tick_{}".format(step)
        if cache_key in self.__cached_dataframes.keys():
            return self.__cached_dataframes[cache_key].loc[start:stop].copy()
        df_window = self.df.loc[start:stop]
        nlines = df_window.shape[0]
        strides = np.empty(nlines + 1, dtype=np.int32)
        ticks_parser = df_window["TickVol"].astype(float).to_numpy()
        nbars = utils.indexing_cusum(nlines, ticks_parser, strides, step)
        return self.__reassemble_df(df_window, strides[:nbars])

    def volume_bars(self, start=None, stop=None, step=None):
        """
        Set dataframe bars for every `step`-volume traded in stock market

        Parameter
        ---
        `start` : ``pandas.Timestamp`` or ``int``
            Initial time instant or index of 1-minute dataframe
        `stop` : ``pandas.Timestamp`` or ``int``
            Final time instant or index of 1-minute dataframe
        `step` : ``int``
            volume required to form a new bar (dataframe row)
            Hint : To use some reasonable value compute the mean of
            `Volume` column in daily intervals
            By default use half of day-average

        Return
        ---
        ``pandas.DataFrame``
            dataframe sampled in volume

        Warning
        ---
        Avoid using typically small values for `step` according to the symbol
        since it induces small time intervals increasing fluctuations. These
        fluctuations are due to number of trades variations in 1-minute bars
        that is the fundamental data used for any other bar type

        """
        start, stop = self.assert_window(start, stop)
        try:
            step = int(step)
        except Exception:
            pass
        if step is None or step == 1:
            step = self.__appropriate_step(start, stop, "volume")
        cache_key = "volume_{}".format(step)
        if cache_key in self.__cached_dataframes.keys():
            return self.__cached_dataframes[cache_key].loc[start:stop].copy()
        df_window = self.df.loc[start:stop]
        nlines = df_window.shape[0]
        strides = np.empty(nlines + 1, dtype=np.int32)
        vol_parser = df_window["Volume"].astype(float).to_numpy()
        nbars = utils.indexing_cusum(nlines, vol_parser, strides, step)
        return self.__reassemble_df(df_window, strides[:nbars])

    def money_bars(self, start=None, stop=None, step=None):
        """
        Set dataframe bars for every `step`-money traded in stock market

        Parameter
        ---
        `start` : ``pandas.Timestamp`` or ``int``
            Initial time instant or index of 1-minute dataframe
        `stop` : ``pandas.Timestamp`` or ``int``
            Final time instant or index of 1-minute dataframe
        `step` : ``float``
            money volume required to form a new bar (dataframe row)
            Hint : To use some reasonable value compute the mean of
            close price multiplied by the volume in daily intervals
            By default use half of day-average

        Return
        ---
        ``pandas.DataFrame``
            dataframe sampled according to `step` money exchanged

        Warning
        ---
        Avoid using typically small values for `step` according to the symbol
        since it induces small time intervals increasing fluctuations. These
        fluctuations are due to number of trades variations in 1-minute bars
        that is the fundamental data used for any other bar type

        """
        start, stop = self.assert_window(start, stop)
        try:
            step = int(step)
        except Exception:
            pass
        if step is None or step == 1:
            step = self.__appropriate_step(start, stop, "money")
        cache_key = "money_{}".format(step)
        if cache_key in self.__cached_dataframes.keys():
            return self.__cached_dataframes[cache_key].loc[start:stop].copy()
        df_window = self.df.loc[start:stop]
        nlines = df_window.shape[0]
        strides = np.empty(nlines + 1, dtype=np.int32)
        money_parser = (
            df_window["Volume"] * (df_window["Close"] + df_window["Open"]) / 2
        ).to_numpy()
        nbars = utils.indexing_cusum(nlines, money_parser, strides, step)
        return self.__reassemble_df(df_window, strides[:nbars])

    def daily_bars(self, start=None, stop=None):
        """
        Convert 1-minute bars dataframe to daily bars

        Parameter
        ---------
        `start` : ``pandas.Timestamp`` or ``int``
            Initial time instant or index of 1-minute dataframe
        `stop` : ``pandas.Timestamp`` or ``int``
            Final time instant or index of 1-minute dataframe

        Return
        ------
        ``pandas.DataFrame``
            open-high-low-close in daily bars

        """
        start, stop = self.assert_window(start, stop)
        cache_key = "time_day"
        if cache_key in self.__cached_dataframes.keys():
            return self.__cached_dataframes[cache_key].loc[start:stop].copy()
        df_window = self.df.loc[start:stop]
        nlines = df_window.shape[0]
        strides = np.empty(nlines + 1, dtype=np.int32)
        days_parser = df_window.index.day.to_numpy().astype(np.int32)
        nbars = utils.indexing_new_days(nlines, days_parser, strides)
        daily_df = self.__reassemble_df(df_window, strides[:nbars])
        date_index = pd.to_datetime(daily_df.index.date)
        daily_df.index = date_index
        daily_df.index.name = "DateTime"
        return daily_df.drop("OpenTime", axis=1)

    def time_bars(self, start=None, stop=None, step=1):
        """
        Set time interval elapsed in open-high-low-close bars

        Parameters
        ----------
        `start` : ``pandas.Timestamp`` or ``int``
            Initial time instant or index of 1-minute dataframe
        `stop` : ``pandas.Timestamp`` or ``int``
            Final time instant or index of 1-minute dataframe
        `step` : ``int`` or "day"
            In minutes if integer

        Return
        ------
        ``pandas.DataFrame``
            dataframe with open-high-low-close bars in `step`-minutes

        WARNING
        -------
        In case step is 5, 10, 15, 30, 60 this function is highly
        demanding to compute if the respective databases are not
        provided, since use the 1-minute dataframes to group new
        bars. For 5 years long series it may take up to minutes

        """
        start, stop = self.assert_window(start, stop)
        if step not in self.available_time_steps:
            raise ValueError(
                "Time step requested not in ", self.available_time_steps
            )
        cache_key = "time_{}".format(step)
        if cache_key in self.__cached_dataframes.keys():
            return self.__cached_dataframes[cache_key].loc[start:stop].copy()
        try:
            df_from_db = self.__get_data_from_db(
                self.__format_new_interval_db_name(step)
            )
            return df_from_db.loc[start:stop]
        except Exception:
            pass
        if step == "day":
            return self.daily_bars(start, stop)
        work_df = self.df.loc[start:stop]
        nlines = work_df.shape[0]
        strides = np.empty(nlines + 1, dtype=np.int32)
        days_parser = work_df.index.day.to_numpy().astype(np.int32)
        ndays = utils.indexing_new_days(nlines, days_parser, strides)
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

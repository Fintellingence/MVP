import os
import pandas as pd
import datetime as dt
import sqlite3 as sql3
import pandas_datareader as pdr

__all__ = ["RawData", "DailyDataYahoo"]

class RawData:
    """
    Class to read symbol from minute-1 database and set as data-frame. Provide
    methods to refactor the data-frame in other formats than linearly time
    spaced bar which may exhibit better statistical properties.
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


    def __get_data_from_db(self):
        conn = sql3.connect(self.db_path)
        df = pd.read_sql("SELECT * FROM {}".format(self.symbol), conn)
        conn.close()
        df.index = pd.to_datetime(df["DateTime"])
        df.drop(["DateTime"], axis=1, inplace=True)
        return df


    def tick_bars(self, bar_size_th=1000):
        """
        Convert 1-minute time spaced data-frame to
        (approximately) `bar_size_th` ticks spaced

        Parameter
        ---------
        `bar_size_th` : ``int``
            number of ticks/deals to form a new candle-stick (data-frame row)

        Return
        `new_df` : ``pandas.DataFrame``

        """
        accum_ticks = 0
        n_bars = 0
        bar_list = []
        last_index = 0
        bar_initial_time = self.df.iloc[0].name
        initial_time_index = []
        for idx in range(self.df.index.size):
            accum_ticks = accum_ticks + self.df.iloc[idx]["TickVol"]
            if accum_ticks > bar_size_th:
                bar_final_time = self.df.iloc[idx].name
                bar_max = self.df.iloc[last_index : (idx + 1)]["High"].max()
                bar_min = self.df.iloc[last_index : (idx + 1)]["Low"].min()
                bar_vol = self.df.iloc[last_index : (idx + 1)]["Volume"].sum()
                bar_opn = self.df.iloc[last_index]["Open"]
                bar_cls = self.df.iloc[idx]["Close"]
                bar_list.append(
                    [
                        bar_final_time,
                        bar_opn,
                        bar_max,
                        bar_min,
                        bar_cls,
                        accum_ticks,
                        bar_vol,
                    ]
                )
                initial_time_index.append(bar_initial_time)
                accum_ticks = 0
                last_index = idx + 1
                bar_initial_time = bar_final_time + pd.Timedelta(minutes=1)
                n_bars = n_bars + 1
        newDF = pd.DataFrame(
            bar_list,
            columns=[
                "BarCloseTime",
                "Open",
                "High",
                "Low",
                "Close",
                "TickVol",
                "Volume",
            ],
            index=initial_time_index,
        )
        newDF.index.name = "BarOpenTime"
        return newDF

    def volume_bars(self, bar_size_th=1e6):
        """
        Convert 1-minute time spaced data-frame to
        (approximately) `bar_size_th` volume spaced

        Parameter
        ---------
        `bar_size_th` : ``int``
            volume required to form a new candle-stick (data-frame row)

        Return
        `new_df` : ``pandas.DataFrame``

        """
        accum_vol = 0
        n_bars = 0
        bar_list = []
        last_index = 0
        bar_initial_time = self.df.iloc[0].name
        initial_time_index = []
        for idx in range(self.df.index.size):
            accum_vol = accum_vol + self.df.iloc[idx]["Volume"]
            if accum_vol > bar_size_th:
                bar_final_time = self.df.iloc[idx].name
                bar_tks = self.df.iloc[last_index : (idx + 1)]["TickVol"].sum()
                bar_max = self.df.iloc[last_index : (idx + 1)]["High"].max()
                bar_min = self.df.iloc[last_index : (idx + 1)]["Low"].min()
                bar_opn = self.df.iloc[last_index]["Open"]
                bar_cls = self.df.iloc[idx]["Close"]
                bar_list.append(
                    [
                        bar_final_time,
                        bar_opn,
                        bar_max,
                        bar_min,
                        bar_cls,
                        bar_tks,
                        accum_vol,
                    ]
                )
                initial_time_index.append(bar_initial_time)
                accum_vol = 0
                last_index = idx + 1
                bar_initial_time = bar_final_time + pd.Timedelta(minutes=1)
                n_bars = n_bars + 1
        new_df = pd.DataFrame(
            bar_list,
            columns=[
                "BarCloseTime",
                "Open",
                "High",
                "Low",
                "Close",
                "TickVol",
                "Volume",
            ],
            index=initial_time_index,
        )
        new_df.index.name = "BarOpenTime"
        return new_df


    def money_bars(self, bar_size_th=1e6):
        """
        Convert 1-minute time spaced data-frame to
        (approximately) `bar_size_th` money spaced

        Parameter
        ---------
        `bar_size_th` : ``int``
            money volume required to form a new candle-stick (data-frame row)

        Return
        `new_df` : ``pandas.DataFrame``

        """
        money = 0
        n_bars = 0
        bar_list = []
        last_index = 0
        bar_initial_time = self.df.iloc[0].name
        initial_time_index = []
        for idx in range(self.df.index.size):
            mean_price = 0.5 * (
                    self.df.iloc[idx]["Close"] + self.df.iloc[idx]["Open"]
            )
            money = money + self.df.iloc[idx]["Volume"] * mean_price
            if money > bar_size_th:
                bar_final_time = self.df.iloc[idx].name
                bar_vol = self.df.iloc[last_index : (idx + 1)]["Volume"].sum()
                bar_tks = self.df.iloc[last_index : (idx + 1)]["TickVol"].sum()
                bar_max = self.df.iloc[last_index : (idx + 1)]["High"].max()
                bar_min = self.df.iloc[last_index : (idx + 1)]["Low"].min()
                bar_opn = self.df.iloc[last_index]["Open"]
                bar_cls = self.df.iloc[idx]["Close"]
                bar_list.append(
                    [
                        bar_final_time,
                        bar_opn,
                        bar_max,
                        bar_min,
                        bar_cls,
                        bar_tks,
                        bar_vol,
                    ]
                )
                initial_time_index.append(bar_initial_time)
                money = 0
                last_index = idx + 1
                bar_initial_time = bar_final_time + pd.Timedelta(minutes=1)
                n_bars = n_bars + 1
        new_df = pd.DataFrame(
            bar_list,
            columns=[
                "BarCloseTime",
                "Open",
                "High",
                "Low",
                "Close",
                "TickVol",
                "Volume",
            ],
            index=initial_time_index,
        )
        new_df.index.name = "BarOpenTime"
        return new_df


    def daily_bars(self):
        """
        Convert 1-minute time spaced data-frame to daily spaced

        Return
        `new_df` : ``pandas.DataFrame``

        """
        last_index = 0
        n_bars = 0
        bar_list = []
        date_index = []
        current_day = self.df.index[0].day
        for idx in range(self.df.index.size):
            if self.df.index[idx].day != current_day:
                bar_max = self.df.iloc[last_index:idx]["High"].max()
                bar_min = self.df.iloc[last_index:idx]["Low"].min()
                bar_vol = self.df.iloc[last_index:idx]["Volume"].sum()
                bar_tck = self.df.iloc[last_index:idx]["TickVol"].sum()
                bar_opn = self.df.iloc[last_index]["Open"]
                bar_cls = self.df.iloc[idx - 1]["Close"]
                bar_date = self.df.index[idx - 1].date()
                bar_list.append(
                    [bar_opn, bar_max, bar_min, bar_cls, bar_tck, bar_vol]
                )
                date_index.append(bar_date)
                current_day = self.df.index[idx].day
                last_index = idx
                n_bars = n_bars + 1
        date_index = pd.to_datetime(date_index)
        new_df = pd.DataFrame(
            bar_list,
            columns=["Open", "High", "Low", "Close", "TickVol", "Volume"],
            index=date_index,
        )
        new_df.index.name = "Date"
        return new_df


    def change_time_window(self, time_step=60):
        """
        Return a dataframe resizing the sample time interval

        Parameters
        ----------
        `time_step` : ``int``
            New time interval of the sample in minutes.
            Available ones are 5, 10, 15, 30, 60

        """
        available_step = [5, 10, 15, 30, 60]
        if time_step not in available_step:
            raise ValueError("New time step requested not in ", available_step)
        last_index = 0
        bar_list = []
        close_time_index = []
        n_bars = 0
        current_day = self.df.index[0].hour
        new_day_first_minute = True
        for idx in range(self.df.index.size):
            if (current_day != self.df.index[idx].day):
                current_day = self.df.index[idx].day
                new_day_first_minute = True
            if (self.df.index[idx].minute % time_step == 0
                    and not new_day_first_minute):
                bar_final_time = self.df.iloc[idx].name
                bar_vol = self.df.iloc[last_index : (idx + 1)]["Volume"].sum()
                bar_tck = self.df.iloc[last_index : (idx + 1)]["TickVol"].sum()
                bar_max = self.df.iloc[last_index : (idx + 1)]["High"].max()
                bar_min = self.df.iloc[last_index : (idx + 1)]["Low"].min()
                bar_opn = self.df.iloc[last_index]["Open"]
                bar_cls = self.df.iloc[idx]["Close"]
                bar_list.append(
                    [bar_opn, bar_max, bar_min, bar_cls, bar_tck, bar_vol]
                )
                close_time_index.append(bar_final_time)
                last_index = idx + 1
                n_bars = n_bars + 1
                minutes_elapsed = 0
            new_day_first_minute = False

        new_df = pd.DataFrame(
            bar_list,
            columns=["Open", "High", "Low", "Close", "TickVol", "Volume"],
            index=close_time_index,
        )
        new_df.index.name = "DateTime"
        return new_df


class DailyDataYahoo():
    """
    Class to read symbol from day-1 Yahoo database and set as data-frame
    """

    def __init__(self, symbol, db_path):
        self.symbol = symbol
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
            print(e, "Trying to download with pandas-datareader"
                    " from {} to {}".format(init_day, final_day)
            )
            df = pdr.DataReader(
                    self.symbol + ".SA",
                    "yahoo",
                    init_day,
                    final_day
            )
            df.rename(columns={"Adj Close": "AdjClose"}, inplace=True)
        conn.close()
        return df

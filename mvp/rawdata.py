import os
import pandas as pd
import sqlite3 as sql3


class RawData:
    """
    Class to read symbol from database and set as data-frame. Provide methods
    to refactor the data-frame in other formats than linearly time spaced bar
    which may exhibit better statistical properties.
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

    def tick_bars_df(self, bar_size_th=1000):
        """
        Convert 1-minute time spaced data-frame to
        (approximately) `bar_size_th` ticks spaced

        Parameter
        ---------
        `bar_size_th` : ``int``
            number of ticks/deals to form a new candle-stick (data-frame row)

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

    def volume_bars_df(self, bar_size_th=1e6):
        """
        Convert 1-minute time spaced data-frame to
        (approximately) `bar_size_th` volume spaced

        Parameter
        ---------
        `bar_size_th` : ``int``
            volume required to form a new candle-stick (data-frame row)

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


    def money_bars_df(self, bar_size_th=1e6):
        """
        Convert 1-minute time spaced data-frame to
        (approximately) `bar_size_th` money spaced

        Parameter
        ---------
        `bar_size_th` : ``int``
            money volume required to form a new candle-stick (data-frame row)

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

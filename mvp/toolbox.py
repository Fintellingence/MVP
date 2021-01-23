import os
import sqlite3
import pandas as pd


__all__ = ["load_from_db_m1", "load_from_db_d1", "from_m1_to_d1",
           "new_time_frequency_df", "time_window_df", "tick_bars_df",
           "volume_bars_df"]


def load_from_db_m1(symbol, db_path):
    if not os.path.isfile(db_path):
        raise IOError(
            "MetaTrader database file {} does not exists".format(db_path)
        )
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM {}".format(symbol), con=conn)
    df.index = pd.to_datetime(df["DateTime"])
    df.drop(["DateTime"], axis=1, inplace=True)
    conn.close()
    return df


def load_from_db_d1(symbol, db_path):
    if not os.path.isfile(db_path):
        raise IOError(
            "MetaTrader database file {} does not exists".format(db_path)
        )
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM {}".format(symbol), con=conn)
    df.index = pd.to_datetime(df["Date"])
    df.drop(["Date"], axis=1, inplace=True)
    conn.close()
    return df


def from_m1_to_d1(df_m1):
    """
    Return a dataframe resizing the sample time interval to 1 day

    """
    last_index = 0
    n_bars = 0
    bar_list = []
    date_index = []
    current_day = df_m1.index[0].day
    for idx in range(df_m1.index.size):
        if df_m1.index[idx].day != current_day:
            bar_max = df_m1.iloc[last_index:idx]["High"].max()
            bar_min = df_m1.iloc[last_index:idx]["Low"].min()
            bar_vol = df_m1.iloc[last_index:idx]["Volume"].sum()
            bar_tck = df_m1.iloc[last_index:idx]["TickVol"].sum()
            bar_opn = df_m1.iloc[last_index]["Open"]
            bar_cls = df_m1.iloc[idx - 1]["Close"]
            bar_date = df_m1.index[idx - 1].date()
            bar_list.append(
                [bar_opn, bar_max, bar_min, bar_cls, bar_tck, bar_vol]
            )
            date_index.append(bar_date)
            current_day = df_m1.index[idx].day
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


def new_time_frequency_df(df_m1, time_step=60):
    """Return a dataframe resizing the sample time interval

    Parameters
    ----------
    db_path : ``int``
        New time interval of the sample in minutes.

    """
    available_step = [1, 5, 10, 15, 30, 60]
    if time_step not in available_step:
        raise ValueError("New time step requested not in ", available_step)
    last_index = 0
    bar_list = []
    close_time_index = []
    accum_ticks = 0
    accum_vol = 0
    n_bars = 0
    for idx in range(df_m1.index.size):
        accum_vol = accum_vol + df_m1.iloc[idx]["Volume"]
        accum_ticks = accum_ticks + df_m1.iloc[idx]["TickVol"]
        if df_m1.index[idx].minute % time_step == 0:
            bar_final_time = df_m1.iloc[idx].name
            bar_max = df_m1.iloc[last_index : (idx + 1)]["High"].max()
            bar_min = df_m1.iloc[last_index : (idx + 1)]["Low"].min()
            bar_vol = df_m1.iloc[last_index : (idx + 1)]["Volume"].sum()
            bar_opn = df_m1.iloc[last_index]["Open"]
            bar_cls = df_m1.iloc[idx]["Close"]
            bar_list.append(
                [bar_opn, bar_max, bar_min, bar_cls, accum_ticks, bar_vol]
            )
            close_time_index.append(bar_final_time)
            accum_vol = 0
            accum_ticks = 0
            last_index = idx + 1
            n_bars = n_bars + 1
    new_df = pd.DataFrame(
        bar_list,
        columns=["Open", "High", "Low", "Close", "TickVol", "Volume"],
        index=close_time_index,
    )
    new_df.index.name = "DateTime"
    return new_df


def time_window_df(df_m1, start, stop, time_step=1):
    """Create a new DataFrame changing sample time step

    Parameters
    ----------
    start : ``datetime.datetime``
        Initial time instant (Pandas Time-stamp).
    stop : ``datetime.datetime``
        Final time instant (Pandas Time-stamp).
    time_step : ``int``
        In minutes.

    """
    if start > stop:
        raise ValueError("initial time must be smaller than the final one")
    return new_time_frequency_df(df_m1.loc[start:stop], time_step)


def tick_bars_df(df, bar_size_th=1000):
    """
    Return dataframe labeled by time instants which accumulate a 'threshold'
    of deals (tick volume)

    """
    accum_ticks = 0
    n_bars = 0
    bar_list = []
    last_index = 0
    bar_initial_time = df.iloc[0].name
    close_time_index = []
    for idx in range(df.index.size):
        accum_ticks = accum_ticks + df.iloc[idx]["TickVol"]
        if accum_ticks > bar_size_th:
            bar_final_time = df.iloc[idx].name
            bar_max = df.iloc[last_index : (idx + 1)]["High"].max()
            bar_min = df.iloc[last_index : (idx + 1)]["Low"].min()
            bar_vol = df.iloc[last_index : (idx + 1)]["Volume"].sum()
            bar_opn = df.iloc[last_index]["Open"]
            bar_cls = df.iloc[idx]["Close"]
            bar_list.append(
                [
                    bar_initial_time,
                    bar_opn,
                    bar_max,
                    bar_min,
                    bar_cls,
                    accum_ticks,
                    bar_vol,
                ]
            )
            close_time_index.append(bar_final_time)
            accum_ticks = 0
            last_index = idx + 1
            bar_initial_time = bar_final_time
            n_bars = n_bars + 1
    newDF = pd.DataFrame(
        bar_list,
        columns=[
            "OpenTime",
            "Open",
            "High",
            "Low",
            "Close",
            "TickVol",
            "Volume",
        ],
        index=close_time_index,
    )
    newDF.index.name = "CloseTime"
    return newDF


def volume_bars_df(df, bar_size_th=1e6):
    """
    Return dataframe labeled by time instants which accumulate a 'threshold'
    of  volume.

    """
    accum_vol = 0.0
    accum_ticks = 0
    n_bars = 0
    bar_list = []
    last_index = 0
    bar_initial_time = df.iloc[0].name
    close_time_index = []
    for idx in range(df.index.size):
        accum_vol = accum_vol + df.iloc[idx]["Volume"]
        accum_ticks = accum_ticks + df.iloc[idx]["TickVol"]
        if accum_vol > bar_size_th:
            bar_final_time = df.iloc[idx].name
            bar_max = df.iloc[last_index : (idx + 1)]["High"].max()
            bar_min = df.iloc[last_index : (idx + 1)]["Low"].min()
            bar_opn = df.iloc[last_index]["Open"]
            bar_cls = df.iloc[idx]["Close"]
            bar_list.append(
                [
                    bar_initial_time,
                    bar_opn,
                    bar_max,
                    bar_min,
                    bar_cls,
                    accum_ticks,
                    accum_vol,
                ]
            )
            close_time_index.append(bar_final_time)
            accum_ticks = 0
            accum_vol = 0.0
            last_index = idx + 1
            bar_initial_time = bar_final_time
            n_bars = n_bars + 1
    new_df = pd.DataFrame(
        bar_list,
        columns=[
            "OpenTime",
            "Open",
            "High",
            "Low",
            "Close",
            "TickVol",
            "Volume",
        ],
        index=close_time_index,
    )
    new_df.index.name = "CloseTime"
    return new_df

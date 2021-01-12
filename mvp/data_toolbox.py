import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams["figure.subplot.hspace"] = 0.0
DEFAULT_CUP = "#5CFF19"
DEFAULT_CDOWN = "#FF2254"


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
    """ Return a dataframe resizing the sample time interval to 1 day """
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
    """
    Return a dataframe resizing the sample time interval
    ====================================================
    time_step : New time interval of the sample in minutes
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
    ================================================
    start     : Initial time instant (Pandas Time-stamp)
    stop      : final time instant (Pandas Time-stamp)
    time_step : In minutes
    """
    if start > stop:
        raise ValueError("initial time must be smaller than the final one")
    return new_time_frequency_df(df_m1.loc[start:stop], time_step)


def tick_bars_df(df, bar_size_th=1000):
    """
    Return dataframe labeled by time instants which
    accumulate a 'threshold' of deals (tick volume)
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


def volumeBarsDF(df, bar_size_th=1e6):
    """
    Return dataframe labeled by time instants
    which accumulate a 'threshold' of  volume
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


def draw_candle_stick(
    axis, data, color_up=DEFAULT_CUP, color_down=DEFAULT_CDOWN
):
    """Auxiliar function to draw a candlestick in matplotlib.
    See 'plot_time_candles'

    """
    if data["Close"] > data["Open"]:
        color = color_up
    else:
        color = color_down
    axis.plot(
        [data["SeqNum"], data["SeqNum"]],
        [data["Low"], data["High"]],
        lw=1.5,
        color="black",
        solid_capstyle="round",
        zorder=2,
    )
    x0 = data["SeqNum"] - 0.25
    y0 = data["Open"]
    width = 0.5
    height = data["Close"] - data["Open"]
    candle_body = mpl.patches.Rectangle(
        (x0, y0),
        width,
        height,
        facecolor=color,
        edgecolor="black",
        lw=1,
        zorder=3,
    )
    axis.add_patch(candle_body)
    return axis


def plotTimeCandles(df_m1, start, stop, time_step=1):
    """Display candlestick plot in a time window [start,stop]
    ======================================================
    df_m1      : 1-minute sample DataFrame
    start     : Initial time instant (Pandas Time-stamp)
    stop      : final time instant (Pandas Time-stamp)
    time_step : In minutes
    """
    plot_df = time_window_df(df_m1, start, stop, time_step)
    plot_df["SeqNum"] = pd.Series(
        np.arange(plot_df.shape[0]), index=plot_df.index
    )
    fig, ax = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        gridspec_kw={"height_ratios": [1, 3]},
        sharex=True,
    )
    current_day = plot_df.index[0].day
    for time_index in plot_df.index:
        ax[1] = draw_candle_stick(ax[1], plot_df.loc[time_index])
        if time_index.day > current_day:
            ax[1].axvline(
                plot_df.loc[time_index]["SeqNum"],
                lw=2,
                ls="--",
                zorder=1,
                color="black",
            )
            current_day = time_index.day
    # Plot volume above the prices
    ax[0].vlines(
        plot_df["SeqNum"],
        np.zeros(plot_df.shape[0]),
        plot_df["Volume"],
        lw=4,
        zorder=2,
    )
    ax[0].set_ylim(0, plot_df["Volume"].max())
    # Set tick parameters
    tick_freq = int(plot_df.shape[0] / 10) + 1
    ax[1].set_xticks(list(plot_df["SeqNum"])[::tick_freq])
    labels = [t.strftime("%H:%M") for t in plot_df.index.time][::tick_freq]
    ax[1].set_xticklabels(labels, rotation=50, ha="right")
    ax[1].grid(ls="--", color="gray", alpha=0.3, zorder=1)
    ax[0].grid(ls="--", color="gray", alpha=0.3, zorder=1)
    ax[1].tick_params(axis="y", pad=0.4)
    ax[1].tick_params(axis="x", pad=0.2)
    ax[0].tick_params(axis="y", pad=0.4)
    ax[1].tick_params(axis="both", which="major", direction="in")
    ax[0].tick_params(axis="both", which="major", direction="in")
    plt.show()

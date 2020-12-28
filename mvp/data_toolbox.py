import os
import sqlite3
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt

from database_builder import DEFAULT_DB_PATH

# CONFIGURE PLOTTING PARAMETERS
mpl.rcParams["figure.subplot.hspace"] = 0.0
# Default colors to draw candles
DEFAULT_CUP = "#5CFF19"  # a shinning green
DEFAULT_CDOWN = "#FF2254"  # a shinning red


def loadFromDB_M1(symbol, db_filename="BRSharesMetaTrader_M1.db"):
    full_db_path = DEFAULT_DB_PATH + db_filename
    exist_err_msg = "ERROR : MetaTrader database file {} does not exists".format(
        full_db_path
    )
    if not os.path.isfile(full_db_path):
        raise IOError(exist_err_msg)
    conn = sqlite3.connect(full_db_path)
    df = pd.read_sql("SELECT * FROM {}".format(symbol), con=conn)
    df.index = pd.to_datetime(df["DateTime"])  # set date/time as index
    df.drop(["DateTime"], axis=1, inplace=True)  # no longer needed
    conn.close()
    return df


def loadFromDB_D1(symbol, db_filename="BRSharesYahoo_D1.db"):
    full_db_path = DEFAULT_DB_PATH + db_filename
    exist_err_msg = "ERROR : MetaTrader database file {} does not exists".format(
        full_db_path
    )
    if not os.path.isfile(full_db_path):
        raise IOError(exist_err_msg)
    conn = sqlite3.connect(full_db_path)
    df = pd.read_sql("SELECT * FROM {}".format(symbol), con=conn)
    df.index = pd.to_datetime(df["Date"])  # set date/time as index
    df.drop(["Date"], axis=1, inplace=True)  # no longer needed
    conn.close()
    return df


def fromM1_to_D1(df1M):
    """ Return a dataframe resizing the sample time interval to 1 day """
    last_DFindex = 0
    accum_ticks = 0
    accum_vol = 0
    nbars = 0
    bar_list = []
    date_index = []
    current_day = df1M.index[0].day
    for DFindex in range(df1M.index.size):
        if df1M.index[DFindex].day != current_day:
            bar_max = df1M.iloc[last_DFindex:DFindex]["High"].max()
            bar_min = df1M.iloc[last_DFindex:DFindex]["Low"].min()
            bar_vol = df1M.iloc[last_DFindex:DFindex]["Volume"].sum()
            bar_tck = df1M.iloc[last_DFindex:DFindex]["TickVol"].sum()
            bar_opn = df1M.iloc[last_DFindex]["Open"]
            bar_cls = df1M.iloc[DFindex - 1]["Close"]
            bar_date = df1M.index[DFindex - 1].date()
            bar_list.append([bar_opn, bar_max, bar_min, bar_cls, bar_tck, bar_vol])
            date_index.append(bar_date)
            # reset variables to next cycle
            current_day = df1M.index[DFindex].day
            last_DFindex = DFindex
            nbars = nbars + 1
    date_index = pd.to_datetime(date_index)
    newDF = pd.DataFrame(
        bar_list,
        columns=["Open", "High", "Low", "Close", "TickVol", "Volume"],
        index=date_index,
    )
    newDF.index.name = "Date"
    return newDF


def newTimeFrequencyDF(df1M, time_step=60):
    """
    Return a dataframe resizing the sample time interval
    ====================================================
    time_step : New time interval of the sample in minutes
    """
    available_tstep = [1, 5, 10, 15, 30, 60]
    if time_step not in available_tstep:
        raise ValueError("New time step requested not in ", available_tstep)
    last_DFindex = 0
    bar_list = []
    closeTime_index = []
    accum_ticks = 0
    accum_vol = 0
    nbars = 0
    for DFindex in range(df1M.index.size):
        accum_vol = accum_vol + df1M.iloc[DFindex]["Volume"]
        accum_ticks = accum_ticks + df1M.iloc[DFindex]["TickVol"]
        if df1M.index[DFindex].minute % time_step == 0:
            bar_finaltime = df1M.iloc[DFindex].name
            bar_max = df1M.iloc[last_DFindex : DFindex + 1]["High"].max()
            bar_min = df1M.iloc[last_DFindex : DFindex + 1]["Low"].min()
            bar_vol = df1M.iloc[last_DFindex : DFindex + 1]["Volume"].sum()
            bar_opn = df1M.iloc[last_DFindex]["Open"]
            bar_cls = df1M.iloc[DFindex]["Close"]
            bar_list.append([bar_opn, bar_max, bar_min, bar_cls, accum_ticks, bar_vol])
            closeTime_index.append(bar_finaltime)
            accum_vol = 0
            accum_ticks = 0
            last_DFindex = DFindex + 1
            bar_initialtime = bar_finaltime
            nbars = nbars + 1
    newDF = pd.DataFrame(
        bar_list,
        columns=["Open", "High", "Low", "Close", "TickVol", "Volume"],
        index=closeTime_index,
    )
    newDF.index.name = "DateTime"
    return newDF


def timeWindowDF(df1M, start, stop, time_step=1):
    """Create a new DataFrame changing sample time step
    ================================================
    start     : Initial time instant (Pandas Time-stamp)
    stop      : final time instant (Pandas Time-stamp)
    time_step : In minutes
    """
    if start > stop:
        raise ValueError("\ninitial time must be smaller than the final one")
    return newTimeFrequencyDF(df1M.loc[start:stop], time_step)


def tickBarsDF(df, barsize_threshold=1000):
    """
    Return dataframe labeled by time instants which
    accumulate a 'threshold' of deals (tick volume)
    """
    accum_ticks = 0
    nbars = 0
    bar_list = []
    last_DFindex = 0
    bar_initialtime = df.iloc[0].name
    closeTime_index = []
    for DFindex in range(df.index.size):
        accum_ticks = accum_ticks + df.iloc[DFindex]["TickVol"]
        if accum_ticks > barsize_threshold:
            bar_finaltime = df.iloc[DFindex].name
            bar_max = df.iloc[last_DFindex : DFindex + 1]["High"].max()
            bar_min = df.iloc[last_DFindex : DFindex + 1]["Low"].min()
            bar_vol = df.iloc[last_DFindex : DFindex + 1]["Volume"].sum()
            bar_opn = df.iloc[last_DFindex]["Open"]
            bar_cls = df.iloc[DFindex]["Close"]
            bar_list.append(
                [
                    bar_initialtime,
                    bar_opn,
                    bar_max,
                    bar_min,
                    bar_cls,
                    accum_ticks,
                    bar_vol,
                ]
            )
            closeTime_index.append(bar_finaltime)
            accum_ticks = 0
            last_DFindex = DFindex + 1
            bar_initialtime = bar_finaltime
            nbars = nbars + 1
    newDF = pd.DataFrame(
        bar_list,
        columns=["OpenTime", "Open", "High", "Low", "Close", "TickVol", "Volume"],
        index=closeTime_index,
    )
    newDF.index.name = "CloseTime"
    return newDF


def volumeBarsDF(df, barsize_threshold=1e6):
    """
    Return dataframe labeled by time instants
    which accumulate a 'threshold' of  volume
    """
    accum_vol = 0.0
    accum_ticks = 0
    nbars = 0
    bar_list = []
    last_DFindex = 0
    bar_initialtime = df.iloc[0].name
    closeTime_index = []
    for DFindex in range(df.index.size):
        accum_vol = accum_vol + df.iloc[DFindex]["Volume"]
        accum_ticks = accum_ticks + df.iloc[DFindex]["TickVol"]
        if accum_vol > barsize_threshold:
            bar_finaltime = df.iloc[DFindex].name
            bar_max = df.iloc[last_DFindex : DFindex + 1]["High"].max()
            bar_min = df.iloc[last_DFindex : DFindex + 1]["Low"].min()
            bar_opn = df.iloc[last_DFindex]["Open"]
            bar_cls = df.iloc[DFindex]["Close"]
            bar_list.append(
                [
                    bar_initialtime,
                    bar_opn,
                    bar_max,
                    bar_min,
                    bar_cls,
                    accum_ticks,
                    accum_vol,
                ]
            )
            closeTime_index.append(bar_finaltime)
            accum_ticks = 0
            accum_vol = 0.0
            last_DFindex = DFindex + 1
            bar_initialtime = bar_finaltime
            nbars = nbars + 1
    newDF = pd.DataFrame(
        bar_list,
        columns=["OpenTime", "Open", "High", "Low", "Close", "TickVol", "Volume"],
        index=closeTime_index,
    )
    newDF.index.name = "CloseTime"
    return newDF


def drawCandleStick(axis, data, color_up=DEFAULT_CUP, color_down=DEFAULT_CDOWN):
    """ Auxiliar function to draw a candlestick in matplotlib. See 'plotTimeCandles' """
    # define the candle body color
    if data["Close"] > data["Open"]:
        color = color_up
    else:
        color = color_down
    # plot vertical line segment corresponding to maximun/minimum deals
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
    candleBody = mpl.patches.Rectangle(
        (x0, y0), width, height, facecolor=color, edgecolor="black", lw=1, zorder=3
    )
    axis.add_patch(candleBody)
    return axis


def plotTimeCandles(df1M, start, stop, time_step=1):
    """Display candlestick plot in a time window [start,stop]
    ======================================================
    df1M      : 1-minute sample DataFrame
    start     : Initial time instant (Pandas Time-stamp)
    stop      : final time instant (Pandas Time-stamp)
    time_step : In minutes
    """
    plotDF = timeWindowDF(df1M, start, stop, time_step)
    plotDF["SeqNum"] = pd.Series(np.arange(plotDF.shape[0]), index=plotDF.index)
    fig, ax = plt.subplots(
        2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [1, 3]}, sharex=True
    )
    currentDay = plotDF.index[0].day
    for timeIndex in plotDF.index:
        ax[1] = drawCandleStick(ax[1], plotDF.loc[timeIndex])
        if timeIndex.day > currentDay:
            # mark a new day beginning with a vertical dashed line
            ax[1].axvline(
                plotDF.loc[timeIndex]["SeqNum"], lw=2, ls="--", zorder=1, color="black"
            )
            currentDay = timeIndex.day
    # Plot volume above the prices
    ax[0].vlines(
        plotDF["SeqNum"], np.zeros(plotDF.shape[0]), plotDF["Volume"], lw=4, zorder=2
    )
    ax[0].set_ylim(0, plotDF["Volume"].max())
    # Set tick parameters
    tickFreq = int(plotDF.shape[0] / 10) + 1
    ax[1].set_xticks(list(plotDF["SeqNum"])[::tickFreq])
    labels = [t.strftime("%H:%M") for t in plotDF.index.time][::tickFreq]
    ax[1].set_xticklabels(labels, rotation=50, ha="right")
    ax[1].grid(ls="--", color="gray", alpha=0.3, zorder=1)
    ax[0].grid(ls="--", color="gray", alpha=0.3, zorder=1)
    ax[1].tick_params(axis="y", pad=0.4)
    ax[1].tick_params(axis="x", pad=0.2)
    ax[0].tick_params(axis="y", pad=0.4)
    ax[1].tick_params(axis="both", which="major", direction="in")
    ax[0].tick_params(axis="both", which="major", direction="in")
    plt.show()

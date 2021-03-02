import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# from mvp.toolbox import time_window_df


mpl.rcParams["figure.subplot.hspace"] = 0.0
DEFAULT_CUP = "#5CFF19"
DEFAULT_CDOWN = "#FF2254"


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
        lw=0.5,
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
        lw=0.25,
        zorder=3,
    )
    axis.add_patch(candle_body)
    return axis


def plot_time_candles(raw_data, start, stop, time_step=1):
    """
    Display candlestick plot in a time window [start,stop]

    Parameters
    ----------
        raw_data : ``mvp.RawData``
            Provide 1-minute dataframe and methods to format the data.
        start : ``datetime.datetime``
            Initial time instant (pandas.Timestamp).
        stop : ``datetime.datetime``
            Final time instant (pandas.Timestamp).
        time_step : ``int``
            In minutes. Available values [1,5,10,15,30,60] (default 1)

    """
    plot_df = raw_data.time_window(start, stop, time_step)
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

    ax[0].vlines(
        plot_df["SeqNum"],
        np.zeros(plot_df.shape[0]),
        plot_df["Volume"],
        lw=4,
        zorder=2,
    )
    ax[0].set_ylim(0, plot_df["Volume"].max())

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


def plot_two_series(series_a, series_b):
    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Series A", color=color)
    ax1.plot(series_a, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "#0E0C0C"
    ax2.set_ylabel("Series B", color=color)
    ax2.plot(series_b, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    return None


def plot_bollinger(model, labels, linewidth=0.2):
    MA_name = "MA_" + str(model.model_parameters["MA"][0])
    DEV_name = "DEV_" + str(model.model_parameters["DEV"][0])
    K_value = model.model_parameters["K_value"]
    plot_data = model.feature_data.df_curated.copy()
    plot_data = pd.concat([plot_data, labels], axis=1).copy()
    plot_data["UpBand"] = plot_data[MA_name] + K_value * plot_data[DEV_name]
    plot_data["DownBand"] = plot_data[MA_name] - K_value * plot_data[DEV_name]
    buy_profit = plot_data[
        (plot_data["Suggestion"] == 1) & (plot_data["Label"] == 1)
    ][["Close"]]
    buy_loss = plot_data[
        (plot_data["Suggestion"] == 1) & (plot_data["Label"] == -1)
    ][["Close"]]
    buy_neutral = plot_data[
        (plot_data["Suggestion"] == 1) & (plot_data["Label"] == 0)
    ][["Close"]]
    plt.scatter(buy_profit.index, buy_profit["Close"], c="g", s=0.1)
    plt.scatter(buy_loss.index, buy_loss["Close"], c="r", s=0.1)
    plt.scatter(buy_neutral.index, buy_neutral["Close"], c="b", s=0.1)
    plot_data.Close.plot(linewidth=linewidth)
    plot_data[MA_name].plot(linewidth=linewidth)
    plot_data.UpBand.plot(linewidth=linewidth)
    plot_data.DownBand.plot(linewidth=linewidth)
    plt.show()
    return None


def plot_crossing_MA(model, labels, linewidth=0.2):
    MA_name1 = "MA_" + str(model.model_parameters["MA"][0])
    MA_name2 = "MA_" + str(model.model_parameters["MA"][1])
    plot_data = model.feature_data.df_curated.copy()
    plot_data = pd.concat([plot_data, labels], axis=1).copy()
    buy_profit = plot_data[
        (plot_data["Suggestion"] == 1) & (plot_data["Label"] == 1)
    ][["Close"]]
    buy_loss = plot_data[
        (plot_data["Suggestion"] == 1) & (plot_data["Label"] == -1)
    ][["Close"]]
    buy_neutral = plot_data[
        (plot_data["Suggestion"] == 1) & (plot_data["Label"] == 0)
    ][["Close"]]
    sell_profit = plot_data[
        (plot_data["Suggestion"] == -1) & (plot_data["Label"] == 1)
    ][["Close"]]
    sell_loss = plot_data[
        (plot_data["Suggestion"] == -1) & (plot_data["Label"] == -1)
    ][["Close"]]
    sell_neutral = plot_data[
        (plot_data["Suggestion"] == -1) & (plot_data["Label"] == 0)
    ][["Close"]]
    plt.scatter(
        buy_profit.index, buy_profit["Close"], c="g", marker=".", s=0.5
    )
    plt.scatter(buy_loss.index, buy_loss["Close"], c="r", marker=".", s=0.5)
    plt.scatter(
        buy_neutral.index, buy_neutral["Close"], c="b", marker=".", s=0.5
    )
    plt.scatter(
        sell_profit.index, sell_profit["Close"], c="g", marker=".", s=0.5
    )
    plt.scatter(sell_loss.index, sell_loss["Close"], c="r", marker=".", s=0.5)
    plt.scatter(
        sell_neutral.index, sell_neutral["Close"], c="b", marker=".", s=0.5
    )
    plot_data.Close.plot(linewidth=linewidth)
    plot_data[MA_name1].plot(linewidth=linewidth)
    plot_data[MA_name2].plot(linewidth=linewidth)
    plt.show()
    return None


def plot_classical_filter(model, labels, linewidth=0.2):
    plot_data = model.feature_data.df_curated.copy()
    plot_data = pd.concat([plot_data, labels], axis=1).copy()
    buy_profit = plot_data[
        (plot_data["Suggestion"] == 1) & (plot_data["Label"] == 1)
    ][["Close"]]
    buy_loss = plot_data[
        (plot_data["Suggestion"] == 1) & (plot_data["Label"] == -1)
    ][["Close"]]
    buy_neutral = plot_data[
        (plot_data["Suggestion"] == 1) & (plot_data["Label"] == 0)
    ][["Close"]]
    sell_profit = plot_data[
        (plot_data["Suggestion"] == -1) & (plot_data["Label"] == 1)
    ][["Close"]]
    sell_loss = plot_data[
        (plot_data["Suggestion"] == -1) & (plot_data["Label"] == -1)
    ][["Close"]]
    sell_neutral = plot_data[
        (plot_data["Suggestion"] == -1) & (plot_data["Label"] == 0)
    ][["Close"]]
    plt.scatter(
        buy_profit.index, buy_profit["Close"], c="g", marker=".", s=0.5
    )
    plt.scatter(buy_loss.index, buy_loss["Close"], c="r", marker=".", s=0.5)
    plt.scatter(
        buy_neutral.index, buy_neutral["Close"], c="b", marker=".", s=0.5
    )
    plt.scatter(
        sell_profit.index, sell_profit["Close"], c="g", marker=".", s=0.5
    )
    plt.scatter(sell_loss.index, sell_loss["Close"], c="r", marker=".", s=0.5)
    plt.scatter(
        sell_neutral.index, sell_neutral["Close"], c="b", marker=".", s=0.5
    )
    plot_data.Close.plot(linewidth=linewidth)
    plt.show()
    return None


def plot_model(model, labels, linewidth=0.2):
    """
    Displays the close time-series along with the indicators used by
    the primary models, also highlights Buy/Sell suggestions and their
    success (label = 1, or -1)

    Parameters
    ----------
        `model` : ``mvp.primary.PrimaryModel``
            provides all data for plotting signals and indicators
        `labels`: ``pd.DataFrame()``
            contains a the labels of events, could come from mvp.labels.Labels.labeled_df
    """
    if model.model_type == "bollinger-bands":
        plot_bollinger(model, labels)
    if model.model_type == "crossing-MA":
        plot_crossing_MA(model, labels)
    if model.model_type == "classical-filter":
        plot_classical_filter(model, labels)
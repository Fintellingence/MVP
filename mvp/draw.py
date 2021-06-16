import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import mvp
import seaborn as sns

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


def plot_symbol(refined_data, time_step=1):
    """
    Basic Auxiliar function to plot symbols via refined_data.RefinedData()
    objects.
    """
    data = refined_data.change_sample_interval(step=time_step)
    fig, ax = plt.subplots()
    with sns.axes_style("darkgrid"):
        ax.plot(data.index, data.Close, label="Close Price")
        ax.set_xlabel("DateTime")
        ax.set_ylabel("Close Price")
    with sns.axes_style("dark"):
        ax2 = ax.twinx()
        ax2.plot(data.index, data.Volume, c="r", alpha=0.35, label="Volume")
        ax2.set_ylabel("Volume")
    plt.title(refined_data.symbol)
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


def plot_equity(book, linewidth=0.8):
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots()
        ax.plot(
            book.index,
            book["EquityCurve"].values,
            c="royalblue",
            linewidth=linewidth,
        )
        ax.set_xlabel("Trades")
        ax.set_ylabel("Equity %")
        plt.title("Equity curve")
        plt.show()
    pass


def plot_bollinger(model, labels, linewidth=1.0, point_size=15):
    MA_name = "MA_" + str(model.features["MA"])
    DEV_name = "DEV_" + str(model.features["DEV"])
    K_value = model.features["K_value"]
    plot_data = model.feature_data.copy()
    plot_data = pd.concat([plot_data, labels], axis=1).copy()
    plot_data["UpBand"] = plot_data[MA_name] + K_value * plot_data[DEV_name]
    plot_data["DownBand"] = plot_data[MA_name] - K_value * plot_data[DEV_name]
    buy_profit = plot_data[
        (plot_data["Side"] == 1) & (plot_data["Label"] == 1)
    ][["Close"]]
    buy_loss = plot_data[
        (plot_data["Side"] == 1) & (plot_data["Label"] == -1)
    ][["Close"]]
    buy_neutral = plot_data[
        (plot_data["Side"] == 1) & (plot_data["Label"] == 0)
    ][["Close"]]
    sell_profit = plot_data[
        (plot_data["Side"] == -1) & (plot_data["Label"] == 1)
    ][["Close"]]
    sell_loss = plot_data[
        (plot_data["Side"] == -1) & (plot_data["Label"] == -1)
    ][["Close"]]
    sell_neutral = plot_data[
        (plot_data["Side"] == -1) & (plot_data["Label"] == 0)
    ][["Close"]]
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots()
        ax.scatter(
            buy_profit.index,
            buy_profit.Close,
            c="limegreen",
            marker="o",
            s=point_size,
        )
        ax.scatter(
            buy_loss.index,
            buy_loss["Close"],
            c="orangered",
            marker="o",
            s=point_size,
        )
        ax.scatter(
            buy_neutral.index,
            buy_neutral["Close"],
            c="navy",
            marker="o",
            s=point_size,
        )
        ax.scatter(
            sell_profit.index,
            sell_profit["Close"],
            c="limegreen",
            marker="o",
            s=point_size,
        )
        ax.scatter(
            sell_loss.index,
            sell_loss["Close"],
            c="orangered",
            marker="o",
            s=point_size,
        )
        ax.scatter(
            sell_neutral.index,
            sell_neutral["Close"],
            c="navy",
            marker="o",
            s=point_size,
        )
        ax.plot(
            plot_data.index,
            plot_data.Close,
            c="black",
            linewidth=linewidth * 0.8,
        )
        ax.plot(
            plot_data.index,
            plot_data["UpBand"].values,
            c="tab:blue",
            alpha=0.7,
            label="UpBand: " + MA_name + "+" + str(K_value) + "$\sigma$",
        )
        ax.plot(
            plot_data.index,
            plot_data["DownBand"].values,
            c="tab:cyan",
            alpha=0.7,
            label="DownBand: " + MA_name + "-" + str(K_value) + "$\sigma$",
        )
        ax.set_xlabel("DateTime")
        ax.set_ylabel("Price")
        ax.legend()
        plt.title("Bollinger Bands (" + model.symbol + ")")
        plt.show()
    return None


def plot_crossing_ma(
    refined_obj,
    primary_data,
    op_params,
    kwargs={},
    linewidth=1.0,
    point_size=15,
):
    primary_data.values()
    slow_window = max(primary_data.values())
    fast_window = min(primary_data.values())
    slow_ma = refined_obj.get_sma(slow_window, **kwargs)
    fast_ma = refined_obj.get_sma(fast_window, **kwargs)
    MA_fast = "MA_" + str(fast_window)
    MA_slow = "MA_" + str(slow_window)
    events = mvp.primary.crossing_ma(
        refined_obj, **primary_data, kwargs=kwargs
    )
    close_data = refined_obj.get_close(**kwargs)
    if "step" in kwargs.keys():
        labels = mvp.labels.event_label_series(
            events, refined_obj.time_bars(step=kwargs["step"]), **op_params
        )
    else:
        labels = mvp.labels.event_label_series(
            events, refined_obj.df, **op_params
        )
    plot_data = pd.concat(
        [
            close_data.to_frame(),
            labels,
            fast_ma.to_frame(name=MA_fast),
            slow_ma.to_frame(name=MA_slow),
        ],
        axis=1,
    ).copy()
    buy_profit = plot_data[
        (plot_data["Side"] == 1) & (plot_data["Label"] == 1)
    ][["Close"]]
    buy_loss = plot_data[
        (plot_data["Side"] == 1) & (plot_data["Label"] == -1)
    ][["Close"]]
    buy_neutral = plot_data[
        (plot_data["Side"] == 1) & (plot_data["Label"] == 0)
    ][["Close"]]
    sell_profit = plot_data[
        (plot_data["Side"] == -1) & (plot_data["Label"] == 1)
    ][["Close"]]
    sell_loss = plot_data[
        (plot_data["Side"] == -1) & (plot_data["Label"] == -1)
    ][["Close"]]
    sell_neutral = plot_data[
        (plot_data["Side"] == -1) & (plot_data["Label"] == 0)
    ][["Close"]]
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots()
        ax.scatter(
            buy_profit.index,
            buy_profit.Close,
            c="limegreen",
            marker="o",
            s=point_size,
        )
        ax.scatter(
            buy_loss.index,
            buy_loss["Close"],
            c="orangered",
            marker="o",
            s=point_size,
        )
        ax.scatter(
            buy_neutral.index,
            buy_neutral["Close"],
            c="navy",
            marker="o",
            s=point_size,
        )
        ax.scatter(
            sell_profit.index,
            sell_profit["Close"],
            c="limegreen",
            marker="o",
            s=point_size,
        )
        ax.scatter(
            sell_loss.index,
            sell_loss["Close"],
            c="orangered",
            marker="o",
            s=point_size,
        )
        ax.scatter(
            sell_neutral.index,
            sell_neutral["Close"],
            c="navy",
            marker="o",
            s=point_size,
        )
        ax.plot(
            plot_data.index,
            plot_data.Close,
            c="black",
            linewidth=linewidth * 0.8,
        )
        ax.plot(
            plot_data.index,
            plot_data[MA_fast].values,
            c="tab:blue",
            alpha=0.7,
            label=MA_fast,
        )
        ax.plot(
            plot_data.index,
            plot_data[MA_slow].values,
            c="tab:cyan",
            alpha=0.7,
            label=MA_slow,
        )
        ax.set_xlabel("DateTime")
        ax.set_ylabel("Price")
        ax.legend()
        plt.title("Crossing-MA (" + refined_obj.symbol + ")")
        plt.show()
    return None


def plot_classical_filter(model, labels, linewidth=1.0, point_size=15):
    plot_data = model.feature_data.copy()
    plot_data = pd.concat([plot_data, labels], axis=1).copy()
    buy_profit = plot_data[
        (plot_data["Side"] == 1) & (plot_data["Label"] == 1)
    ][["Close"]]
    buy_loss = plot_data[
        (plot_data["Side"] == 1) & (plot_data["Label"] == -1)
    ][["Close"]]
    buy_neutral = plot_data[
        (plot_data["Side"] == 1) & (plot_data["Label"] == 0)
    ][["Close"]]
    sell_profit = plot_data[
        (plot_data["Side"] == -1) & (plot_data["Label"] == 1)
    ][["Close"]]
    sell_loss = plot_data[
        (plot_data["Side"] == -1) & (plot_data["Label"] == -1)
    ][["Close"]]
    sell_neutral = plot_data[
        (plot_data["Side"] == -1) & (plot_data["Label"] == 0)
    ][["Close"]]
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots()
        ax.scatter(
            buy_profit.index,
            buy_profit.Close,
            c="limegreen",
            marker="o",
            s=point_size,
        )
        ax.scatter(
            buy_loss.index,
            buy_loss["Close"],
            c="orangered",
            marker="o",
            s=point_size,
        )
        ax.scatter(
            buy_neutral.index,
            buy_neutral["Close"],
            c="navy",
            marker="o",
            s=point_size,
        )
        ax.scatter(
            sell_profit.index,
            sell_profit["Close"],
            c="limegreen",
            marker="o",
            s=point_size,
        )
        ax.scatter(
            sell_loss.index,
            sell_loss["Close"],
            c="orangered",
            marker="o",
            s=point_size,
        )
        ax.scatter(
            sell_neutral.index,
            sell_neutral["Close"],
            c="navy",
            marker="o",
            s=point_size,
        )
        ax.plot(
            plot_data.index,
            plot_data.Close,
            c="black",
            linewidth=linewidth * 0.8,
            label="CUSUM th: " + str(model.features["threshold"]),
        )
        ax.set_xlabel("DateTime")
        ax.set_ylabel("Price")
        ax.legend()
        plt.title("Classical-Filter (" + model.symbol + ")")
        plt.show()
    return None


def plot_model(
    refined_obj,
    primary_params,
    op_params,
    kwargs,
    linewidth=0.8,
    point_size=15,
):
    """
    Displays the target (usually Close) time-series along with the indicators
    used by the primary models to generate triggers, also highlights Buy/Sell
    Sides and their success (label = 1, or -1)

    Parameters
    ----------
        `model` : ``mvp.primary.PrimaryModel``
            provides all data for plotting signals and indicators
        `operation_parameters`: ``dict``
            Inside `OperationParameters` key we have to provide another dict
            containing three values:
             - StopLoss (SL)
             - TakeProfit (TP)
             - InvestmentHorizon (IH)
             - MarginMode (margin_mode)
            These values should be provided like the following:
                {'SL': 0.01, 'TP': 0.01, 'IH': 1000,'margin_mode':'percent'}}
        `event_filter`" ``pandas.DataFrame``
            DataFrame containing the output of a Metamodel which is used to
            filter the trade suggestions of primary model
        `linewidth`: ``float``
            specifies linewidth parameter for line plots (series/indicators)
        `point_size`: ``float``
            specifies point size parameter for scatter plots (events)
    """
    if strategy == "bollinger-bands":
        plot_bollinger(events, label_data, close_data, linewidth, point_size)
    if strategy == "crossing-MA":
        plot_crossing_MA(events, label_data, close_data, linewidth, point_size)
    if strategy == "classical-filter":
        plot_classical_filter(
            events, label_data, close_data, linewidth, point_size
        )

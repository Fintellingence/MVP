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


# def plot_time_candles(df_m1, start, stop, time_step=1):
#     """Display candlestick plot in a time window [start,stop]
#     Parameters
#     ----------
#         df_m1 : ``DataFrame``
#             1-minute sample DataFrame.
#         start : ``datetime.datetime``
#             Initial time instant (Pandas Time-stamp).
#         stop : ``datetime.datetime``
#             Final time instant (Pandas Time-stamp).
#         time_step : ``int``
#             In minutes.
#     """
#     plot_df = time_window_df(df_m1, start, stop, time_step)
#     plot_df["SeqNum"] = pd.Series(
#         np.arange(plot_df.shape[0]), index=plot_df.index
#     )
#     fig, ax = plt.subplots(
#         2,
#         1,
#         figsize=(10, 6),
#         gridspec_kw={"height_ratios": [1, 3]},
#         sharex=True,
#     )
#     current_day = plot_df.index[0].day
#     for time_index in plot_df.index:
#         ax[1] = draw_candle_stick(ax[1], plot_df.loc[time_index])
#         if time_index.day > current_day:
#             ax[1].axvline(
#                 plot_df.loc[time_index]["SeqNum"],
#                 lw=2,
#                 ls="--",
#                 zorder=1,
#                 color="black",
#             )
#             current_day = time_index.day

#     ax[0].vlines(
#         plot_df["SeqNum"],
#         np.zeros(plot_df.shape[0]),
#         plot_df["Volume"],
#         lw=4,
#         zorder=2,
#     )
#     ax[0].set_ylim(0, plot_df["Volume"].max())

#     tick_freq = int(plot_df.shape[0] / 10) + 1
#     ax[1].set_xticks(list(plot_df["SeqNum"])[::tick_freq])
#     labels = [t.strftime("%H:%M") for t in plot_df.index.time][::tick_freq]
#     ax[1].set_xticklabels(labels, rotation=50, ha="right")
#     ax[1].grid(ls="--", color="gray", alpha=0.3, zorder=1)
#     ax[0].grid(ls="--", color="gray", alpha=0.3, zorder=1)
#     ax[1].tick_params(axis="y", pad=0.4)
#     ax[1].tick_params(axis="x", pad=0.2)
#     ax[0].tick_params(axis="y", pad=0.4)
#     ax[1].tick_params(axis="both", which="major", direction="in")
#     ax[0].tick_params(axis="both", which="major", direction="in")
#     plt.show()

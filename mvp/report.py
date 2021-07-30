import pandas as pd
import matplotlib.pyplot as plt
from mvp.draw import plot_equity


def trade_book(close_data, events, label_data):
    entries = label_data.index
    exits = label_data["PositionEnd"]
    entry_df = (
        close_data.loc[entries]
        .reset_index()
        .rename(columns={"Close": "EntryPrice", "DateTime": "EntryDate"})
    )
    exit_df = (
        close_data.loc[exits]
        .reset_index()
        .rename(columns={"Close": "ExitPrice", "DateTime": "ExitDate"})
    )
    side_df = label_data[["Side"]].reset_index().drop(columns=["DateTime"])
    trades_df = pd.concat([entry_df, exit_df, side_df], axis=1)
    trades_df["Profit"] = trades_df["Side"] * (
        trades_df["ExitPrice"] - trades_df["EntryPrice"]
    )
    trades_df["NetProfit"] = trades_df["Profit"].cumsum(skipna=False)
    trades_df["RelativeProfit"] = (trades_df["Profit"] * 100) / trades_df[
        "EntryPrice"
    ]
    trades_df["EquityCurve"] = 100 + trades_df["RelativeProfit"].cumsum(
        skipna=False
    )
    return trades_df.copy()

def basic_strategy_info(book):
    plot_equity(book)
    print("Min: ",book["EquityCurve"].min())
    print("Sharpe: ",book["RelativeProfit"].mean()/book["RelativeProfit"].std())
    print("Avg Holding Period: ",(book["ExitDate"] - book["EntryDate"]).mean())
    print("Distribution of returns: ")
    plt.hist(book["RelativeProfit"],bins = 20)

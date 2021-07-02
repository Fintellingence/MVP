import pandas as pd


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


def avg_holding_time(book):
    return (book["ExitDate"] - book["EntryDate"]).mean()


def gross_profit(book):
    return book[book["Profit"] > 0]["Profit"].sum()


def gross_loss(book):
    return book[book["Profit"] < 0]["Profit"].sum()


def net_profit(book):
    return gross_profit(book) + gross_loss(book)


def best_trade(book):
    return book[book["Profit"] == book["Profit"].max()].drop(
        columns=["NetProfit"]
    )


def worst_trade(book):
    return book[book["Profit"] == book["Profit"].min()].drop(
        columns=["NetProfit"]
    )


def time_range(primary_model):
    return primary_model.index[0], primary_model.index[-1]

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


def basic_strategy_info(book, display=True):
    if display:
        plot_equity(book)
        print("Min: ", book["EquityCurve"].min())
        print(
            "Sharpe: ",
            book["RelativeProfit"].mean() / book["RelativeProfit"].std(),
        )
        print(
            "Avg Holding Period: ",
            (book["ExitDate"] - book["EntryDate"]).mean(),
        )
        print("Final Equity", str(book["EquityCurve"].iloc[-1]))
        print("Number of trades:", len(book.index))
        print("Distribution of returns: ")
        plt.hist(book["RelativeProfit"], bins=20)
    return dict(
        min=book["EquityCurve"].min(),
        sharpe=book["RelativeProfit"].mean() / book["RelativeProfit"].std(),
        avg_hold_time=(book["ExitDate"] - book["EntryDate"]).mean(),
        final_equity=book["EquityCurve"].iloc[-1],
        n_trades=len(book.index)
    )


def filter_book(book, meta_filters):
    original_book = book.copy()
    original_book = original_book.set_index("EntryDate")
    filter_pass = meta_filters[meta_filters == 1].index
    filtered_book = original_book.loc[filter_pass]
    filtered_book = filtered_book.drop(
        columns=["Profit", "RelativeProfit", "EquityCurve"]
    )
    filtered_book["Profit"] = filtered_book["Side"] * (
        filtered_book["ExitPrice"] - filtered_book["EntryPrice"]
    )
    filtered_book["NetProfit"] = filtered_book["Profit"].cumsum(skipna=False)
    filtered_book["RelativeProfit"] = (
        filtered_book["Profit"] * 100
    ) / filtered_book["EntryPrice"]
    filtered_book["EquityCurve"] = 100 + filtered_book[
        "RelativeProfit"
    ].cumsum(skipna=False)
    return filtered_book.reset_index()

def compare_observable(observable, primary_stats, meta_stats):
    plt.hist(primary_stats[observable],color='b') ,plt.hist(meta_stats[observable],color='r')
    print('[+] primary:') 
    print('mean: ', primary_stats[observable].mean())
    print('std: ',primary_stats[observable].std())
    print('[+] metamodel:') 
    print('mean: ', meta_stats[observable].mean())
    print('std: ',meta_stats[observable].std())

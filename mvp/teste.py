import mvp
import pandas as pd


def autocorr_teste(data):
    df_curated = data.df
    next_df = df_curated["Close"].shift(periods=1)
    returns_df = df_curated["Close"] - next_df
    print("\nEsta é a série original:")
    print(df_curated["Close"])
    print("\nEsta é a série Returns:")
    print(returns_df)
    correlation = returns_df.corr(df_curated["Close"])
    print("\nA autocorrelação é: " + str(correlation))
    return None


db_path = "/Users/rogeriocamargo/FintelligenceData/MetaTrader_M1.db"
sym_path = "/Users/rogeriocamargo/FintelligenceData/stocks.txt"
symbols = mvp.helper.get_symbols(sym_path)[1:]

list_raw_data_objects = []
for symbol in symbols[:1]:
    temp = mvp.rawdata.RawData(
        symbol,
        db_path,
    )
    list_raw_data_objects.append(temp)

autocorr_teste(list_raw_data_objects[0])

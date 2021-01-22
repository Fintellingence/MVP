import curated, helper, rawdata
import pandas as pd
import numpy as np

# We have to push to database processed data
TXT_PATH = "/media/naga/Backup Plus/Data/stocks.txt"
DB_PATH = "/media/naga/Backup Plus/Data/BRSharesMetaTrader_M1.db"
tickers = helper.get_tickers(TXT_PATH)[1:]

listData = []

for ticker in tickers[:1]:
    temp = rawdata.RawData(ticker, DB_PATH)
    listData.append(temp)

parameters = {
    "MA": [10],
    "DEV": [10],
    "ACF": [10, 500, 30000],
    "FD": 0.3,
    "RSI": [20],
    "STAT": False,
}

test = curated.CuratedData(listData[0], parameters)
print(test.df_curated.tail(20))
import mvp
import pandas as pd
import numpy as np

# We have to push to database processed data
TXT_PATH = "/Users/rogeriocamargo/Library/Mobile Documents/com~apple~CloudDocs/Documents/Trabalho/BlackDonalds/MVP/scripts/database/stocks.txt"

tickers = mvp.helper.get_tickers(TXT_PATH)

listData = []

for ticker in tickers[:1]:
    temp = mvp.rawdata.RawData(ticker)
    listData.append(temp)

parameters = {
    "MA": [10],
    "DEV": [10],
    "ACF": [10, 500, 30000],
    "FD": 0.3,
    "RSI": 20,
    "STAT": False,
}

test = mvp.curated.CuratedData(listData[0], parameters)
print(test.dfCurated)
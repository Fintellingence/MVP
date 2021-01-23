import mvp
import numpy as np

# We have to push to database processed data
TXT_PATH = "/Users/rogeriocamargo/Library/Mobile Documents/com~apple~CloudDocs/Documents/Trabalho/BlackDonalds/MVP/scripts/database/stocks.txt"
DB_PATH = "/Users/rogeriocamargo/Library/Mobile Documents/com~apple~CloudDocs/Documents/Trabalho/BlackDonalds/MVP/scripts/database/BRSharesMetaTrader_M1.db"
symbols = mvp.helper.get_symbols(TXT_PATH)[1:]

list_raw_data_objects = []
for symbol in symbols[:1]:
    temp = mvp.rawdata.RawData(symbol, DB_PATH)
    list_raw_data_objects.append(temp)

parameters = {
    "MA": [10],
    "DEV": [10],
    "RSI": [5, 15],
}

list_curated_data_objects = []
for raw_data in list_raw_data_objects:
    temp = mvp.curated.CuratedData(raw_data, parameters)
    list_curated_data_objects.append(temp)

# Test:
print(list_curated_data_objects[0].df_curated.tail(20))
from classes.CuratedData import CuratedData
from classes.rawData import rawData
from classes.Helper import Helper
import pandas as pd
import numpy as np

# We have to push to database processed data

tickers = Helper.getTickers('database/stocks.txt')

listData = []

for ticker in tickers[:1]:
     temp = rawData(ticker)
     listData.append(temp)

parameters = {'MA': [10,15,20], 'DEV': 15, 'ACF':True, 'FD': 0.3, 'RSI': 20, 'STAT':False}

test = CuratedData(listData[0],parameters)
print(test.dfCurated)
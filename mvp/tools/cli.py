from CuratedData import CuratedData
from rawData import rawData
import Helper

# We have to push to database processed data

tickers = Helper.get_tickers('MVP/mvp/tools/stocks.txt')

listData = []

for ticker in tickers:
     temp = rawData(ticker)
     listData.append(temp)

listCuratedData = []
periods = [10*i for i in range(1,100)]

for data in listData:
    temp = CuratedData(data,30)
    listCuratedData.append(temp)


#push to database -> curatedPETR.simpleMA

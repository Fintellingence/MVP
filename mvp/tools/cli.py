from CuratedData import CuratedData
from rawData import rawData
import Helper
import pandas as pd
import numpy as np

# We have to push to database processed data

tickers = Helper.get_tickers('MVP/mvp/tools/stocks.txt')

listData = []

for ticker in tickers[:3]:
     temp = rawData(ticker)
     listData.append(temp)

teste = listData[0].volume.values
teste2 = list(teste)
print((teste2))

listCuratedData = []
#periods = [10*i for i in range(1,100)]

#for data in listData[:3]:
    #temp = CuratedData(data,30)
    #listCuratedData.append(temp)


#push to database -> curatedPETR.simpleMA

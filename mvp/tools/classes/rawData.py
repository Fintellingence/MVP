import sqlite3 as sql3
import pandas as pd

class rawData:
    def __init__(self, ticker):
        self.ticker = ticker
        self.fullData = self.getFullData()
        self.volume = self.getVolume()
        self.high = self.getHigh()
        self.low = self.getLow()
        self.close = self.getClose()
        self.open = self.getOpen()

    def getFullData(self):
        conn = sql3.connect('database/BRShares_Intraday1M.db')
        df = pd.read_sql("SELECT * FROM {}".format(self.ticker),conn).drop(columns=['<TICKVOL>'])
        df.rename(columns={"<HIGH>": "High", "<LOW>": "Low", "<OPEN>":"Open", "<CLOSE>":"Close", "<VOL>":"Vol"}, inplace=True)
        return df.set_index('DateTime')

    def getVolume(self):
        return self.fullData["Vol"]

    def getHigh(self):
        return self.fullData["High"]

    def getLow(self):
        return self.fullData["Low"]
    
    def getClose(self):
        return self.fullData["Close"]
    
    def getOpen(self):
        return self.fullData["Open"]
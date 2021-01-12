import sqlite3 as sql3
import pandas as pd

class rawData:
    def __init__(self, ticker):
        self.ticker = ticker
        self.df = self.getFullData()
        

    def getFullData(self):
        conn = sql3.connect('database/BRShares_Intraday1M.db')
        df = pd.read_sql("SELECT * FROM {}".format(self.ticker),conn).drop(columns=['<TICKVOL>'])
        conn.close()
        df.rename(columns={"<HIGH>": "High", "<LOW>": "Low", "<OPEN>":"Open", "<CLOSE>":"Close", "<VOL>":"Vol"}, inplace=True) 
        return df.set_index('DateTime')

    
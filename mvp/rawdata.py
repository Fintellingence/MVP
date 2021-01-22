import sqlite3 as sql3
import pandas as pd


class RawData:
    def __init__(self, ticker, db_path):
        self.ticker = ticker
        self.db_path = db_path
        self.df = self.getFullData()

    def getFullData(self):
        conn = sql3.connect(self.db_path)
        df = pd.read_sql("SELECT * FROM {}".format(self.ticker), conn).drop(
            columns=["TickVol"]
        )
        conn.close()
        return df.set_index("DateTime")

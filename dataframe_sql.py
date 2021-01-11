import datetime as dt
import pandas as pd
import numpy as np
import sqlite3
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import scipy.optimize as spo

def default_data_frame(start_date,end_date):
    #SPY is standard in accounting for valid trading days
    df = web.DataReader('SPY','yahoo',start_date,end_date)
    df.to_csv('SPY.csv')
    dfout = pd.read_csv("SPY.csv",index_col = "Date",usecols = ['Date','Adj Close'],
                         parse_dates = True,na_values = ['nan'])
    dfout = dfout.rename(columns = {'Adj Close':'SPY'})
    return dfout

start = dt.datetime(2009,1,1)
end = dt.datetime(2011,12,31)

df = default_data_frame(start,end)

conn = sqlite3.connect('SP500.db')
c = conn.cursor()
df.to_sql('SPY',con =conn,if_exists='replace')


df = web.DataReader('GOOG','yahoo',start,end)
df.to_sql('GOOG',con = conn,if_exists='replace')
conn.close()
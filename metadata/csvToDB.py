import os
import glob
import csv
import pandas as pd
import numpy as np
import sqlite3

def get_tickers(file):
    with open(file) as f:
        tickers = f.readlines()
    stripper = lambda x: (x.strip())
    return list(map(stripper, tickers))

tickers = get_tickers('topBRinNYSE.txt')
CWD = os.getcwd()
conn = sqlite3.connect('Recommendations.db')

for ticker in tickers:
    tickerDir = CWD+'/'+ticker
    os.chdir(tickerDir)
    ticker_df = pd.read_csv(ticker+'Recommendations.csv').set_index('DateTime')
    ticker_df.to_sql(ticker, con =conn, if_exists='replace')
    print('Recommendation histpry for '+ticker+" successfully saved in Reccomendations.db")
conn.close()
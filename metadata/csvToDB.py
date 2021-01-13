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

def recomendations_to_database():
    conn = sqlite3.connect('Recommendations.db')
    for ticker in tickers:
        tickerDir = CWD+'/'+ticker
        os.chdir(tickerDir)
        ticker_df = pd.read_csv(ticker+'Recommendations.csv').set_index('DateTime')
        ticker_df.to_sql(ticker, con =conn, if_exists='replace')
        print('Recommendation history for '+ticker+" successfully saved in Reccomendations.db")
    conn.close()
    return 0

def news_to_database():
    conn = sqlite3.connect('News.db')
    for ticker in tickers:
        tickerDir = CWD+'/'+ticker
        os.chdir(tickerDir)
        ticker_df = pd.read_csv(ticker+'News.csv').set_index('DateTime')
        ticker_df.to_sql(ticker, con =conn, if_exists='replace')
        print('Recommendation history for '+ticker+" successfully saved in Reccomendations.db")
    conn.close()
    return 0



tickers = get_tickers('topBRinNYSE.txt')
CWD = os.getcwd()
news_to_database()




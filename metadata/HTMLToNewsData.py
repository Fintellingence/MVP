from bs4 import BeautifulSoup
import os
import glob
import csv
import pandas as pd
import numpy as np

CSS_CLASS = {'up':'body-table-rating-upgrade', 'neutral':'body-table-rating-neutral','down':'body-table-rating-downgrade'}
CWD = os.getcwd()

def map_level(f, item, level):
    if level == 0:
        return f(item)
    else:
        return [map_level(f, i, level - 1) for i in item]

def get_tickers(file):
    with open(file) as f:
        tickers = f.readlines()
    stripper = lambda x: (x.strip())
    return list(map(stripper, tickers))

def string_to_YYYYMMDD_HHMMP(dateTimeString):
    months = ['Jan','Fev','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    date = dateTimeString.split(' ')[0]
    time = dateTimeString.split(' ')[1]
    decomposed = date.split('-')
    for month in months:
        if month == decomposed[0]:
            return dateTimeString
        if month == decomposed[1]:
            return decomposed[1]+'-'+decomposed[0]+'-'+decomposed[2]+' '+time


    
tickers = get_tickers('topBRinNYSE.txt')
print(tickers)

for ticker in tickers:
    print("scraping "+ticker+" pages")
    tickerDir = CWD+'/'+ticker
    os.chdir(tickerDir)
    HTMLpages = glob.glob("*.html")
    ticker_df = pd.DataFrame(columns=['DateTime', 'News'])
    for page in HTMLpages:
        with open(page) as html_doc:
            soup = BeautifulSoup(html_doc,'html.parser')
        print("scraping page "+page)
        tableNews = soup.find('table', class_='fullview-news-outer')
        if tableNews is None:
            break
        tableRows = tableNews.find_all('tr')
        tableRows = list(map(lambda element: element.contents,tableRows))
        tableData = list(map(lambda element: [(element[0].string)[:-2],element[1].find('a').string],tableRows))
        page_df = pd.DataFrame(tableData)
        page_df.columns = ['DateTime', 'News']
        ticker_df = pd.concat([ticker_df,page_df])
    
    ticker_df = ticker_df.reset_index()
    ticker_df = ticker_df.drop(columns='index')
    i=1
    while i< len(ticker_df.index):
        if len(ticker_df['DateTime'].iloc[i]) < len('MMM-DD-YY HH:MMXM'):
            ticker_df['DateTime'].iloc[i] = ticker_df['DateTime'].iloc[i-1][:10]+ticker_df['DateTime'].iloc[i]
        i+=1

    ticker_df['DateTime'] = ticker_df['DateTime'].map(string_to_YYYYMMDD_HHMMP)
    ticker_df['DateTime'] = pd.to_datetime(ticker_df['DateTime'],format='%b-%d-%y %H:%M%p')  #https://www.dataindependent.com/pandas/pandas-to-datetime/
    ticker_df.set_index('DateTime', inplace=True)
    ticker_df.sort_values(by='DateTime',inplace=True)
    ticker_df = ticker_df.drop_duplicates()
    ticker_df.to_csv(ticker+'News.csv')    
    print("News for "+ticker+" successfully scraped.")


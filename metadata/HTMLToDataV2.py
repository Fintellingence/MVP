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

def string_to_YYYYMMDD(string):
    MONTHS = {'Jan':'01','Fev':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
    decomposed = string.split('-')
    for month in MONTHS.keys():
        if month == decomposed[0]:
            return '20'+decomposed[2]+'/'+MONTHS[month]+'/'+decomposed[1]
        if month == decomposed[1]:
            return '20'+decomposed[2]+'/'+MONTHS[month]+'/'+decomposed[0]


    
tickers = get_tickers('topBRinNYSE.txt')
print(tickers)

for ticker in tickers:
    print("scraping "+ticker+" pages")
    tickerDir = CWD+'/'+ticker
    os.chdir(tickerDir)
    HTMLpages = glob.glob("*.html")
    globalRecommendations = list()
    ticker_df = pd.DataFrame(columns=['DateTime', 'Status','Company','Recommendation','Target'])
    for page in HTMLpages:
        with open(page) as html_doc:
            soup = BeautifulSoup(html_doc,'html.parser')
        print("scraping page "+page)
        tableRatings = soup.find('table', class_='fullview-ratings-outer')
        if tableRatings is None:
            break
        innerRatingTable = tableRatings.find_all('td', class_='fullview-ratings-inner')
        upgradeFilter = lambda innerRatingTable: innerRatingTable.find('tr', class_=CSS_CLASS['up'])
        neutralFilter = lambda innerRatingTable: innerRatingTable.find('tr', class_=CSS_CLASS['neutral'])
        downgradeFilter = lambda innerRatingTable: innerRatingTable.find('tr', class_=CSS_CLASS['down'])
        elementFilter = lambda td: td.find_all('td')
        contentFilter = lambda tag: tag.string

        upgradeTables = list(filter(None,list(map(upgradeFilter,innerRatingTable))))
        neutralTables = list(filter(None,list(map(neutralFilter,innerRatingTable))))
        downgradeTables = list(filter(None,list(map(downgradeFilter,innerRatingTable))))

        upgradeContent = list(map(elementFilter,upgradeTables))
        neutralContent = list(map(elementFilter,neutralTables))
        downgradeContent = list(map(elementFilter,downgradeTables))

        upRecommend = list(map_level(contentFilter,upgradeContent,2))
        neutralRecommend = list(map_level(contentFilter,neutralContent,2))
        downRecommend = list(map_level(contentFilter,downgradeContent,2))



        recommendations = upRecommend + neutralRecommend + downRecommend
        page_df = pd.DataFrame(recommendations)
        page_df.columns = ['DateTime', 'Status','Company','Recommendation','Target']
        ticker_df = pd.concat([ticker_df,page_df])
    ticker_df['DateTime'] = ticker_df['DateTime'].map(string_to_YYYYMMDD)
    ticker_df['DateTime'] = pd.to_datetime(ticker_df['DateTime'],format='%Y/%m/%d')
    ticker_df.set_index('DateTime', inplace=True)
    ticker_df.sort_values(by='DateTime',inplace=True)
    ticker_df = ticker_df.drop_duplicates()
    ticker_df.to_csv(ticker+'Recommendations.csv')    
    print("Recommendations for "+ticker+" successfully scraped.")


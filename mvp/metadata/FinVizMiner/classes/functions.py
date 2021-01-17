from bs4 import BeautifulSoup
from classes.helper import *
import pandas as pd
import numpy as np
import sqlite3

CSS_CLASS = {'up':'body-table-rating-upgrade', 'neutral':'body-table-rating-neutral','down':'body-table-rating-downgrade'}

def recommendations(html_doc):
    soup = BeautifulSoup(html_doc,'html.parser')
    tableRatings = soup.find('table', class_='fullview-ratings-outer')
    if tableRatings is None:
        return None
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
    return page_df

def news(html_doc):
    soup = BeautifulSoup(html_doc,'html.parser')
    tableNews = soup.find('table', class_='fullview-news-outer')
    if tableNews is None:
        return None
    tableRows = tableNews.find_all('tr')
    tableRows = list(map(lambda element: element.contents,tableRows))
    tableData = list(map(lambda element: [(element[0].string)[:-2],element[1].find('a').string],tableRows))
    page_df = pd.DataFrame(tableData)
    page_df.columns = ['DateTime', 'News']
    return page_df

def clean_recommendations(df_recommendation):
    df_recommendation['DateTime'] = df_recommendation['DateTime'].map(string_to_YYYYMMDD)
    df_recommendation['DateTime'] = pd.to_datetime(df_recommendation['DateTime'],format='%Y/%m/%d')
    df_recommendation.set_index('DateTime', inplace=True)
    df_recommendation.sort_values(by='DateTime',inplace=True)
    df_recommendation = df_recommendation.drop_duplicates()
    return df_recommendation



def clean_news(df_news):
    df_news = df_news.reset_index()
    df_news = df_news.drop(columns='index')
    i=1
    while i< len(df_news.index):
        if len(df_news['DateTime'].iloc[i]) < len('MMM-DD-YY HH:MMXM'):
            df_news['DateTime'].iloc[i] = df_news['DateTime'].iloc[i-1][:10]+df_news['DateTime'].iloc[i]
        i+=1

    df_news['DateTime'] = df_news['DateTime'].map(string_to_YYYYMMDD_HHMMP)
    df_news['DateTime'] = pd.to_datetime(df_news['DateTime'],format='%b-%d-%y %H:%M%p')  #https://www.dataindependent.com/pandas/pandas-to-datetime/
    df_news.set_index('DateTime', inplace=True)
    df_news.sort_values(by='DateTime',inplace=True)
    df_news = df_news.drop_duplicates()
    return df_news

def recommendations_to_database(clean_recommendations, ticker):
    conn = sqlite3.connect('Data/Recommendations.db')
    clean_recommendations.to_sql(ticker, con =conn, if_exists='replace')
    conn.close()
    return None

def news_to_database(clean_news, ticker):
    conn = sqlite3.connect('Data/News.db')
    clean_news.to_sql(ticker, con =conn, if_exists='replace')
    conn.close()
    return None


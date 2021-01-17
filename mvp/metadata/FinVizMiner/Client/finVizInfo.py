from classes.helper import *
from classes.functions import *
import pandas as pd

for ticker in get_tickers('Data/NYSEtickers.txt'):
    df_recommendation = pd.DataFrame(columns = ['DateTime', 'Status','Company','Recommendation','Target'])
    df_news = pd.DataFrame(columns = ['DateTime', 'News'])
    for URL in ticker_to_way_back_URL_list(ticker):
        html_doc = fetch_URL(URL)
        df_recommendation = pd.concat([df_recommendation, recommendations(html_doc)])       
        df_news = pd.concat([df_news, news(html_doc)])
    recommendations_to_database(clean_recommendations(df_recommendation),ticker)
    news_to_database(clean_news(df_news),ticker)
    print("Database entry for Recommendations ands News created, ticker: "+ticker)
    
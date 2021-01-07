import requests
import pprint

API_KEY = 'bvqusnn48v6ptdtkvk3g'
COMPANY_NEWS_END = 'https://finnhub.io/api/v1/stock/recommendation'

with open('finhubTickers.txt') as f:
    tickers = f.readlines()

payload = {'symbol': 'AMZN', 'token': API_KEY}
r = requests.get(COMPANY_NEWS_END, params=payload)
data = r.json()
pprint.pprint(data)
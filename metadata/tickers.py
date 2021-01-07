import requests
import pprint


API_KEY = 'bvqusnn48v6ptdtkvk3g'
LOOK_UP_END = 'https://finnhub.io/api/v1/search'

with open('stocks.txt') as f:
    stocks = f.readlines()

stripper = lambda x: (x.strip())
stocks = list(map(stripper, stocks))
print(len(stocks))

finhubTickers = []
for stock in stocks:
    payload = {'q': stock, 'token': API_KEY}
    r = requests.get(LOOK_UP_END, params=payload)
    data = r.json()
    for element in data['result']:
        if element['type'] == 'Common Stock':
            if element['displaySymbol'][-3:] == '.SA':
                finhubTickers.append(element)
                print(element['displaySymbol'])

with open('finhubTickers.txt', 'w') as filehandle:
    for ticker in finhubTickers:
        filehandle.write('%s\n' % ticker)
print(len(finhubTickers))
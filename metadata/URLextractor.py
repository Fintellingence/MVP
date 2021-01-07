import requests
import subprocess

FINVIZ_BASE_URL = 'finviz.com/quote.ashx'

with open('topBRinNYSE.txt') as f:
    tickers = f.readlines()

stripper = lambda x: (x.strip())
tickers = list(map(stripper, tickers))

finvizURLs = []
for ticker in tickers:
    finvizURL = FINVIZ_BASE_URL+'?t='+ticker
    finvizURLs.append(finvizURL)

with open('finvizURLs.txt', 'w') as filehandle:
    for URL in finvizURLs:
        filehandle.write('%s\n' % URL)

f = open("blah.txt", "w")
rc = subprocess.call(["wayback-machine-scraper ", "-a", "\'finviz.com/quote.ashx?t=VALE$\'","-f", "20100101", "-t", "20200101", "finviz.com/quote.ashx?t=VALE", "-v"])


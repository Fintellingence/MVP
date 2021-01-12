import requests
import subprocess
import os

CWD = os.getcwd()
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

for ticker in tickers:
    tickerDir = CWD+'/'+ticker
    if not os.path.isdir(tickerDir):
        try:
             os.mkdir(tickerDir)
        except OSError:
            print ("Creation of the directory %s failed" % tickerDir)
        else:
            print ("Successfully created the directory %s " % tickerDir)

for URL in zip(finvizURLs,tickers):
    tickerDir = CWD+'/'+URL[1]
    os.chdir(tickerDir)
    process = subprocess.run(["wayback-machine-scraper", "-a", "\'"+URL[0]+"$\'", "-f", "20100101", "-t", "20200101", URL[0],"-v"], capture_output=True)
    with open(URL[1]+'_out.txt','wb') as filehandle:
        filehandle.write(process.stderr)
        print("All pages fetched for "+URL[1])
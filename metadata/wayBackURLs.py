import re
import requests
import os

CWD = os.getcwd()

with open('output.txt') as f:
    lines = f.readlines()
stripper = lambda x: (x.strip())
lines = list(map(stripper, lines))

filtered = []
for line in lines:
    try:
        filtElement = re.search('\<(.+?)\>', line).group(1)
    except AttributeError:
        filtElement = ''
    filtered.append(filtElement)

filtered = list(set(filtered))
filterGet = lambda x: x[4:]
filtered = list(map(filterGet,filtered[1:]))

with open('topBRinNYSE.txt') as f:
    tickers = f.readlines()
stripper = lambda x: (x.strip())
tickers = list(map(stripper, tickers))


URLdict = {}
for ticker in tickers:
    tickerList = list()
    for URL in filtered:
        if URL[-len(ticker):] == ticker:
            tickerList.append(URL)
    URLdict[ticker] = tickerList

for ticker in URLdict.keys():
    tickerDir = CWD+'/'+ticker
    if not os.path.isdir(tickerDir):
        try:
             os.mkdir(tickerDir)
        except OSError:
            print ("Creation of the directory %s failed" % tickerDir)
        else:
            print ("Successfully created the directory %s " % tickerDir)
     

for ticker in URLdict.keys():
    tickerDir = CWD+'/'+ticker
    os.chdir(tickerDir)
    for URL in URLdict[ticker]:
        waybackPage = requests.get(URL)
        if waybackPage.status_code == 200:
            HTMLName = re.search('web/(.+?)id_', URL).group(1)
            with open(HTMLName+'.html', 'w+') as filehandle:
                 filehandle.write('%s\n' % waybackPage.content)
                 print("Successfully fetched the URL "+URL)

        else:
            print("Faild to fetch the URL "+URL)

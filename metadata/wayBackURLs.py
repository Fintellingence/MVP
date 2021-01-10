import re
import requests
import os

CWD = os.getcwd()


with open('topBRinNYSE.txt') as f:
    tickers = f.readlines()
stripper = lambda x: (x.strip())
tickers = list(map(stripper, tickers))


for ticker in tickers:
    tickerDir = CWD+'/'+ticker
    os.chdir(tickerDir)

    with open(ticker+'_out.txt') as f:
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
    
    for URL in filtered:
        try:
            waybackPage = requests.get(URL)
            if waybackPage.status_code == 200:
                HTMLName = re.search('web/(.+?)id_', URL).group(1)
                with open(HTMLName+'.html', 'w+') as filehandle:
                    filehandle.write('%s\n' % waybackPage.content)
                    print("Successfully fetched the URL "+URL)
            else:
                print("Failed to fetch the URL "+URL+" status code"+waybackPage.status_code)
        except:
            print("Failure to request "+waybackPage.url)
        

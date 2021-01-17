import requests
import subprocess
import re

def get_tickers(file):
    with open(file) as f:
        tickers = f.readlines()
    stripper = lambda x: (x.strip())
    return list(map(stripper, tickers))

def ticker_to_way_back_URL_list(ticker):
    FINVIZ_BASE_URL = 'finviz.com/quote.ashx'
    finvizURL = FINVIZ_BASE_URL+'?t='+ticker
    process = subprocess.run(["wayback-machine-scraper", "-a", "\'"+finvizURL+"$\'", \
        "-f", "20100101", "-t", "20200101", finvizURL,"-v"], capture_output=True)
    wayBackOut = process.stderr.decode("utf-8").splitlines()
    filtered = []
    for line in wayBackOut:
        try:
            filtElement = re.search('\<(.+?)\>', line).group(1)
        except AttributeError:
            filtElement = ''
        filtered.append(filtElement)
    filtered = list(set(filtered))
    filterGet = lambda x: x[4:]
    filtered = list(map(filterGet,filtered[1:]))
    return filtered

def fetch_URL(URL):
    try:
        r = requests.get(URL)
        if r.status_code == 200:
            print("Successfully fetched the URL "+URL)
        else:
            print("Failed to fetch "+URL+" status code "+ str(r.status_code))
        return r.content
    except:
        return None

def map_level(f, item, level):
    if level == 0:
        return f(item)
    else:
        return [map_level(f, i, level - 1) for i in item]

def string_to_YYYYMMDD(string):
    MONTHS = {'Jan':'01','Fev':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
    decomposed = string.split('-')
    for month in MONTHS.keys():
        if month == decomposed[0]:
            return '20'+decomposed[2]+'/'+MONTHS[month]+'/'+decomposed[1]
        if month == decomposed[1]:
            return '20'+decomposed[2]+'/'+MONTHS[month]+'/'+decomposed[0]

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

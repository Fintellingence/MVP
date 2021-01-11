from bs4 import BeautifulSoup
import os
import glob
import csv

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


tickers = get_tickers('topBRinNYSE.txt')
print(tickers)

for ticker in tickers:
    print("scraping "+ticker+" pages")
    tickerDir = CWD+'/'+ticker
    os.chdir(tickerDir)
    HTMLpages = glob.glob("*.html")
    globalRecommendations = list()
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
        globalRecommendations.append(recommendations)
    with open(ticker+"Recomendation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(globalRecommendations)
        print(globalRecommendations)
    print("Recommendations for "+ticker+" successfully scraped.")


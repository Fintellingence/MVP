from bs4 import BeautifulSoup

CSS_CLASS = {'up':'body-table-rating-upgrade', 'neutral':'body-table-rating-neutral','down':'body-table-rating-downgrade'}

def map_level(f, item, level):
    if level == 0:
        return f(item)
    else:
        return [map_level(f, i, level - 1) for i in item]

def map_two_args(f, list, param):
    return [f(x) for x in zip(list,param)]

with open('20190703230135.html') as html_doc:
    soup = BeautifulSoup(html_doc,'html.parser')

tableRatings = soup.find('table', class_='fullview-ratings-outer')
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



recommendations = {'up':upRecommend, 'neutral': neutralRecommend, 'down': downRecommend}
print(recommendations['up'])
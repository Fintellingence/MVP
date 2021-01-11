from bs4 import BeautifulSoup


with open('20190703230135.html') as html_doc:
    soup = BeautifulSoup(html_doc,'html.parser')

tableRatings = soup.find('table', class_='fullview-ratings-outer')
innerRatingTable = tableRatings.find_all('td', class_='fullview-ratings-inner')
contentTable = innerRatingTable[0].find('tr', class_='body-table-rating-upgrade')
print(contentTable.prettify())

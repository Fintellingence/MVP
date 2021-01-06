import requests

API_KEY = 'bvqusnn48v6ptdtkvk3g'
LOOK_UP_END = 'https://finnhub.io/api/v1/search'

payload = {'q': 'VALE3', 'token': API_KEY}
r = requests.get(LOOK_UP_END, params=payload)
print(r.json())
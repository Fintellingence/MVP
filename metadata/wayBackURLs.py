import re

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
print(len(filtered))

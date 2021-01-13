def string_to_YYYYMMDD(string):
    MONTHS = {'Jan':'01','Fev':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
    decomposed = string.split('-')
    print(decomposed)
    for month in MONTHS.keys():
        if month == decomposed[0]:
            print('20'+decomposed[2]+'-'+MONTHS[month]+'-'+decomposed[1])
            return '20'+decomposed[2]+'-'+MONTHS[month]+'-'+decomposed[1]
        if month == decomposed[1]:
            print('20'+decomposed[2]+'-'+MONTHS[month]+'-'+decomposed[0])
            return '20'+decomposed[2]+'-'+MONTHS[month]+'-'+decomposed[0]

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

print(string_to_YYYYMMDD_HHMMP('Dec-01-13 05:28PM'))
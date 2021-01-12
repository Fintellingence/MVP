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


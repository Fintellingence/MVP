class Helper:
    
    @staticmethod
    def getTickers(file):
        with open(file) as f:
            tickers = f.readlines()
        stripper = lambda x: (x.strip())
        return list(map(stripper, tickers))
   
    @staticmethod
    def getAverage(values):
        return 0

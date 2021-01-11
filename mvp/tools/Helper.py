class Helper:
    
    @staticmethod
    def get_tickers(file):
        with open(file) as f:
            tickers = f.readlines()
        stripper = lambda x: (x.strip())
        return list(map(stripper, tickers))
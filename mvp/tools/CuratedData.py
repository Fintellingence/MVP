import pandas as pd

class CuratedData:
    # In this model, data is provided as a class.
    def __init__(self, data, n):
        self.ticker = data.ticker
        self.volume = data.volume
        self.high = data.high
        self.low = data.low
        self.close = data.close
        self.open = data.open
        self.period = n
        #=========================================
        # Statistics #
        #=========================================
        self.simpleMA = self.getSimpleMA()
        self.deviation = self.getDeviation()
        self.ACF = self.getACF()
        self.fracDiff = self.getFracDiff(level)
        self.RSI = self.getRSI()
        self.stationarityScore = self.getStationarity()

    def getSimpleMA(self):
        return 0

    def getDeviation(self):
        return 0
    
    def getACF(self):
        return 0
    
    def getFracDiff(self,level):
        return 0
    
    def getRSI(self):
        return 0

    def getStationarity(self):
        return 0
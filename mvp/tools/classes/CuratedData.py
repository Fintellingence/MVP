import pandas as pd

class CuratedData:
    # In this model, data is provided as a class.
    def __init__(self, data, parameters):
        self.ticker = data.ticker
        self.dfCurated = data.df
        self.parameters = parameters
        #=========================================
        # Statistics #
        #=========================================
        for paramMA in self.parameters['MA']:
            self.dfCurated['MA'+str(paramMA)] = self.getSimpleMA(paramMA)
        
        self.stationarityScore = self.getStationarity()

    def getSimpleMA(self, paramMA):
        return  self.dfCurated['Close'].rolling(window=paramMA).mean()

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


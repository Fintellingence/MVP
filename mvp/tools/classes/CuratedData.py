import pandas as pd

class CuratedData:
    # In this model, data is provided as a class.
    def __init__(self, data, parameters):
        self.ticker = data.ticker
        self.dfCurated = data.df
        self.parameters = parameters
        self.returns = self.getReturns()
        #=========================================
        # Statistics #
        #=========================================
        for paramMA in self.parameters['MA']:
            self.dfCurated['MA'+str(paramMA)] = self.getSimpleMA(paramMA)
        
        for paramDEV in self.parameters['DEV']:
            self.dfCurated['DEV'+str(paramDEV)] = self.getDeviation(paramDEV)
        
        #for paramACF in self.parameters['ACF']:
            #self.dfCurated['ACF'+str(paramACF)] = self.getACF(paramACF)
        
        self.stationarityScore = self.getStationarity()

    def getSimpleMA(self, paramMA):
        return self.dfCurated['Close'].rolling(window=paramMA).mean()

    def getDeviation(self, paramDEV):
        return self.dfCurated['Close'].rolling(window=paramDEV).std()
    
    def getACF(self,paramACF):
        return 0 #self.returns.autocorr(lag = paramACF)
    
    def getFracDiff(self,level):
        return 0
    
    def getReturns(self):
        return 0 #(self.dfCurated['Close']-self.dfCurated['Close'].shift(1))/self.dfCurated['Close'].shift(1)

    def getRSI(self):
        return 0

    def getStationarity(self):
        return 0


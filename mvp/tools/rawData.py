import sqlite3

class rawData:
    def __init__(self, ticker):
        self.volume = self.getVolume()
        self.high = self.getHigh()
        self.low = self.getLow()
        self.close = self.getClose()
        self.open = self.getOpen()

    def getVolume(self):
        return 0

    def getHigh(self):
        return 0

    def getLow(self):
        return 0
    
    def getClose(self):
        return 0
    
    def getOpen(self):
        return 0
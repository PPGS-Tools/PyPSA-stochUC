import numpy as np
from pandas import DataFrame
from pandas.tseries.frequencies import to_offset
import json

class SeasonalityFeatures():
    def __init__(self,periods_n:list=[24]) -> None:
        self.periods_n = periods_n
        self.fitted = False

    def fit(self,y,freq=None):
        """index of y is used as a reference point for the sinewaves
        Params:
         - freq: If y has missing values the freq has to be set manualy
        """
        if freq ==None and y.index.freq == None:
            raise Exception("No frequency given. Please set param freq")
        self.fitted = True
        self.startInx = y.index[0]
        if freq == None:
            self.freq = y.index.freq
        else:
            self.freq = to_offset(freq)

    def __dateToIdx(self,date):
        return int((date - self.startInx)/self.freq)

    def addSeasonalFeatures(self,X:DataFrame)->DataFrame:
        X_ = X.copy()
        assert self.fitted
        t = X_.index.map(self.__dateToIdx).values
        for period in self.periods_n:
            X_[f"C{period}"] = np.cos(t*2*np.pi/period)
            X_[f"S{period}"] = np.sin(t*2*np.pi/period)
        return X_
    def __str__(self) -> str:
        return f"startIdx: {self.startInx}    freq: {self.freq.freqstr}    periods_n: {self.periods_n}"
    
    def toJson(self)->str:
        out = {"startInx":str(self.startInx),
               "freq":self.freq.freqstr,
               "periods_n":self.periods_n}
        return json.dumps(out, sort_keys=True, indent=4)
    
    @staticmethod
    def fromJson(jsonStr:str):
        data = json.loads(jsonStr)
        seas = SeasonalityFeatures()
        seas.startInx = data["startInx"]
        seas.freq = to_offset(data["freq"])
        seas.periods_n = data["periods_n"]
        return seas
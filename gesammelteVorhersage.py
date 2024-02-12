import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import r2_score,mean_squared_error
from sklearn.cluster import KMeans

from Daten.Zeitreihe import ladeVorverarbeiteteDaten,splitTrainTest,SeasonalityFeatures
from epftoolbox.data import DataScaler

from statsmodels.tsa.statespace.sarimax import SARIMAX,SARIMAXResults

import yaml

OK = '✅'
FAIL = '❌'
PRICES = ("pos_aFRR_[EURO/MW]","neg_aFRR_[EURO/MW]","Da_[EUR/MWh]")


class Vorhersage:
    def __init__(self,name:str,fit=False,disp=True) -> None:
        self.name = name
        self.endTrain = "2020"
        self.disp = disp
        with open("Vorhersage.yaml", "r",encoding="utf-8") as stream:
            try:
                conf = yaml.full_load(stream)
                if disp: 
                    print(OK+" Konfiguration wurde geladen")
            except yaml.YAMLError as exc:
                print(exc)
        vorhersage = conf["Vorhersagen"][name]
        self.arima_order = vorhersage["arima_order"]
        self.target = vorhersage["target"]
        self.seasonalitiesStr = vorhersage["seasonalities"]
        self.exog = vorhersage["exog"]
        self.freq = str(vorhersage["freq"])
        self.start_date = str(vorhersage["start_date"])
        self.lowTol = False
        if "lowTol" in vorhersage.keys():
            self.lowTol = vorhersage["lowTol"]
        self.fill = True
        if "fill" in vorhersage.keys():
            self.fill = vorhersage["fill"]

        self.fitScalers()
        self.fitSeasonality()

        if fit:
            self.model = self.fitModel()
            self.saveModel()
        else:
            self.model = self.loadModel()


    @property
    def __df(self):
        df = ladeVorverarbeiteteDaten(self.freq,filled=True)[self.start_date:]

        # Add additional Timeseries
        df["Residual_Load_[MW]"] = df["Load_Forecast_[MW]"] - df["Wind_[MW]"] - df["Solar_[MW]"]
        df["Weekend"] = df.index.map(lambda x: int(x.day_of_week > 4))
        return df
    
    @property
    def __dfForY(self):
        df = ladeVorverarbeiteteDaten(self.freq,filled=self.fill)[self.start_date:]

        # Add additional Timeseries
        df["Residual_Load_[MW]"] = df["Load_Forecast_[MW]"] - df["Wind_[MW]"] - df["Solar_[MW]"]
        df["Weekend"] = df.index.map(lambda x: int(x.day_of_week > 4))
        return df
    
    def fitScalers(self):
        y_train = self.y[:self.endTrain]
        X_train = self.X[:self.endTrain]

        self.__scalerY = DataScaler("Std")
        if self.target in PRICES:
            self.__scalerY = DataScaler("Invariant")

        self.__scalerXOther = DataScaler("Std")
        self.__scalerXPrices = DataScaler("Invariant")

        self.__scalerY.fit_transform(y_train.values)

        self.__scalerXPrices.fit_transform(X_train.loc[:,X_train.columns.isin(PRICES)].values)
        self.__scalerXOther.fit_transform(X_train.loc[:,~X_train.columns.isin(PRICES)].values)

    def getScalers(self):
        return (self.__scalerY,self.__scalerXOther,self.__scalerXPrices)

    def fitSeasonality(self):
        sF = SeasonalityFeatures(self.seasonalitiesStr)
        sF.fit(self.y[:self.endTrain])
        self.__seasonalityFeatures = sF

    def fitModel(self):
        y_train = self.y_scaled[:self.endTrain]
        X_train = self.X_scaled[:self.endTrain]
        X_seas = self.seasonalityFeatures.addSeasonalFeatures(X_train)

        if self.lowTol:
            model = SARIMAX(y_train,X_seas,order=self.arima_order).fit(disp=False,factr=10e9,maxiter=150)
        else:
            model = SARIMAX(y_train,X_seas,order=self.arima_order).fit(disp=False,maxiter=150)
        return model
    
    def saveModel(self):
        self.model.save(f"Vorhersage/Vorhersagemodelle/SARIMAX_Modelle/{self.name}.pickle",remove_data=True)
        print(OK+f" Model {self.name} Saved")

    def loadModel(self):
        return SARIMAXResults.load(f"Vorhersage/Vorhersagemodelle/SARIMAX_Modelle/{self.name}.pickle")

    
    def upsample(self,df):
        lastValue = df.shift(freq=self.freq).tail(1) # Künstlicher Wert, welcher am Ende temporär angehängt werden kann
        df = pd.concat([df,lastValue],axis=0)        # Anhängen
        df = df.resample("1H").ffill()               # Upsampling mitteils halten des letzten Wertes
        df.drop(df.tail(1).index,inplace=True)       # Entfernen des Endpunktes
        df /= 4                                      # Auf Preis pro Stunde umrechnen
        return df

    
    def __inverse_transform(self,y):
        y_inv = y.copy()
        y_inv = y_inv.apply(lambda x: self.__scalerY.inverse_transform(x.values.reshape(-1,1)).flatten())
        return y_inv
    
    def getClusteredForecast(self,
                marketClosureOnDa,
                start_date,
                daysToForecast = 1,
                n_samples = 1000,
                n_clusters=5):
        y_pred = self.__forecast(marketClosureOnDa,start_date,daysToForecast=daysToForecast,n_samples=n_samples)
        y_clust,p = self.__cluster(y_pred,n_clusters)
        y_clust = self.__inverse_transform(y_clust)
        if self.freq == "4H":
            y_clust = self.upsample(y_clust)
        return y_clust,p
    
    def __cluster(self,y_pred,n_clusters)->tuple:
        kmeans = KMeans(n_clusters, random_state=42, n_init="auto").fit(y_pred.values.T)
        lables = kmeans.labels_

        l = pd.Series(lables)
        p = l.groupby(l).count()
        p /= p.sum()
        p

        # define Clusters as nearest to mean
        centers = kmeans.cluster_centers_
        reduced = []
        for center in centers:
            distanceToCenter = y_pred.apply(lambda x: np.linalg.norm(center-x))    # calculate distance to mean
            szenarioClosestToCenter = distanceToCenter.sort_values().index[0]       # find closest to mean
            reduced.append(y_pred[szenarioClosestToCenter])                        # save this one
            # reduced.append(pd.Series(center,index=y_pred.index))  #alternativ wird Clustermittelpunkt genommen
        reduced = pd.DataFrame(reduced).reset_index(drop=True).T

        return reduced,p

    
    def __forecast(self,
                marketClosureOnDa,
                start_date,
                daysToForecast = 1,
                n_samples = 1000)->pd.DataFrame:
        
        y2 = self.y_scaled
        X2 = self.seasonalityFeatures.addSeasonalFeatures(self.X_scaled)

        marketClosureOnDa = pd.to_datetime(start_date).replace(hour=marketClosureOnDa) - pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        startPrediction = marketClosureOnDa.replace(hour=0,minute=0,second=0) + pd.Timedelta(days=1)
        endPrediction = marketClosureOnDa.replace(hour=23,minute=59,second=59) + pd.Timedelta(days=daysToForecast)

        if self.disp:
            print(f"Market closure time: {marketClosureOnDa}")
            print(f"Forcast Horizon: {startPrediction} - {endPrediction}")


        y_train = y2.loc[:marketClosureOnDa]
        y_test = y2.loc[marketClosureOnDa:endPrediction]
        X_train = X2.loc[:marketClosureOnDa]
        X_test = X2.loc[marketClosureOnDa:endPrediction]
        
        model = self.model.apply(y_train,X_train)

        n_steps = len(y_test)
        y_pred = model.simulate(n_steps,repetitions=n_samples,anchor="end",exog=X_test)[startPrediction:endPrediction]
        return y_pred
    
    def forecast(self,
                marketClosureOnDa,
                start_date,
                daysToForecast = 1,
                n_samples = 1000)->pd.DataFrame:
                
        y2 = self.y_scaled
        X2 = self.seasonalityFeatures.addSeasonalFeatures(self.X_scaled)

        marketClosureOnDa = pd.to_datetime(start_date).replace(hour=marketClosureOnDa) - pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        startPrediction = marketClosureOnDa.replace(hour=0,minute=0,second=0) + pd.Timedelta(days=1)
        endPrediction = marketClosureOnDa.replace(hour=23,minute=59,second=59) + pd.Timedelta(days=daysToForecast)

        if self.disp:
            print(f"Market closure time: {marketClosureOnDa}")
            print(f"Forcast Horizon: {startPrediction} - {endPrediction}")


        y_train = y2.loc[:marketClosureOnDa]
        y_test = y2.loc[marketClosureOnDa:endPrediction]
        X_train = X2.loc[:marketClosureOnDa]
        X_test = X2.loc[marketClosureOnDa:endPrediction]
        
        model = self.model.apply(y_train,X_train)

        n_steps = len(y_test)
        y_pred = model.simulate(n_steps,repetitions=n_samples,anchor="end",exog=X_test)
        return y_pred
    
    def inverse_transform(self,y):
        return self.__inverse_transform(y)
    
    def cluster(self,y_pred,n_clusters)->tuple:
        return self.__cluster(y_pred,n_clusters)

    @property
    def X(self):
        return self.__df[self.exog].copy()
    @property
    def y(self):
        if self.fill:
            return self.__df[[self.target]].copy()
        else:
            return self.__dfForY[[self.target]].copy()
    @property
    def X_scaled(self):
        X_scaled=self.X.copy()
        X_scaled.loc[:,X_scaled.columns.isin(PRICES)] = self.__scalerXPrices.transform(X_scaled.loc[:,X_scaled.columns.isin(PRICES)].values)
        X_scaled.loc[:,~X_scaled.columns.isin(PRICES)] = self.__scalerXOther.transform(X_scaled.loc[:,~X_scaled.columns.isin(PRICES)].values)
        return X_scaled
    @property
    def y_scaled(self):
        y_scaled=self.y.copy()
        y_scaled[:] = self.__scalerY.transform(y_scaled.values)
        return y_scaled
    @property
    def seasonalityFeatures(self)->SeasonalityFeatures:
        return self.__seasonalityFeatures
    
    

if __name__ == "__main__":
    v1 = Vorhersage("aFRR_pos")
    print(v1.getClusteredForecast(10,"2022-04-21")[0].index)
    



class GesammelteVorhersage:

    def __init__(self,fit=False,disp=True) -> None:
        # https://stackoverflow.com/a/1774043
        with open("Vorhersage.yaml", "r") as stream:
            try:
                self.__conf = yaml.full_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.vorhersagen = {}
        self.__disp = disp
        if fit:
            self.__fitModels()
        else:
            self.__loadModels()

    def __fitModels(self):
        n_models = len(self.__conf["Vorhersagen"])
        if n_models > 0:
            print(f"Fitting {n_models} models:")
            for name,vorhersage in self.__conf["Vorhersagen"].items():
                order = tuple(vorhersage["arima_order"])
                print(f"\t- {name:<10} ARIMA{order}")
        for name,vorhersage in self.__conf["Vorhersagen"].items():
            self.vorhersagen[name] = Vorhersage(name,fit=True,disp=False)

    def __loadModels(self):
        self.vorhersagen = {}
        for name in self.__conf["Vorhersagen"]:
            self.vorhersagen[name] = Vorhersage(name,disp=False)

    def __printModelSummary(self):
        n_models = len(self.__conf["Vorhersagen"])
        if n_models > 0:
            print(f"{n_models} Modelle:")
            for name,vorhersage in self.__conf["Vorhersagen"].items():
                order = tuple(vorhersage["arima_order"])
                print(f"\t- {name:<10} ARIMA{order}")

    def getClusteredForecast(self,
                marketClosureOnDa,
                start_date,
                daysToForecast = 1,
                n_samples = 1000,
                n_clusters=5):
        clusters = {}
        if self.__disp:
            print("Erstellen der Vorhersagen für:")
            self.__printModelSummary()
        for name,vorhersage in self.vorhersagen.items():
            clusters[name] = vorhersage.getClusteredForecast(marketClosureOnDa,start_date,daysToForecast=daysToForecast,n_samples=n_samples,n_clusters=n_clusters)
            if self.__disp:
                print(OK + f" {name}")
        return clusters
    
    def plotClusters(self,preds,date,prop=None):
        fig, axs = plt.subplots(2,2,sharex=True)
        for (predName,pred),ax in zip(preds.items(),axs.ravel()):
            p = pred[1]
            pred = pred[0]
            einheit = self.vorhersagen[predName].target.split("_")[-1].replace("[","").replace("]","")
            name = predName.split("_")
            subscript = "" if len(name) <= 1 else name[1]
            name = f"${name[0]}_{{{subscript}}}$"
            ax.plot(pred)
            ax.legend(p,loc="upper left",prop={'size': 7})
            ax.set_title(name)
            ax.set_ylabel(einheit)
            ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24,6)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.suptitle(f"Vorhersagen für den {date}")
        fig.tight_layout()
        plt.show()
    
# if __name__ == "__main__":
#     gV = GesammelteVorhersage()
#     print(gV.getClusteredForecast(10,"2022-04-20"))
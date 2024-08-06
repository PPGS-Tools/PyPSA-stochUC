import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
src = os.path.dirname(current) 
sys.path.append(src) 

from inv_scaler import StandardScaler, InvScaler
from seasonality_features import SeasonalityFeatures


import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.cluster import KMeans
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults


class Forecast:
    def __init__(self,y:Series,X:DataFrame | Series,arima_order,seasonalitiesList:list[int],lowTol = False, price_columns:list[str]=list(),freq="1H") -> None:
        self.freq = freq
        if isinstance(X,Series):
            X = X.to_frame()
        self.X = X.asfreq(freq)
        if isinstance(y,DataFrame):
            assert len(y.columns)==1
            y = y.iloc[:,0]
        self.y = y.asfreq(freq)
        self.y_name = y.name
        self.X_columns = X.columns
        self.arima_order = arima_order
        self.seasonalitiesList = seasonalitiesList
        self.start_date = y.index[0]
        self.lowTol = lowTol
        self.price_columns = price_columns

        self.fitScalers()
        self.fitSeasonality()


        self.model:SARIMAXResults = self.fitModel()

    def fitScalers(self):

        self.__scalerY = StandardScaler()
        if self.y.name in self.price_columns:
            self.__scalerY = InvScaler()

        self.__scalerXOther = StandardScaler()
        self.__scalerXPrices = InvScaler()

        self.__scalerY.fit_transform(self.y.values.reshape(-1, 1))

        X_prices = self.X.loc[:,self.X.columns.isin(self.price_columns)].values
        if X_prices.size>0:
            self.__scalerXPrices.fit_transform(X_prices)
        X_other = self.X.loc[:,~self.X.columns.isin(self.price_columns)].values
        if X_other.size>0:
            self.__scalerXOther.fit_transform(X_other)

    @property
    def X_scaled(self):
        X_scaled=self.X.copy()
        X_prices = self.X.loc[:,self.X.columns.isin(self.price_columns)]
        X_other = self.X.drop(columns=X_prices.columns)
        if X_prices.size>0:
            X_scaled[X_prices.columns] = self.__scalerXPrices.transform(X_prices.values)
        if X_other.size>0:
            X_scaled[X_other.columns] = self.__scalerXOther.transform(X_other.values)
        return X_scaled
    @property
    def y_scaled(self):
        y_scaled=self.y.copy()
        y_scaled[:] = self.__scalerY.transform(y_scaled.values.reshape(-1,1)).flatten()
        return y_scaled

    def getScalers(self):
        return (self.__scalerY,self.__scalerXOther,self.__scalerXPrices)

    def fitSeasonality(self):
        sF = SeasonalityFeatures(self.seasonalitiesList)
        sF.fit(self.y)
        self.seasonalityFeatures = sF

    def fitModel(self)->SARIMAXResults:
        y_train = self.y_scaled
        X_train = self.X_scaled
        X_seas = self.seasonalityFeatures.addSeasonalFeatures(X_train)

        if self.lowTol:
            model = SARIMAX(y_train,X_seas,order=self.arima_order).fit(disp=False,factr=10e9,maxiter=150)
        else:
            model = SARIMAX(y_train,X_seas,order=self.arima_order).fit(disp=False,maxiter=150)
        return model
    def save(self, fname, remove_data=False):
        """
        Save a pickle of this instance.

        Parameters
        ----------
        fname : {str, handle}
            A string filename or a file handle.
        remove_data : bool
            If False (default), then the instance is pickled without changes.
            If True, then all arrays with length nobs are set to None before
            pickling. See the remove_data method.
            In some cases not all arrays will be set to None.

        Notes
        -----
        If remove_data is true and the model result does not implement a
        remove_data method then this will raise an exception.
        """

        self.model.remove_data()
        if remove_data:
            self.remove_data()

        from statsmodels.iolib.smpickle import save_pickle
        save_pickle(self, fname)

    def remove_data(self):
        self.X = None
        self.y = None

    @classmethod
    def load(cls, fname)->"Forecast":
        """
        Load a pickled results instance

        .. warning::

           Loading pickled models is not secure against erroneous or
           maliciously constructed data. Never unpickle data received from
           an untrusted or unauthenticated source.

        Parameters
        ----------
        fname : {str, handle, pathlib.Path}
            A string filename or a file handle.

        Returns
        -------
        Results
            The unpickled results instance.
        """

        from statsmodels.iolib.smpickle import load_pickle
        return load_pickle(fname)


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
        y_pred = self.forecast(marketClosureOnDa,start_date,daysToForecast=daysToForecast,n_samples=n_samples,invTransform=False)
        y_clust,p = self.cluster(y_pred,n_clusters)
        y_clust = self.__inverse_transform(y_clust)
        # if self.freq == "4H":
        #     y_clust = self.upsample(y_clust)
        return y_clust,p

    def cluster(self,y_pred,n_clusters)->tuple:
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


    def forecast(self,
                marketClosureOnDa,
                start_date,
                daysToForecast = 1,
                n_samples = 1000,invTransform=True)->pd.DataFrame:

        y2 = self.y_scaled
        X2 = self.seasonalityFeatures.addSeasonalFeatures(self.X_scaled)

        marketClosureOnDa = pd.to_datetime(start_date).replace(hour=marketClosureOnDa) - pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        startPrediction = marketClosureOnDa.replace(hour=0,minute=0,second=0) + pd.Timedelta(days=1)
        endPrediction = marketClosureOnDa.replace(hour=23,minute=59,second=59) + pd.Timedelta(days=daysToForecast)

        print(f"Market closure time: {marketClosureOnDa}")
        print(f"Forcast Horizon: {startPrediction} - {endPrediction}")


        y_train = y2.loc[:marketClosureOnDa]
        y_test = y2.loc[marketClosureOnDa:endPrediction]
        X_train = X2.loc[:marketClosureOnDa]
        X_test = X2.loc[marketClosureOnDa:endPrediction]

        model = self.model.apply(y_train,X_train)

        n_steps = len(y_test)
        y_pred = model.simulate(n_steps,repetitions=n_samples,anchor="end",exog=X_test)[startPrediction:endPrediction]
        if invTransform:
            y_pred = self.__inverse_transform(y_pred)
        return y_pred

if __name__ == "__main__":
    from util import testData
    df = testData()
    y = df[["da_EUR_MWh"]]
    X = df[["residual_load_MW"]]
    v = Forecast(y,X,(1,1,4),[8,12,24,168,8760],price_columns=["da_EUR_MWh"],lowTol=True)
    v.save('test')

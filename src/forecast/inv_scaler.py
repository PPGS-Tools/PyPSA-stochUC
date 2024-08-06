import numpy as np
from numpy import ndarray
from pandas import DataFrame
from scipy.sparse import spmatrix
from sklearn.preprocessing import RobustScaler as StandardScaler

class InvScaler(StandardScaler):
    """Scaler which performs $\sinh^{-1}$ on the data before transforming.
    It uses median as mean and median absolute deviation as standard deviation.
    """

    def fit(self, X: ndarray | DataFrame | spmatrix) -> StandardScaler:
        X_copy = X.copy()
        return super().fit(X_copy)

    def fit_transform(self, X: ndarray | DataFrame | spmatrix) -> ndarray:
        super().fit(X)
        return self.transform(X)

    def transform(self, X: ndarray | DataFrame) -> ndarray:
        """Center, scale and transform the data.
        
        Uses $\sinh^{-1}$ to transform data before fitting.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the specified axis.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        X = super().transform(X)
        X = np.arcsinh(X)
        return X
    
    def inverse_transform(self, X: ndarray | DataFrame) -> ndarray:
        """Inverse Transform and scale back the data to the original representation.

        Uses $\sinh$ to transform data before fitting.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The rescaled data to be transformed back.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        X = np.sinh(X)
        X = super().inverse_transform(X)
        return X
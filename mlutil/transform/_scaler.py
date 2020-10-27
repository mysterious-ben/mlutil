import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from typing import Union


class SigmaClipper(TransformerMixin, BaseEstimator):
    """Clip values exceeding specified sigma levels from median
    
    :param low_sigma: low sigma level (aabsolute value)
        Must be non-negative
    :param max_sigma: high sigma level (aabsolute value)
        Must be non-negative
    :param mean_fun: 'median' | 'mean'
    """

    def __init__(self, low_sigma: float = 3., high_sigma: float = 3., mean_fun: str = 'median'):
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma
        self.mean_fun = mean_fun

    def fit(self, X: Union[np.array, pd.DataFrame], y=None):
        assert self.low_sigma >= 0
        assert self.high_sigma >= 0
        if self.mean_fun == 'median':
            mean = np.nanmedian
        elif self.mean_fun == 'mean':
            mean = np.nanmean
        else:
            raise ValueError(self.mean_fun)
        X_ = np.asarray(X)
        self.mean_ = mean(X_, axis=0)
        self.std_ = np.sqrt(mean((X_ - self.mean_) ** 2, axis=0))
        self.high_ = self.mean_ + self.low_sigma * self.std_
        self.low_ = self.mean_ - self.high_sigma * self.std_
        return self
        
    def transform(self, X: Union[np.array, pd.DataFrame]) -> Union[np.array, pd.DataFrame]:
        if isinstance(X, pd.DataFrame):
            X = X.clip(self.low_, self.high_, axis=1)
        else:
            X = np.clip(X, self.low_[np.newaxis, :], self.high_[np.newaxis, :])
        return X

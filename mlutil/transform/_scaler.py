import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from typing import Union, Optional


class SigmaClipper(TransformerMixin, BaseEstimator):
    """Clip values exceeding specified sigma levels from median
    
    :param sigma: both low_sigma and high_sigma
    :param low_sigma: low sigma level (aabsolute value)
        Overrides sigma
    :param max_sigma: high sigma level (aabsolute value)
        Overrides sigma
    :param mean_fun: 'median' | 'mean'
    """

    def __init__(self, sigma: Optional[float] = 3.,
                 low_sigma: Optional[float] = None,
                 high_sigma: Optional[float] = None,
                 mean_fun: str = 'median'):
        self.sigma = sigma
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma
        self.mean_fun = mean_fun

    def fit(self, X: Union[np.array, pd.DataFrame], y=None):
        self.low_sigma_ = self.sigma if self.low_sigma is None else self.low_sigma
        self.high_sigma_ = self.sigma if self.high_sigma is None else self.high_sigma
        assert self.low_sigma_ >= 0
        assert self.high_sigma_ >= 0
        
        if self.mean_fun == 'median':
            mean = np.nanmedian
        elif self.mean_fun == 'mean':
            mean = np.nanmean
        else:
            raise ValueError(self.mean_fun)
        X_ = np.asarray(X)
        self.mean_ = mean(X_, axis=0)
        self.std_ = np.sqrt(mean((X_ - self.mean_) ** 2, axis=0))
        self.high_ = self.mean_ + self.low_sigma_ * self.std_
        self.low_ = self.mean_ - self.high_sigma_ * self.std_
        return self
        
    def transform(self, X: Union[np.array, pd.DataFrame]) -> Union[np.array, pd.DataFrame]:
        if isinstance(X, pd.DataFrame):
            X = X.clip(self.low_, self.high_, axis=1)
        else:
            X = np.clip(X, self.low_[np.newaxis, :], self.high_[np.newaxis, :])
        return X

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from typing import Optional, List


class ColumnSelector(TransformerMixin, BaseEstimator):
    """Transformer affecting only selected columns
    
    :param transformer: scikit-learn transformer
    :param columns: select specific DataFrame columns
    :param columns_regex: select DataFrame columns via regex
        mutually exclusive with parameters `columns` and `columns_like`
    :param columns_like: select DataFrame columns via like
        mutually exclusive with parameters `columns` and `columns_regex`
    """

    def __init__(self, transformer=None,
                 columns: Optional[List[str]] = None,
                 columns_regex: Optional[str] = None,
                 columns_like: Optional[str] = None,
                 remainder: str = 'passthrough',
                 copy: bool = True):
        self.transformer = transformer
        self.columns = columns
        self.columns_regex = columns_regex
        self.columns_like = columns_like
        self.remainder = remainder
        self.copy = copy

    def fit(self, X: pd.DataFrame, y=None):
        if self.transform is None:
            return self
        if (self.columns is None) and (self.columns_regex is None) and (self.columns_like is None):
            self.columns_ = X.columns
        else:
            self.columns_ = X.filter(
                items=self.columns, regex=self.columns_regex, like=self.columns_like, axis=1
            ).columns
        self.transformer.fit(X=X[self.columns_], y=y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.transform is None:
            return X
        if self.remainder != 'passthrough':
            raise NotImplementedError
        if self.copy:
            X = X.copy()
        X_transformed = self.transformer.transform(X=X[self.columns_])
        if isinstance(X_transformed, pd.DataFrame):
            for c in self.columns_:
                X[c] = X_transformed[c]
        else:
            X.loc[:, self.columns_] = X_transformed
        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.transform is None:
            return X
        if hasattr(self.transformer, 'inverse_transform'):
            if self.copy:
                X = X.copy()
            X_orig = self.transformer.inverse_transform(X=X[self.columns_])
            if isinstance(X_orig, pd.DataFrame):
                for c in self.columns_:
                    X[c] = X_orig[c]
            else:
                X.loc[:, self.columns_] = X_orig
            return X
        else:
            return None

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from typing import Optional, List


class ColumnSelector(TransformerMixin, BaseEstimator):
    """Transformer affecting only selected columns"""

    def __init__(self, transformer=None,
                 columns: Optional[List[str]] = None,
                 remainder: str = 'passthrough',
                 copy: bool = True):
        self.transformer = transformer
        self.columns = columns
        self.remainder = remainder
        self.copy = copy

    def fit(self, X: pd.DataFrame, y=None):
        if self.transform is None:
            return self
        self.columns_ = self.columns if self.columns is not None else X.columns
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

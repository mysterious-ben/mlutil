from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pygam import GAM as _GAM
from pygam import f, l, s, te  # noqa: F401
from pygam.distributions import Distribution
from pygam.terms import Term
from sklearn.base import BaseEstimator, RegressorMixin


class GAM(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        terms: Union[str, Term] = "auto",
        distribution: Union[str, Distribution] = "normal",
        link: str = "identity",
        fit_intercept: bool = True,
        fit_linear: bool = False,
        fit_splines: bool = True,
        constraints: Optional[List] = None,
        lam: Union[float, List] = 0.6,
        n_splines: int = 25,
        spline_order: int = 3,
        penalties: Union[str, List] = "auto",
        dtype: str = "numerical",
        max_iter: int = 100,
        tol: float = 1e-4,
        callbacks: Tuple[str, ...] = ("deviance", "diffs"),
        verbose: bool = False,
        **kwargs,
    ):
        self.terms = terms
        self.distribution = distribution
        self.link = link
        self.fit_intercept = fit_intercept
        self.fit_linear = fit_linear
        self.fit_splines = fit_splines
        self.constraints = constraints
        self.lam = lam
        self.n_splines = n_splines
        self.spline_order = spline_order
        self.penalties = penalties
        self.dtype = dtype
        self.max_iter = max_iter
        self.tol = tol
        self.callbacks = callbacks
        self.verbose = verbose
        self.kwargs = kwargs

    @property
    def _is_fitted(self):
        return hasattr(self, "estimator_") and self.estimator_._is_fitted

    @property
    def coef_(self):
        return self.estimator_.coef_

    @property
    def statistics_(self):
        return self.estimator_.statistics_

    @property
    def logs_(self):
        return self.estimator_.logs_

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        weights: Optional[Union[np.ndarray, pd.Series]] = None,
    ):
        self.estimator_ = _GAM(
            terms=self.terms,
            distribution=self.distribution,
            link=self.link,
            fit_intercept=self.fit_intercept,
            fit_linear=self.fit_linear,
            fit_splines=self.fit_splines,
            constraints=self.constraints,
            lam=self.lam,
            n_splines=self.n_splines,
            spline_order=self.spline_order,
            penalties=self.penalties,
            dtype=self.dtype,
            max_iter=self.max_iter,
            tol=self.tol,
            callbacks=self.callbacks,
            verbose=self.verbose,
            **self.kwargs,
        )
        self._feature_names = (
            list(X.columns) if isinstance(X, pd.DataFrame) else list(range(X.shape[1]))
        )
        self.estimator_.fit(X=X, y=y, weights=weights)
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self._is_fitted:
            raise AttributeError("GAM has not been fitted. Call fit first.")
        return self.estimator_.predict(X)

    def loglikelihood(self, X, y, weights=None) -> np.ndarray:
        if not self._is_fitted:
            raise AttributeError("GAM has not been fitted. Call fit first.")
        return self.estimator_.loglikelihood(X, y, weights=weights)

    def confidence_intervals(
        self, X, width: Optional[float] = 0.95, quantiles: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        if not self._is_fitted:
            raise AttributeError("GAM has not been fitted. Call fit first.")
        return self.estimator_.confidence_intervals(X, width=width, quantiles=quantiles)

    def generate_X_grid(
        self, term: int, n: int = 100, meshgrid: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        if not self._is_fitted:
            raise AttributeError("GAM has not been fitted. Call fit first.")
        return self.estimator_.generate_X_grid(term=term, n=n, meshgrid=meshgrid)

    def partial_dependence(
        self,
        term: int,
        X=None,
        width: Optional[float] = None,
        quantiles: Optional[Tuple[float, float]] = None,
        meshgrid: bool = False,
    ) -> Tuple[np.ndarray, List]:
        if not self._is_fitted:
            raise AttributeError("GAM has not been fitted. Call fit first.")
        return self.estimator_.partial_dependence(
            term=term, X=X, width=width, quantiles=quantiles, meshgrid=meshgrid
        )

    def summary(self) -> None:
        if not self._is_fitted:
            raise AttributeError("GAM has not been fitted. Call fit first.")
        return self.estimator_.summary()

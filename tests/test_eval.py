import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from mlutil.eval import TimeSeriesSplit


def test_dummy():
    assert True


def test_TimeSeriesSplit():
    X = np.vstack([np.random.normal(size=100), np.random.normal(size=100)]).T
    y = np.random.normal(size=100)
    cv = TimeSeriesSplit(test_period_len=50)
    m = LinearRegression()
    scores = cross_validate(m, X, y, scoring=['neg_mean_squared_error'], cv=cv)
    assert len(scores['test_neg_mean_squared_error'] > 0)

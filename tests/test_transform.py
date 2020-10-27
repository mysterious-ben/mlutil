import pytest
import numpy as np
import pandas as pd
from mlutil.transform import SigmaClipper, ColumnSelector


@pytest.mark.parametrize(
    'X, X_new, sigma, sigma_low, sigma_high',
    [
        (
            pd.DataFrame({
                'a': [np.nan, -1., 2., 1., 1., 302.],
                'b': [-2., 1., 3., 2., -201, np.nan],
            }),
            pd.DataFrame({
                'a': [np.nan, -1., 2., 1., 1., 4.],
                'b': [-2., 1., 3., 2., -5, np.nan],
            }),
            3., None, None,
        ),
        (
            pd.DataFrame({
                'a': [np.nan, -1., 2., 1., 1., 302.],
                'b': [-2., 1., 3., 2., -201, np.nan],
            }),
            pd.DataFrame({
                'a': [np.nan, -1., 2., 1., 1., 4.],
                'b': [-2., 1., 3., 2., -5, np.nan],
            }),
            1., 3., 3.,
        ),
        (
            np.array([
                [np.nan, -1., 2., 1., 1., 302.],
                [-2., 1., 3., 2., -201, np.nan],
            ]).T,
            np.array([
                [np.nan, -1., 2., 1., 1., 4.],
                [-2., 1., 3., 2., -5, np.nan],
            ]).T,
            3., None, None,
        ),
    ]
)
def test_SigmaClipper(X, X_new, sigma, sigma_low, sigma_high):
    t = SigmaClipper(sigma=sigma, low_sigma=sigma_low, high_sigma=sigma_high)
    X_new_ = t.fit_transform(X)
    if isinstance(X, np.ndarray):
        np.testing.assert_allclose(X_new_, X_new)
    elif isinstance(X, pd.DataFrame):
        np.testing.assert_allclose(X_new_.values, X_new.values)
    else:
        raise TypeError(type(X))


@pytest.mark.parametrize(
    'X, X_new, columns',
    [
        (
            pd.DataFrame({
                'a': [np.nan, -1., 2., 1., 1., 302.],
                'b': [-2., 1., 3., 2., -201, np.nan],
            }),
            pd.DataFrame({
                'a': [np.nan, -1., 2., 1., 1., 4.],
                'b': [-2., 1., 3., 2., -5, np.nan],
            }),
            None,
        ),
        (
            pd.DataFrame({
                'a': [np.nan, -1., 2., 1., 1., 302.],
                'b': [-2., 1., 3., 2., -201, np.nan],
            }),
            pd.DataFrame({
                'a': [np.nan, -1., 2., 1., 1., 4.],
                'b': [-2., 1., 3., 2., -201, np.nan],
            }),
            ['a'],
        ),
    ]
)
def test_ColumnSelector(X, X_new, columns):
    t = ColumnSelector(
        SigmaClipper(low_sigma=3, high_sigma=3),
        columns=columns,
    )
    X_new_ = t.fit_transform(X)
    if isinstance(X, np.ndarray):
        np.testing.assert_allclose(X_new_, X_new)
    elif isinstance(X, pd.DataFrame):
        np.testing.assert_allclose(X_new_.values, X_new.values)
    else:
        raise TypeError(type(X))

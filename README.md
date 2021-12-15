# mlutil

Utilities for ML models.

Whenever possibles, modules are made to be compatible with `scikit-learn`. 

## Installation

```
pip install mlutil
```

## Modules

- `eval` (cross-validation)
- `model` (ML models)
- `transform` (feature transformers)
- `tune` (TBD: ML model tuning)

### Eval

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from mlutil.eval import TimeSeriesSplit

X = np.vstack([np.random.normal(size=100), np.random.normal(size=100)]).T
y = np.random.normal(size=100)
m = LinearRegression()
cv = TimeSeriesSplit(test_size=50)
scores = cross_validate(m, X, y, scoring=["neg_mean_squared_error"], cv=cv)
```

### Model

```python
import numpy as np
from mlutil.model import GAM

m = GAM(n_splines=5)
X = np.arange(20)[:, None]
y = np.arange(20) + np.random.normal(scale=0.1, size=20)
m.fit(X, y)
X_test = np.arange(15, 25)[:, None]
y_test = np.arange(15, 25)
y_hat = m.predict(X_test)
np.testing.assert_allclose(y_test, y_hat, atol=1.0)
m.summary()
```

### Transform

```python
import numpy as np
import pandas as pd
from mlutil.transform import ColumnSelector, SigmaClipper

X = pd.DataFrame({
    "a": [np.nan, -1.0, 2.0, 1.0, 1.0, 302.0],
    "b": [-2.0, 1.0, 3.0, 2.0, -201, np.nan],
})
t = ColumnSelector(SigmaClipper(low_sigma=3, high_sigma=3))
X_new_ = t.fit_transform(X)
```

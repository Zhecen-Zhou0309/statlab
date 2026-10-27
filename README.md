
# statlab

A simple Python package created for STAT 159 Lab 09 at UC Berkeley.

This package provides functions for data cleaning, standardization, rolling averages, linear regression, and RMSE calculation.

## Functions

### Data Preparation
- `clean_anomaly_data()`: Removes missing values and outliers.
- `compute_z_score()`: Computes z-score standardization.

### Statistics
- `compute_rolling_mean()`: Calculates a rolling average.
- `fit_linear_trend()`: Fits a simple linear regression model.
- `compute_rmse()`: Calculates root mean squared error.

## Installation

Install the package in editable mode:

```bash
python -m pip install -e .
```

## Testing

Run all unit tests with:

```bash
python -m pytest -v
```

## Example Usage

```python
from statlab import compute_rmse
import numpy as np

y_true = np.array([1, 2, 3])
y_pred = np.array([1, 2, 4])

rmse = compute_rmse(y_true, y_pred)
print(rmse)
```

## Author

Zhecen Zhou

STAT 159, Fall 2026

University of California, Berkeley

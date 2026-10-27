import numpy as np


def compute_rolling_mean(series, window=5):
    """Computes rolling mean with a given window size."""
    if window <= 0:
        raise ValueError('Window size must be a positive integer.')
    return series.rolling(window=window, min_periods=1).mean()


def fit_linear_trend(x, y):
    """Fits OLS linear regression y = slope * x + intercept."""
    if len(x) != len(y):
        raise ValueError('Input vectors x and y must have the same length.')
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    intercept = y_mean - slope * x_mean
    y_pred = slope * x + intercept
    return slope, intercept, y_pred


def compute_rmse(y_true, y_pred):
    """Calculates Root Mean Squared Error (RMSE)."""
    if len(y_true) != len(y_pred):
        raise ValueError('Vectors must be of identical length.')
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
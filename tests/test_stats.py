import pytest
import numpy as np
from statlab.stats import fit_linear_trend, compute_rmse


def test_fit_linear_trend_exact_line():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x + 3.0
    slope, intercept, y_pred = fit_linear_trend(x, y)
    assert pytest.approx(slope) == 2.0
    assert pytest.approx(intercept) == 3.0
    np.testing.assert_allclose(y_pred, y)


def test_compute_rmse_zero_for_identical_arrays():
    y = np.array([1.0, 2.0, 3.0])
    assert pytest.approx(compute_rmse(y, y)) == 0.0


def test_compute_rmse_length_mismatch_raises_error():
    with pytest.raises(ValueError):
        compute_rmse(np.array([1, 2]), np.array([1, 2, 3]))
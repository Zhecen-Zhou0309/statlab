import pytest
import numpy as np
import pandas as pd
from statlab.data_prep import clean_anomaly_data, compute_z_score


def test_clean_anomaly_data_removes_nans_and_outliers():
    df = pd.DataFrame({'month': [1, 2, 3, 4], 'anomaly': [1.0, np.nan, 99.0, 2.0]})
    cleaned = clean_anomaly_data(df, 'anomaly', min_val=-5, max_val=5)
    assert len(cleaned) == 2
    assert list(cleaned['anomaly']) == [1.0, 2.0]


def test_compute_z_score_mean_zero_std_one():
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    z = compute_z_score(s)
    assert pytest.approx(z.mean(), abs=1e-6) == 0.0
    assert pytest.approx(z.std(), abs=1e-6) == 1.0


def test_compute_z_score_zero_variance_raises_error():
    s = pd.Series([5.0, 5.0, 5.0])
    with pytest.raises(ValueError, match='Standard deviation is zero'):
        compute_z_score(s)
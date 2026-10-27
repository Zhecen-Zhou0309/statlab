def clean_anomaly_data(df, col_name='anomaly', min_val=-5.0, max_val=5.0):
    """Removes NaNs and filters extreme outliers from a dataframe column."""
    df_clean = df.copy()
    # Remove NaNs
    df_clean = df_clean.dropna(subset=[col_name])
    # Filter outliers
    mask = (df_clean[col_name] >= min_val) & (df_clean[col_name] <= max_val)
    return df_clean[mask].reset_index(drop=True)


def compute_z_score(series):
    """Computes z-score standardization for a pandas Series."""
    std_dev = series.std()
    if std_dev == 0:
        raise ValueError('Standard deviation is zero; cannot compute z-score.')
    return (series - series.mean()) / std_dev
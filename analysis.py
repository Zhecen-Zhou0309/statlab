import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statlab import (
    clean_anomaly_data,
    compute_z_score,
    compute_rolling_mean,
    fit_linear_trend,
    compute_rmse
)

# Generate synthetic climate anomaly data
np.random.seed(159)

months = np.arange(1, 101)
true_trend = 0.03 * months + 14.0
noise = np.random.normal(loc=0.0, scale=0.5, size=100)
anomalies = true_trend + noise

# Introduce missing values and an outlier
anomalies[10] = np.nan
anomalies[45] = 99.9

raw_df = pd.DataFrame({
    'month': months,
    'anomaly': anomalies
})

# Run the analysis using the statlab package
clean_df = clean_anomaly_data(
    raw_df, 'anomaly', min_val=10, max_val=20
)

clean_df['z_score'] = compute_z_score(clean_df['anomaly'])
clean_df['rolling_avg'] = compute_rolling_mean(
    clean_df['anomaly'], window=5
)

slope, intercept, y_pred = fit_linear_trend(
    clean_df['month'].values,
    clean_df['anomaly'].values
)

clean_df['trend_pred'] = y_pred
rmse = compute_rmse(clean_df['anomaly'].values, y_pred)

print(f'Cleaned Observations: {len(clean_df)}')
print(f'Fitted Slope: {slope:.4f} °C/month')
print(f'Fitted Intercept: {intercept:.4f} °C')
print(f'RMSE: {rmse:.4f}')

# Visualization
plt.figure(figsize=(10, 5))

plt.plot(
    clean_df['month'],
    clean_df['anomaly'],
    'o',
    label='Cleaned Observed Data',
    alpha=0.6
)

plt.plot(
    clean_df['month'],
    clean_df['rolling_avg'],
    '-',
    label='5-Month Rolling Average',
    linewidth=2
)

plt.plot(
    clean_df['month'],
    clean_df['trend_pred'],
    '--',
    label=f'Linear Trend (Slope={slope:.3f})',
    color='red'
)

plt.title('Climate Temperature Anomaly Analysis')
plt.xlabel('Month')
plt.ylabel('Temperature Anomaly (°C)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

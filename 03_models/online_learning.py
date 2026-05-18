import os
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, '02_data', 'improved_wait_dataset.csv')
OUT_PATH = os.path.join(PROJECT_ROOT, '05_results', 'models', 'online_learning_results.csv')
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

FEATURES = [
    'job_gpu', 'total_free', 'queue_length', 'running_jobs', 'max_free_node', 'variance_free',
    'can_fit_now', 'gpu_fit_ratio', 'fragmentation', 'queue_pressure', 'node_availability', 'avg_free_per_node'
]

def main():
    df = pd.read_csv(DATA_PATH).sample(frac=1, random_state=42).reset_index(drop=True)

    split = int(len(df) * 0.8)
    initial = df.iloc[:split]
    stream = df.iloc[split:].copy()

    drift_n = int(len(stream) * 0.2)
    stream.loc[:drift_n, 'queue_length'] = stream.loc[:drift_n, 'queue_length'] * 1.35
    stream.loc[:drift_n, 'queue_pressure'] = stream.loc[:drift_n, 'queue_pressure'] * 1.25

    x_init = initial[FEATURES].values
    y_init = initial['wait_time'].values
    x_stream = stream[FEATURES].values
    y_stream = stream['wait_time'].values

    scaler = StandardScaler()
    x_init_s = scaler.fit_transform(x_init)

    model = SGDRegressor(random_state=42, max_iter=4000, tol=1e-4)
    model.fit(x_init_s, y_init)

    pred_before = model.predict(scaler.transform(x_stream))
    mae_before = mean_absolute_error(y_stream, pred_before)
    r2_before = r2_score(y_stream, pred_before)

    batch_size = max(32, len(x_stream) // 8)
    for i in range(0, len(x_stream), batch_size):
        xb = x_stream[i:i + batch_size]
        yb = y_stream[i:i + batch_size]
        model.partial_fit(scaler.transform(xb), yb)

    pred_after = model.predict(scaler.transform(x_stream))
    mae_after = mean_absolute_error(y_stream, pred_after)
    r2_after = r2_score(y_stream, pred_after)

    out = pd.DataFrame([
        {'stage': 'without_retraining', 'mae': mae_before, 'r2': r2_before},
        {'stage': 'online_retrained', 'mae': mae_after, 'r2': r2_after},
    ])
    out.to_csv(OUT_PATH, index=False)

    print(out.to_string(index=False, formatters={'mae': '{:.4f}'.format, 'r2': '{:.4f}'.format}))
    print('Saved:', OUT_PATH)

if __name__ == '__main__':
    main()

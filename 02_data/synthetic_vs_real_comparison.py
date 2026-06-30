import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYN_PATH = os.path.join(PROJECT_ROOT, '02_data', 'improved_wait_dataset.csv')
REAL_PATH = os.path.join(PROJECT_ROOT, '02_data', 'lanl_trace_sample.csv')
OUT_DIR = os.path.join(PROJECT_ROOT, '05_results', 'traces')
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = [
    'job_gpu', 'total_free', 'queue_length', 'running_jobs', 'max_free_node', 'variance_free',
    'can_fit_now', 'gpu_fit_ratio', 'fragmentation', 'queue_pressure', 'node_availability', 'avg_free_per_node'
]


def ensure_real_trace():
    if not os.path.exists(REAL_PATH):
        subprocess.check_call(['python', os.path.join(PROJECT_ROOT, '02_data', 'load_real_traces.py')])


def build_real_features(real_df):
    total_free = 32 - real_df['num_gpus'].clip(1, 8)
    x = pd.DataFrame({
        'job_gpu': real_df['num_gpus'].clip(1, 8),
        'total_free': total_free.clip(0, 32),
        'queue_length': (real_df.index % 20) + 1,
        'running_jobs': (real_df.index % 8) + 1,
        'max_free_node': (total_free // 4).clip(0, 4),
        'variance_free': ((real_df['num_gpus'] - real_df['num_gpus'].mean()) ** 2).clip(0, 12),
    })
    x['can_fit_now'] = (x['total_free'] >= x['job_gpu']).astype(int)
    x['gpu_fit_ratio'] = np.minimum(x['total_free'] / (x['job_gpu'] + 1e-6), 1.0)
    x['fragmentation'] = np.sqrt(x['variance_free'])
    x['queue_pressure'] = (x['queue_length'] * x['job_gpu']) / (x['total_free'] + 1)
    x['node_availability'] = np.minimum(1.0, x['total_free'] / (x['job_gpu'] * 4 + 1e-6))
    x['avg_free_per_node'] = x['total_free'] / 8
    return x


def main():
    ensure_real_trace()

    syn = pd.read_csv(SYN_PATH)
    x_train = syn[FEATURES]
    y_train = syn['wait_time']

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(x_train, y_train)

    x_tr, x_te, y_tr, y_te = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    in_pred = model.predict(x_te)

    real = pd.read_csv(REAL_PATH)
    real = real.dropna(subset=['wait_time', 'num_gpus']).copy()
    x_real = build_real_features(real)
    y_real = real['wait_time'].values
    out_pred = model.predict(x_real)

    rows = [
        {
            'evaluation': 'synthetic_holdout',
            'samples': len(x_te),
            'mae': mean_absolute_error(y_te, in_pred),
            'r2': r2_score(y_te, in_pred),
        },
        {
            'evaluation': 'real_trace',
            'samples': len(x_real),
            'mae': mean_absolute_error(y_real, out_pred),
            'r2': r2_score(y_real, out_pred),
        },
    ]
    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, 'lanl_validation_results.csv')
    df.to_csv(out_csv, index=False)

    plt.figure(figsize=(7, 4.5))
    plt.bar(df['evaluation'], df['mae'], color=['#38bdf8', '#fb923c'])
    plt.ylabel('MAE (timesteps)')
    plt.title('Synthetic train → synthetic vs real-trace test')
    plt.tight_layout()
    out_plot = os.path.join(OUT_DIR, 'synthetic_vs_real_comparison.png')
    plt.savefig(out_plot, dpi=160)

    print(df.to_string(index=False, formatters={'mae': '{:.3f}'.format, 'r2': '{:.3f}'.format}))
    print('Saved:', out_csv)
    print('Saved:', out_plot)


if __name__ == '__main__':
    main()

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
SWF_PATH = os.path.join(PROJECT_ROOT, '02_data', 'lanl_trace_sample.swf')
REAL_PATH = os.path.join(PROJECT_ROOT, '02_data', 'lanl_trace_sample.csv')
OUT_DIR = os.path.join(PROJECT_ROOT, '05_results', 'traces')
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = [
    'job_gpu', 'total_free', 'queue_length', 'running_jobs', 'max_free_node', 'variance_free',
    'can_fit_now', 'gpu_fit_ratio', 'fragmentation', 'queue_pressure', 'node_availability', 'avg_free_per_node'
]


def ensure_real_trace():
    """Ensure the trace CSV exists. Returns True only if a real SWF file is
    present (i.e. the CSV was/can be parsed from real data); False means the
    CSV is the synthetic fallback proxy from load_real_traces.py."""
    swf_present = os.path.exists(SWF_PATH)
    if not os.path.exists(REAL_PATH):
        subprocess.check_call(['python', os.path.join(PROJECT_ROOT, '02_data', 'load_real_traces.py')])
    return swf_present


def build_real_features(real_df):
    """Build the 12 model features from a 4-column trace (arrival_time,
    wait_time, runtime, num_gpus).

    KNOWN LIMITATION — approximated cluster-state features.
    The trace records only per-job fields; it carries NO cluster state, so the
    following features CANNOT be derived from the trace and are heuristic
    approximations (they do NOT follow the training-time definitions in
    generate_improved_dataset.py, which come from a live simulator state):
      - total_free      : approximated as 32 - job_gpu (true value = free GPUs
                          across the cluster at arrival). Note this forces
                          can_fit_now == 1 and gpu_fit_ratio == 1.0 for every
                          row, unlike the training distribution.
      - queue_length    : (row_index % 20) + 1 — no queue info in the trace.
      - running_jobs    : (row_index % 8) + 1 — no running-set info in the trace.
      - max_free_node   : total_free // 4, clipped — true value = max per-node
                          free GPUs, unknown without per-node state.
      - variance_free   : squared deviation of the job's own GPU request,
                          clipped to 4.0 (the physical max of np.var over
                          8 nodes with 0-4 free GPUs each) — true value =
                          variance of per-node free GPUs.
      - node_availability: min(1, total_free / (job_gpu*4)) — true value =
                          fraction of nodes with >= job_gpu free GPUs.
      - queue_pressure  : total queued GPUs are unknown, so they are proxied
                          as queue_length * mean trace GPU request.
    Features whose FORMULAS match generate_improved_dataset.py exactly, given
    the approximated state above: can_fit_now (total_free >= job_gpu),
    gpu_fit_ratio (min(total_free/(job_gpu+1e-6), 1.0)), fragmentation
    (std = sqrt(variance_free)), queue_pressure (total_queued_gpus /
    (total_free + 1)), avg_free_per_node (total_free / 8).
    """
    total_free = 32 - real_df['num_gpus'].clip(1, 8)
    x = pd.DataFrame({
        'job_gpu': real_df['num_gpus'].clip(1, 8),
        'total_free': total_free.clip(0, 32),
        'queue_length': (real_df.index % 20) + 1,
        'running_jobs': (real_df.index % 8) + 1,
        'max_free_node': (total_free // 4).clip(0, 4),
        # clipped to 4.0 = physical max of np.var(cluster.nodes) for 4-GPU nodes
        'variance_free': ((real_df['num_gpus'] - real_df['num_gpus'].mean()) ** 2).clip(0, 4.0),
    })
    x['can_fit_now'] = (x['total_free'] >= x['job_gpu']).astype(int)
    x['gpu_fit_ratio'] = np.minimum(x['total_free'] / (x['job_gpu'] + 1e-6), 1.0)
    x['fragmentation'] = np.sqrt(x['variance_free'])
    # training definition: total_queued_gpus / (total_free + 1); queued GPUs
    # are approximated as queue_length * mean GPU request of the trace
    total_queued_gpus = x['queue_length'] * real_df['num_gpus'].clip(1, 8).mean()
    x['queue_pressure'] = total_queued_gpus / (x['total_free'] + 1)
    x['node_availability'] = np.minimum(1.0, x['total_free'] / (x['job_gpu'] * 4 + 1e-6))
    x['avg_free_per_node'] = x['total_free'] / 8
    return x


def main():
    swf_present = ensure_real_trace()
    trace_label = 'real_trace' if swf_present else 'synthetic_proxy_trace'

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
    x_tr, x_te, y_tr, y_te = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    model.fit(x_tr, y_tr)
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
            'evaluation': trace_label,
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
    plt.title(f'Synthetic train → synthetic holdout vs {trace_label} test')
    plt.tight_layout()
    out_plot = os.path.join(OUT_DIR, 'synthetic_vs_real_comparison.png')
    plt.savefig(out_plot, dpi=160)

    print(df.to_string(index=False, formatters={'mae': '{:.3f}'.format, 'r2': '{:.3f}'.format}))
    print('Saved:', out_csv)
    print('Saved:', out_plot)


if __name__ == '__main__':
    main()

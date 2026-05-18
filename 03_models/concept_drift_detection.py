import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, '02_data', 'improved_wait_dataset.csv')
OUT_DIR = os.path.join(PROJECT_ROOT, '05_results', 'models')
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = [
    'job_gpu', 'total_free', 'queue_length', 'running_jobs', 'max_free_node', 'variance_free',
    'can_fit_now', 'gpu_fit_ratio', 'fragmentation', 'queue_pressure', 'node_availability', 'avg_free_per_node'
]

def rolling_mae(y_true, y_pred, win=80):
    errs = np.abs(y_true - y_pred)
    out = []
    for i in range(len(errs)):
        st = max(0, i - win + 1)
        out.append(float(np.mean(errs[st:i + 1])))
    return np.array(out)

def main():
    df = pd.read_csv(DATA_PATH).sample(frac=1, random_state=41).reset_index(drop=True)
    x = df[FEATURES].copy()
    y = df['wait_time'].values

    drift_start = int(len(x) * 0.7)
    x.loc[drift_start:, 'queue_pressure'] *= 1.2
    x.loc[drift_start:, 'fragmentation'] *= 1.15

    split = int(len(x) * 0.6)
    x_train = x.iloc[:split].values
    y_train = y[:split]
    x_stream = x.iloc[split:].values
    y_stream = y[split:]

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    model = SGDRegressor(random_state=42, max_iter=3000, tol=1e-4)
    model.fit(x_train_s, y_train)

    preds = []
    triggers = []
    threshold = np.std(y_train) * 0.55

    for i in range(len(x_stream)):
        p = float(model.predict(scaler.transform(x_stream[i:i+1]))[0])
        preds.append(p)
        if i > 30:
            current_mae = np.mean(np.abs(y_stream[max(0, i - 30):i + 1] - np.array(preds[max(0, i - 30):i + 1])))
            if current_mae > threshold:
                model.partial_fit(scaler.transform(x_stream[max(0, i - 40):i + 1]), y_stream[max(0, i - 40):i + 1])
                triggers.append(i)

    preds = np.array(preds)
    rmae = rolling_mae(y_stream, preds)

    out = pd.DataFrame({
        'stream_index': np.arange(len(y_stream)),
        'actual_wait': y_stream,
        'predicted_wait': preds,
        'rolling_mae': rmae,
        'drift_trigger': [1 if i in triggers else 0 for i in range(len(y_stream))],
    })

    out_csv = os.path.join(OUT_DIR, 'concept_drift_results.csv')
    out.to_csv(out_csv, index=False)

    plt.figure(figsize=(10, 4.8))
    plt.plot(out['stream_index'], out['rolling_mae'], color='#38bdf8', label='Rolling MAE')
    plt.axhline(threshold, color='#fb923c', linestyle='--', label='Retrain trigger threshold')
    if triggers:
        plt.scatter(triggers, out.loc[triggers, 'rolling_mae'], color='#34d399', s=25, label='Adaptive retrain trigger')
    plt.title('Concept drift detection with adaptive retraining triggers')
    plt.xlabel('Stream index')
    plt.ylabel('Rolling MAE')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out_plot = os.path.join(OUT_DIR, 'concept_drift_triggers.png')
    plt.savefig(out_plot, dpi=160)

    print(f'Drift triggers fired: {len(triggers)}')
    print('Saved:', out_csv)
    print('Saved:', out_plot)

if __name__ == '__main__':
    main()

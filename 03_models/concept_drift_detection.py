import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, PROJECT_ROOT)
from vizstyle import (  # noqa: E402,F401  (shared house style; runs from any cwd)
    figure, finish, save_both, PALETTE, color_of, label_of, bar_ends, legend_roles,
)

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

    # ── figure (presentation only; nothing below changes a number) ───────────
    out_plot = os.path.join(OUT_DIR, 'concept_drift_triggers.png')
    plot_stem = out_plot[:-len('.png')]
    drift_onset = drift_start - split      # where the injected shift enters the stream

    for mode in ('light', 'dark'):
        p = PALETTE[mode]
        fig, ax = figure(mode, figsize=(10, 5.2))

        # One chart, one story: the error curve is the subject (series_1), the
        # retrain events are the second series (series_2). Reference lines are
        # neutral chrome, solid hairlines, direct-labelled instead of legended.
        ax.plot(out['stream_index'], out['rolling_mae'], color=p['series_1'],
                linewidth=1.9, label='Rolling MAE (80-job window)')

        # NB: the trigger test is evaluated on a 30-job window, the plotted curve
        # is the 80-job window written to the CSV, so markers can sit below this
        # line. The label says which window the line belongs to rather than
        # implying the two are the same series.
        ax.axhline(threshold, color=p['muted'], linewidth=1.0, zorder=1)
        ax.text(0.998, threshold, 'Retrain trigger threshold (tested on a 30-job window)  ',
                transform=ax.get_yaxis_transform(), ha='right', va='bottom',
                fontsize=8.5, color=p['muted'])

        if 0 < drift_onset < len(y_stream):
            ax.axvline(drift_onset, color=p['muted'], linewidth=1.0, zorder=1)
            ax.text(drift_onset, 0.985, '  Synthetic drift injected',
                    transform=ax.get_xaxis_transform(), ha='left', va='top',
                    fontsize=8.5, color=p['muted'])

        if triggers:
            ax.scatter(triggers, out.loc[triggers, 'rolling_mae'],
                       color=p['series_2'], s=34, zorder=3,
                       label='Adaptive retrain trigger')

        ax.set_xlabel('Stream position (jobs seen after the training split)')
        ax.set_ylabel('Rolling MAE (simulation time steps)')
        ax.set_xlim(0, len(y_stream) - 1)
        ax.set_axisbelow(True)
        ax.legend(loc='upper left')

        fig.tight_layout(rect=(0, 0.03, 1, 0.85))
        finish(
            fig, mode,
            title='Concept drift detection with adaptive retraining triggers',
            subtitle=(f'{len(triggers)} retrain events over the {len(y_stream)}-job stream; '
                      'markers overlap where consecutive jobs trigger'),
            source='02_data/improved_wait_dataset.csv -> 05_results/models/concept_drift_results.csv',
        )
        save_both(fig, plot_stem, mode)

    print(f'Drift triggers fired: {len(triggers)}')
    print('Saved:', out_csv)
    print('Saved:', out_plot)

if __name__ == '__main__':
    main()

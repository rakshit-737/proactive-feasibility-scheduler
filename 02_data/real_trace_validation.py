# real_trace_validation.py
#
# Validate the wait-time prediction approach on TWO real SWF traces
# (LANL CM-5 1994, SDSC SP2 1998) using the reconstructed per-arrival
# datasets produced by build_real_trace_datasets.py.
#
# Per trace, two evaluations:
#   (a) TRANSFER of the synthetic 12-feature model (03_models/wait_model_v2.pkl)
#       onto the real machine mapped to 8 virtual nodes of capacity/8 procs.
#       The SWF logs record no per-node placement, so node-level features
#       (max_free_node, variance_free, fragmentation, node_availability) are
#       APPROXIMATED with a uniform-occupancy split of total_free over the 8
#       virtual nodes (variance/fragmentation are therefore near zero).  The
#       model outputs synthetic simulator ticks, the trace waits are seconds,
#       so ONE affine calibration (a, b) is least-squares fitted on the FIRST
#       1000 jobs' (prediction, actual_wait_minutes) and metrics are reported
#       on the remainder.  This is a BEST-CASE transfer protocol and is
#       labelled as such in the output.
#   (b) RETRAIN on the trace itself (the headline test): XGBoost (same
#       hyperparameter family as 03_models/train_improved_model.py) on the 8
#       honestly-reconstructable features, target wait_log_minutes,
#       CHRONOLOGICAL 80/20 split (no shuffle -- realistic deployment).
#       Compared against predict-global-median and predict-rolling-mean
#       (last 100 jobs) baselines.
#
# Metric conventions (also stated in the notes column of the CSV):
#   * retrained + baselines: r2 in log space (wait_log_minutes),
#     mae_minutes back-transformed to minutes.
#   * transfer_calibrated: r2/mae directly in calibrated minutes space.
#
# Outputs (05_results/traces/):
#   real_trace_validation.csv
#   real_pred_vs_actual_{lanl,sdsc}.png       (+ -dark.png twin)
#   real_feature_importance_{lanl,sdsc}.png   (+ -dark.png twin)
#
# Figures are styled through the repo-wide vizstyle module: neither chart is
# about scheduling policies, so both use exactly ONE series hue (series_1) and
# neutral ink for reference marks and text.

import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from vizstyle import (figure, finish, save_both, PALETTE, color_of,  # noqa: E402
                      label_of, bar_ends, legend_roles)

DATA_DIR = os.path.join(PROJECT_ROOT, '02_data')
MODEL_PATH = os.path.join(PROJECT_ROOT, '03_models', 'wait_model_v2.pkl')
OUT_DIR = os.path.join(PROJECT_ROOT, '05_results', 'traces')
os.makedirs(OUT_DIR, exist_ok=True)

TRACES = [
    ('lanl', 'real_trace_dataset_lanl.csv', 1024),
    ('sdsc', 'real_trace_dataset_sdsc.csv', 128),
]

# --- FULL config (do not change for smoke tests; smoke uses the smaller CSVs
#     produced by build_real_trace_datasets.py --max-lines N) ---
CALIB_N = 1000            # transfer: jobs used for affine calibration
TRAIN_FRAC = 0.80         # retrain: chronological split
ROLLING_WINDOW = 100      # rolling-mean baseline window

RETRAIN_FEATURES = [
    'job_procs', 'total_free', 'queue_length', 'running_jobs',
    'can_fit_now', 'fit_ratio', 'queue_pressure', 'free_frac',
]

# --- presentation only: readable names for figure text -----------------------
TRACE_TITLE = {
    'lanl': 'LANL CM-5 (1994)',
    'sdsc': 'SDSC SP2 (1998)',
}

# Axes are labelled by feature NAME, never by column index.
FEATURE_LABEL = {
    'job_procs': 'Job size (processors)',
    'total_free': 'Free processors',
    'queue_length': 'Queue length',
    'running_jobs': 'Running jobs',
    'can_fit_now': 'Job fits right now',
    'fit_ratio': 'Free / requested ratio',
    'queue_pressure': 'Queued demand / free',
    'free_frac': 'Free fraction of machine',
}

SYNTH_FEATURES_ORDER = [
    'job_gpu', 'total_free', 'queue_length', 'running_jobs',
    'max_free_node', 'variance_free',
    'can_fit_now', 'gpu_fit_ratio', 'fragmentation',
    'queue_pressure', 'node_availability', 'avg_free_per_node',
]


# ── (a) transfer: map trace state onto the synthetic 12-feature schema ──────
def build_transfer_matrix(df, capacity):
    """Rescale the real machine into the synthetic 8-node x 4-GPU (32-unit)
    coordinate system the model was trained on, then apply a uniform-occupancy
    approximation for the node-level features.

    wait_model_v2 was trained with total_free in [0,32], max_free_node in
    [0,4], job_gpu in [1,8] etc.; feeding raw processor counts (up to 1024 on
    LANL) would saturate every tree split and measure garbage-in rather than
    distribution shift.  So: capacity -> 32 units, jobs -> 1..8 units
    (quarter-machine and larger saturates at 8 — documented), free capacity
    split evenly over 8 virtual nodes of 4 units.  queue_length/running_jobs
    stay as raw counts (there is no principled scale factor for job counts;
    their saturation is part of the honest sim-to-real gap).  Node-level
    variance/fragmentation/availability come from the even split because SWF
    has no placement info.  Output-side affine calibration (fit on the first
    CALIB_N jobs) absorbs the ts-vs-minutes unit difference.  Best-case
    protocol; labelled as such in the results.
    """
    s = 32.0 / capacity                                  # machine -> 32 units
    tf_raw = df['total_free'].to_numpy(dtype=float)      # already clamped >= 0
    procs = df['job_procs'].to_numpy(dtype=float)

    tf = tf_raw * s                                      # 0..32 synthetic units
    demand = procs * s                                   # job size, same units
    job_gpu = np.clip(np.ceil(demand), 1, 8)             # 1..8 like training

    # even split of ~integer tf over 8 virtual nodes of size 4:
    T = np.rint(tf)
    r = np.mod(T, 8.0)                                   # r nodes get +1 unit
    lo = np.floor(T / 8.0)
    hi = np.minimum(lo + (r > 0), 4.0)
    max_free_node = np.minimum(np.ceil(T / 8.0), 4.0)
    variance_free = r * (8.0 - r) / 64.0                 # var of the even split
    fragmentation = np.sqrt(variance_free)

    can_fit_now = (tf >= demand).astype(int)
    gpu_fit_ratio = np.minimum(tf / (demand + 1e-6), 1.0)

    # queue_pressure in scaled units: recover raw queued procs from the stored
    # ratio (qp_raw = sum_queued/(tf_raw+1)), rescale, re-normalise.
    qp_raw = df['queue_pressure'].to_numpy(dtype=float)
    queued_procs = qp_raw * (tf_raw + 1.0)
    queue_pressure = (queued_procs * s) / (tf + 1.0)

    node_availability = (r * (hi >= job_gpu).astype(float)
                         + (8.0 - r) * (lo >= job_gpu).astype(float)) / 8.0
    avg_free_per_node = tf / 8.0                         # 0..4 like training

    X = np.column_stack([
        job_gpu, tf, df['queue_length'].to_numpy(dtype=float),
        df['running_jobs'].to_numpy(dtype=float),
        max_free_node, variance_free,
        can_fit_now, gpu_fit_ratio, fragmentation,
        queue_pressure, node_availability, avg_free_per_node,
    ])
    return X


def eval_transfer(df, capacity, model, features):
    assert features == SYNTH_FEATURES_ORDER, \
        f'wait_model_v2 feature order changed: {features}'
    X = build_transfer_matrix(df, capacity)
    pred_raw = model.predict(X).astype(float)            # synthetic tick units
    actual_min = df['wait_time'].to_numpy(dtype=float) / 60.0

    n_cal = min(CALIB_N, max(len(df) // 10, 2))
    # affine calibration actual ~= a*pred + b on the FIRST n_cal jobs
    A = np.column_stack([pred_raw[:n_cal], np.ones(n_cal)])
    (a, b), *_ = np.linalg.lstsq(A, actual_min[:n_cal], rcond=None)
    pred_cal = a * pred_raw[n_cal:] + b
    y = actual_min[n_cal:]

    r2 = r2_score(y, pred_cal)
    mae = mean_absolute_error(y, pred_cal)
    print(f'  [transfer] affine calib a={a:.4f} b={b:.4f} on first {n_cal} jobs')
    print(f'  [transfer] calibrated R2={r2:.4f}  MAE={mae:.2f} min  (n={len(y)})')
    return {
        'method': 'transfer_calibrated', 'r2': r2, 'mae_minutes': mae,
        'n_train': n_cal, 'n_test': len(y),
        'notes': ('best-case transfer protocol; node-level features '
                  'uniform-occupancy approximated (no placement in SWF); '
                  f'affine calib a={a:.4g} b={b:.4g}; r2/mae in raw minutes'),
    }


# ── (b) retrain on the trace itself ─────────────────────────────────────────
def eval_retrained(df, name):
    n = len(df)
    n_train = int(n * TRAIN_FRAC)
    train, test = df.iloc[:n_train], df.iloc[n_train:]
    y_train_log = train['wait_log_minutes'].to_numpy(dtype=float)
    y_test_log = test['wait_log_minutes'].to_numpy(dtype=float)
    y_test_min = test['wait_time'].to_numpy(dtype=float) / 60.0

    # same hyperparameter family as 03_models/train_improved_model.py
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(train[RETRAIN_FEATURES], y_train_log)

    pred_log = model.predict(test[RETRAIN_FEATURES]).astype(float)
    pred_min = np.expm1(pred_log)                        # back-transform
    r2 = r2_score(y_test_log, pred_log)
    mae = mean_absolute_error(y_test_min, np.maximum(pred_min, 0.0))
    print(f'  [retrained] chrono 80/20  R2(log)={r2:.4f}  MAE={mae:.2f} min '
          f'(train={n_train}, test={n - n_train})')

    rows = [{
        'method': 'retrained', 'r2': r2, 'mae_minutes': mae,
        'n_train': n_train, 'n_test': n - n_train,
        'notes': ('XGBoost on 8 reconstructable features, target '
                  'wait_log_minutes, chronological split; r2 in log space, '
                  'mae back-transformed to minutes'),
    }]

    # baseline 1: predict-global-median (train median; median commutes with
    # the monotone log1p transform, so one constant serves both spaces)
    med_min = float(np.median(train['wait_time'] / 60.0))
    med_log = float(np.log1p(med_min))
    r2_med = r2_score(y_test_log, np.full_like(y_test_log, med_log))
    mae_med = mean_absolute_error(y_test_min, np.full_like(y_test_min, med_min))
    print(f'  [baseline_median] R2(log)={r2_med:.4f}  MAE={mae_med:.2f} min '
          f'(constant {med_min:.2f} min)')
    rows.append({
        'method': 'baseline_median', 'r2': r2_med, 'mae_minutes': mae_med,
        'n_train': n_train, 'n_test': n - n_train,
        'notes': (f'constant train-median prediction ({med_min:.2f} min); '
                  'constant predictor => r2 <= 0 by construction'),
    })

    # baseline 2: rolling mean of the last 100 ACTUAL waits (causal: for each
    # test job, mean over the preceding 100 jobs incl. train tail).  Rolling
    # mean of log values for R2(log); rolling mean of minutes for MAE.
    all_log = df['wait_log_minutes'].to_numpy(dtype=float)
    all_min = df['wait_time'].to_numpy(dtype=float) / 60.0
    s_log = pd.Series(all_log).rolling(ROLLING_WINDOW, min_periods=1).mean().shift(1)
    s_min = pd.Series(all_min).rolling(ROLLING_WINDOW, min_periods=1).mean().shift(1)
    pred_roll_log = s_log.to_numpy()[n_train:]
    pred_roll_min = s_min.to_numpy()[n_train:]
    r2_roll = r2_score(y_test_log, pred_roll_log)
    mae_roll = mean_absolute_error(y_test_min, pred_roll_min)
    print(f'  [baseline_rolling] R2(log)={r2_roll:.4f}  MAE={mae_roll:.2f} min')
    rows.append({
        'method': 'baseline_rolling', 'r2': r2_roll, 'mae_minutes': mae_roll,
        'n_train': n_train, 'n_test': n - n_train,
        'notes': (f'causal rolling mean of last {ROLLING_WINDOW} actual waits; '
                  'log-space mean for r2, minute-space mean for mae'),
    })

    # ── plots ────────────────────────────────────────────────────────────
    # Presentation only below this line: every quantity plotted is already
    # computed above.  Each figure is rendered twice, light and dark.
    trace_title = TRACE_TITLE.get(name, name.upper())
    source = f'02_data/real_trace_dataset_{name}.csv'

    # holdout scatter, log axes (+1 min offset so zero waits are plottable)
    x_actual = y_test_min + 1.0
    y_pred = np.maximum(pred_min, 0.0) + 1.0
    lims = [1.0, max(float(np.max(y_test_min)), float(np.max(pred_min)), 1.0) + 1.0]
    scatter_stem = os.path.join(OUT_DIR, f'real_pred_vs_actual_{name}')
    for mode in ('light', 'dark'):
        p = PALETTE[mode]
        fig, ax = figure(mode, figsize=(7, 6.6))
        ax.scatter(x_actual, y_pred, s=4,
                   alpha=0.15 if mode == 'light' else 0.22,
                   color=p['series_1'], edgecolors='none', zorder=3)
        # reference rule, not a series: neutral ink, solid hairline
        ax.plot(lims, lims, color=p['muted'], linewidth=1.1, zorder=2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Actual wait (minutes + 1, log scale)')
        ax.set_ylabel('Predicted wait (minutes + 1, log scale)')
        ax.set_axisbelow(True)
        ax.legend(handles=[
            Line2D([], [], marker='o', linestyle='none', markersize=5,
                   color=p['series_1'], label='Held-out job'),
            Line2D([], [], color=p['muted'], linewidth=1.1,
                   label='Perfect prediction (y = x)'),
        ], loc='upper left')
        fig.tight_layout(rect=(0, 0.02, 1, 0.87))
        finish(fig, mode,
               title=f'{trace_title}: predicted vs actual wait, holdout',
               subtitle=(f'XGBoost retrained on the trace, chronological 80/20 '
                         f'split  ·  R²(log) = {r2:.3f}  ·  MAE = {mae:.1f} min  '
                         f'·  n = {n - n_train} held-out jobs'),
               source=source)
        print(f'  Saved: {save_both(fig, scatter_stem, mode)}')

    # feature importances -- one series colour; bars labelled by feature NAME
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]
    imp_stem = os.path.join(OUT_DIR, f'real_feature_importance_{name}')
    bar_order = order[::-1]                              # barh draws bottom-up
    bar_names = [FEATURE_LABEL.get(RETRAIN_FEATURES[i], RETRAIN_FEATURES[i])
                 for i in bar_order]
    top_imp = max(float(imp.max()), 1e-9)                # display scale only
    for mode in ('light', 'dark'):
        p = PALETTE[mode]
        fig, ax = figure(mode, figsize=(8.4, 5.2))
        ax.barh(bar_names, imp[bar_order], color=p['series_1'], height=0.68)
        bar_ends(ax, 'h')
        ax.set_xlabel('Relative importance (share of total gain)')
        ax.set_xlim(0, top_imp * 1.18)
        # direct-label the leading feature only, never every bar
        ax.text(imp[order[0]] + top_imp * 0.02, len(bar_names) - 1,
                f'{imp[order[0]]:.2f}', va='center', ha='left',
                color=p['ink_2'], fontsize=9)
        fig.tight_layout(rect=(0, 0.02, 1, 0.87))
        finish(fig, mode,
               title=f'{trace_title}: what carries the wait signal',
               subtitle=('Gain-based feature importance, XGBoost retrained on '
                         'the 8 reconstructable trace features'),
               source=source)
        print(f'  Saved: {save_both(fig, imp_stem, mode)}')
    print('  Importances: ' + ', '.join(
        f'{RETRAIN_FEATURES[i]}={imp[i]:.3f}' for i in order))

    return rows


def main():
    random.seed(42)      # run_index 0 -> seed 42 (repo convention)
    np.random.seed(42)

    with open(MODEL_PATH, 'rb') as f:
        bundle = pickle.load(f)
    synth_model = bundle['model']
    synth_features = bundle['features']
    print(f'Loaded wait_model_v2 | features: {synth_features}')

    all_rows = []
    for name, csv_file, expected_capacity in TRACES:
        csv_path = os.path.join(DATA_DIR, csv_file)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f'{csv_path} missing -- run build_real_trace_datasets.py first')
        df = pd.read_csv(csv_path)
        print(f'\n=== {name.upper()} ({len(df)} jobs, capacity {expected_capacity}) ===')
        neg_pct = 100.0 * df['neg_free_flag'].mean()
        print(f'  Reconstructed free-capacity was negative (clamped) for '
              f'{neg_pct:.2f}% of arrivals')

        rows = [eval_transfer(df, expected_capacity, synth_model, synth_features)]
        rows += eval_retrained(df, name)
        for r in rows:
            r['trace'] = name
            if r['method'] == 'transfer_calibrated' and neg_pct > 0:
                r['notes'] += f'; {neg_pct:.1f}% arrivals had clamped negative free'
        all_rows.extend(rows)

    out = pd.DataFrame(all_rows)[
        ['trace', 'method', 'r2', 'mae_minutes', 'n_train', 'n_test', 'notes']]
    out_path = os.path.join(OUT_DIR, 'real_trace_validation.csv')
    out.to_csv(out_path, index=False, encoding='utf-8')
    print(f'\nSaved: {out_path}')
    print(out[['trace', 'method', 'r2', 'mae_minutes', 'n_train', 'n_test']]
          .to_string(index=False))


if __name__ == '__main__':
    main()

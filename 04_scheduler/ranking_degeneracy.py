"""Ranking degeneracy of a wait-time model used as a queue-ordering policy.

THE CLAIM
---------
The proactive scheduler ranks the queue by predicted wait. At any single
dispatch instant every queued job observes the SAME cluster: total free
processors, queue length, number of running jobs, fragmentation, and so on are
properties of the machine, not of the job. In the 12-feature synthetic vector
(and the 8-feature per-trace vector) the only inputs that differ between two
jobs waiting side by side are:

    job_gpu / job_procs        the requested size
    can_fit_now                = 1[total_free >= job_gpu]
    gpu_fit_ratio / fit_ratio  = min(total_free / job_gpu, 1)
    node_availability          = |{nodes with >= job_gpu free}| / num_nodes
    queue_pressure             = (queued_procs - job_gpu) / (total_free + 1)

and every one of those is a deterministic function of the requested size once
the cluster state is fixed. Therefore, holding the instant fixed, the model's
score is a function of requested size ALONE:

    predicted_wait(job | state) = g_state(job_gpu)

and the ranking it induces is the ranking of g_state over the queue's sizes.
The eleven cluster-state features shift every score by the same amount and
cannot reorder anything. The learned model, used this way, is a per-instant
lookup table from requested size to priority -- not a policy that reasons
about individual jobs.

CONSEQUENCE
-----------
The reported mean-wait improvement over FCFS is attributable to size-based
ordering, not to what the model learned about queue dynamics. If the argument
is right, an ML-free "smallest requested size first" policy must reproduce the
proactive scheduler's decisions, and the ML pipeline (dataset generation,
training, inference, SHAP explanation, drift monitoring) buys nothing over a
one-line sort key.

WHAT THIS SCRIPT MEASURES
-------------------------
It instruments REAL dispatch instants (via the RANK_OBSERVER hook in both
benchmarks -- no reimplementation of the simulators) and reports, over every
instant with at least two queued jobs:

  1. per-feature coefficient of variation ACROSS the queue -- how many features
     actually differ between co-queued jobs;
  2. functional dependence -- whether two jobs with equal requested size ever
     receive different predictions (they cannot, if the claim holds);
  3. Kendall tau between the model's order and the smallest-size-first order;
  4. the fraction of instants where the two orders are IDENTICAL;
  5. the fraction where the model's order is identical to plain arrival order
     (the degenerate-to-FCFS case, which happens when tree plateaus give every
     queued job the same score).

Outputs: 05_results/degeneracy/{ranking_degeneracy.csv,
         feature_variation.csv, ranking_degeneracy.png}

Usage:
  python ranking_degeneracy.py              # synthetic + both traces
  python ranking_degeneracy.py --quick      # fewer runs/windows
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, '05_results', 'degeneracy')
os.makedirs(OUT_DIR, exist_ok=True)

SIZE_COL = 0     # requested size is feature 0 in both feature vectors


class Collector:
    """Accumulates statistics over observed ranking decisions."""

    def __init__(self, setting, feature_names):
        self.setting = setting
        self.feature_names = feature_names
        self.instants = 0
        self.same_as_size = 0
        self.same_as_arrival = 0
        self.taus = []
        self.n_distinct_pred = []
        self.queue_lens = []
        self.violations = 0          # equal size, different prediction
        self.feat_varies = np.zeros(len(feature_names), dtype=np.int64)
        self.monotone_instants = 0   # score non-decreasing in size?
        self.size_score = []         # (size, within-instant normalised score)

    def __call__(self, policy, queue, x, pred):
        n = len(queue)
        if n < 2:
            return
        self.instants += 1
        self.queue_lens.append(n)

        # (1) which features actually differ across co-queued jobs?
        spread = x.max(axis=0) - x.min(axis=0)
        self.feat_varies += (np.abs(spread) > 1e-12).astype(np.int64)

        # (2) functional dependence on size: equal size => equal prediction
        sizes = x[:, SIZE_COL]
        for s in np.unique(sizes):
            p = pred[sizes == s]
            if len(p) > 1 and (p.max() - p.min()) > 1e-9:
                self.violations += 1

        self.n_distinct_pred.append(len(np.unique(np.round(pred, 9))))

        # (2b) the score IS a lookup table over size -- recover it. Is that
        # table monotone increasing (i.e. exactly smallest-first), or has the
        # model learned a non-trivial size preference?
        uniq = np.unique(sizes)
        table = np.array([pred[sizes == s].mean() for s in uniq])
        if len(uniq) > 1:
            if np.all(np.diff(table) >= -1e-9):
                self.monotone_instants += 1
            lo, hi = table.min(), table.max()
            if hi > lo:
                for s, v in zip(uniq, (table - lo) / (hi - lo)):
                    self.size_score.append((float(s), float(v)))

        # (3) rank agreement with smallest-size-first
        if len(np.unique(sizes)) > 1 and len(np.unique(pred)) > 1:
            tau = stats.kendalltau(pred, sizes).statistic
            if not np.isnan(tau):
                self.taus.append(float(tau))

        # (4)/(5) identical resulting dispatch orders (with the same tie-breaks
        # the schedulers actually use: (score, arrival, id))
        arrivals = np.array([j.arrival_time for j in queue], dtype=float)
        ids = np.array([j.job_id for j in queue], dtype=float)
        model_order = sorted(range(n), key=lambda i: (float(pred[i]), arrivals[i], ids[i]))
        size_order = sorted(range(n), key=lambda i: (float(sizes[i]), arrivals[i], ids[i]))
        arrival_order = sorted(range(n), key=lambda i: (arrivals[i], ids[i]))
        if model_order == size_order:
            self.same_as_size += 1
        if model_order == arrival_order:
            self.same_as_arrival += 1

    def summary(self):
        return {
            'setting': self.setting,
            'ranking_instants': self.instants,
            'mean_queue_len': float(np.mean(self.queue_lens)) if self.queue_lens else float('nan'),
            'features_total': len(self.feature_names),
            'features_varying_mean': float(
                (self.feat_varies / max(self.instants, 1)).sum()),
            'equal_size_diff_pred_violations': self.violations,
            'kendall_tau_vs_size_mean': float(np.mean(self.taus)) if self.taus else float('nan'),
            'kendall_tau_vs_size_min': float(np.min(self.taus)) if self.taus else float('nan'),
            'pct_order_identical_to_size': 100.0 * self.same_as_size / max(self.instants, 1),
            'pct_order_identical_to_arrival': 100.0 * self.same_as_arrival / max(self.instants, 1),
            'pct_size_table_monotone': 100.0 * self.monotone_instants / max(self.instants, 1),
            'mean_distinct_predictions': float(np.mean(self.n_distinct_pred)) if self.n_distinct_pred else float('nan'),
        }

    def size_curve(self):
        """Mean within-instant normalised score as a function of requested size:
        the learned size -> priority table, averaged over dispatch instants."""
        if not self.size_score:
            return pd.DataFrame(columns=['setting', 'size', 'norm_score', 'n'])
        d = pd.DataFrame(self.size_score, columns=['size', 'norm_score'])
        g = d.groupby('size', as_index=False).agg(
            norm_score=('norm_score', 'mean'), n=('norm_score', 'size'))
        g.insert(0, 'setting', self.setting)
        return g

    def feature_table(self):
        return pd.DataFrame({
            'setting': self.setting,
            'feature': self.feature_names,
            'pct_instants_varying_across_queue':
                100.0 * self.feat_varies / max(self.instants, 1),
        })


# ─────────────────────────────────────────────────────────────────────────────

def run_synthetic(n_runs):
    """Instrument the synthetic 14-scheduler benchmark's PROACTIVE policy."""
    import random
    import multi_scheduler_benchmark as msb

    col = Collector('synthetic (12 features)', list(msb.FEATURES))
    msb.RANK_OBSERVER = col
    nn_model = None
    for run in range(n_runs):
        random.seed(1000 + run)
        np.random.seed(1000 + run)
        jobs = msb.generate_jobs()
        msb.assign_estimates(jobs, random.Random(msb.EST_SEED_BASE + run))
        msb.run_once(jobs, 'proactive', nn_model)
    msb.RANK_OBSERVER = None
    return col


def run_trace(trace_key, n_windows, warmup_days, measure_days):
    """Instrument the trace-driven benchmark's PROACTIVE policy on real data."""
    import trace_driven_benchmark as tdb

    meta = tdb.TRACES[trace_key]
    capacity, df, _ = tdb.parse_swf_jobs(os.path.join(tdb.DATA_DIR, meta['swf']))
    t_min, t_max = int(df['submit'].min()), int(df['submit'].max())
    split_time = int(t_min + tdb.TRAIN_FRACTION * (t_max - t_min))
    model, _ = tdb.train_trace_model(trace_key, split_time, df, False, capacity)

    col = Collector(f'{meta["label"]} (8 features)', list(tdb.BASE_FEATURES))
    tdb.RANK_OBSERVER = col
    windows = tdb.carve_windows(df, split_time, n_windows, warmup_days, measure_days)
    for widx, w0, measure_start, w1, sel in windows:
        jobs, _ = tdb.make_jobs(sel, w0, measure_start, capacity)
        if not jobs:
            continue
        tdb.simulate(jobs, 'PROACTIVE', capacity, model,
                     {'trace': trace_key, 'window': widx, 'capacity': capacity,
                      'jobs_total': len(jobs), 'jobs_dropped_oversized': 0,
                      'offered_load': 0.0})
    tdb.RANK_OBSERVER = None
    return col


def make_figure(summary, feats, curves, path):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    s = summary.copy()
    idx = np.arange(len(s))
    axes[0].barh(idx - 0.2, s['pct_order_identical_to_size'], height=0.38,
                 color='#f472b6', label='identical to smallest-size-first')
    axes[0].barh(idx + 0.2, s['pct_order_identical_to_arrival'], height=0.38,
                 color='#94a3b8', label='identical to arrival order (FCFS)')
    axes[0].set_yticks(idx)
    axes[0].set_yticklabels(s['setting'])
    axes[0].set_xlabel('% of dispatch instants')
    axes[0].set_xlim(0, 100)
    axes[0].set_title('ML queue order vs ML-free orders')
    axes[0].legend(loc='lower right')
    axes[0].invert_yaxis()

    for setting, grp in feats.groupby('setting', sort=False):
        g = grp.sort_values('pct_instants_varying_across_queue', ascending=False)
        axes[1].plot(range(len(g)), g['pct_instants_varying_across_queue'],
                     marker='o', label=setting)
    axes[1].set_xlabel('feature (sorted by variability)')
    axes[1].set_ylabel('% of instants the feature differs across the queue')
    axes[1].set_title('Only the size-derived features vary between co-queued jobs')
    axes[1].set_ylim(-3, 103)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    for setting, grp in curves.groupby('setting', sort=False):
        g = grp.sort_values('size')
        axes[2].plot(g['size'], g['norm_score'], marker='o', ms=4, label=setting)
    axes[2].set_xscale('symlog', linthresh=8)
    axes[2].set_xlabel('requested size (processors / GPUs)')
    axes[2].set_ylabel('normalised predicted-wait score')
    axes[2].set_title('The recovered size $\\rightarrow$ priority table')
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle('Ranking degeneracy: a wait-time model used as a queue-ordering '
                 'score is a function of requested size alone')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=20, help='synthetic runs')
    ap.add_argument('--windows', type=int, default=6, help='trace windows')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.runs, args.windows = 5, 2

    collectors = [run_synthetic(args.runs)]
    for key in ('sdsc', 'lanl'):
        try:
            collectors.append(run_trace(key, args.windows, 3, 7))
        except FileNotFoundError as exc:
            print(f'skipping {key}: {exc}')

    summary = pd.DataFrame([c.summary() for c in collectors])
    feats = pd.concat([c.feature_table() for c in collectors], ignore_index=True)
    curves = pd.concat([c.size_curve() for c in collectors], ignore_index=True)

    sum_path = os.path.join(OUT_DIR, 'ranking_degeneracy.csv')
    feat_path = os.path.join(OUT_DIR, 'feature_variation.csv')
    curve_path = os.path.join(OUT_DIR, 'size_priority_table.csv')
    fig_path = os.path.join(OUT_DIR, 'ranking_degeneracy.png')
    summary.to_csv(sum_path, index=False)
    feats.to_csv(feat_path, index=False)
    curves.to_csv(curve_path, index=False)
    make_figure(summary, feats, curves, fig_path)

    print('\n' + '=' * 78)
    print('RANKING DEGENERACY OF THE WAIT-TIME MODEL USED AS A QUEUE-ORDERING SCORE')
    print('=' * 78)
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('\nPer-feature variability across co-queued jobs (% of instants):')
    for setting, grp in feats.groupby('setting', sort=False):
        print(f'\n  {setting}')
        for r in grp.sort_values('pct_instants_varying_across_queue',
                                 ascending=False).itertuples(index=False):
            print(f'    {r.feature:22s} {r.pct_instants_varying_across_queue:6.1f}%')

    viol = int(summary['equal_size_diff_pred_violations'].sum())
    print(f'\nEqual-size / different-prediction violations across all settings: {viol}')
    print('(zero => the score is exactly a function of requested size, as argued)')
    for p in (sum_path, feat_path, curve_path, fig_path):
        print('Saved:', p)


if __name__ == '__main__':
    main()

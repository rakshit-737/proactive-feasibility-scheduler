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
The remaining EIGHT features describe only the cluster, so they shift every
score by the same amount and cannot reorder anything. The learned model, used
this way, is a per-instant lookup table from requested size to priority -- not
a policy that reasons about individual jobs.

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
sys.path.insert(0, PROJECT_ROOT)
from vizstyle import figure, finish, save_both, bar_ends, PALETTE  # noqa: E402

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


def make_figure(summary, feats, curves, stem):
    """The evidence figure, written as a light/dark pair.

    Two panels, each with one job:
      (a) the MECHANISM -- which features can differ between two jobs waiting
          side by side. Features are named, never indexed; the eight that sit at
          exactly 0.0% are the point, so their zero is direct-labelled rather
          than rendered as an invisible bar.
      (b) the CONSEQUENCE for ordering -- how often the ML order is literally the
          smallest-first order, or literally arrival order.

    Colour follows the entity across the whole repository: orange is the ML-free
    control (smallest-first), grey is a classical baseline (arrival/FCFS).
    """
    for mode in ('light', 'dark'):
        p = PALETTE[mode]
        fig, axes = figure(mode, figsize=(14.5, 6.2), ncols=2,
                           gridspec_kw={'width_ratios': [1.15, 1],
                                        'wspace': 0.42})

        # ── (a) which features can distinguish two co-queued jobs? ──────────
        syn = feats[feats['setting'].str.startswith('synthetic')]
        g = syn.sort_values('pct_instants_varying_across_queue', ascending=True)
        vals = g['pct_instants_varying_across_queue'].to_numpy()
        names = g['feature'].tolist()
        colors = [p['series_1'] if v > 0 else p['muted'] for v in vals]
        y = np.arange(len(g))
        axes[0].barh(y, vals, height=0.66, color=colors)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(names, fontsize=9)
        axes[0].set_xlim(0, 100)
        axes[0].set_xlabel('% of dispatch instants the feature differs across the queue')
        axes[0].set_title('(a) Only size-derived features vary between co-queued jobs',
                          loc='left')
        bar_ends(axes[0], 'h')
        for yi, v in zip(y, vals):
            zero = v <= 0
            axes[0].text(v + 1.5, yi, '0.0%' if zero else f'{v:.1f}%',
                         va='center', fontsize=8.5,
                         color=p['muted'] if zero else p['ink_2'])
        n_zero = int((vals <= 0).sum())
        axes[0].text(0.99, 0.02,
                     f'{n_zero} of {len(vals)} features are identical for every\n'
                     'queued job — they cannot affect any ranking',
                     transform=axes[0].transAxes, ha='right', va='bottom',
                     fontsize=9, color=p['ink_2'])

        # ── (b) does the ML order equal an ML-free order? ────────────────────
        s = summary.copy()
        idx = np.arange(len(s))
        h = 0.34
        axes[1].barh(idx - h / 2 - 0.02, s['pct_order_identical_to_size'],
                     height=h, color=p['series_2'],
                     label='identical to smallest-first order')
        axes[1].barh(idx + h / 2 + 0.02, s['pct_order_identical_to_arrival'],
                     height=h, color=p['muted'],
                     label='identical to arrival order (FCFS)')
        axes[1].set_yticks(idx)
        axes[1].set_yticklabels([t.replace(' (', '\n(') for t in s['setting']],
                                fontsize=9)
        axes[1].set_xlim(0, 100)
        axes[1].set_xlabel('% of dispatch instants')
        axes[1].set_title('(b) The resulting queue order is an ML-free order',
                          loc='left')
        bar_ends(axes[1], 'h')
        axes[1].invert_yaxis()
        for i, (a, b) in enumerate(zip(s['pct_order_identical_to_size'],
                                       s['pct_order_identical_to_arrival'])):
            axes[1].text(a + 1.5, i - h / 2 - 0.02, f'{a:.0f}%', va='center',
                         fontsize=8.5, color=p['ink_2'])
            axes[1].text(b + 1.5, i + h / 2 + 0.02, f'{b:.0f}%', va='center',
                         fontsize=8.5, color=p['ink_2'])
        axes[1].legend(loc='lower right')

        total = int(summary['ranking_instants'].sum())
        viol = int(summary['equal_size_diff_pred_violations'].sum())
        finish(fig, mode,
               title='A learned wait-time score cannot tell two queued jobs apart '
                     'by anything but size',
               subtitle=f'Every queued job sees the same cluster, so only per-job '
                        f'features can differ — and each of those is a function of '
                        f'requested size.\nAcross {total:,} real dispatch instants, '
                        f'two equally-sized jobs received different scores '
                        f'{viol} times.',
               source='05_results/degeneracy/ — 04_scheduler/ranking_degeneracy.py')
        fig.subplots_adjust(top=0.78, bottom=0.11, left=0.135, right=0.985)
        save_both(fig, stem, mode)


def make_size_table_figure(curves, stem):
    """The recovered size -> priority table, as small multiples.

    One panel per trained model rather than three lines on shared axes: the
    settings have different size domains, and small multiples let each keep a
    single series colour instead of spending three hues on facets.
    """
    settings = list(dict.fromkeys(curves['setting']))
    for mode in ('light', 'dark'):
        p = PALETTE[mode]
        fig, axes = figure(mode, figsize=(14.5, 3.9), ncols=len(settings),
                           gridspec_kw={'wspace': 0.22})
        axes = np.atleast_1d(axes)
        for ax, setting in zip(axes, settings):
            g = curves[curves['setting'] == setting].sort_values('size')
            ax.plot(g['size'], g['norm_score'], marker='o', ms=4.5,
                    color=p['series_1'], linewidth=1.8)
            ax.set_xscale('log')
            ax.set_title(setting, loc='left', fontsize=10)
            ax.set_xlabel('requested size (processors)')
            ax.set_ylim(-0.05, 1.08)
            ax.set_axisbelow(True)
        axes[0].set_ylabel('normalised predicted-wait score')
        finish(fig, mode,
               title='What the model actually learned: a lookup table from '
                     'requested size to priority',
               subtitle='Score against requested size, averaged over dispatch '
                        'instants. Read the SHAPE, not the wiggle: SDSC spans 51 '
                        'distinct sizes with few\nobservations each, so its mean '
                        'curve is noisy — per-instant the table is monotone 52% of '
                        'the time there vs 62% on LANL and 67% synthetic.',
               source='05_results/degeneracy/size_priority_table.csv — '
                      'monotone fractions in ranking_degeneracy.csv')
        fig.subplots_adjust(top=0.70, bottom=0.15, left=0.055, right=0.99)
        save_both(fig, stem, mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=20, help='synthetic runs')
    ap.add_argument('--windows', type=int, default=6, help='trace windows')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--figures-only', action='store_true',
                    help='re-render the figures from the saved CSVs without '
                         're-running any simulation (for iterating on styling)')
    args = ap.parse_args()
    if args.quick:
        args.runs, args.windows = 5, 2

    if args.figures_only:
        summary = pd.read_csv(os.path.join(OUT_DIR, 'ranking_degeneracy.csv'))
        feats = pd.read_csv(os.path.join(OUT_DIR, 'feature_variation.csv'))
        curves = pd.read_csv(os.path.join(OUT_DIR, 'size_priority_table.csv'))
        make_figure(summary, feats, curves, os.path.join(OUT_DIR, 'ranking_degeneracy'))
        make_size_table_figure(curves, os.path.join(OUT_DIR, 'size_priority_table'))
        print('Re-rendered figures from existing CSVs in', OUT_DIR)
        return

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
    fig_stem = os.path.join(OUT_DIR, 'ranking_degeneracy')
    table_stem = os.path.join(OUT_DIR, 'size_priority_table')
    fig_path = fig_stem + '.png'
    summary.to_csv(sum_path, index=False)
    feats.to_csv(feat_path, index=False)
    curves.to_csv(curve_path, index=False)
    make_figure(summary, feats, curves, fig_stem)
    make_size_table_figure(curves, table_stem)

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
    for p in (sum_path, feat_path, curve_path, fig_path,
              table_stem + '.png'):
        print('Saved:', p)


if __name__ == '__main__':
    main()

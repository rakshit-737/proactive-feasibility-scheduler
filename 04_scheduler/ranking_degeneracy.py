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
The remaining SEVEN features (of 12; four of 8 in the trace vector) describe
only the cluster, so they shift every score by the same amount and cannot
reorder anything. The learned model, used
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
It instruments dispatch instants (via the RANK_OBSERVER hook in both
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
     queued job the same score);
  6. the fraction where EVERY queued job receives the same score. (5) is an
     upper bound on (6) BY CONSTRUCTION, not by assumption: both read the same
     quantised score (see TIE_DECIMALS below), so one score covering the whole
     queue collapses the sort key (score, arrival, id) to (arrival, id) and the
     induced order IS arrival order. The converse fails -- distinct scores can
     also happen to rank the queue in arrival order -- so the two columns are
     kept apart and the strict measure is never read off the weaker one.

Instants are NOT all "real": the synthetic setting is a simulator, and the two
trace settings replay recorded Parallel Workloads Archive schedules. The figure
captions split the total accordingly rather than calling all of it real.

Outputs: 05_results/degeneracy/{ranking_degeneracy.csv,
         ranking_degeneracy_totals.csv, feature_variation.csv,
         size_priority_table.csv, ranking_degeneracy.png,
         size_priority_table.png}

Usage:
  python ranking_degeneracy.py                  # synthetic + both traces
  python ranking_degeneracy.py --quick          # fewer runs/windows
  python ranking_degeneracy.py --allow-partial  # tolerate a missing trace
  python ranking_degeneracy.py --figures-only   # re-render from saved CSVs

Anything but the bare invocation produces a PARTIAL run: the totals CSV says so
in its `partial` column, and both figures say so in their caption.
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

TRACE_KEYS = ('sdsc', 'lanl')
# Synthetic + one setting per trace. The published instant total is the sum over
# all of them, so a run that yields fewer settings has under-reported it and must
# say so rather than writing a smaller number into the same CSV.
EXPECTED_SETTINGS = 1 + len(TRACE_KEYS)

# THE PUBLISHED PROTOCOL, as module constants rather than as literals hidden in
# argparse. 20 synthetic runs and 20 trace windows are what produced the instant
# total the manuscript quotes (and 20 windows is the trace-driven benchmark's own
# N_WINDOWS), so any other value produces a DIFFERENT total under the same column
# name. `partial` below is derived from these, which is the only way a run
# invoked as `--windows 2` can know that it is not the published one.
PUBLISHED_RUNS = 20
PUBLISHED_WINDOWS = 20

# Sentinel written into the totals CSV when no trace was missing. NOT the empty
# string: pd.read_csv turns an empty field into NaN, so a consumer comparing to
# '' silently never matches and every read of the column has to special-case
# float('nan'). 'none' round-trips as itself.
NO_MISSING_TRACES = 'none'

# ONE definition of "these two scores are the same score".
#
# Predictions are quantised to TIE_DECIMALS decimal places, and EVERY comparison
# of predictions in this file reads that quantised value: the equal-size /
# different-prediction violation check, the all-scores-tied counter, the
# distinct-prediction count, and -- crucially -- the score that induces the
# model's dispatch order. Quantising in one place is what makes the reported
# bound true by construction:
#
#     one quantised score across the whole queue
#         => the sort key (score, arrival, id) reduces to (arrival, id)
#         => the induced order IS arrival order,
#
# so all_tied <= same_as_arrival at every instant and therefore
# pct_all_scores_tied <= pct_order_identical_to_arrival in the summary.
#
# Counting ties on a rounded value while ORDERING on the raw float does not give
# that, and used not to: two scores 3e-10 apart quantise to one value (counted
# as tied) yet still induce an order of their own, so pct_all_scores_tied could
# in principle exceed the column documented as its upper bound.
#
# Quantisation is never LOOSER than the |spread| <= 1e-9 test it replaces. The
# bins are 1e-9 wide, so two values that land in one bin differ by less than
# 1e-9 and the spread test would have called them equal too. The guarantee is
# made self-consistent, never widened.
TIE_DECIMALS = 9
TIE_ATOL = 10.0 ** -TIE_DECIMALS


def tie_keys(pred):
    """The quantised scores every prediction comparison in this module uses."""
    return np.round(np.asarray(pred, dtype=float), TIE_DECIMALS)


class RunScope:
    """What one invocation actually covered, and whether that is the published
    protocol.

    Every consumer of "is this the published number?" reads this object: the
    totals CSV, both figure captions and the closing console block. Keeping the
    predicate in one place is the point -- the previous version recomputed it as
    `bool(missing) or bool(args.quick)`, which reported partial=False beside a
    total produced by `--windows 2`.
    """

    def __init__(self, runs, windows, missing=(), quick=False):
        self.runs = int(runs)
        self.windows = int(windows)
        self.missing = list(missing)
        self.quick = bool(quick)

    @property
    def partial(self):
        """True unless this run reproduces the published protocol exactly."""
        return bool(self.missing) or self.quick \
            or self.runs != PUBLISHED_RUNS or self.windows != PUBLISHED_WINDOWS

    def reasons(self):
        """Every reason this run is not the published one, in plain words."""
        out = []
        if self.missing:
            out.append(f"missing traces: {', '.join(self.missing)}")
        if self.runs != PUBLISHED_RUNS:
            out.append(f'--runs {self.runs} (published: {PUBLISHED_RUNS})')
        if self.windows != PUBLISHED_WINDOWS:
            out.append(f'--windows {self.windows} (published: {PUBLISHED_WINDOWS})')
        if self.quick and not out:
            out.append('--quick')
        return out

    def missing_field(self):
        """The totals CSV's missing_traces cell, with an explicit sentinel."""
        return ';'.join(self.missing) if self.missing else NO_MISSING_TRACES

    def caption_prefix(self, summary):
        """Leading caption line naming the settings present, or '' when full.

        A partial run's figure is a diagnostic, not a publication artefact, and
        the caption is the only part of a PNG a reader ever sees -- so the
        smaller total must not be presented in the same words as the published
        one.
        """
        if not self.partial:
            return ''
        present = ' + '.join(str(s).split(' (')[0] for s in summary['setting'])
        why = '; '.join(self.reasons()) or 'reduced protocol'
        return (f'PARTIAL RUN: {present} only — NOT the published totals '
                f'({why}).\n')


class Collector:
    """Accumulates statistics over observed ranking decisions."""

    def __init__(self, setting, feature_names):
        self.setting = setting
        self.feature_names = feature_names
        self.instants = 0
        self.same_as_size = 0
        self.same_as_arrival = 0
        self.all_tied = 0            # every queued job got the same score
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

        # The single quantised view of this instant's scores. Everything below
        # that compares predictions reads `keys`, never `pred` -- see
        # TIE_DECIMALS for why the counters and the induced order must agree.
        keys = tie_keys(pred)

        # (1) which features actually differ across co-queued jobs?
        spread = x.max(axis=0) - x.min(axis=0)
        self.feat_varies += (np.abs(spread) > 1e-12).astype(np.int64)

        # (2) functional dependence on size: equal size => equal prediction
        sizes = x[:, SIZE_COL]
        for s in np.unique(sizes):
            k = keys[sizes == s]
            if len(k) > 1 and len(np.unique(k)) > 1:
                self.violations += 1

        n_distinct = len(np.unique(keys))
        self.n_distinct_pred.append(n_distinct)

        # (5b) the STRICT all-scores-tied case, measured rather than inferred.
        # Because `keys` is also the sort key below, all-tied here IMPLIES the
        # induced order equals arrival order -- but NOT conversely: distinct
        # scores can rank the queue in arrival order too. So same_as_arrival is
        # a genuine upper bound on all_tied, and both counters are kept so the
        # claim "all scores tie" is never read off the weaker column.
        if n_distinct == 1:
            self.all_tied += 1

        # (2b) the score IS a lookup table over size -- recover it. Is that
        # table monotone increasing (i.e. exactly smallest-first), or has the
        # model learned a non-trivial size preference?
        uniq = np.unique(sizes)
        table = np.array([keys[sizes == s].mean() for s in uniq])
        if len(uniq) > 1:
            if np.all(np.diff(table) >= -TIE_ATOL):
                self.monotone_instants += 1
            lo, hi = table.min(), table.max()
            if hi > lo:
                for s, v in zip(uniq, (table - lo) / (hi - lo)):
                    self.size_score.append((float(s), float(v)))

        # (3) rank agreement with smallest-size-first
        if len(np.unique(sizes)) > 1 and n_distinct > 1:
            tau = stats.kendalltau(keys, sizes).statistic
            if not np.isnan(tau):
                self.taus.append(float(tau))

        # (4)/(5) identical resulting dispatch orders (with the same tie-breaks
        # the schedulers actually use: (score, arrival, id))
        #
        # ONE deliberate difference from the schedulers, stated because it has a
        # real semantic consequence: they sort on the RAW float
        # (multi_scheduler_benchmark.rank_queue uses
        # `key=lambda i: (float(pred[i]), arrival, job_id)`), whereas
        # `model_order` below sorts on the QUANTISED key. Within a single 1e-9
        # bin the two can therefore disagree -- the scheduler may separate two
        # jobs this diagnostic reports as tied, so model_order is not always
        # byte-for-byte the dispatch order the benchmark produced.
        #
        # That is the right trade HERE, and only here. This module exists to
        # measure how often the learned score CANNOT tell two queued jobs apart;
        # an ordering conjured out of a sub-1e-9 float difference is precisely
        # the artefact being measured, not a decision worth reproducing. Reading
        # the counters and the induced order off one quantised value is also
        # what makes pct_all_scores_tied <= pct_order_identical_to_arrival true
        # by construction rather than by assumption (see TIE_DECIMALS). The
        # schedulers' own behaviour is untouched; only this diagnostic's notion
        # of "the same score" is.
        arrivals = np.array([j.arrival_time for j in queue], dtype=float)
        ids = np.array([j.job_id for j in queue], dtype=float)
        model_order = sorted(range(n), key=lambda i: (float(keys[i]), arrivals[i], ids[i]))
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
            'pct_all_scores_tied': 100.0 * self.all_tied / max(self.instants, 1),
            'pct_size_table_monotone': 100.0 * self.monotone_instants / max(self.instants, 1),
            'mean_distinct_predictions': (float(np.mean(self.n_distinct_pred))
                                          if self.n_distinct_pred
                                          else float('nan')),
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
    # No trace_key argument: the trace enters training only as `df`, so there is
    # no name to get wrong silently.
    model, _ = tdb.train_trace_model(split_time, df, False, capacity)

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


def instant_provenance(summary):
    """(total, synthetic, trace) instant counts.

    The caption used to call the whole sum "real dispatch instants". Roughly
    3,646 of them come from the synthetic simulator, which is not a real
    machine, so the total is split here instead of being renamed.
    """
    total = int(summary['ranking_instants'].sum())
    is_synthetic = summary['setting'].astype(str).str.startswith('synthetic')
    synthetic = int(summary.loc[is_synthetic, 'ranking_instants'].sum())
    return total, synthetic, total - synthetic


def make_figure(summary, feats, curves, stem, scope):
    """The evidence figure, written as a light/dark pair.

    Two panels, each with one job:
      (a) the MECHANISM -- which features can differ between two jobs waiting
          side by side. Features are named, never indexed; the seven that sit at
          exactly 0.0% are the point, so their zero is direct-labelled rather
          than rendered as an invisible bar.
      (b) the CONSEQUENCE for ordering -- how often the ML order is literally the
          smallest-first order, or literally arrival order.

    Colour follows the entity across the whole repository: orange is the ML-free
    control (smallest-first), grey is a classical baseline (arrival/FCFS).

    `scope` is not optional. The caption quotes an instant total, and a total
    from a reduced run reads exactly like the published one unless the figure
    itself says otherwise -- which is the defect the hard-fail on a missing
    trace was added to close and which --allow-partial reopened.
    """
    prefix = scope.caption_prefix(summary)
    total, synthetic, from_traces = instant_provenance(summary)
    viol = int(summary['equal_size_diff_pred_violations'].sum())
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

        finish(fig, mode,
               title='A learned wait-time score cannot tell two queued jobs apart '
                     'by anything but size',
               subtitle=f'{prefix}Every queued job sees the same cluster, so only '
                        f'per-job features can differ — and each of those is a '
                        f'function of requested size.\nAcross {total:,} dispatch '
                        f'instants ({synthetic:,} synthetic, {from_traces:,} '
                        f'replayed from the real SWF traces), two equally-sized '
                        f'jobs received different scores {viol} times.',
               source='05_results/degeneracy/ — 04_scheduler/ranking_degeneracy.py')
        fig.subplots_adjust(top=0.74 if prefix else 0.78,
                            bottom=0.11, left=0.135, right=0.985)
        save_both(fig, stem, mode)


def make_size_table_figure(curves, summary, stem, scope):
    """The recovered size -> priority table, as small multiples.

    One panel per trained model rather than three lines on shared axes: the
    settings have different size domains, and small multiples let each keep a
    single series colour instead of spending three hues on facets.

    Carries the same `scope` banner as make_figure: this figure names the
    settings it drew, so a run short of a setting must say that the panel is
    absent by design rather than leave a reader to notice a missing column.
    """
    prefix = scope.caption_prefix(summary)
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
        # Monotone fractions come from the run's own summary so the caption can
        # never go stale against the CSVs it cites.
        mono = {r['setting']: r['pct_size_table_monotone']
                for _, r in summary.iterrows()}
        mono_txt = ', '.join(
            f"{s.split(' (')[0]} {mono[s]:.0f}%" for s in settings if s in mono)
        finish(fig, mode,
               title='What the model actually learned: a lookup table from '
                     'requested size to priority',
               subtitle=f'{prefix}Score against requested size, averaged over '
                        'dispatch instants. Read the SHAPE, not the wiggle: SDSC '
                        'spans many distinct sizes with few\nobservations each, so '
                        'its mean curve is noisy — per-instant monotone fractions: '
                        f'{mono_txt}.',
               source='05_results/degeneracy/size_priority_table.csv — '
                      'monotone fractions in ranking_degeneracy.csv')
        fig.subplots_adjust(top=0.64 if prefix else 0.70,
                            bottom=0.15, left=0.055, right=0.99)
        save_both(fig, stem, mode)


# ─────────────────────────────────────────────────────────────────────────────
# The totals file: what a re-render is allowed to trust
# ─────────────────────────────────────────────────────────────────────────────

TOTALS_COLUMNS = ('settings_expected', 'settings_present', 'missing_traces',
                  'runs', 'published_runs', 'windows', 'published_windows',
                  'partial', 'total_instants', 'total_violations')


def _as_bool(value):
    """CSV round-trip safe truthiness: 'False' is a string, and bool('False')
    is True, which is exactly how a partial run would sneak past a guard."""
    if isinstance(value, str):
        return value.strip().lower() in ('true', '1', 'yes')
    return bool(value)


def totals_row(summary, scope, n_settings):
    """The single-row totals table describing this run."""
    return pd.DataFrame([{
        'settings_expected': EXPECTED_SETTINGS,
        'settings_present': n_settings,
        'missing_traces': scope.missing_field(),
        # What actually produced the numbers beside them, so the file does not
        # depend on anyone remembering the command line.
        'runs': scope.runs,
        'published_runs': PUBLISHED_RUNS,
        'windows': scope.windows,
        'published_windows': PUBLISHED_WINDOWS,
        'partial': scope.partial,
        'total_instants': int(summary['ranking_instants'].sum()),
        'total_violations': int(summary['equal_size_diff_pred_violations'].sum()),
    }], columns=list(TOTALS_COLUMNS))


def scope_from_totals(summary, totals_path):
    """Validate the saved totals against the saved per-setting table.

    --figures-only re-renders PUBLICATION figures whose captions quote a total,
    and it does not recompute anything -- so the two CSVs it reads have to be
    checked against each other here or nowhere. Every failure below is a
    SystemExit rather than a warning, because the output of proceeding is a
    finished PNG that looks exactly like the real one.
    """
    if not os.path.exists(totals_path):
        raise SystemExit(
            f'ERROR: {totals_path} does not exist.\n'
            f'--figures-only re-renders the published figures from the saved '
            f'CSVs and recomputes nothing, so the totals file is the ONLY '
            f'record of whether those CSVs came from a complete run. Refusing '
            f'to render a caption that quotes a total nothing vouches for. '
            f'Re-run ranking_degeneracy.py without --figures-only.')
    totals = pd.read_csv(totals_path)
    if len(totals) != 1:
        raise SystemExit(
            f'ERROR: {totals_path} holds {len(totals)} rows; exactly 1 is '
            f'expected. Re-run ranking_degeneracy.py without --figures-only.')
    absent = [c for c in TOTALS_COLUMNS if c not in totals.columns]
    if absent:
        raise SystemExit(
            f'ERROR: {totals_path} is missing {absent}: it predates the columns '
            f'that record what produced it, so it cannot be checked. Re-run '
            f'ranking_degeneracy.py without --figures-only.')

    row = totals.iloc[0]
    if _as_bool(row['partial']):
        raise SystemExit(
            f'ERROR: {totals_path} says partial=True '
            f'(missing_traces={row["missing_traces"]}, runs={row["runs"]}, '
            f'windows={row["windows"]}).\n'
            f'Re-rendering the publication figures from a partial run is the '
            f'exact mistake --allow-partial is fenced off to prevent: the '
            f'caption would present a reduced total in the published figure\'s '
            f'own words. Re-run the full protocol first.')

    sum_instants = int(summary['ranking_instants'].sum())
    sum_violations = int(summary['equal_size_diff_pred_violations'].sum())
    if (sum_instants != int(row['total_instants'])
            or sum_violations != int(row['total_violations'])):
        raise SystemExit(
            f'ERROR: {totals_path} disagrees with ranking_degeneracy.csv.\n'
            f'  totals says   instants={int(row["total_instants"])}, '
            f'violations={int(row["total_violations"])}\n'
            f'  per-setting sums instants={sum_instants}, '
            f'violations={sum_violations}\n'
            f'One of the two files is stale -- most likely the totals row, '
            f'which an earlier --figures-only run would have left untouched '
            f'while re-rendering figures from a newer table. Re-run '
            f'ranking_degeneracy.py without --figures-only.')

    missing_cell = str(row['missing_traces']).strip()
    missing = ([] if missing_cell in (NO_MISSING_TRACES, '', 'nan')
               else missing_cell.split(';'))
    return RunScope(runs=int(row['runs']), windows=int(row['windows']),
                    missing=missing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=PUBLISHED_RUNS,
                    help=f'synthetic runs (published protocol: {PUBLISHED_RUNS})')
    # PUBLISHED_WINDOWS matches the trace-driven benchmark's published protocol,
    # so the default pipeline invocation reproduces the manuscript's instant
    # counts and ANY other value marks the run partial.
    ap.add_argument('--windows', type=int, default=PUBLISHED_WINDOWS,
                    help=f'trace windows (published protocol: {PUBLISHED_WINDOWS})')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--allow-partial', action='store_true',
                    help='continue when a trace cannot be loaded. Without this '
                         'flag a missing trace is a hard error, so the published '
                         'instant total can never be silently under-reported.')
    ap.add_argument('--figures-only', action='store_true',
                    help='re-render the figures from the saved CSVs without '
                         're-running any simulation (for iterating on styling). '
                         'Refuses if the saved totals row is absent, partial, or '
                         'disagrees with ranking_degeneracy.csv.')
    args = ap.parse_args()
    if args.quick:
        args.runs, args.windows = 5, 2

    sum_path = os.path.join(OUT_DIR, 'ranking_degeneracy.csv')
    totals_path = os.path.join(OUT_DIR, 'ranking_degeneracy_totals.csv')
    feat_path = os.path.join(OUT_DIR, 'feature_variation.csv')
    curve_path = os.path.join(OUT_DIR, 'size_priority_table.csv')
    fig_stem = os.path.join(OUT_DIR, 'ranking_degeneracy')
    table_stem = os.path.join(OUT_DIR, 'size_priority_table')
    fig_path = fig_stem + '.png'

    if args.figures_only:
        summary = pd.read_csv(sum_path)
        feats = pd.read_csv(feat_path)
        curves = pd.read_csv(curve_path)
        # Raises SystemExit unless the totals row exists, says the run was
        # complete, and adds up to the table being re-rendered.
        scope = scope_from_totals(summary, totals_path)
        make_figure(summary, feats, curves, fig_stem, scope)
        make_size_table_figure(curves, summary, table_stem, scope)
        print(f'Totals row agrees with ranking_degeneracy.csv '
              f'({int(summary["ranking_instants"].sum()):,} instants, partial=False)')
        print('Re-rendered figures from existing CSVs in', OUT_DIR)
        return

    collectors = [run_synthetic(args.runs)]
    missing = []
    for key in TRACE_KEYS:
        try:
            collectors.append(run_trace(key, args.windows, 3, 7))
        except FileNotFoundError as exc:
            if not args.allow_partial:
                raise SystemExit(
                    f'ERROR: trace {key!r} could not be run: {exc}\n'
                    f'The published ranking-instant total is the sum over '
                    f'{EXPECTED_SETTINGS} settings (synthetic + '
                    f'{len(TRACE_KEYS)} traces); dropping one silently reports a '
                    f'smaller total under the same column name. Restore the SWF '
                    f'traces in 02_data/*.swf.gz (they are tracked in git), or '
                    f're-run with --allow-partial. A partial table must NOT be '
                    f'committed.')
            missing.append(key)
            print(f'WARNING: trace {key!r} unavailable, continuing because '
                  f'--allow-partial was passed: {exc}')

    summary = pd.DataFrame([c.summary() for c in collectors])
    feats = pd.concat([c.feature_table() for c in collectors], ignore_index=True)
    curves = pd.concat([c.size_curve() for c in collectors], ignore_index=True)

    # What this run actually covered, persisted next to the per-setting table.
    # It is a SEPARATE file: make_figure draws one bar pair per row of
    # ranking_degeneracy.csv, so a TOTAL row there would render as a fourth,
    # non-existent setting.
    #
    # A missing trace is NOT the only way to under-report the total: --quick and
    # user-supplied --runs/--windows change it too, and their defaults ARE the
    # published protocol, so the scope object derives `partial` from all of them
    # instead of from the missing list alone.
    scope = RunScope(runs=args.runs, windows=args.windows, missing=missing,
                     quick=args.quick)
    totals = totals_row(summary, scope, len(collectors))
    total_instants = int(totals.iloc[0]['total_instants'])
    total_violations = int(totals.iloc[0]['total_violations'])

    summary.to_csv(sum_path, index=False)
    totals.to_csv(totals_path, index=False)
    feats.to_csv(feat_path, index=False)
    curves.to_csv(curve_path, index=False)
    make_figure(summary, feats, curves, fig_stem, scope)
    make_size_table_figure(curves, summary, table_stem, scope)

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

    print(f'\nEqual-size / different-prediction violations across all settings: '
          f'{total_violations}')
    print('(zero => the score is exactly a function of requested size, as argued)')
    print(f'Total ranking instants across {len(collectors)}/{EXPECTED_SETTINGS} '
          f'settings: {total_instants:,} (violations: {total_violations}, '
          f'partial={scope.partial})')
    if scope.partial:
        print('PARTIAL RUN — this total is NOT the published number; do not commit '
              'these artefacts.')
        for why in scope.reasons():
            print(f'  reason: {why}')
    for p in (sum_path, totals_path, feat_path, curve_path, fig_path,
              table_stem + '.png'):
        print('Saved:', p)


if __name__ == '__main__':
    main()

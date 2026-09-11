"""Robustness of the ranking-degeneracy claim: four adversarial attacks.

WHAT IS BEING ATTACKED
----------------------
04_scheduler/ranking_degeneracy.py publishes a negative result. At a fixed
dispatch instant every queued job observes the SAME cluster, so in the 8-feature
trace vector every per-job input is a deterministic function of the job's
requested size given that state. The learned score is therefore g_state(size),
the induced ranking is a permutation of the size order, and the ML pipeline is
statistically equivalent to "sort by requested size". Measured over 45,432
dispatch instants with ZERO counterexamples.

This module does NOT touch that result. It adds a PARALLEL experiment that tries
to break the claim four ways, in the order a reviewer would propose them, and
reports every attempt whether it succeeded or not. The published instrumentation
(ranking_degeneracy.Collector) is IMPORTED, never re-implemented, so an attack
and the published measurement are read off the same counters.

THE FOUR ATTACKS
----------------
A1  A DIFFERENT LEARNING OBJECTIVE (pairwise / listwise learning-to-rank).
    THE ARGUMENT FIRST. The degeneracy is a property of the INPUTS, not of the
    loss. If every per-job feature is a function of size given the state, then
    every measurable function of those features is too -- including the scoring
    function a rank:pairwise or rank:ndcg objective converges to. A training
    objective changes WHICH function of size is learned; it cannot manufacture
    an input that distinguishes two equally-sized co-queued jobs.
    THE MEASUREMENT: train XGBRanker with rank:pairwise and with rank:ndcg over
    the identical 8 features and run the identical instrumentation.

A2  A MONOTONE TRANSFORM of the score. Fails BY CONSTRUCTION: a strictly
    monotone map is order-preserving, so it cannot reorder a queue and cannot
    split a tie. Demonstrated anyway on the PUBLISHED model's own predictions
    (no retraining) under two transforms, because the point is to have tried and
    because a non-zero result here would mean the instrumentation is broken.

A3  A HISTORY WINDOW. A model that sees the cluster state k ticks ago as well as
    now. Genuinely worth measuring -- but lagged cluster state is still SHARED
    by every co-queued job at the dispatch instant, so the argument predicts the
    lagged columns land in exactly the same "identical for every queued job"
    bucket as the contemporaneous ones.

A4  SCORE AT ENQUEUE TIME, CACHED. THIS ONE SHOULD WORK. If a job is scored when
    it ENTERS the queue and the score is then cached, two jobs in one queue were
    scored against DIFFERENT cluster states. The shared-state premise fails and
    the ranking need not be a size permutation.
    THEN THE SECOND QUESTION, which is the one that matters: does the
    non-degenerate variant actually SCHEDULE better? A cached enqueue-time score
    is a DIFFERENT POLICY, not a different implementation of the same one, and
    staleness is a cost as well as a source of variation. Its mean wait is
    compared against PROACTIVE and SJF_USEREST on the same windows, and written
    to a separate utility CSV.

WHAT COUNTS AS "BROKEN"
-----------------------
The published claim is falsified by a COUNTEREXAMPLE: two equally-sized jobs,
co-queued at one instant, receiving different scores. That is the
`equal_size_diff_pred_violations` counter, and it is the verdict criterion here.
The supporting columns (mean Kendall tau against size, the all-tied fraction,
the mean number of distinct score levels, the fraction of instants whose induced
order IS the smallest-first order) are reported alongside because an attack can
weaken the claim without falsifying it.

SEED
----
ROBUST_SEED = 90210, used only for the models trained in this file. The
protected seed families in this repository are 42+i, 1000+run, 20000+run,
800+run, 4000+id+nodes, 5000+i, 7000+run, SEED=42 (traces) and POWER_SEED=31337.
90210 falls in none of them, so no existing experiment can collide with it.

PROTOCOL
--------
The published trace protocol is followed exactly and is imported rather than
restated: trace_driven_benchmark.N_WINDOWS windows, WARMUP_DAYS warm-up,
MEASURE_DAYS measured, TRAIN_FRACTION chronological split. --windows reduces the
count for a smoke run; when it does, every row of the artefact says so in its
`windows` column and `partial` is True, rather than a smaller number being
written under the published column name.

Outputs: 05_results/degeneracy/robustness_attacks.csv
         05_results/degeneracy/robustness_attack_utility.csv

Usage:
  python 04_scheduler/robustness_attacks.py
  python 04_scheduler/robustness_attacks.py --windows 3     # smoke run
  python 04_scheduler/robustness_attacks.py --traces sdsc
"""

import argparse
import os
import sys
from collections import deque

import numpy as np
import pandas as pd
from xgboost import XGBRanker, XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import trace_driven_benchmark as tdb                      # noqa: E402
from ranking_degeneracy import Collector, tie_keys        # noqa: E402

OUT_DIR = os.path.join(PROJECT_ROOT, '05_results', 'degeneracy')
os.makedirs(OUT_DIR, exist_ok=True)

ATTACKS_CSV = os.path.join(OUT_DIR, 'robustness_attacks.csv')
UTILITY_CSV = os.path.join(OUT_DIR, 'robustness_attack_utility.csv')

# New experiment, new seed -- see the SEED note in the module docstring.
ROBUST_SEED = 90210

# A3: how far back the history window reaches. "Small k" as the reviewer would
# propose it; the lagged block is the four CLUSTER-only columns, since the
# per-job columns lagged would be a different job's features and meaningless.
LAG_K = 5
CLUSTER_COLS = ['total_free', 'queue_length', 'running_jobs', 'free_frac']
LAG_FEATURES = tdb.BASE_FEATURES + [f'{c}_lag{LAG_K}' for c in CLUSTER_COLS]

# A1: learning-to-rank needs query groups and graded relevance, neither of which
# the wait-prediction dataset carries. Rows are grouped into hour-long buckets of
# submit time (the coarsest grouping that still puts genuinely co-queued jobs in
# one group), and log1p(wait) is cut into RELEVANCE_GRADES global quantile bins.
# SHORTEST wait gets the HIGHEST grade, so "most relevant" means "dispatch first"
# and the ranker's preferred order is directly comparable with the published
# pointwise model's ascending-predicted-wait order.
RANK_GROUP_SECONDS = 3600
RELEVANCE_GRADES = 5

VERDICT_BROKEN = 'BROKEN'
VERDICT_INTACT = 'NOT BROKEN'


# ─────────────────────────────────────────────────────────────────────────────
# Score wrappers. tdb.rank_queue calls model.predict(x) and sorts ASCENDING, so
# any variant that can be expressed as a function of the contemporaneous feature
# matrix is run through the PUBLISHED simulation loop by handing it one of these
# instead of the published regressor. Nothing in trace_driven_benchmark.py is
# modified to make that work -- the hook is the one it already exposes.
# ─────────────────────────────────────────────────────────────────────────────

class NegatedModel:
    """A ranker read as a dispatch order.

    XGBRanker scores HIGHER for MORE relevant, and this file grades shortest
    wait as most relevant, so the dispatch order is DESCENDING score. Negating
    turns it into the ascending-score convention tdb.rank_queue sorts on. The
    negation is strictly monotone, so it moves no counter that A2 is about: it
    changes the SIGN of Kendall tau against size, nothing else.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, x):
        return -np.asarray(self.model.predict(x), dtype=float)


class TransformedModel:
    """The published model's score pushed through a monotone map (A2)."""

    def __init__(self, model, fn):
        self.model = model
        self.fn = fn

    def predict(self, x):
        return np.asarray(self.fn(np.asarray(self.model.predict(x),
                                             dtype=float)), dtype=float)


class LaggedModel:
    """A model over [current 8 features | cluster state LAG_K instants ago] (A3).

    The lag buffer is per-simulation state, so one instance serves exactly one
    window and is rebuilt for the next. Until LAG_K instants have been seen the
    lag block repeats the oldest state available, which is the only choice that
    does not invent history.
    """

    def __init__(self, model):
        self.model = model
        self.history = deque(maxlen=LAG_K + 1)

    def reset(self):
        self.history.clear()

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        # Columns 1,2,3,7 of BASE_FEATURES are the cluster-only block and are
        # identical down every row by construction; row 0 is the cluster state.
        idx = [tdb.BASE_FEATURES.index(c) for c in CLUSTER_COLS]
        now = x[0, idx]
        self.history.append(now)
        lag = self.history[0]
        block = np.repeat(lag[None, :], x.shape[0], axis=0)
        return np.asarray(self.model.predict(np.hstack([x, block])), dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Training the attack models
# ─────────────────────────────────────────────────────────────────────────────

def replay_training_frame(df, capacity, split_time):
    """The published training frame: replayed features before the split.

    Identical construction to trace_driven_benchmark.train_trace_model -- same
    replay function, same chronological cut -- but returned rather than fitted,
    so the three attack models are trained on exactly the rows the published
    model was trained on.
    """
    jobs = [{'job_id': int(r.job_id), 'submit': int(r.submit),
             'wait': int(r.recorded_wait), 'run': int(r.runtime),
             'procs': int(r.procs)} for r in df.itertuples(index=False)]
    frame, _ = tdb.replay_trace_features(jobs, capacity)
    frame = frame[frame['submit_time'] < split_time].copy()
    return frame.sort_values('submit_time').reset_index(drop=True)


def train_rank_model(frame, objective):
    """A1: learning-to-rank over the SAME 8 features (rank:pairwise/rank:ndcg)."""
    x = frame[tdb.BASE_FEATURES].to_numpy(dtype=np.float64)
    wait = np.log1p(frame['wait_time'].clip(lower=0).to_numpy(dtype=np.float64))
    qid = (frame['submit_time'].to_numpy(dtype=np.int64) // RANK_GROUP_SECONDS)

    # Shortest wait => highest grade => dispatched first (see RELEVANCE_GRADES).
    ranks = pd.Series(wait).rank(method='average', pct=True).to_numpy()
    grade = (RELEVANCE_GRADES - 1) - np.clip(
        (ranks * RELEVANCE_GRADES).astype(int), 0, RELEVANCE_GRADES - 1)

    order = np.argsort(qid, kind='stable')
    model = XGBRanker(objective=objective, n_estimators=300, max_depth=6,
                      learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
                      random_state=ROBUST_SEED, n_jobs=4, verbosity=0)
    model.fit(x[order], grade[order], qid=qid[order])
    return NegatedModel(model)


def train_lagged_model(frame):
    """A3: the same regressor over current features plus the state LAG_K back."""
    x_now = frame[tdb.BASE_FEATURES].to_numpy(dtype=np.float64)
    lag = frame[CLUSTER_COLS].shift(LAG_K).bfill().to_numpy(dtype=np.float64)
    x = np.hstack([x_now, lag])
    y = np.log1p(frame['wait_time'].clip(lower=0).to_numpy(dtype=np.float64))
    model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08,
                         subsample=0.9, colsample_bytree=0.9,
                         random_state=ROBUST_SEED, n_jobs=4, verbosity=0)
    model.fit(x, y)
    return LaggedModel(model)


# ─────────────────────────────────────────────────────────────────────────────
# A4: the enqueue-time cached-score policy. This is the one variant that CANNOT
# be expressed as a function of the contemporaneous feature matrix, so it needs
# its own dispatch loop. The loop mirrors trace_driven_benchmark.simulate for
# the non-backfill, non-preemptive case and reuses that module's TraceJob,
# TraceCluster, build_feature_matrix and summarize, so the only thing that
# differs from the published simulation is WHEN the score is computed.
# ─────────────────────────────────────────────────────────────────────────────

def simulate_cached_enqueue(jobs_in, capacity, model, window_meta,
                            observer=None):
    """Score each job ONCE, at the instant it joins the queue, then cache it."""
    jobs = [tdb.TraceJob(j.job_id, j.arrival_time, j.num_gpus, j.runtime,
                         j.est_runtime, j.measured) for j in jobs_in]
    cluster = tdb.TraceCluster(capacity)
    queue, running, completed = [], [], []
    cached = {}
    n = len(jobs)
    ai = 0
    t = jobs[0].arrival_time
    t0 = t
    util_area = 0.0

    while len(completed) < n:
        if running:
            still = []
            for job in running:
                if job.end_time <= t:
                    cluster.release(job)
                    completed.append(job)
                else:
                    still.append(job)
            running = still

        while ai < n and jobs[ai].arrival_time <= t:
            queue.append(jobs[ai])
            ai += 1

        if queue:
            x = tdb.build_feature_matrix(queue, cluster, running, False)
            # Jobs that have never been scored are scored NOW, against the state
            # they see on arrival. Jobs already holding a score keep it -- that
            # staleness IS the policy, and is what makes the variant a different
            # policy rather than a different implementation.
            fresh = [i for i, j in enumerate(queue) if id(j) not in cached]
            if fresh:
                pred_fresh = model.predict(x[fresh])
                for i, p in zip(fresh, pred_fresh):
                    cached[id(queue[i])] = float(p)
            pred = np.array([cached[id(j)] for j in queue], dtype=float)
            if observer is not None:
                observer('PROACTIVE_ENQUEUE_CACHED', queue, x, pred)

            order = sorted(range(len(queue)),
                           key=lambda i: (float(pred[i]), queue[i].arrival_time,
                                          queue[i].job_id))
            started = []
            for i in order:
                job = queue[i]
                if cluster.total_free_gpus() < job.num_gpus:
                    continue
                if cluster.allocate(job, t):
                    started.append(job)
            if started:
                started_ids = {id(j) for j in started}
                queue = [j for j in queue if id(j) not in started_ids]
                for j in started:
                    cached.pop(id(j), None)
                running.extend(started)

        next_t = None
        if running:
            next_t = min(j.end_time for j in running)
        if ai < n:
            next_t = (jobs[ai].arrival_time if next_t is None
                      else min(next_t, jobs[ai].arrival_time))
        if next_t is None:
            break
        util_area += (capacity - cluster.total_free_gpus()) * (next_t - t)
        t = next_t

    return tdb.summarize(completed, util_area, max(t - t0, 1), capacity,
                         'PROACTIVE_ENQUEUE_CACHED', window_meta)


# ─────────────────────────────────────────────────────────────────────────────
# Running one attack over one trace
# ─────────────────────────────────────────────────────────────────────────────

def instrument_via_published_loop(label, model, windows, capacity):
    """Run the PUBLISHED simulation loop with a substituted score function.

    Used for the baseline reference, A1, A2 and A3 -- everything whose score is
    a function of the contemporaneous feature matrix. The observer hook and the
    dispatch loop are trace_driven_benchmark's own.
    """
    col = Collector(label, list(tdb.BASE_FEATURES))
    tdb.RANK_OBSERVER = col
    try:
        for widx, w0, measure_start, w1, sel in windows:
            jobs, _ = tdb.make_jobs(sel, w0, measure_start, capacity)
            if not jobs:
                continue
            if hasattr(model, 'reset'):
                model.reset()
            tdb.simulate(jobs, 'PROACTIVE', capacity, model,
                         {'trace': label, 'window': widx, 'capacity': capacity,
                          'jobs_total': len(jobs), 'jobs_dropped_oversized': 0,
                          'offered_load': 0.0})
    finally:
        tdb.RANK_OBSERVER = None
    return col


def instrument_cached_enqueue(label, model, windows, capacity):
    """A4's instrumentation, plus the per-window scheduling rows it produces."""
    col = Collector(label, list(tdb.BASE_FEATURES))
    rows = []
    for widx, w0, measure_start, w1, sel in windows:
        jobs, _ = tdb.make_jobs(sel, w0, measure_start, capacity)
        if not jobs:
            continue
        rows.append(simulate_cached_enqueue(
            jobs, capacity, model,
            {'window': widx, 'capacity': capacity, 'jobs_total': len(jobs)},
            observer=col))
    return col, rows


def baseline_policy_rows(policy, model, windows, capacity):
    """Per-window rows for a published policy, for the A4 utility comparison."""
    rows = []
    for widx, w0, measure_start, w1, sel in windows:
        jobs, _ = tdb.make_jobs(sel, w0, measure_start, capacity)
        if not jobs:
            continue
        rows.append(tdb.simulate(jobs, policy, capacity, model,
                                 {'window': widx, 'capacity': capacity,
                                  'jobs_total': len(jobs)}))
    return rows


def attack_row(attack_id, attack, trace, col, note, n_windows, partial):
    """One artefact row, read straight off the published Collector's summary."""
    s = col.summary()
    violations = int(s['equal_size_diff_pred_violations'])
    return {
        'attack_id': attack_id,
        'attack': attack,
        'trace': trace,
        'ranking_instants': int(s['ranking_instants']),
        'violations': violations,
        'mean_kendall_tau_vs_size': s['kendall_tau_vs_size_mean'],
        'pct_all_scores_tied': s['pct_all_scores_tied'],
        'mean_distinct_levels': s['mean_distinct_predictions'],
        'pct_order_identical_to_size': s['pct_order_identical_to_size'],
        'windows': n_windows,
        'partial': partial,
        'verdict': VERDICT_BROKEN if violations > 0 else VERDICT_INTACT,
        'note': note,
    }


def run_trace_attacks(trace_key, n_windows, partial):
    meta = tdb.TRACES[trace_key]
    label = meta['label']
    print(f"\n{'=' * 74}\n{label} -- robustness attacks\n{'=' * 74}")

    capacity, df, _ = tdb.parse_swf_jobs(
        os.path.join(tdb.DATA_DIR, meta['swf']))
    t_min, t_max = int(df['submit'].min()), int(df['submit'].max())
    split_time = int(t_min + tdb.TRAIN_FRACTION * (t_max - t_min))
    windows = tdb.carve_windows(df, split_time, n_windows, tdb.WARMUP_DAYS,
                                tdb.MEASURE_DAYS)
    print(f'  capacity {capacity}  windows {len(windows)}')

    # The PUBLISHED model, trained exactly as the published pipeline trains it.
    base_model, _ = tdb.train_trace_model(split_time, df, False, capacity)
    frame = replay_training_frame(df, capacity, split_time)
    print(f'  training rows {len(frame)}')

    rows = []

    # ── reference: the published pointwise model, same protocol ─────────────
    print('  A0 baseline (published pointwise regressor) ...')
    col = instrument_via_published_loop(label, base_model, windows, capacity)
    rows.append(attack_row(
        'A0', 'baseline: published pointwise regressor', trace_key, col,
        'Reference row, not an attack. Reproduces the published measurement '
        'under this run\'s window count so every attack below is read against '
        'a like-for-like control.', len(windows), partial))

    # ── A1: a different learning objective ──────────────────────────────────
    for obj, short in (('rank:pairwise', 'pairwise'), ('rank:ndcg', 'ndcg')):
        print(f'  A1 learning-to-rank {obj} ...')
        model = train_rank_model(frame, obj)
        col = instrument_via_published_loop(label, model, windows, capacity)
        rows.append(attack_row(
            'A1', f'different objective: XGBRanker {obj}', trace_key, col,
            f'Argument: degeneracy is a property of the INPUTS, not the loss -- '
            f'if every per-job feature is a function of size given the state, '
            f'so is any function of them, whatever the training objective. '
            f'Measured with a {short} learning-to-rank model over the identical '
            f'8 features; scores negated so ascending order is dispatch order '
            f'(monotone, so it moves no counter but the sign of tau). '
            f'Query groups are {RANK_GROUP_SECONDS}s submit-time buckets, '
            f'relevance is {RELEVANCE_GRADES} global quantile grades of '
            f'log1p(wait) with shortest wait most relevant.',
            len(windows), partial))

    # ── A2: monotone transforms of the published score ──────────────────────
    transforms = (
        ('log1p', lambda p: np.log1p(p - p.min() + 1e-9)),
        ('sigmoid', lambda p: 1.0 / (1.0 + np.exp(-p))),
    )
    for name, fn in transforms:
        print(f'  A2 monotone transform {name} ...')
        model = TransformedModel(base_model, fn)
        col = instrument_via_published_loop(label, model, windows, capacity)
        rows.append(attack_row(
            'A2', f'monotone transform of the published score: {name}',
            trace_key, col,
            'Argument: a strictly monotone map is order-preserving, so it can '
            'neither reorder a queue nor split a tie -- this attack fails by '
            'construction and a non-zero violation count here would mean the '
            'instrumentation is broken, not that the claim is. Applied to the '
            'PUBLISHED model\'s own predictions; nothing is retrained.',
            len(windows), partial))

    # ── A3: a history window ────────────────────────────────────────────────
    print(f'  A3 history window (lag {LAG_K}) ...')
    model = train_lagged_model(frame)
    col = instrument_via_published_loop(label, model, windows, capacity)
    rows.append(attack_row(
        'A3', f'history window: cluster state {LAG_K} ticks ago', trace_key, col,
        f'Argument: lagged cluster state is still SHARED by every co-queued job '
        f'at the dispatch instant, so the lagged block lands in the same '
        f'"identical for every queued job" bucket as the contemporaneous one. '
        f'Measured with the 8 features plus {CLUSTER_COLS} as of {LAG_K} back. '
        f'Honest caveat: a "tick" is one training row back when fitting and one '
        f'dispatch instant back when simulating, and the lag block repeats the '
        f'oldest available state until {LAG_K} instants have been seen.',
        len(windows), partial))

    # ── A4: score at enqueue time, cached ───────────────────────────────────
    print('  A4 enqueue-time cached score ...')
    col, cached_rows = instrument_cached_enqueue(label, base_model, windows,
                                                 capacity)
    rows.append(attack_row(
        'A4', 'score computed at enqueue time and cached', trace_key, col,
        'Argument: if a job is scored when it ENTERS the queue and the score is '
        'then cached, two co-queued jobs were scored against DIFFERENT cluster '
        'states, the shared-state premise fails, and the ranking need not be a '
        'size permutation. This is a DIFFERENT POLICY, not a different '
        'implementation of the published one: staleness is a cost as well as a '
        'source of variation, so see robustness_attack_utility.csv for whether '
        'it actually schedules better or merely becomes non-degenerate.',
        len(windows), partial))

    # ── A4's second question: does it schedule better? ──────────────────────
    util = []
    for r in cached_rows:
        util.append({'trace': trace_key, 'scheduler': 'PROACTIVE_ENQUEUE_CACHED',
                     'window': r['window'], 'mean_wait': r['mean_wait'],
                     'median_wait': r['median_wait'],
                     'mean_bounded_slowdown': r['mean_bounded_slowdown'],
                     'jobs_measured': r['jobs_measured']})
    for policy in ('PROACTIVE', 'SJF_USEREST'):
        print(f'  A4 utility comparison: {policy} ...')
        for r in baseline_policy_rows(policy, base_model, windows, capacity):
            util.append({'trace': trace_key, 'scheduler': policy,
                         'window': r['window'], 'mean_wait': r['mean_wait'],
                         'median_wait': r['median_wait'],
                         'mean_bounded_slowdown': r['mean_bounded_slowdown'],
                         'jobs_measured': r['jobs_measured']})

    return rows, util


# ─────────────────────────────────────────────────────────────────────────────

def summarise_utility(util_df, partial, n_windows):
    """Per-trace per-policy means, with the delta against PROACTIVE."""
    g = (util_df.groupby(['trace', 'scheduler'], as_index=False)
         .agg(windows=('window', 'nunique'),
              jobs_measured=('jobs_measured', 'sum'),
              mean_wait=('mean_wait', 'mean'),
              median_wait=('median_wait', 'mean'),
              mean_bounded_slowdown=('mean_bounded_slowdown', 'mean')))
    ref = (g[g['scheduler'] == 'PROACTIVE']
           .set_index('trace')['mean_wait'].to_dict())
    g['mean_wait_vs_proactive_pct'] = [
        100.0 * (w - ref[tr]) / ref[tr] if ref.get(tr) else float('nan')
        for tr, w in zip(g['trace'], g['mean_wait'])]
    g['partial'] = partial
    g['windows_requested'] = n_windows
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--traces', nargs='+', default=list(tdb.TRACES),
                    choices=list(tdb.TRACES))
    ap.add_argument('--windows', type=int, default=tdb.N_WINDOWS,
                    help=f'evaluation windows per trace (published: '
                         f'{tdb.N_WINDOWS}); any other value marks every row '
                         f'partial')
    args = ap.parse_args()

    partial = (args.windows != tdb.N_WINDOWS
               or sorted(args.traces) != sorted(tdb.TRACES))

    all_rows, all_util = [], []
    for trace_key in args.traces:
        rows, util = run_trace_attacks(trace_key, args.windows, partial)
        all_rows.extend(rows)
        all_util.extend(util)

    attacks = pd.DataFrame(all_rows)
    attacks.to_csv(ATTACKS_CSV, index=False)

    util_df = pd.DataFrame(all_util)
    summarise_utility(util_df, partial, args.windows).to_csv(UTILITY_CSV,
                                                             index=False)

    print(f'\nWrote {ATTACKS_CSV}')
    print(f'Wrote {UTILITY_CSV}')
    if partial:
        print('PARTIAL RUN: not the published protocol; every row says so.')
    cols = ['attack_id', 'trace', 'violations', 'mean_kendall_tau_vs_size',
            'pct_order_identical_to_size', 'verdict']
    print(attacks[cols].to_string(index=False))
    print(summarise_utility(util_df, partial, args.windows).to_string(
        index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())


# Referenced so the tie-quantisation helper the Collector uses is an explicit
# import rather than an accident of module layout: every prediction comparison
# in the artefact above is on tie_keys' quantised value, exactly as published.
_ = tie_keys

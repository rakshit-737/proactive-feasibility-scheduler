"""Does adding a per-job feature that is NOT a function of requested size
actually break the ranking degeneracy -- and does it make the scheduler good?

WHY THIS EXISTS
---------------
`04_scheduler/ranking_degeneracy.py` establishes the project's negative result:
at a fixed dispatch instant every queued job sees the SAME cluster state, so
every feature in the 8-feature wait-time set is a deterministic function of the
job's requested size given that state. The learned score collapses to
g_S(size), the induced ranking is a permutation of the size order, and the
pipeline is TOST-equivalent to "sort by requested size". That is measured, with
zero counterexamples, and NOTHING in this module touches it.

The manuscript also STATES the converse condition -- "a wait-time feature set
can only produce a meaningful ranking if it contains a per-job attribute that
is not a function of requested size given the cluster state" -- but never
demonstrates it. This module demonstrates it, on a controlled sweep over the
two real traces, and then closes the loop that makes it a finding rather than a
demonstration: non-degeneracy is NECESSARY for the model to rank anything, and
it is NOT SUFFICIENT for the model to be USEFUL.

WHAT IS ADDED, AND WHY EACH ONE IS NOT A FUNCTION OF SIZE
---------------------------------------------------------
  est_runtime        SWF field 9, the user's requested time. Two jobs of the
                     same size can request wildly different walltimes.
  user_hist_wait     mean wait of that user's STRICTLY EARLIER COMPLETED jobs.
  user_hist_runtime  mean runtime of that user's strictly earlier completed jobs.
  queue_id           SWF field 15, categorical.
  user_id            SWF field 12, categorical.

SWF field 16 (partition) is deliberately NOT included: it is CONSTANT on both
committed traces, so it varies across no pair of co-queued jobs and cannot
break anything. Saying so is more useful than quietly leaving it out.

CAUSALITY -- THE POINT ON WHICH THIS EXPERIMENT LIVES OR DIES
--------------------------------------------------------------
A user-history feature computed with a full-trace groupby mean leaks the future
into every row and would MANUFACTURE a fake non-degeneracy: the "feature" would
carry information no scheduler could ever have, violations would appear, and
the experiment would prove precisely nothing. So the history for a job
submitted at instant t uses only that user's jobs that had already COMPLETED
before t -- COMPLETED, not merely submitted earlier, because a job's wait time
is not observable until it starts, and its runtime is not observable until it
ends. Completion is taken from the RECORDED schedule
(submit + recorded_wait + runtime), which is exactly what the real machine
would have known at t. `tests/test_non_degeneracy.py` pins this down with a
leakage test that goes red under the full-trace groupby.

COLD START: a job whose user has no completed prior job gets NaN, which XGBoost
handles natively by learning a default branch direction. NaN is preferred over
imputing the global running mean for two reasons: (a) it keeps "this user is
unseen" DISTINGUISHABLE from "this user happens to average the global mean",
which an imputed constant destroys; and (b) an imputed global constant is
itself a pooled statistic that would have to be maintained causally too, adding
a second thing to get wrong for no modelling benefit.

HOW THE EXISTING MACHINERY IS REUSED RATHER THAN COPIED
--------------------------------------------------------
* The degeneracy instrumentation is `ranking_degeneracy.Collector`, IMPORTED.
  The two experiments therefore agree by construction on what a violation is,
  what counts as a tie, and how the induced order is formed.
* The simulation, window carving, job construction, SWF parsing and the
  8 base feature columns are `trace_driven_benchmark`'s, IMPORTED.
* `build_feature_matrix` is MONKEY-PATCHED for the duration of a variant rather
  than reimplemented: the patch calls the original for columns 0..7 and appends
  the extra columns. The published module is not edited and, because the patch
  is installed and removed inside a context manager, is not left altered.
* Pairing and TOST are `simstats.equivalence_table` / `pairwise_significance`,
  paired by window LABEL.

WHAT IS NECESSARILY DUPLICATED, AND WHY
----------------------------------------
`train_variant_model` below repeats the body of
`trace_driven_benchmark.train_trace_model`: same chronological split, same
`replay_trace_features` replay, same log1p(wait) target, same XGBRegressor
hyper-parameters, same SEED=42. It cannot simply call it because that function
hard-codes `BASE_FEATURES (+ EST_FEATURE)` as the column list and this sweep's
whole purpose is to vary that list. The duplication is the feature list and
nothing else; `tests/test_non_degeneracy.py` asserts the hyper-parameters here
match the published ones field by field.

SEEDS: no new seed is introduced. Model training uses SEED=42, imported from
trace_driven_benchmark, exactly as the published trace models do. Nothing in
this experiment is stochastic beyond that -- windows are deterministic, the
simulation is deterministic, and the feature construction is a deterministic
sweep over the trace.

USAGE (the pipeline step)
-------------------------
    python 04_scheduler/non_degeneracy_sweep.py

    --windows N   evaluation windows per trace (default 20, the published
                  protocol). Any other value is recorded in the artefact's
                  `n_windows` column and announced on stdout.
    --traces      subset of {sdsc,lanl} (default both).
    --quick       shorthand for --windows 3, for smoke tests.
"""

import argparse
import os
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, '02_data'))

import trace_driven_benchmark as tdb            # noqa: E402
from ranking_degeneracy import Collector        # noqa: E402
from simstats import equivalence_table          # noqa: E402
from swf_io import open_swf                     # noqa: E402
from build_real_trace_datasets import replay_trace_features  # noqa: E402

OUT_DIR = os.path.join(PROJECT_ROOT, '05_results', 'degeneracy')
os.makedirs(OUT_DIR, exist_ok=True)

SWEEP_CSV = os.path.join(OUT_DIR, 'non_degeneracy_sweep.csv')
UTILITY_CSV = os.path.join(OUT_DIR, 'non_degeneracy_utility.csv')

# The five candidate per-job attributes, and the feature sets built from them.
# `est_runtime` is read off the TraceJob the simulator already carries (it is
# parsed by trace_driven_benchmark.parse_swf_jobs, median-imputed there when the
# trace recorded -1); the other four come from the per-job table below.
EXTRA_FEATURES = ['est_runtime', 'user_hist_wait', 'user_hist_runtime',
                  'queue_id', 'user_id']

FEATURE_SETS = (
    [('baseline', [])]
    + [(f'+{name}', [name]) for name in EXTRA_FEATURES]
    + [('+all', list(EXTRA_FEATURES))]
)

# Reference policies the ML variants are judged against.
DEGENERACY_REFERENCE = 'SMALLEST_FIRST'   # what degeneracy makes PROACTIVE equal to
UTILITY_REFERENCE = 'SJF_USEREST'         # the simple runtime heuristic


# ─────────────────────────────────────────────────────────────────────────────
# Per-job attributes (the part that must be causal)
# ─────────────────────────────────────────────────────────────────────────────

def parse_swf_identity(path):
    """job_id -> (user_id, queue_id) from SWF fields 12 and 15.

    A SEPARATE, minimal pass rather than a fork of
    trace_driven_benchmark.parse_swf_jobs: that function is published and must
    not be touched, and its filtering is authoritative. This pass applies NO
    filtering at all -- it is a lookup table, and the job population is whatever
    parse_swf_jobs kept. Fields are 1-indexed in the SWF spec, so field 12 is
    parts[11] and field 15 is parts[14]. A short row (some cleaned traces stop
    at field 11) yields -1, the SWF's own "unknown" sentinel.
    """
    ident = {}
    with open_swf(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            parts = line.split()
            if len(parts) < 11:
                continue
            job_id = int(parts[0])
            uid = int(float(parts[11])) if len(parts) > 11 else -1
            qid = int(float(parts[14])) if len(parts) > 14 else -1
            ident[job_id] = (uid, qid)
    return ident


def causal_user_history(df):
    """Per-job (user_hist_wait, user_hist_runtime) using only COMPLETED past.

    `df` must carry job_id, submit, recorded_wait, runtime, user_id and be
    sorted by submit. For the job submitted at t, the history is the mean over
    that user's jobs whose RECORDED completion instant

        submit_k + recorded_wait_k + runtime_k

    is <= t. Strictly-earlier submission is NOT enough: at instant t a job that
    was submitted at t-1 and has not started yet has no observable wait, and one
    that is running has no observable runtime. Using either would leak.

    Returns two float arrays aligned with df's row order; NaN where the user has
    no completed prior job (see the module docstring on cold start).

    O(n log n): one min-heap of pending completions, drained up to t before each
    job is scored, feeding per-user running sums.
    """
    import heapq

    job_ids = df['job_id'].to_numpy()
    submits = df['submit'].to_numpy(dtype=np.int64)
    waits = df['recorded_wait'].to_numpy(dtype=np.float64)
    runs = df['runtime'].to_numpy(dtype=np.float64)
    users = df['user_id'].to_numpy()

    n = len(df)
    hist_wait = np.full(n, np.nan, dtype=np.float64)
    hist_run = np.full(n, np.nan, dtype=np.float64)

    pending = []                 # (completion_time, seq, user, wait, runtime)
    sums = {}                    # user -> [wait_sum, run_sum, count]

    for i in range(n):
        t = submits[i]
        # Absorb every job that COMPLETED at or before t. `<= t` (not `< t`) is
        # deliberate and harmless either way: a job completing exactly at t is
        # observable at t.
        while pending and pending[0][0] <= t:
            _c, _s, u, w, r = heapq.heappop(pending)
            acc = sums.setdefault(u, [0.0, 0.0, 0])
            acc[0] += w
            acc[1] += r
            acc[2] += 1
        acc = sums.get(users[i])
        if acc is not None and acc[2] > 0:
            hist_wait[i] = acc[0] / acc[2]
            hist_run[i] = acc[1] / acc[2]
        # This job becomes observable only at its own recorded completion, which
        # is strictly after t because runtime > 0 is enforced by parse_swf_jobs.
        heapq.heappush(pending, (int(submits[i] + waits[i] + runs[i]), int(i),
                                 users[i], waits[i], runs[i]))

    assert len(job_ids) == n
    return hist_wait, hist_run


def build_job_attributes(df, swf_path):
    """Attach user_id, queue_id and the two causal history columns to `df`.

    Returns a NEW frame; `df` (the published parse) is not mutated.

    user_id / queue_id are used as raw SWF integer codes. They are categorical,
    and XGBoost splits a tree on thresholds of the code, which is an arbitrary
    but CONSISTENT partition of the categories -- adequate here because the
    question is only whether a per-job attribute that is not a function of size
    can separate co-queued jobs at all, not whether an optimal encoding of user
    identity would separate them better. A one-hot expansion over 83/93 users
    would change the answer only in the direction of MORE separation, so the
    integer code is the conservative choice.
    """
    ident = parse_swf_identity(swf_path)
    out = df.copy()
    out['user_id'] = [ident.get(int(j), (-1, -1))[0] for j in out['job_id']]
    out['queue_id'] = [ident.get(int(j), (-1, -1))[1] for j in out['job_id']]
    out = out.sort_values(['submit', 'job_id']).reset_index(drop=True)
    hw, hr = causal_user_history(out)
    out['user_hist_wait'] = hw
    out['user_hist_runtime'] = hr
    return out


def attribute_lookup(attr_df, names):
    """job_id -> np.array of the `names` columns, for use inside the simulator."""
    if not names:
        return {}
    sub = attr_df[['job_id'] + [n for n in names if n != 'est_runtime']]
    cols = [n for n in names if n != 'est_runtime']
    vals = sub[cols].to_numpy(dtype=np.float64) if cols else np.empty((len(sub), 0))
    return {int(j): vals[i] for i, j in enumerate(sub['job_id'].to_numpy())}


# ─────────────────────────────────────────────────────────────────────────────
# The monkey-patched feature matrix
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def extended_features(names, lookup):
    """Install a build_feature_matrix that appends `names` after the 8 base
    columns, for the duration of the block, then restore the original.

    The original is CALLED for columns 0..7, never re-derived, so the base
    features -- and therefore SIZE_COL=0 being requested size, which the
    Collector depends on -- are bit-identical to the published pipeline.

    `with_est` from the caller is forced False: this sweep controls the feature
    list itself, and 'est_runtime' enters as one of `names` when selected. A
    variant list that contains it puts it in the SAME position the published
    PROACTIVE_EST model uses (column 8) whenever it is the only extra.
    """
    original = tdb.build_feature_matrix

    def patched(queue, cluster, running, with_est):
        base = original(queue, cluster, running, False)
        if not names:
            return base
        extra = np.empty((len(queue), len(names)), dtype=np.float64)
        other = [n for n in names if n != 'est_runtime']
        for i, job in enumerate(queue):
            row = lookup.get(int(job.job_id))
            k = 0
            for name in names:
                if name == 'est_runtime':
                    extra[i, k] = job.est_runtime
                else:
                    extra[i, k] = (row[other.index(name)] if row is not None
                                   else np.nan)
                k += 1
        return np.hstack([base, extra])

    tdb.build_feature_matrix = patched
    try:
        yield
    finally:
        tdb.build_feature_matrix = original


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

# Hyper-parameters copied verbatim from trace_driven_benchmark.train_trace_model.
# Kept as a dict so the test file can assert equality against that function's
# source rather than against a remembered value.
XGB_PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.08,
                  subsample=0.9, colsample_bytree=0.9,
                  random_state=tdb.SEED, n_jobs=4, verbosity=0)


def train_variant_model(split_time, attr_df, capacity, extra_names):
    """Retrain the wait model on the trace's own early period with
    BASE_FEATURES + extra_names.

    Mirrors trace_driven_benchmark.train_trace_model exactly except for the
    column list -- see the module docstring on what is duplicated and why.
    """
    jobs = [{'job_id': int(r.job_id), 'submit': int(r.submit),
             'wait': int(r.recorded_wait), 'run': int(r.runtime),
             'procs': int(r.procs)} for r in attr_df.itertuples(index=False)]
    df, _ = replay_trace_features(jobs, capacity)
    df = df[df['submit_time'] < split_time]
    if extra_names:
        df = df.merge(attr_df[['job_id'] + extra_names], on='job_id', how='inner')
    feats = list(tdb.BASE_FEATURES) + list(extra_names)
    x = df[feats].to_numpy(dtype=np.float64)
    y = np.log1p(df['wait_time'].clip(lower=0).to_numpy(dtype=np.float64))
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(x, y)
    return model, len(df), feats


# ─────────────────────────────────────────────────────────────────────────────
# The sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_variant(trace_key, capacity, attr_df, split_time, windows,
                set_name, extra_names):
    """Train one variant, replay every window under PROACTIVE with the
    degeneracy instrumentation live, and return (summary_row, wait_rows)."""
    model, n_train, feats = train_variant_model(split_time, attr_df, capacity,
                                                extra_names)
    lookup = attribute_lookup(attr_df, extra_names)
    col = Collector(f'{tdb.TRACES[trace_key]["label"]} / {set_name}', feats)

    wait_rows = []
    with extended_features(extra_names, lookup):
        tdb.RANK_OBSERVER = col
        try:
            for widx, w0, measure_start, _w1, sel in windows:
                jobs, dropped = tdb.make_jobs(sel, w0, measure_start, capacity)
                if not jobs:
                    continue
                meta = {'trace': trace_key, 'window': widx,
                        'capacity': capacity, 'jobs_total': len(jobs),
                        'jobs_dropped_oversized': dropped, 'offered_load': 0.0}
                out = tdb.simulate(jobs, 'PROACTIVE', capacity, model, meta)
                out['scheduler'] = f'PROACTIVE::{set_name}'
                wait_rows.append(out)
        finally:
            tdb.RANK_OBSERVER = None

    summary = col.summary()
    summary.update({'trace': trace_key, 'feature_set': set_name,
                    'n_features': len(feats),
                    'extra_features': '|'.join(extra_names) or 'none',
                    'train_rows': n_train})
    return summary, wait_rows


def run_reference(trace_key, capacity, windows, policy):
    """Replay every window under a model-free reference policy."""
    rows = []
    for widx, w0, measure_start, _w1, sel in windows:
        jobs, dropped = tdb.make_jobs(sel, w0, measure_start, capacity)
        if not jobs:
            continue
        meta = {'trace': trace_key, 'window': widx, 'capacity': capacity,
                'jobs_total': len(jobs), 'jobs_dropped_oversized': dropped,
                'offered_load': 0.0}
        rows.append(tdb.simulate(jobs, policy, capacity, None, meta))
    return rows


def run_trace(trace_key, n_windows):
    """Full sweep for one trace. Returns (sweep_rows, utility_rows)."""
    meta = tdb.TRACES[trace_key]
    swf_path = os.path.join(tdb.DATA_DIR, meta['swf'])
    capacity, df, _ = tdb.parse_swf_jobs(swf_path)
    attr_df = build_job_attributes(df, swf_path)

    n_users = attr_df['user_id'].nunique()
    n_queues = attr_df['queue_id'].nunique()
    cold = float(attr_df['user_hist_wait'].isna().mean())
    print(f"\n{'='*74}\n{meta['label']}  --  {meta['swf']}\n{'='*74}")
    print(f'  capacity {capacity}, jobs {len(attr_df)}, '
          f'users {n_users}, queues {n_queues}, '
          f'cold-start (NaN history) {100*cold:.1f}%')

    t_min, t_max = int(attr_df['submit'].min()), int(attr_df['submit'].max())
    split_time = int(t_min + tdb.TRAIN_FRACTION * (t_max - t_min))
    windows = tdb.carve_windows(attr_df, split_time, n_windows,
                                tdb.WARMUP_DAYS, tdb.MEASURE_DAYS)
    print(f'  evaluation windows: {len(windows)}')

    # The two model-free references, replayed once: they do not depend on the
    # feature set, so running them per variant would be identical work.
    ref_rows = {}
    for policy in (DEGENERACY_REFERENCE, UTILITY_REFERENCE):
        ref_rows[policy] = run_reference(trace_key, capacity, windows, policy)
        mw = np.mean([r['mean_wait'] for r in ref_rows[policy]])
        print(f'  {policy:16s} mean wait {mw/60:9.1f} min')

    sweep_rows, utility_rows = [], []
    for set_name, extra_names in FEATURE_SETS:
        summary, wait_rows = run_variant(trace_key, capacity, attr_df,
                                         split_time, windows, set_name,
                                         extra_names)
        sched = f'PROACTIVE::{set_name}'

        # (1) Does the degeneracy equivalence against SMALLEST_FIRST survive?
        runs = pd.DataFrame(wait_rows + ref_rows[DEGENERACY_REFERENCE])
        eq = equivalence_table(runs, [(sched, DEGENERACY_REFERENCE)],
                               metric='mean_wait', unit='window')
        summary['tost_vs_smallest_first_equivalent'] = bool(eq['equivalent'].iloc[0])
        summary['tost_vs_smallest_first_p'] = float(eq['p_tost'].iloc[0])
        summary['mean_wait_min'] = float(np.mean([r['mean_wait'] for r in wait_rows]) / 60.0)
        summary['smallest_first_mean_wait_min'] = float(eq['mean_b'].iloc[0] / 60.0)
        summary['n_windows'] = len(wait_rows)
        sweep_rows.append(summary)

        # (2) Utility: is it better than the simple runtime heuristic?
        runs_u = pd.DataFrame(wait_rows + ref_rows[UTILITY_REFERENCE])
        equ = equivalence_table(runs_u, [(sched, UTILITY_REFERENCE)],
                               metric='mean_wait', unit='window')
        r = equ.iloc[0]
        utility_rows.append({
            'trace': trace_key, 'feature_set': set_name,
            'extra_features': '|'.join(extra_names) or 'none',
            'n_windows': int(r['n']),
            'ml_mean_wait_min': float(r['mean_a'] / 60.0),
            'sjf_userest_mean_wait_min': float(r['mean_b'] / 60.0),
            'pct_vs_sjf_userest': float(r['pct_diff']),
            'mean_diff_min': float(r['mean_diff'] / 60.0),
            'ci_low_min': float(r['ci_low'] / 60.0),
            'ci_high_min': float(r['ci_high'] / 60.0),
            'p_tost': float(r['p_tost']),
            'tost_equivalent_to_sjf': bool(r['equivalent']),
            # The half that matters: does the ML variant WIN? A win requires the
            # paired mean difference to be negative (lower wait) AND the
            # equivalence interval to exclude zero on the good side.
            'beats_sjf_userest': bool(r['ci_high'] < 0.0),
        })
        print(f"    {set_name:20s} viol={summary['equal_size_diff_pred_violations']:6d}  "
              f"tau={summary['kendall_tau_vs_size_mean']:6.3f}  "
              f"tied={summary['pct_all_scores_tied']:6.2f}%  "
              f"distinct={summary['mean_distinct_predictions']:6.2f}  "
              f"TOST(size)={'EQUIV' if summary['tost_vs_smallest_first_equivalent'] else 'BROKEN'}  "
              f"wait={summary['mean_wait_min']:8.1f}m "
              f"({utility_rows[-1]['pct_vs_sjf_userest']:+.1f}% vs SJF)")

    return sweep_rows, utility_rows


SWEEP_COLUMNS = ['trace', 'feature_set', 'extra_features', 'n_features',
                 'n_windows', 'train_rows', 'ranking_instants',
                 'mean_queue_len', 'equal_size_diff_pred_violations',
                 'kendall_tau_vs_size_mean', 'kendall_tau_vs_size_min',
                 'mean_distinct_predictions', 'pct_all_scores_tied',
                 'pct_order_identical_to_size', 'pct_order_identical_to_arrival',
                 'pct_size_table_monotone', 'features_varying_mean',
                 'mean_wait_min', 'smallest_first_mean_wait_min',
                 'tost_vs_smallest_first_equivalent',
                 'tost_vs_smallest_first_p']


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--windows', type=int, default=tdb.N_WINDOWS,
                    help='evaluation windows per trace (default 20 = published)')
    ap.add_argument('--traces', nargs='+', default=list(tdb.TRACES),
                    choices=list(tdb.TRACES))
    ap.add_argument('--quick', action='store_true',
                    help='smoke test: 3 windows')
    args = ap.parse_args()

    n_windows = 3 if args.quick else args.windows
    if n_windows != tdb.N_WINDOWS:
        print(f'*** REDUCED PROTOCOL: {n_windows} windows per trace, not the '
              f'published {tdb.N_WINDOWS}. The n_windows column records this. ***')

    sweep, utility = [], []
    for trace_key in args.traces:
        s, u = run_trace(trace_key, n_windows)
        sweep.extend(s)
        utility.extend(u)

    sdf = pd.DataFrame(sweep)
    sdf = sdf[[c for c in SWEEP_COLUMNS if c in sdf.columns]]
    sdf.to_csv(SWEEP_CSV, index=False)
    udf = pd.DataFrame(utility)
    udf.to_csv(UTILITY_CSV, index=False)

    print(f'\nwrote {SWEEP_CSV}')
    print(f'wrote {UTILITY_CSV}')

    # The deliverable sentence, with the numbers behind both halves.
    print('\n' + '=' * 74)
    broke = sdf[(sdf['feature_set'] != 'baseline')
                & (sdf['equal_size_diff_pred_violations'] > 0)]
    base = sdf[sdf['feature_set'] == 'baseline']
    print(f'baseline variants with ZERO violations : '
          f'{int((base["equal_size_diff_pred_violations"] == 0).sum())}/{len(base)}')
    print(f'augmented variants with violations > 0 : {len(broke)}/'
          f'{len(sdf) - len(base)}')
    wins = udf[udf['beats_sjf_userest']]
    print(f'augmented variants that BEAT {UTILITY_REFERENCE} : '
          f'{len(wins)}/{len(udf[udf["feature_set"] != "baseline"])}')
    if len(broke) and not len(wins):
        print('\nAdding a per-job feature that is not a function of size BREAKS '
              'the degeneracy,\nbut does NOT by itself make the scheduler better '
              'than a simple runtime heuristic.')
    elif not len(broke):
        print('\nMAJOR FINDING: the degeneracy did NOT break where theory says '
              'it should. Read the table.')


if __name__ == '__main__':
    main()

"""Trace-driven scheduler benchmark on real Parallel Workloads Archive traces.

WHY THIS EXISTS
---------------
Every scheduler number in this project up to v3.3 came from a synthetic
generator (uniform arrivals, `randint(1,8)` processors, `randint(5,20)` tick
runtimes). Real batch workloads are heavy-tailed and diurnal, and the project's
own sim-to-real study showed the synthetic-trained wait model transfers at
R^2 ~ 0 to real traces. A scheduler ranking measured on the synthetic generator
therefore carries no evidence about real machines. This harness replays two
REAL traces (LANL CM-5 1994, SDSC SP2 1998; Parallel Workloads Archive,
cleaned versions) through the same policy set.

WHAT REAL TRACES BUY US THAT THE SYNTHETIC STUDY COULD NOT
----------------------------------------------------------
1. Real arrivals / runtimes / sizes (heavy-tailed, diurnal, bursty).
2. **Real user runtime estimates.** SWF field 9 (requested time) is what the
   user actually typed at submit. v3.3 had to *simulate* estimate error with
   the f-model (est = runtime * U(1,C), C=5). On these traces the real error is
   both larger and differently shaped: SDSC over-estimates by 6.4x at the
   median; LANL's median is only 1.5x but 36% of jobs UNDER-estimate, which the
   f-model cannot produce at all (it is over-estimate-only by construction).
   So classical estimate-driven policies are evaluated here on the estimates
   they would really receive, not on a favourable synthetic model.
3. A fidelity check: the recorded wait times are in the trace, so simulated
   FCFS can be compared against what the real machine actually did.

SIMULATION MODEL (and its documented limits)
--------------------------------------------
* Event-driven, exact to the second: state only changes at arrivals and
  completions, so the simulator steps between those instants instead of
  quantising time into ticks. No tick-rounding bias.
* Capacity-only allocator (`TraceCluster`): a job needs N processors out of
  MaxProcs. SWF records no per-node placement, so node-level fragmentation is
  NOT modelled -- inventing it would be fabricating data. This makes the
  simulated machine slightly *easier* to pack than the real one, which is
  exactly why the recorded-vs-simulated fidelity check is reported.
* Warm-up: each window simulates WARMUP_DAYS before the measured period, and
  metrics are computed only over jobs arriving in the measured period, so the
  cold empty-cluster start does not flatter any policy.
* Jobs requesting more than MaxProcs, or with non-positive runtime, are
  dropped (counted and reported).

POLICIES (12) -- the guide's question, asked on real data
----------------------------------------------------------
  FCFS              first-come first-served + unrestricted first-fit
  FIFO_STRICT       textbook FIFO; a blocked head blocks the queue
  SJF_ORACLE        shortest job first on TRUE runtime (unattainable bound)
  SJF_USEREST       shortest job first on the REAL user estimate  <- deployable
  HRRN_USEREST      highest response ratio next on the real estimate
  SMALLEST_FIRST    order by REQUESTED SIZE, no ML, no runtime information.
                    The control for the ranking-degeneracy result: the wait
                    model's per-job features are all functions of requested
                    size, so this policy is what PROACTIVE reduces to.
  EASY_USEREST      canonical two-condition EASY backfill, real estimates
                    (this is what production SLURM/EASY actually does)
  EASY_ORACLE       same, with true runtimes (estimate-quality upper bound)
  CONS_BF_USEREST   conservative backfill, real estimates
  SRPT_ORACLE       preemptive shortest-remaining-processing-time + checkpoint
                    cost (strongest preemptive classical bound)
  PROACTIVE         ML wait-time model RETRAINED ON THIS TRACE, ranking the
                    queue by predicted wait. Uses NO runtime information --
                    the regime v3.3 identified as its niche.
  PROACTIVE_EST     the same model retrained WITH the user estimate as an
                    extra feature. This is the decisive control: if ML adds
                    anything beyond the estimate that SJF already exploits,
                    it must show up as PROACTIVE_EST beating SJF_USEREST.

The ML model is retrained per trace on a strictly earlier time slice than any
evaluation window (chronological split, no leakage), i.e. the ML scheduler is
given its BEST case -- the project already showed zero-shot transfer fails.

Outputs (05_results/trace_schedulers/):
  trace_scheduler_windows.csv       per-window x per-scheduler raw metrics
  trace_scheduler_summary.csv       per-trace x per-scheduler means
  trace_scheduler_significance.csv  Holm-adjusted paired tests
  trace_estimate_quality.csv        real user-estimate error vs the f-model
  trace_scheduler_comparison.png    summary figure
  trace_fidelity.csv                recorded vs simulated FCFS waits

Usage:
  python trace_driven_benchmark.py                 # both traces, full run
  python trace_driven_benchmark.py --smoke         # 2 short windows, fast
  python trace_driven_benchmark.py --traces sdsc
"""

import argparse
import gzip
import io
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vizstyle import (figure, finish, save_both, bar_ends, color_of,  # noqa: E402
                      label_of, PALETTE)

from sjf_scheduler import order_queue as order_sjf, order_queue_estimated as order_sjf_est
from hrrn_scheduler import order_queue_estimated as order_hrrn_est
from size_scheduler import order_queue as order_smallest
from backfill_scheduler import easy_backfill_dispatch, conservative_backfill_dispatch
from simstats import gini, pairwise_significance, equivalence_table

# Optional diagnostic hook: callable(policy, queue, feature_matrix, predictions)
# invoked at every ML ranking decision. Set by ranking_degeneracy.py to
# instrument real dispatch instants without duplicating the simulation loop.
RANK_OBSERVER = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, '02_data')
OUT_DIR = os.path.join(PROJECT_ROOT, '05_results', 'trace_schedulers')
os.makedirs(OUT_DIR, exist_ok=True)

TRACES = {
    'sdsc': {
        'swf': 'SDSC-SP2-1998-4.2-cln.swf',
        'features_csv': 'real_trace_dataset_sdsc.csv',
        'label': 'SDSC SP2 (1998)',
    },
    'lanl': {
        'swf': 'LANL-CM5-1994-4.1-cln.swf',
        'features_csv': 'real_trace_dataset_lanl.csv',
        'label': 'LANL CM-5 (1994)',
    },
}

DAY = 86400
TRAIN_FRACTION = 0.60     # first 60% of trace time trains the wait model
N_WINDOWS = 20            # evaluation windows carved from the remaining 40%
WARMUP_DAYS = 3           # simulated but NOT measured (fills the cluster)
MEASURE_DAYS = 7          # measured period
PREEMPT_OVERHEAD = 60     # SRPT checkpoint/restore penalty, seconds
SLOWDOWN_TAU = 60         # bounded-slowdown floor, seconds (Feitelson convention)
SEED = 42

# Feature order used by the retrained per-trace model. These are exactly the
# columns build_real_trace_datasets.py reconstructs from the recorded schedule,
# and all of them are computable online inside the simulator.
BASE_FEATURES = ['job_procs', 'total_free', 'queue_length', 'running_jobs',
                 'can_fit_now', 'fit_ratio', 'queue_pressure', 'free_frac']
EST_FEATURE = 'est_runtime'


# ─────────────────────────────────────────────────────────────────────────────
# Trace loading
# ─────────────────────────────────────────────────────────────────────────────

KEEP_STATUS = (0, 1, -1)   # completed / failed / unknown; drop cancelled+partial


def open_swf(path):
    """Open an SWF trace, transparently falling back to the gzipped copy.

    Only the `.swf.gz` files are committed (`.gitignore` excludes `*.swf`), so
    a fresh clone has the compressed trace and not the expanded one. Reading
    either keeps the pipeline runnable straight after `git clone`.
    """
    if os.path.exists(path):
        return open(path, 'r', encoding='utf-8', errors='replace')
    gz = path if path.endswith('.gz') else path + '.gz'
    if os.path.exists(gz):
        return io.TextIOWrapper(gzip.open(gz, 'rb'), encoding='utf-8',
                                errors='replace')
    raise FileNotFoundError(
        f'Neither {path} nor {gz} exists. The Parallel Workloads Archive '
        f'traces ship with the repository as .swf.gz.')


def parse_swf_jobs(path):
    """Parse an SWF file into (capacity, DataFrame of jobs, drop stats).

    Keeps the same filter as build_real_trace_datasets.py so the scheduling
    study and the prediction study operate on the identical job population.
    Field 9 (index 8) is the user's REQUESTED time -- the real runtime
    estimate. A '-1' there means the trace did not record one.
    """
    capacity = None
    max_nodes = None
    rows = []
    stats = {'total': 0, 'bad_procs': 0, 'bad_runtime': 0, 'bad_wait': 0,
             'bad_status': 0, 'kept': 0, 'no_estimate': 0}

    with open_swf(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(';'):
                low = line.lower()
                if low.startswith('; maxprocs:'):
                    capacity = int(line.split(':')[1])
                elif low.startswith('; maxnodes:'):
                    max_nodes = int(line.split(':')[1])
                continue
            parts = line.split()
            if len(parts) < 11:
                continue
            stats['total'] += 1
            job_id = int(parts[0])
            submit = int(parts[1])
            wait = int(float(parts[2]))
            run = int(float(parts[3]))
            alloc = int(parts[4])
            req_procs = int(parts[7])
            req_time = float(parts[8])
            status = int(parts[10])

            procs = alloc if alloc > 0 else req_procs
            if procs <= 0:
                stats['bad_procs'] += 1
                continue
            if run <= 0:
                stats['bad_runtime'] += 1
                continue
            if wait < 0:
                stats['bad_wait'] += 1
                continue
            if status not in KEEP_STATUS:
                stats['bad_status'] += 1
                continue
            stats['kept'] += 1
            if req_time <= 0:
                stats['no_estimate'] += 1
            rows.append({'job_id': job_id, 'submit': submit,
                         'recorded_wait': wait, 'runtime': run,
                         'procs': procs, 'req_time': req_time})

    if capacity is None:
        capacity = max_nodes
    if capacity is None:
        raise ValueError(f'No MaxProcs/MaxNodes header found in {path}')

    df = pd.DataFrame(rows).sort_values(['submit', 'job_id']).reset_index(drop=True)

    # Missing estimate (-1): fall back to the trace's MEDIAN requested time.
    # Deliberately NOT the true runtime -- that would hand estimate-driven
    # policies a free oracle on those jobs (9% of LANL) and inflate them.
    present = df['req_time'] > 0
    median_req = float(df.loc[present, 'req_time'].median()) if present.any() else 3600.0
    df['est_runtime'] = np.where(present, df['req_time'], median_req)
    # An estimate below the trace's floor is meaningless for planning.
    df['est_runtime'] = df['est_runtime'].clip(lower=1.0)
    return capacity, df, stats


def estimate_quality_report(df, trace_key):
    """Characterise the REAL user-estimate error and contrast it with the
    f-model (est = runtime * U(1,C)) that the synthetic v3.3 study assumed."""
    present = df['req_time'] > 0
    d = df.loc[present]
    ratio = (d['est_runtime'] / d['runtime']).to_numpy()
    return {
        'trace': trace_key,
        'jobs': int(len(df)),
        'estimate_present_pct': float(100.0 * present.mean()),
        'ratio_median': float(np.median(ratio)),
        'ratio_mean': float(np.mean(ratio)),
        'ratio_p90': float(np.percentile(ratio, 90)),
        'under_estimate_frac': float((ratio < 1.0).mean()),
        'accurate_within_2x_frac': float(((ratio >= 0.5) & (ratio <= 2.0)).mean()),
        # f-model with C=5 (v3.3 default): factor ~ U(1,5)
        'fmodel_C5_ratio_median': 3.0,
        'fmodel_under_estimate_frac': 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Simulation primitives
# ─────────────────────────────────────────────────────────────────────────────

class TraceJob:
    __slots__ = ('job_id', 'arrival_time', 'num_gpus', 'runtime', 'est_runtime',
                 'start_time', 'end_time', 'allocated_nodes', 'remaining',
                 'measured')

    def __init__(self, job_id, arrival_time, num_gpus, runtime, est_runtime,
                 measured):
        self.job_id = job_id
        self.arrival_time = arrival_time
        self.num_gpus = num_gpus          # processors; name kept for policy reuse
        self.runtime = runtime
        self.est_runtime = est_runtime
        self.measured = measured          # arrived in the measured period?
        self.start_time = None
        self.end_time = None
        self.allocated_nodes = []
        self.remaining = float(runtime)


class TraceCluster:
    """Capacity-only cluster exposing the same interface the backfill helpers
    and the synthetic benchmark's Cluster expose (one node holding MaxProcs).

    SWF records no per-node placement, so modelling nodes would mean inventing
    fragmentation that is not in the data. A single-node capacity pool is the
    honest reading of the trace.
    """

    def __init__(self, capacity):
        self.num_nodes = 1
        self.gpus_per_node = capacity
        self.capacity = capacity
        self.nodes = [capacity]

    def total_free_gpus(self):
        return self.nodes[0]

    def allocate(self, job, t):
        if self.nodes[0] < job.num_gpus:
            return False
        self.nodes[0] -= job.num_gpus
        job.start_time = t
        job.end_time = t + job.runtime
        job.allocated_nodes = [(0, job.num_gpus)]
        return True

    def release(self, job):
        self.nodes[0] += job.num_gpus


def build_feature_matrix(queue, cluster, running, with_est):
    """Feature rows for every queued job, in queue order.

    Matches build_real_trace_datasets.replay_trace_features exactly: the
    features are evaluated with the scored job itself EXCLUDED from the queue
    aggregates, because in training each row was built at the job's submit
    instant, before it joined the queue.
    """
    total_free = cluster.total_free_gpus()
    capacity = cluster.capacity
    n_running = len(running)
    q_len = len(queue)
    total_queued_procs = 0
    for j in queue:
        total_queued_procs += j.num_gpus

    rows = np.empty((q_len, len(BASE_FEATURES) + (1 if with_est else 0)),
                    dtype=np.float64)
    for i, job in enumerate(queue):
        others_procs = total_queued_procs - job.num_gpus
        rows[i, 0] = job.num_gpus
        rows[i, 1] = total_free
        rows[i, 2] = q_len - 1
        rows[i, 3] = n_running
        rows[i, 4] = 1.0 if total_free >= job.num_gpus else 0.0
        rows[i, 5] = min(total_free / job.num_gpus, 1.0)
        rows[i, 6] = others_procs / (total_free + 1.0)
        rows[i, 7] = total_free / capacity
        if with_est:
            rows[i, 8] = job.est_runtime
    return rows


def rank_queue(queue, t, cluster, running, policy, model):
    """Return the queue in dispatch order for the ordering-based policies."""
    if policy in ('FCFS', 'FIFO_STRICT', 'EASY_USEREST', 'EASY_ORACLE',
                  'CONS_BF_USEREST'):   # arrival order; dispatch helpers rank
        return queue                       # arrival order; dispatch helpers rank
    if policy == 'SJF_ORACLE':
        return order_sjf(queue, t)
    if policy == 'SJF_USEREST':
        return order_sjf_est(queue, t)
    if policy == 'HRRN_USEREST':
        return order_hrrn_est(queue, t)
    if policy == 'SMALLEST_FIRST':
        return order_smallest(queue, t)
    if policy in ('PROACTIVE', 'PROACTIVE_EST'):
        with_est = (policy == 'PROACTIVE_EST')
        x = build_feature_matrix(queue, cluster, running, with_est)
        pred = model.predict(x)
        if RANK_OBSERVER is not None:
            RANK_OBSERVER(policy, queue, x, pred)
        order = sorted(range(len(queue)),
                       key=lambda i: (float(pred[i]), queue[i].arrival_time,
                                      queue[i].job_id))
        return [queue[i] for i in order]
    raise ValueError(f'unknown policy {policy}')


def summarize(completed, util_area, horizon, capacity, policy, window_meta,
              preemptions=0):
    """Metrics over MEASURED jobs only (warm-up arrivals are excluded)."""
    m = [j for j in completed if j.measured and j.end_time is not None]
    # Unified wait definition (v3.3): total delay = turnaround - true runtime.
    # Identical to start - arrival for non-preemptive policies; for SRPT it
    # honestly charges requeue time and checkpoint overhead as waiting.
    waits = np.array([j.end_time - j.arrival_time - j.runtime for j in m],
                     dtype=float)
    turnaround = np.array([j.end_time - j.arrival_time for j in m], dtype=float)
    runtimes = np.array([max(j.runtime, SLOWDOWN_TAU) for j in m], dtype=float)
    # Bounded slowdown (Feitelson et al.): turnaround / max(runtime, tau),
    # floored at 1. THE standard batch-scheduling metric -- mean wait alone
    # hides the fact that a 10-minute delay is catastrophic for a 1-minute job
    # and irrelevant for a 12-hour one.
    bsld = np.maximum(turnaround / runtimes, 1.0)

    out = {
        'scheduler': policy,
        'jobs_measured': int(len(m)),
        'mean_wait': float(waits.mean()) if len(waits) else float('nan'),
        'median_wait': float(np.median(waits)) if len(waits) else float('nan'),
        'p95_wait': float(np.percentile(waits, 95)) if len(waits) else float('nan'),
        'max_wait': float(waits.max()) if len(waits) else float('nan'),
        'mean_bounded_slowdown': float(bsld.mean()) if len(bsld) else float('nan'),
        'p95_bounded_slowdown': float(np.percentile(bsld, 95)) if len(bsld) else float('nan'),
        'mean_turnaround': float(turnaround.mean()) if len(turnaround) else float('nan'),
        'fairness_gini': gini(waits),
        'gpu_util': float(util_area / (capacity * horizon)) if horizon > 0 else 0.0,
        'preemptions': int(preemptions),
    }
    out.update(window_meta)
    return out


def simulate(jobs_in, policy, capacity, model, window_meta):
    """Event-driven non-preemptive simulation of one window under one policy."""
    jobs = [TraceJob(j.job_id, j.arrival_time, j.num_gpus, j.runtime,
                     j.est_runtime, j.measured) for j in jobs_in]
    cluster = TraceCluster(capacity)
    queue, running, completed = [], [], []
    n = len(jobs)
    ai = 0
    t = jobs[0].arrival_time
    util_area = 0.0
    t0 = t

    est_fn = None
    if policy in ('EASY_USEREST', 'CONS_BF_USEREST'):
        est_fn = lambda j: j.est_runtime

    while len(completed) < n:
        # 1) release completions due at t
        if running:
            still = []
            for job in running:
                if job.end_time <= t:
                    cluster.release(job)
                    completed.append(job)
                else:
                    still.append(job)
            running = still

        # 2) admit arrivals due at t
        while ai < n and jobs[ai].arrival_time <= t:
            queue.append(jobs[ai])
            ai += 1

        # 3) dispatch
        if queue:
            ordered = rank_queue(queue, t, cluster, running, policy, model)
            if policy in ('EASY_USEREST', 'EASY_ORACLE'):
                started = easy_backfill_dispatch(cluster, ordered, running, t,
                                                 est_runtime_of=est_fn)
            elif policy == 'CONS_BF_USEREST':
                started = conservative_backfill_dispatch(cluster, ordered,
                                                         running, t,
                                                         est_runtime_of=est_fn)
            else:
                started = []
                for job in ordered:
                    if cluster.total_free_gpus() < job.num_gpus:
                        if policy == 'FIFO_STRICT':
                            break      # textbook FIFO: blocked head blocks all
                        continue
                    if cluster.allocate(job, t):
                        started.append(job)
            if started:
                started_ids = {id(j) for j in started}
                queue = [j for j in queue if id(j) not in started_ids]
                running.extend(started)

        # 4) advance to the next event, integrating utilisation over the gap
        next_t = None
        if running:
            next_t = min(j.end_time for j in running)
        if ai < n:
            nxt_arrival = jobs[ai].arrival_time
            next_t = nxt_arrival if next_t is None else min(next_t, nxt_arrival)
        if next_t is None:
            # nothing running, no arrivals left, but queue non-empty =>
            # only possible if a job can never fit; filtered upstream.
            break
        used = capacity - cluster.total_free_gpus()
        util_area += used * (next_t - t)
        t = next_t

    return summarize(completed, util_area, max(t - t0, 1), capacity, policy,
                     window_meta)


def simulate_srpt(jobs_in, capacity, window_meta, overhead=PREEMPT_OVERHEAD):
    """Event-driven preemptive SRPT with a checkpoint/restore penalty.

    Exact despite being event-driven: all running jobs' remaining times shrink
    at the same rate, so the remaining-time ORDER can only change when a job
    arrives or completes -- i.e. exactly at the events we step through.
    """
    jobs = [TraceJob(j.job_id, j.arrival_time, j.num_gpus, j.runtime,
                     j.est_runtime, j.measured) for j in jobs_in]
    n = len(jobs)
    ai = 0
    queued, running, completed = [], [], []
    t = jobs[0].arrival_time
    t0 = t
    util_area = 0.0
    preemptions = 0

    while len(completed) < n:
        # 1) completions
        if running:
            still = []
            for job in running:
                if job.remaining <= 1e-9:
                    job.end_time = t
                    completed.append(job)
                else:
                    still.append(job)
            running = still

        # 2) arrivals
        while ai < n and jobs[ai].arrival_time <= t:
            queued.append(jobs[ai])
            ai += 1

        # 3) SRPT selection: greedy pack by remaining work over all active jobs
        active = running + queued
        active.sort(key=lambda j: (j.remaining, j.arrival_time, j.job_id))
        free = capacity
        selected = set()
        for job in active:
            if job.num_gpus <= free:
                selected.add(id(job))
                free -= job.num_gpus

        new_running, new_queued = [], []
        for job in running:
            if id(job) in selected:
                new_running.append(job)
            else:
                job.remaining += overhead        # checkpoint/restore penalty
                preemptions += 1
                new_queued.append(job)
        for job in queued:
            if id(job) in selected:
                if job.start_time is None:
                    job.start_time = t
                new_running.append(job)
            else:
                new_queued.append(job)
        running, queued = new_running, new_queued

        # 4) next event
        next_t = None
        if running:
            next_t = t + min(j.remaining for j in running)
        if ai < n:
            nxt_arrival = jobs[ai].arrival_time
            next_t = nxt_arrival if next_t is None else min(next_t, nxt_arrival)
        if next_t is None:
            break
        dt = next_t - t
        for job in running:
            job.remaining -= dt
        util_area += sum(j.num_gpus for j in running) * dt
        t = next_t

    return summarize(completed, util_area, max(t - t0, 1), capacity,
                     'SRPT_ORACLE', window_meta, preemptions=preemptions)


# ─────────────────────────────────────────────────────────────────────────────
# Per-trace model training (chronological, no leakage)
# ─────────────────────────────────────────────────────────────────────────────

def train_trace_model(trace_key, split_time, swf_df, with_est, capacity):
    """Retrain the wait-time model on the trace's own EARLY period.

    Features come from build_real_trace_datasets.py, which reconstructs the
    cluster state at each job's submit instant by replaying the RECORDED
    schedule. Only rows with submit_time < split_time are used, and every
    evaluation window starts after split_time, so no evaluation job -- and no
    cluster state influenced by one -- is ever seen in training.

    Target is log1p(wait seconds): the wait distribution is heavy-tailed, and
    the scheduler only consumes the RANKING, which any monotone target
    preserves.
    """
    path = os.path.join(DATA_DIR, TRACES[trace_key]['features_csv'])
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        # The reconstructed-feature CSVs are gitignored (they are large and
        # derived), so rebuild them in memory from the trace itself using the
        # same replay used to produce them. Keeps a fresh clone runnable.
        sys.path.insert(0, os.path.join(PROJECT_ROOT, '02_data'))
        from build_real_trace_datasets import replay_trace_features
        jobs = [{'job_id': int(r.job_id), 'submit': int(r.submit),
                 'wait': int(r.recorded_wait), 'run': int(r.runtime),
                 'procs': int(r.procs)} for r in swf_df.itertuples(index=False)]
        df, _ = replay_trace_features(jobs, capacity)
    df = df[df['submit_time'] < split_time]
    if with_est:
        df = df.merge(swf_df[['job_id', 'est_runtime']], on='job_id', how='inner')
    feats = BASE_FEATURES + ([EST_FEATURE] if with_est else [])
    x = df[feats].to_numpy(dtype=np.float64)
    y = np.log1p(df['wait_time'].clip(lower=0).to_numpy(dtype=np.float64))
    model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08,
                         subsample=0.9, colsample_bytree=0.9,
                         random_state=SEED, n_jobs=4, verbosity=0)
    model.fit(x, y)
    return model, len(df)


# ─────────────────────────────────────────────────────────────────────────────
# Window carving and the per-trace driver
# ─────────────────────────────────────────────────────────────────────────────

def carve_windows(df, split_time, n_windows, warmup_days, measure_days):
    """Evenly spaced, non-overlapping windows over the post-split period.

    Evenly spaced -- NOT the busiest windows. Selecting on load would be
    cherry-picking; instead every window's offered load is reported so the
    result can be read conditionally on congestion.
    """
    t_end = int(df['submit'].max())
    span = warmup_days + measure_days
    total_needed = n_windows * span * DAY
    available = t_end - split_time
    if available < total_needed:
        n_windows = max(1, int(available // (span * DAY)))
    stride = (available - span * DAY) / max(n_windows - 1, 1) if n_windows > 1 else 0

    windows = []
    for k in range(n_windows):
        w0 = int(split_time + k * stride)
        measure_start = w0 + warmup_days * DAY
        w1 = measure_start + measure_days * DAY
        sel = df[(df['submit'] >= w0) & (df['submit'] < w1)]
        if len(sel) < 50:
            continue
        windows.append((k + 1, w0, measure_start, w1, sel))
    return windows


def make_jobs(sel, w0, measure_start, capacity):
    """Window jobs with window-relative arrival times; oversized jobs dropped."""
    jobs, dropped = [], 0
    for row in sel.itertuples(index=False):
        if row.procs > capacity or row.runtime <= 0:
            dropped += 1
            continue
        jobs.append(TraceJob(
            job_id=int(row.job_id),
            arrival_time=int(row.submit - w0),
            num_gpus=int(row.procs),
            runtime=int(row.runtime),
            est_runtime=float(row.est_runtime),
            measured=bool(row.submit >= measure_start),
        ))
    jobs.sort(key=lambda j: (j.arrival_time, j.job_id))
    return jobs, dropped


POLICIES = ['FCFS', 'FIFO_STRICT', 'SJF_ORACLE', 'SJF_USEREST', 'HRRN_USEREST',
            'SMALLEST_FIRST', 'EASY_USEREST', 'EASY_ORACLE', 'CONS_BF_USEREST',
            'SRPT_ORACLE', 'PROACTIVE', 'PROACTIVE_EST']


def run_trace(trace_key, n_windows, warmup_days, measure_days, skip_consbf):
    meta = TRACES[trace_key]
    swf_path = os.path.join(DATA_DIR, meta['swf'])
    print(f"\n{'='*74}\n{meta['label']}  --  {meta['swf']}\n{'='*74}")

    capacity, df, pstats = parse_swf_jobs(swf_path)
    kept_pct = 100.0 * pstats['kept'] / max(pstats['total'], 1)
    print(f"Capacity (MaxProcs)   : {capacity}")
    print(f"Parsed / kept         : {pstats['total']} / {pstats['kept']} ({kept_pct:.1f}%)")
    print(f"  dropped cancelled   : {pstats['bad_status']}")
    print(f"  no user estimate    : {pstats['no_estimate']} "
          f"({100.0*pstats['no_estimate']/max(pstats['kept'],1):.1f}% -> median fallback)")

    eq = estimate_quality_report(df, trace_key)
    print(f"Real user estimates   : median {eq['ratio_median']:.2f}x runtime, "
          f"{100*eq['under_estimate_frac']:.1f}% UNDER-estimate, "
          f"{100*eq['accurate_within_2x_frac']:.1f}% within 2x "
          f"(f-model C=5: 3.00x median, 0.0% under-estimates)")

    t_min, t_max = int(df['submit'].min()), int(df['submit'].max())
    split_time = int(t_min + TRAIN_FRACTION * (t_max - t_min))
    print(f"Chronological split   : train < day "
          f"{(split_time - t_min)/DAY:.0f}, evaluate after")

    print('Retraining wait model on the trace\'s own early period ...')
    model_base, n_train = train_trace_model(trace_key, split_time, df, False, capacity)
    model_est, _ = train_trace_model(trace_key, split_time, df, True, capacity)
    print(f'  trained on {n_train} pre-split jobs '
          f'({len(BASE_FEATURES)} features; +1 with user estimate)')

    windows = carve_windows(df, split_time, n_windows, warmup_days, measure_days)
    print(f'Evaluation windows    : {len(windows)} x '
          f'({warmup_days}d warm-up + {measure_days}d measured)\n')

    policies = [p for p in POLICIES
                if not (skip_consbf and p == 'CONS_BF_USEREST')]
    rows, fidelity = [], []

    for widx, w0, measure_start, w1, sel in windows:
        jobs, dropped = make_jobs(sel, w0, measure_start, capacity)
        if not jobs:
            continue
        n_measured = sum(1 for j in jobs if j.measured)
        # Offered load = processor-seconds demanded / processor-seconds available
        demand = sum(j.num_gpus * j.runtime for j in jobs)
        horizon = (warmup_days + measure_days) * DAY
        offered_load = demand / (capacity * horizon)
        wmeta = {'trace': trace_key, 'window': widx, 'capacity': capacity,
                 'jobs_total': len(jobs), 'jobs_dropped_oversized': dropped,
                 'offered_load': offered_load}

        print(f'  window {widx:02d} | {len(jobs):5d} jobs '
              f'({n_measured} measured) | offered load {offered_load:.2f}')

        for policy in policies:
            model = model_est if policy == 'PROACTIVE_EST' else model_base
            if policy == 'SRPT_ORACLE':
                out = simulate_srpt(jobs, capacity, wmeta)
            else:
                out = simulate(jobs, policy, capacity, model, wmeta)
            rows.append(out)
            print(f"      {policy:16s} wait={out['mean_wait']/60:9.1f} min  "
                  f"bsld={out['mean_bounded_slowdown']:8.2f}  "
                  f"util={out['gpu_util']:.2f}")

        # Fidelity: what the REAL machine did on exactly these measured jobs
        rec = sel[sel['submit'] >= measure_start]['recorded_wait']
        sim_fcfs = [r for r in rows if r['window'] == widx
                    and r['scheduler'] == 'FCFS' and r['trace'] == trace_key]
        fidelity.append({
            'trace': trace_key, 'window': widx,
            'offered_load': offered_load,
            'recorded_mean_wait_min': float(rec.mean() / 60.0) if len(rec) else float('nan'),
            'recorded_median_wait_min': float(rec.median() / 60.0) if len(rec) else float('nan'),
            'sim_fcfs_mean_wait_min': sim_fcfs[0]['mean_wait'] / 60.0 if sim_fcfs else float('nan'),
        })

    return pd.DataFrame(rows), eq, pd.DataFrame(fidelity)


# ─────────────────────────────────────────────────────────────────────────────

def make_figure(summary, stem):
    """Trace-driven results as a light/dark pair.

    Colour follows the ENTITY, not the panel or the rank: a policy keeps its
    colour in every panel of every figure in the repo (an earlier version drew
    the same scheduler blue in one panel and green in the next, so the reader
    could not track it across metrics). Emphasis encoding — blue is the ML
    method, orange is the ML-free control it is being tested against, everything
    classical recedes to neutral grey. Oracle policies are marked in the tick
    LABEL rather than given a fourth hue, because the text already says it.
    """
    traces = list(summary['trace'].unique())
    metrics = [('mean_wait', 'Mean wait (minutes)', 60.0),
               ('mean_bounded_slowdown', 'Mean bounded slowdown', 1.0)]

    for mode in ('light', 'dark'):
        p = PALETTE[mode]
        fig, axes = figure(mode, figsize=(15.5, 5.1 * len(traces)),
                           nrows=len(traces), ncols=2, squeeze=False,
                           gridspec_kw={'wspace': 0.52, 'hspace': 0.36})
        for r, tr in enumerate(traces):
            for c, (col, xlabel, div) in enumerate(metrics):
                ax = axes[r][c]
                sub = summary[summary['trace'] == tr].sort_values(col)
                vals = (sub[col] / div).to_numpy()
                names = [label_of(s) for s in sub['scheduler']]
                colors = [color_of(s, mode) for s in sub['scheduler']]
                y = np.arange(len(sub))
                ax.barh(y, vals, height=0.68, color=colors)
                ax.set_yticks(y)
                ax.set_yticklabels(names, fontsize=8.6)
                ax.set_xlabel(xlabel)
                ax.set_title(f'{TRACES[tr]["label"]} — '
                             f'{"mean wait" if c == 0 else "bounded slowdown"}',
                             loc='left', fontsize=10.5)
                bar_ends(ax, 'h')
                ax.invert_yaxis()
                span = vals.max() if len(vals) else 1.0
                for yi, v in zip(y, vals):
                    ax.text(v + span * 0.012, yi, f'{v:,.1f}', va='center',
                            fontsize=8, color=p['ink_2'])
                ax.set_xlim(0, span * 1.16)
        # Figure-level legend in the title block: inside an axes it overlapped
        # the longest bar (strict FIFO) and clipped its value label.
        fig.legend(
            handles=[Patch(facecolor=p['series_1'], label='Proactive (ML wait-time model)'),
                     Patch(facecolor=p['series_2'], label='Smallest-first (no ML)'),
                     Patch(facecolor=p['muted'], label='Classical baselines')],
            loc='upper left', bbox_to_anchor=(0.011, 0.912), ncol=3,
            fontsize=9, frameon=False, handlelength=1.4, columnspacing=1.8)

        finish(fig, mode,
               title='On real workloads the ML scheduler lands where an ML-free '
                     'size sort lands',
               subtitle='Twelve policies replayed through two Parallel Workloads '
                        'Archive traces, 20 paired 7-day windows each at offered '
                        'load ≈0.70,\nusing the real user runtime estimates the '
                        'traces record. Lower is better in both metrics.',
               source='05_results/trace_schedulers/trace_scheduler_summary.csv — '
                      '04_scheduler/trace_driven_benchmark.py',
               title_y=0.992, subtitle_y=0.972, source_y=0.004)
        fig.subplots_adjust(top=0.845, bottom=0.058, left=0.175, right=0.985)
        save_both(fig, stem, mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', nargs='*', default=list(TRACES),
                    choices=list(TRACES))
    ap.add_argument('--windows', type=int, default=N_WINDOWS)
    ap.add_argument('--warmup-days', type=int, default=WARMUP_DAYS)
    ap.add_argument('--measure-days', type=int, default=MEASURE_DAYS)
    ap.add_argument('--skip-consbf', action='store_true',
                    help='skip conservative backfill (slowest policy)')
    ap.add_argument('--smoke', action='store_true',
                    help='2 short windows per trace for a fast sanity run. NOTE: '
                         'this OVERWRITES the full results in 05_results/'
                         'trace_schedulers/ — re-run without --smoke afterwards')
    ap.add_argument('--figures-only', action='store_true',
                    help='re-render the figures from the saved summary CSV '
                         'without re-running any simulation')
    args = ap.parse_args()

    if args.figures_only:
        summary = pd.read_csv(os.path.join(OUT_DIR, 'trace_scheduler_summary.csv'))
        make_figure(summary, os.path.join(OUT_DIR, 'trace_scheduler_comparison'))
        print('Re-rendered figures from existing CSV in', OUT_DIR)
        return

    if args.smoke:
        args.windows, args.warmup_days, args.measure_days = 2, 1, 2

    np.random.seed(SEED)

    all_rows, all_eq, all_fid = [], [], []
    for key in args.traces:
        rows, eq, fid = run_trace(key, args.windows, args.warmup_days,
                                  args.measure_days, args.skip_consbf)
        all_rows.append(rows)
        all_eq.append(eq)
        all_fid.append(fid)

    runs = pd.concat(all_rows, ignore_index=True)
    runs_path = os.path.join(OUT_DIR, 'trace_scheduler_windows.csv')
    runs.to_csv(runs_path, index=False)

    summary = (runs.groupby(['trace', 'scheduler'], as_index=False)
               .agg({'mean_wait': 'mean', 'median_wait': 'mean',
                     'p95_wait': 'mean', 'max_wait': 'mean',
                     'mean_bounded_slowdown': 'mean',
                     'p95_bounded_slowdown': 'mean',
                     'mean_turnaround': 'mean', 'fairness_gini': 'mean',
                     'gpu_util': 'mean', 'preemptions': 'mean',
                     'jobs_measured': 'mean', 'offered_load': 'mean'})
               .sort_values(['trace', 'mean_wait']))
    summary_path = os.path.join(OUT_DIR, 'trace_scheduler_summary.csv')
    summary.to_csv(summary_path, index=False)

    sig_frames = []
    for key in args.traces:
        sub = runs[runs['trace'] == key]
        for metric in ('mean_wait', 'mean_bounded_slowdown'):
            s = pairwise_significance(sub, references=('PROACTIVE', 'EASY_USEREST'),
                                      metric=metric, unit='window')
            s.insert(0, 'trace', key)
            sig_frames.append(s)
    sig = pd.concat(sig_frames, ignore_index=True)
    sig_path = os.path.join(OUT_DIR, 'trace_scheduler_significance.csv')
    sig.to_csv(sig_path, index=False)

    # Equivalence (TOST). A non-significant difference test does NOT establish
    # that two policies perform the same; these are the claims that need it.
    EQUIV_PAIRS = [
        ('SMALLEST_FIRST', 'PROACTIVE'),   # is the ML pipeline redundant?
        ('PROACTIVE_EST', 'SJF_USEREST'),  # does ML add anything over SJF?
        ('PROACTIVE', 'FCFS'),             # does ML beat doing nothing?
    ]
    eq_frames = []
    for key in args.traces:
        sub = runs[runs['trace'] == key]
        for metric in ('mean_wait', 'mean_bounded_slowdown'):
            e = equivalence_table(sub, EQUIV_PAIRS, metric=metric,
                                  unit='window', margin_frac=0.10)
            e.insert(0, 'trace', key)
            eq_frames.append(e)
    equiv = pd.concat(eq_frames, ignore_index=True)
    equiv_path = os.path.join(OUT_DIR, 'trace_scheduler_equivalence.csv')
    equiv.to_csv(equiv_path, index=False)

    eq_path = os.path.join(OUT_DIR, 'trace_estimate_quality.csv')
    pd.DataFrame(all_eq).to_csv(eq_path, index=False)

    fid_path = os.path.join(OUT_DIR, 'trace_fidelity.csv')
    pd.concat(all_fid, ignore_index=True).to_csv(fid_path, index=False)

    fig_stem = os.path.join(OUT_DIR, 'trace_scheduler_comparison')
    fig_path = fig_stem + '.png'
    make_figure(summary, fig_stem)

    print('\n' + '=' * 74)
    print('TRACE-DRIVEN SUMMARY (mean over windows; wait in minutes)')
    print('=' * 74)
    disp = summary.copy()
    disp['mean_wait_min'] = disp['mean_wait'] / 60.0
    disp['p95_wait_min'] = disp['p95_wait'] / 60.0
    print(disp[['trace', 'scheduler', 'mean_wait_min', 'p95_wait_min',
                'mean_bounded_slowdown', 'fairness_gini', 'gpu_util']]
          .to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    print('\n' + '=' * 74)
    print('EQUIVALENCE TESTS (paired TOST, margin = 10% of the reference)')
    print('=' * 74)
    print(equiv[['trace', 'scheduler_a', 'scheduler_b', 'metric', 'pct_diff',
                 'ci_low', 'ci_high', 'margin', 'p_tost', 'equivalent']]
          .to_string(index=False, float_format=lambda v: f'{v:.4g}'))

    for p in (runs_path, summary_path, sig_path, equiv_path, eq_path, fid_path,
              fig_path):
        print('Saved:', p)


if __name__ == '__main__':
    main()

import os
import random
import pickle
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.neural_network import MLPRegressor

from sjf_scheduler import order_queue as order_sjf, order_queue_estimated as order_sjf_est
from priority_scheduler import order_queue as order_priority
from hrrn_scheduler import order_queue as order_hrrn
from size_scheduler import order_queue as order_smallest
from neural_network_scheduler import order_queue as order_nn, build_feature_vector
from backfill_scheduler import easy_backfill_dispatch, conservative_backfill_dispatch
from simstats import gini, holm_correction, pairwise_significance, equivalence_table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vizstyle import (figure, finish, save_both, bar_ends, color_of,  # noqa: E402
                      label_of, PALETTE)

# Optional diagnostic hook: callable(policy, queue, feature_matrix, predictions)
# invoked at every PROACTIVE ranking decision. Set by ranking_degeneracy.py to
# instrument real dispatch instants without duplicating the simulation loop.
RANK_OBSERVER = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, '03_models', 'wait_model_v2.pkl')
DATA_PATH = os.path.join(PROJECT_ROOT, '02_data', 'improved_wait_dataset.csv')
OUT_DIR = os.path.join(PROJECT_ROOT, '05_results', 'schedulers')
os.makedirs(OUT_DIR, exist_ok=True)

SIM_TIME = 300
NUM_RUNS = 20
NUM_NODES = 8
GPUS_PER_NODE = 4
CAPACITY = NUM_NODES * GPUS_PER_NODE

# User runtime-estimate model ("f-model", Mu'alem & Feitelson 2001):
# est = runtime * (1 + (C-1)*u), u ~ U(0,1)  =>  factor uniform on [1, C].
# C=5 means users over-estimate by 3x on average — the empirically dominant
# regime (real estimates inflate up to an order of magnitude). C=1 recovers
# the perfect-information oracle; estimate_sensitivity.py sweeps C.
EST_OVER_FACTOR = 5.0
# Seed base for the estimate RNG. MUST be a dedicated Random instance so the
# global stream that generates the paired workloads stays untouched (existing
# scheduler results remain bit-identical).
EST_SEED_BASE = 20000

# SRPT preemption cost: checkpoint/restore penalty in ticks added to a job's
# remaining work each time it is preempted.
PREEMPT_OVERHEAD = 1

with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)
wait_model = bundle['model']
FEATURES = bundle['features']

class Job:
    def __init__(self, job_id, arrival_time, num_gpus, runtime, priority_score,
                 est_runtime=None):
        self.job_id = job_id
        self.arrival_time = arrival_time
        self.num_gpus = num_gpus
        self.runtime = runtime
        self.priority_score = priority_score
        # user-supplied runtime estimate (assign_estimates); None until set
        self.est_runtime = est_runtime
        self.start_time = None
        self.end_time = None
        self.allocated_nodes = []

class Cluster:
    def __init__(self, num_nodes, gpus_per_node):
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.nodes = [gpus_per_node] * num_nodes

    def total_free_gpus(self):
        return sum(self.nodes)

    def allocate(self, job, t):
        need = job.num_gpus
        alloc = []
        for i in range(self.num_nodes):
            if need <= 0:
                break
            free = self.nodes[i]
            if free > 0:
                used = min(free, need)
                self.nodes[i] -= used
                alloc.append((i, used))
                need -= used
        if need == 0:
            job.start_time = t
            job.end_time = t + job.runtime
            job.allocated_nodes = alloc
            return True
        for nid, used in alloc:
            self.nodes[nid] += used
        return False

    def release(self, job):
        for nid, used in job.allocated_nodes:
            self.nodes[nid] += used

def get_features(job, cluster, queue, running):
    # `queue` INCLUDES the scored job at every call site here, but the training
    # rows (generate_improved_dataset.extract_features) are snapshotted at the
    # job's arrival, BEFORE it joins the queue. queue_length and queue_pressure
    # therefore exclude the job itself, matching the trained semantics (the
    # trace-side build_feature_matrix does the same).
    tf = cluster.total_free_gpus()
    return np.array([
        job.num_gpus,
        tf,
        len(queue) - 1,
        len(running),
        max(cluster.nodes),
        float(np.var(cluster.nodes)),
        int(tf >= job.num_gpus),
        min(tf / (job.num_gpus + 1e-6), 1.0),
        float(np.std(cluster.nodes)),
        (sum(q.num_gpus for q in queue) - job.num_gpus) / (tf + 1),
        sum(1 for n in cluster.nodes if n >= job.num_gpus) / cluster.num_nodes,
        tf / cluster.num_nodes,
    ])

def generate_jobs(n=110):
    jobs = []
    for i in range(n):
        jobs.append(Job(
            i,
            random.randint(0, SIM_TIME // 2),
            random.randint(1, 8),
            random.randint(5, 20),
            random.randint(1, 10),
        ))
    return sorted(jobs, key=lambda j: j.arrival_time)

def assign_estimates(jobs, rng, over_factor=EST_OVER_FACTOR):
    """Attach user runtime estimates: est = runtime * (1 + (C-1)*u), u~U(0,1).

    Takes a dedicated `rng` (random.Random) so the global workload stream is
    untouched. Drawing u once and scaling by (C-1) couples estimates across
    over_factor values — the sensitivity sweep varies estimate QUALITY, not
    the noise realisation.
    """
    for job in jobs:
        u = rng.random()
        job.est_runtime = job.runtime * (1.0 + (over_factor - 1.0) * u)

def train_nn_predictor():
    df = pd.read_csv(DATA_PATH)
    x = df[FEATURES].values
    y = df['wait_time'].values
    nn = MLPRegressor(hidden_layer_sizes=(24, 12), max_iter=400, random_state=42)
    nn.fit(x, y)
    return nn

def rank_queue(queue, t, cluster, running, scheduler, nn_model):
    if scheduler in ('fifo', 'fifo_strict', 'backfill', 'backfill_est',
                     'cons_bf', 'proactive_bf'):
        # arrival order. NOTE: 'fifo' dispatches ANY fitting job per tick
        # (FCFS + unrestricted first-fit backfilling — the label predates
        # v3.3 and is kept for continuity); 'fifo_strict' blocks on the
        # queue head (textbook FIFO). The backfill dispatch helpers handle
        # reservation + backfill scan themselves.
        return queue
    if scheduler == 'sjf':
        return order_sjf(queue, t)
    if scheduler == 'sjf_est':
        return order_sjf_est(queue, t)
    if scheduler == 'priority':
        return order_priority(queue, t)
    if scheduler == 'hrrn':
        return order_hrrn(queue, t)
    if scheduler == 'smallest':
        return order_smallest(queue, t)
    if scheduler == 'proactive':
        x = np.array([get_features(j, cluster, queue, running) for j in queue])
        pred = wait_model.predict(x)
        if RANK_OBSERVER is not None:
            RANK_OBSERVER('proactive', queue, x, pred)
        idx = sorted(range(len(queue)),
                     key=lambda i: (float(pred[i]), queue[i].arrival_time,
                                    queue[i].job_id))
        return [queue[i] for i in idx]
    return order_nn(queue, cluster, running, nn_model)

def summarize(completed, util, mae_samples, preemptions=0):
    # Unified wait definition: total delay = turnaround - true runtime.
    # For non-preemptive schedulers this equals start - arrival exactly
    # (end = start + runtime), preserving pre-v3.3 numbers bit-for-bit.
    # For preemptive SRPT it honestly counts post-preemption requeue time
    # AND checkpoint-overhead ticks as waiting (time-to-first-dispatch
    # would hide ~29% of SRPT's queueing delay).
    waits = [j.end_time - j.arrival_time - j.runtime
             for j in completed if j.end_time is not None]
    turnarounds = [j.end_time - j.arrival_time for j in completed if j.end_time is not None]
    # bounded slowdown, standard backfill-literature metric: turnaround over
    # true runtime (bound tau=1 tick), >= 1 by construction
    slowdowns = [max((j.end_time - j.arrival_time) / max(j.runtime, 1), 1.0)
                 for j in completed if j.end_time is not None]
    return {
        'mean_wait': float(np.mean(waits)) if waits else float('nan'),
        'max_wait': float(np.max(waits)) if waits else float('nan'),
        'p95_wait': float(np.percentile(waits, 95)) if waits else float('nan'),
        'mean_turnaround': float(np.mean(turnarounds)) if turnarounds else float('nan'),
        'mean_bounded_slowdown': float(np.mean(slowdowns)) if slowdowns else float('nan'),
        'throughput': len(completed) / SIM_TIME,
        'fairness_gini': gini(waits),
        'gpu_util': float(np.mean(util)) if util else 0.0,
        'preemptions': int(preemptions),
        # wait-prediction MAE at dispatch time; NaN for non-ML schedulers
        'pred_wait_mae': float(np.mean(mae_samples)) if mae_samples else float('nan'),
    }

def run_once(jobs_in, scheduler, nn_model):
    cluster = Cluster(NUM_NODES, GPUS_PER_NODE)
    jobs = [Job(j.job_id, j.arrival_time, j.num_gpus, j.runtime, j.priority_score,
                est_runtime=j.est_runtime) for j in jobs_in]
    queue, running, completed = [], [], []
    util, mae_samples = [], []

    for t in range(SIM_TIME):
        for job in running[:]:
            if job.end_time == t:
                cluster.release(job)
                running.remove(job)
                completed.append(job)

        for job in jobs:
            if job.arrival_time == t:
                queue.append(job)

        if queue:
            queue = rank_queue(queue, t, cluster, running, scheduler, nn_model)

        if scheduler in ('backfill', 'backfill_est', 'proactive_bf'):
            # EASY backfill: arrival-order head with a reservation guarantee
            # (anti-starvation); backfill jobs may never delay the head.
            # 'backfill_est' plans with user estimates instead of true runtimes.
            bf_order = None
            hook = None
            est_fn = (lambda j: j.est_runtime) if scheduler == 'backfill_est' else None
            if scheduler == 'proactive_bf':
                # Backfill scan ordered by wait_model_v2 predicted wait
                # (shortest first); head stays arrival order.
                def bf_order(jobs_list, active_jobs, _t=t):
                    scored = []
                    for j in jobs_list:
                        pw = float(wait_model.predict(get_features(j, cluster, jobs_list, active_jobs).reshape(1, -1))[0])
                        scored.append((pw, j.arrival_time, j.job_id, j))
                    scored.sort(key=lambda s: s[:3])
                    return [s[3] for s in scored]

                # Predicted wait at the dispatch decision point, from the
                # PRE-allocation cluster state (same convention as PROACTIVE).
                def hook(job, cl, pending, active, _t=t):
                    pred = float(wait_model.predict(get_features(job, cl, pending, active).reshape(1, -1))[0])
                    mae_samples.append(abs(pred - (_t - job.arrival_time)))

            started = easy_backfill_dispatch(cluster, queue, running, t,
                                             backfill_order=bf_order,
                                             pre_dispatch_hook=hook,
                                             est_runtime_of=est_fn)
            for job in started:
                queue.remove(job)
                running.append(job)
        elif scheduler == 'cons_bf':
            # Conservative backfill: reservation for EVERY queued job in
            # arrival order (perfect estimates — strongest version).
            started = conservative_backfill_dispatch(cluster, queue, running, t)
            for job in started:
                queue.remove(job)
                running.append(job)
        else:
            for job in queue[:]:
                if cluster.total_free_gpus() < job.num_gpus and scheduler == 'fifo_strict':
                    break   # textbook FIFO: a blocked head blocks the queue
                if cluster.total_free_gpus() >= job.num_gpus:
                    # Predicted wait at the dispatch decision point, from the
                    # PRE-allocation cluster state (only ML schedulers predict
                    # wait times; the others have no comparable prediction).
                    pred = float('nan')
                    if scheduler == 'proactive':
                        pred = float(wait_model.predict(get_features(job, cluster, queue, running).reshape(1, -1))[0])
                    elif scheduler == 'nn':
                        pred = float(nn_model.predict(build_feature_vector(job, cluster, queue, running).reshape(1, -1))[0])

                    if cluster.allocate(job, t):
                        running.append(job)
                        queue.remove(job)
                        if not np.isnan(pred):
                            actual_wait = job.start_time - job.arrival_time
                            mae_samples.append(abs(pred - actual_wait))

        util.append((CAPACITY - cluster.total_free_gpus()) / CAPACITY)

    return summarize(completed, util, mae_samples)

def run_once_preemptive(jobs_in, overhead=PREEMPT_OVERHEAD):
    """Preemptive SRPT (shortest remaining processing time) with a checkpoint
    penalty: every tick the jobs with the least remaining work get the GPUs
    (greedy pack in remaining-time order); a running job that loses its slot
    is preempted and pays `overhead` extra ticks of remaining work. Runtime
    oracle (true remaining time) — the strongest preemptive classical
    baseline. Deterministic: ties broken by (remaining, arrival, id), so the
    selection is stable across ticks and jobs do not ping-pong.
    """
    jobs = [Job(j.job_id, j.arrival_time, j.num_gpus, j.runtime, j.priority_score,
                est_runtime=j.est_runtime) for j in jobs_in]
    for j in jobs:
        j.remaining = float(j.runtime)
    queued, running, completed = [], [], []
    util = []
    preemptions = 0

    for t in range(SIM_TIME):
        # completions first (a job that ran its last tick at t-1 releases now,
        # matching the non-preemptive convention end_time = start + runtime)
        for job in running[:]:
            if job.remaining <= 0:
                job.end_time = t
                running.remove(job)
                completed.append(job)

        for job in jobs:
            if job.arrival_time == t:
                queued.append(job)

        # SRPT selection: greedy pack by remaining work over ALL active jobs
        candidates = sorted(running + queued,
                            key=lambda j: (j.remaining, j.arrival_time, j.job_id))
        free = CAPACITY
        selected = set()
        for job in candidates:
            if job.num_gpus <= free:
                selected.add(job.job_id)
                free -= job.num_gpus

        for job in running[:]:
            if job.job_id not in selected:
                running.remove(job)
                queued.append(job)
                job.remaining += overhead   # checkpoint/restore penalty
                preemptions += 1
        for job in queued[:]:
            if job.job_id in selected:
                queued.remove(job)
                running.append(job)
                if job.start_time is None:
                    job.start_time = t      # first dispatch, defines wait

        for job in running:
            job.remaining -= 1.0
        util.append(sum(j.num_gpus for j in running) / CAPACITY)

    return summarize(completed, util, [], preemptions=preemptions)

def main():
    nn_model = train_nn_predictor()
    schedulers = ['fifo', 'fifo_strict', 'sjf', 'sjf_est', 'priority', 'hrrn',
                  'smallest', 'proactive', 'nn',
                  'backfill', 'backfill_est', 'cons_bf', 'proactive_bf',
                  'srpt']
    rows = []

    for run in range(NUM_RUNS):
        # per-run seed: paired workloads, every scheduler sees the same jobs
        # Eval seeds MUST differ from the model's training-data seeds (42+i in
        # generate_improved_dataset.py) or the benchmark evaluates on training
        # workloads. 1000+run also restores comparability with prior results.
        random.seed(1000 + run)
        np.random.seed(1000 + run)
        jobs = generate_jobs()
        # runtime estimates from a DEDICATED rng: global stream untouched, so
        # pre-v3.3 scheduler results reproduce bit-identically
        assign_estimates(jobs, random.Random(EST_SEED_BASE + run))
        for sch in schedulers:
            if sch == 'srpt':
                out = run_once_preemptive(jobs)
            else:
                out = run_once(jobs, sch, nn_model)
            out['run'] = run + 1
            out['scheduler'] = sch.upper()
            rows.append(out)
            print(f"Run {run+1:02d} | {sch.upper():12s} wait={out['mean_wait']:.2f} throughput={out['throughput']:.3f}")

    df = pd.DataFrame(rows)
    runs_path = os.path.join(OUT_DIR, 'multi_scheduler_runs.csv')
    df.to_csv(runs_path, index=False)

    summary = df.groupby('scheduler', as_index=False).agg({
        'pred_wait_mae': 'mean',
        'mean_wait': 'mean',
        'max_wait': 'mean',
        'p95_wait': 'mean',
        'mean_turnaround': 'mean',
        'mean_bounded_slowdown': 'mean',
        'fairness_gini': 'mean',
        'throughput': 'mean',
        'gpu_util': 'mean',
        'preemptions': 'mean',
    }).sort_values('mean_wait')

    summary_path = os.path.join(OUT_DIR, 'multi_scheduler_benchmark.csv')
    summary.to_csv(summary_path, index=False)

    sig = pairwise_significance(df)
    sig_path = os.path.join(OUT_DIR, 'multi_scheduler_significance.csv')
    sig.to_csv(sig_path, index=False)

    # Equivalence (TOST): a non-significant difference is not sameness. The
    # SMALLEST-vs-PROACTIVE pair is the ranking-degeneracy claim's test.
    equiv = pd.concat(
        [equivalence_table(df, [('SMALLEST', 'PROACTIVE'),
                                ('NN', 'PROACTIVE'),
                                ('SMALLEST', 'NN')],
                           metric=m, unit='run', margin_frac=0.10)
         for m in ('mean_wait', 'mean_bounded_slowdown')],
        ignore_index=True)
    equiv_path = os.path.join(OUT_DIR, 'multi_scheduler_equivalence.csv')
    equiv.to_csv(equiv_path, index=False)

    # Previously this plotted mean wait (12-26 ts) and throughput (0.367
    # jobs/ts) as adjacent bars on ONE axis, which rendered every throughput bar
    # invisible. Throughput and utilisation are in fact IDENTICAL across all 14
    # policies here (the workload and capacity are fixed; only the order
    # changes), so they belong in the subtitle as a stated fact, not in a chart.
    # The two panels now carry the two metrics that actually differ.
    plot_stem = os.path.join(OUT_DIR, 'scheduler_comparison')
    metrics = [('mean_wait', 'Mean wait (timesteps)'),
               ('mean_bounded_slowdown', 'Mean bounded slowdown')]
    for mode in ('light', 'dark'):
        p = PALETTE[mode]
        fig, axes = figure(mode, figsize=(14, 6.4), ncols=2,
                           gridspec_kw={'wspace': 0.42})
        for ax, (col, xlabel) in zip(axes, metrics):
            sub = summary.sort_values(col)
            vals = sub[col].to_numpy()
            y = np.arange(len(sub))
            ax.barh(y, vals, height=0.68,
                    color=[color_of(s, mode) for s in sub['scheduler']])
            ax.set_yticks(y)
            ax.set_yticklabels([label_of(s) for s in sub['scheduler']], fontsize=9)
            ax.set_xlabel(xlabel)
            bar_ends(ax, 'h')
            ax.invert_yaxis()
            span = vals.max()
            for yi, v in zip(y, vals):
                ax.text(v + span * 0.012, yi, f'{v:.2f}', va='center',
                        fontsize=8.2, color=p['ink_2'])
            ax.set_xlim(0, span * 1.14)
        fig.legend(handles=[
            Patch(facecolor=p['series_1'], label='Proactive / NN (ML)'),
            Patch(facecolor=p['series_2'], label='Smallest-first (no ML)'),
            Patch(facecolor=p['muted'], label='Classical baselines')],
            loc='upper left', bbox_to_anchor=(0.011, 0.885), ncol=3, fontsize=9,
            frameon=False, handlelength=1.4, columnspacing=1.8)
        finish(fig, mode,
               title='Synthetic benchmark: the ML scheduler and a one-line size '
                     'sort are statistically equivalent',
               subtitle=f'{NUM_RUNS} paired seeded runs, {len(summary)} policies. '
                        'Throughput and GPU utilisation are identical for every '
                        'policy here (same workload, same capacity —\nonly the '
                        'queue order changes), so only the two metrics that '
                        'differ are plotted. Lower is better in both.',
               source='05_results/schedulers/multi_scheduler_benchmark.csv — '
                      '04_scheduler/multi_scheduler_benchmark.py',
               title_y=0.985, subtitle_y=0.945)
        fig.subplots_adjust(top=0.80, bottom=0.085, left=0.19, right=0.985)
        save_both(fig, plot_stem, mode)
    plot_path = plot_stem + '.png'

    print('\n=== Multi-scheduler summary ===')
    print(summary.to_string(index=False, formatters={
        'pred_wait_mae': '{:.3f}'.format,
        'mean_wait': '{:.3f}'.format,
        'max_wait': '{:.3f}'.format,
        'p95_wait': '{:.3f}'.format,
        'mean_turnaround': '{:.3f}'.format,
        'mean_bounded_slowdown': '{:.3f}'.format,
        'fairness_gini': '{:.3f}'.format,
        'throughput': '{:.3f}'.format,
        'gpu_util': '{:.3f}'.format,
        'preemptions': '{:.1f}'.format,
    }))
    print('\n=== Paired significance (mean wait, Holm-adjusted) ===')
    print(sig.to_string(index=False, formatters={
        'mean_wait_sched': '{:.3f}'.format,
        'mean_wait_ref': '{:.3f}'.format,
        'mean_diff': '{:+.3f}'.format,
        'pct_vs_ref': '{:+.2f}'.format,
        'cohens_dz': '{:+.3f}'.format,
        'ttest_p': '{:.3g}'.format,
        'wilcoxon_p': '{:.3g}'.format,
        'ttest_p_holm': '{:.3g}'.format,
        'wilcoxon_p_holm': '{:.3g}'.format,
    }))
    print('\n=== Equivalence (paired TOST, margin = 10% of the reference) ===')
    print(equiv[['scheduler_a', 'scheduler_b', 'metric', 'mean_a', 'mean_b',
                 'pct_diff', 'ci_low', 'ci_high', 'margin', 'p_tost',
                 'equivalent']]
          .to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    print('Saved:', runs_path)
    print('Saved:', summary_path)
    print('Saved:', sig_path)
    print('Saved:', equiv_path)
    print('Saved:', plot_path)

if __name__ == '__main__':
    main()

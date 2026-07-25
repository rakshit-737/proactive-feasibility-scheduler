import os
import sys
import random
import pickle
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from vizstyle import (figure, finish, save_both, bar_ends, legend_roles,  # noqa: E402
                      color_of, label_of, PALETTE)

MODEL_PATH = os.path.join(PROJECT_ROOT, '03_models', 'wait_model_v2.pkl')
OUT_FAIR = os.path.join(PROJECT_ROOT, '05_results', 'fairness')
OUT_SCHED = os.path.join(PROJECT_ROOT, '05_results', 'schedulers')
os.makedirs(OUT_FAIR, exist_ok=True)
os.makedirs(OUT_SCHED, exist_ok=True)

SIM_TIME = 300
NUM_RUNS = 20
NUM_NODES = 8
GPUS_PER_NODE = 4

with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)
model = bundle['model']

class Job:
    def __init__(self, job_id, arrival_time, num_gpus, runtime):
        self.job_id = job_id
        self.arrival_time = arrival_time
        self.num_gpus = num_gpus
        self.runtime = runtime
        self.start_time = None
        self.end_time = None
        self.allocated_nodes = []

class Cluster:
    def __init__(self, num_nodes=NUM_NODES, gpus_per_node=GPUS_PER_NODE):
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
                used = min(need, free)
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

def gini(values):
    x = np.array([v for v in values if v >= 0], dtype=float)
    if len(x) == 0 or np.sum(x) == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    c = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(c) / c[-1]) / n)

def get_features(job, cluster, queue, running):
    # `queue` includes the scored job at dispatch time; training rows exclude
    # it (snapshot at arrival, before enqueue), so queue_length/queue_pressure
    # subtract the job's own contribution to match the trained semantics.
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

def size_bucket(job):
    if job.num_gpus <= 2:
        return 'small'
    if job.num_gpus <= 5:
        return 'medium'
    return 'large'

def generate_jobs(n=110):
    return sorted([
        Job(i, random.randint(0, SIM_TIME // 2), random.randint(1, 8), random.randint(5, 20)) for i in range(n)
    ], key=lambda j: j.arrival_time)

def run_scheduler(jobs_in, mode='fifo'):
    cluster = Cluster()
    jobs = [Job(j.job_id, j.arrival_time, j.num_gpus, j.runtime) for j in jobs_in]
    queue, running, done = [], [], []

    for t in range(SIM_TIME):
        for j in running[:]:
            if j.end_time == t:
                cluster.release(j)
                running.remove(j)
                done.append(j)

        for j in jobs:
            if j.arrival_time == t:
                queue.append(j)

        if queue and mode.startswith('proactive'):
            scored = [(float(model.predict(get_features(j, cluster, queue, running).reshape(1, -1))[0]), j) for j in queue]
            scored.sort(key=lambda x: (x[0], x[1].arrival_time))
            queue = [j for _, j in scored]

        if mode == 'proactive_starvation':
            starving = [j for j in queue if (t - j.arrival_time) > (3 * j.runtime)]
            non_starving = [j for j in queue if (t - j.arrival_time) <= (3 * j.runtime)]
            queue = starving + non_starving

        for j in queue[:]:
            if cluster.total_free_gpus() >= j.num_gpus and cluster.allocate(j, t):
                running.append(j)
                queue.remove(j)

    waits = [j.start_time - j.arrival_time for j in done if j.start_time is not None]
    starvation_count = sum(1 for j in done if (j.start_time - j.arrival_time) > (3 * j.runtime))

    by_size = []
    for grp in ['small', 'medium', 'large']:
        grp_jobs = [j for j in done if size_bucket(j) == grp]
        completed = len(grp_jobs)
        by_size.append({
            'scheduler': mode,
            'job_size': grp,
            'completed_jobs': completed,
            'completion_rate': completed / max(1, len([x for x in jobs if size_bucket(x) == grp])),
            'mean_wait': float(np.mean([j.start_time - j.arrival_time for j in grp_jobs])) if grp_jobs else np.nan,
        })

    return {
        'scheduler': mode,
        'mean_wait': float(np.mean(waits)) if waits else np.nan,
        'max_wait': float(np.max(waits)) if waits else np.nan,
        'starvation_count': starvation_count,
        'fairness_gini': gini(waits),
        'completed_jobs': len(done),
        'waits': waits,
        'by_size': by_size,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Presentation only: how each policy in this experiment is coloured and named.
#
# Colour follows the ENTITY, not the panel: FIFO is the grey classical baseline
# it is in every other figure in the repo, and BOTH proactive variants are the
# ML policy, so both take the ML blue. The starvation-guard variant is told
# apart by its tick label, not by a fourth hue -- the same convention vizstyle
# uses for oracle policies, where the label already carries the identity.
# ─────────────────────────────────────────────────────────────────────────────
POLICY_KEY = {
    'fifo': 'FIFO',
    'proactive': 'PROACTIVE',
    'proactive_starvation': 'PROACTIVE',
}
POLICY_TICK = {
    'fifo': label_of('FIFO'),
    'proactive': label_of('PROACTIVE'),
    'proactive_starvation': 'Proactive + starvation guard',
}
SIZE_PANEL = {
    'small': 'Small jobs (1-2 GPUs)',
    'medium': 'Medium jobs (3-5 GPUs)',
    'large': 'Large jobs (6-8 GPUs)',
}


def make_size_figure(by_size_df, stem):
    """Mean wait per job-size bucket, as small multiples sharing one wait scale.

    One panel per size bucket and one bar per policy, so a policy keeps a single
    colour across all three panels; the shared x-axis keeps the cross-panel
    comparison honest. Only the 'large' panel is direct-labelled -- that bucket
    is where the ML ordering moves the delay to, and labelling all nine bars
    would just be a table drawn as a chart.
    """
    order = ['fifo', 'proactive', 'proactive_starvation']
    sizes = ['small', 'medium', 'large']

    def mean_wait(sched, size):
        row = by_size_df[(by_size_df['scheduler'] == sched) & (by_size_df['job_size'] == size)]
        return float(row['mean_wait'].iloc[0]) if len(row) else np.nan

    values = {size: [mean_wait(s, size) for s in order] for size in sizes}
    vmax = float(np.nanmax([v for vals in values.values() for v in vals]))

    light_path = None
    for mode in ('light', 'dark'):
        p = PALETTE[mode]
        fig, axes = figure(mode, figsize=(13, 4.4), ncols=3, sharex=True, sharey=True,
                           gridspec_kw={'wspace': 0.07})
        y = np.arange(len(order))
        colors = [color_of(POLICY_KEY[s], mode) for s in order]

        for ax, size in zip(axes, sizes):
            vals = values[size]
            ax.barh(y, vals, height=0.62, color=colors)
            ax.set_title(SIZE_PANEL[size], loc='left')
            ax.set_xlim(0, vmax * 1.18)
            bar_ends(ax, 'h')
            if size == 'large':
                for yi, v in zip(y, vals):
                    if np.isfinite(v):
                        ax.text(v + vmax * 0.015, yi, f'{v:.1f}', va='center',
                                fontsize=8.5, color=p['ink_2'])

        axes[0].set_yticks(y)
        axes[0].set_yticklabels([POLICY_TICK[s] for s in order], fontsize=9)
        axes[0].invert_yaxis()
        axes[1].set_xlabel('mean wait before start (simulation time steps)')
        legend_roles(axes[0], mode, roles=('ml', 'baseline'), loc='lower right')

        finish(fig, mode,
               title='Ordering the queue by predicted wait moves the delay onto the '
                     'largest jobs',
               subtitle='Mean queueing delay per job-size bucket, averaged over '
                        f'{NUM_RUNS} seeded runs of a {NUM_NODES}x{GPUS_PER_NODE}-GPU '
                        'cluster. Small and medium jobs start almost immediately '
                        'under the ML order;\nthe large bucket absorbs the wait. '
                        'The starvation guard hands part of it back.',
               source='05_results/fairness/completion_by_size.csv')
        fig.subplots_adjust(top=0.73, bottom=0.16, left=0.155, right=0.99)
        path = save_both(fig, stem, mode)
        if mode == 'light':
            light_path = path
    return light_path


def main():
    modes = ['fifo', 'proactive', 'proactive_starvation']
    rows, size_rows, all_wait_rows = [], [], []

    for run in range(NUM_RUNS):
        random.seed(800 + run)
        np.random.seed(800 + run)
        jobs = generate_jobs()
        for mode in modes:
            out = run_scheduler(jobs, mode)
            rows.append({k: v for k, v in out.items() if k not in ('waits', 'by_size')})
            size_rows.extend([{**x, 'run': run + 1} for x in out['by_size']])
            all_wait_rows.extend([{'scheduler': mode, 'run': run + 1, 'wait_time': w} for w in out['waits']])
            print(f"Run {run+1:02d} | {mode:20s} mean_wait={out['mean_wait']:.2f} gini={out['fairness_gini']:.3f}")

    summary = pd.DataFrame(rows).groupby('scheduler', as_index=False).mean(numeric_only=True)
    by_size_df = pd.DataFrame(size_rows).groupby(['scheduler', 'job_size'], as_index=False).mean(numeric_only=True)
    waits_df = pd.DataFrame(all_wait_rows)

    metrics_csv_1 = os.path.join(OUT_FAIR, 'fairness_metrics.csv')
    metrics_csv_2 = os.path.join(OUT_SCHED, 'fairness_metrics.csv')
    by_size_csv = os.path.join(OUT_FAIR, 'completion_by_size.csv')
    summary.to_csv(metrics_csv_1, index=False)
    summary.to_csv(metrics_csv_2, index=False)
    by_size_df.to_csv(by_size_csv, index=False)

    plot_path = make_size_figure(
        by_size_df, os.path.join(OUT_FAIR, 'wait_time_distribution_by_size'))

    print('\n=== Fairness summary ===')
    print(summary.to_string(index=False, formatters={
        'mean_wait': '{:.3f}'.format,
        'max_wait': '{:.3f}'.format,
        'fairness_gini': '{:.3f}'.format,
        'completed_jobs': '{:.2f}'.format,
        'starvation_count': '{:.2f}'.format,
    }))
    print('Saved:', metrics_csv_1)
    print('Saved:', by_size_csv)
    print('Saved:', plot_path)

if __name__ == '__main__':
    main()

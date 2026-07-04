# ─────────────────────────────────────────────────────────────────────────────
# fairness_budget_sweep.py
#
# Bounded-fairness wait-budget sweep (Pareto frontier).
#
# Policy "proactive with hard wait-budget B": each tick, BEFORE dispatch, the
# queue is partitioned:
#   ESCALATED = jobs whose current wait (t - arrival_time) > B
#               -> kept in arrival order, placed at the queue HEAD
#   REST      = everything else
#               -> sorted by wait_model_v2 predicted wait, ascending
#
# Special cases:
#   B = None  -> nothing ever escalates: pure proactive ordering.
#   B = 0     -> every job that has waited at least one tick escalates, so the
#               queue head is arrival-ordered and the policy degenerates to
#               FIFO. (Only nuance: jobs arriving on the current tick have
#               wait == 0, which is NOT > 0, so same-tick arrivals are ordered
#               by predicted wait for their first tick before escalating. A
#               separately simulated true-FIFO baseline is run on the same
#               seeds for reference deltas.)
#
# Dispatch follows the repo convention (see proactive_Schedule_v2.py /
# multi_scheduler_benchmark.py): release -> arrivals -> order -> dispatch ALL
# jobs that fit this tick, greedy allocation across per-node GPU lists.
#
# Outputs:
#   05_results/fairness/budget_sweep.csv   (one row per budget B)
#   05_results/fairness/budget_pareto.png  (mean vs max wait; mean vs Gini)
# ─────────────────────────────────────────────────────────────────────────────

import os
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths (PROJECT_ROOT pattern used across the repo) ────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, '03_models', 'wait_model_v2.pkl')
OUT_DIR = os.path.join(PROJECT_ROOT, '05_results', 'fairness')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Full experiment config ───────────────────────────────────────────────────
SIM_TIME = 300
NUM_RUNS = 20
NUM_NODES = 8
GPUS_PER_NODE = 4
NUM_JOBS = 110
CAPACITY = NUM_NODES * GPUS_PER_NODE
BUDGETS = [0, 10, 20, 30, 40, 60, 80, 120, None]   # None = pure proactive
STARVATION_FACTOR = 3.0                            # starved: wait > 3x run mean

# ── Load wait_model_v2 (clean 12-feature model) ──────────────────────────────
with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)
wait_model = bundle['model']
FEATURES = bundle['features']
print(f"Loaded wait_model_v2 | features: {FEATURES}")

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
    """Feature extraction — must exactly match generate_improved_dataset.py."""
    tf = cluster.total_free_gpus()
    return [
        job.num_gpus,
        tf,
        len(queue),
        len(running),
        max(cluster.nodes),
        float(np.var(cluster.nodes)),
        int(tf >= job.num_gpus),
        min(tf / (job.num_gpus + 1e-6), 1.0),
        float(np.std(cluster.nodes)),
        sum(q.num_gpus for q in queue) / (tf + 1),
        sum(1 for n in cluster.nodes if n >= job.num_gpus) / cluster.num_nodes,
        tf / cluster.num_nodes,
    ]

def gini(vals):
    x = np.array([v for v in vals if v >= 0], dtype=float)
    if len(x) == 0 or np.sum(x) == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)

def generate_jobs(num_jobs, max_time):
    jobs = []
    for i in range(num_jobs):
        arrival_time = random.randint(0, max_time // 2)
        num_gpus = random.randint(1, 8)
        runtime = random.randint(5, 20)
        jobs.append(Job(i, arrival_time, num_gpus, runtime))
    return sorted(jobs, key=lambda x: x.arrival_time)

def order_queue_budget(queue, t, cluster, running, budget):
    """Wait-budget policy: escalated jobs (wait > B) at the head in arrival
    order; the rest sorted by predicted wait ascending. budget=None => pure
    proactive (no escalation)."""
    if budget is None:
        escalated, rest = [], list(queue)
    else:
        escalated = [j for j in queue if (t - j.arrival_time) > budget]
        rest = [j for j in queue if (t - j.arrival_time) <= budget]
    escalated.sort(key=lambda j: (j.arrival_time, j.job_id))
    if rest:
        feats = np.array([get_features(j, cluster, queue, running) for j in rest])
        preds = wait_model.predict(feats)
        scored = sorted(zip(preds, rest),
                        key=lambda x: (x[0], x[1].arrival_time, x[1].job_id))
        rest = [j for _, j in scored]
    return escalated + rest

def run_once(jobs_in, policy, budget=None):
    """policy: 'fifo' (true FIFO reference) or 'budget' (proactive with hard
    wait-budget; budget=None = pure proactive). Returns per-job wait list."""
    cluster = Cluster(NUM_NODES, GPUS_PER_NODE)
    jobs = [Job(j.job_id, j.arrival_time, j.num_gpus, j.runtime) for j in jobs_in]
    queue, running, completed = [], [], []

    for t in range(SIM_TIME):
        for job in running[:]:
            if job.end_time == t:
                cluster.release(job)
                running.remove(job)
                completed.append(job)

        for job in jobs:
            if job.arrival_time == t:
                queue.append(job)

        if queue and policy == 'budget':
            queue = order_queue_budget(queue, t, cluster, running, budget)

        # Dispatch ALL fitting jobs each tick (repo convention)
        for job in queue[:]:
            if cluster.total_free_gpus() >= job.num_gpus:
                if cluster.allocate(job, t):
                    running.append(job)
                    queue.remove(job)

    waits = [j.start_time - j.arrival_time for j in completed if j.start_time is not None]
    return waits

def wait_metrics(waits):
    w = np.array(waits, dtype=float)
    m = float(np.mean(w))
    return {
        'mean_wait': m,
        'p95_wait': float(np.percentile(w, 95)),
        'max_wait': float(np.max(w)),
        'gini': gini(waits),
        'starvation': int(np.sum(w > STARVATION_FACTOR * m)),
    }

def budget_label(b):
    return 'None' if b is None else str(b)

def main():
    # per-run metrics: fifo_runs[run] and budget_runs[B][run]
    fifo_runs = []
    budget_runs = {b: [] for b in BUDGETS}

    for run in range(NUM_RUNS):
        # per-run seed so runs differ but the whole set is reproducible;
        # every policy sees the SAME job trace (paired runs)
        # Out-of-training seeds (training data uses 42+i) — avoids evaluating
        # the wait model on the exact workloads it was trained on.
        random.seed(5000 + run)
        np.random.seed(5000 + run)
        jobs = generate_jobs(NUM_JOBS, SIM_TIME)

        fifo_m = wait_metrics(run_once(jobs, 'fifo'))
        fifo_runs.append(fifo_m)

        parts = [f"Run {run + 1:2d}: FIFO mean={fifo_m['mean_wait']:.2f} max={fifo_m['max_wait']:.0f}"]
        for b in BUDGETS:
            m = wait_metrics(run_once(jobs, 'budget', budget=b))
            budget_runs[b].append(m)
            parts.append(f"B={budget_label(b)}: mean={m['mean_wait']:.2f} max={m['max_wait']:.0f}")
        print(' | '.join(parts))

    # ── Aggregate across runs ────────────────────────────────────────────────
    fifo_mean_waits = np.array([m['mean_wait'] for m in fifo_runs])
    fifo_agg = {k: float(np.mean([m[k] for m in fifo_runs]))
                for k in ('mean_wait', 'p95_wait', 'max_wait', 'gini', 'starvation')}

    rows = []
    for b in BUDGETS:
        runs = budget_runs[b]
        mean_waits = np.array([m['mean_wait'] for m in runs])
        # paired per-run improvement vs FIFO on the same seed
        impr = (fifo_mean_waits - mean_waits) / fifo_mean_waits * 100.0
        rows.append({
            'budget': budget_label(b),
            'mean_wait': float(np.mean(mean_waits)),
            'mean_wait_std': float(np.std(mean_waits)),
            'p95_wait': float(np.mean([m['p95_wait'] for m in runs])),
            'max_wait': float(np.mean([m['max_wait'] for m in runs])),
            'gini': float(np.mean([m['gini'] for m in runs])),
            'starvation': float(np.mean([m['starvation'] for m in runs])),
            'mean_improvement_vs_fifo_pct': float(np.mean(impr)),
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, 'budget_sweep.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')

    print('\n=== FIFO reference (same seeds) ===')
    print(f"mean={fifo_agg['mean_wait']:.3f} p95={fifo_agg['p95_wait']:.3f} "
          f"max={fifo_agg['max_wait']:.3f} gini={fifo_agg['gini']:.3f} "
          f"starvation={fifo_agg['starvation']:.2f}")
    print('\n=== Budget sweep summary (mean over runs) ===')
    print(df.to_string(index=False, formatters={
        'mean_wait': '{:.3f}'.format,
        'mean_wait_std': '{:.3f}'.format,
        'p95_wait': '{:.3f}'.format,
        'max_wait': '{:.3f}'.format,
        'gini': '{:.4f}'.format,
        'starvation': '{:.2f}'.format,
        'mean_improvement_vs_fifo_pct': '{:.2f}'.format,
    }))

    # ── Pareto plot ─────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    order = df.sort_values('mean_wait')
    for ax, ycol, ylabel in ((ax1, 'max_wait', 'Max wait (ticks)'),
                             (ax2, 'gini', 'Gini coefficient of waits')):
        ax.plot(order['mean_wait'], order[ycol], '-', color='#94a3b8',
                zorder=1, linewidth=1.2)
        ax.scatter(df['mean_wait'], df[ycol], s=55, color='#38bdf8',
                   zorder=2, label='wait-budget policy')
        for _, r in df.iterrows():
            is_endpoint = r['budget'] == 'None'
            ax.annotate(f"B={r['budget']}", (r['mean_wait'], r[ycol]),
                        textcoords='offset points', xytext=(6, 6), fontsize=8,
                        fontweight='bold' if is_endpoint else 'normal')
        # highlight endpoints: pure proactive (B=None) and true FIFO reference
        pp = df[df['budget'] == 'None'].iloc[0]
        ax.scatter([pp['mean_wait']], [pp[ycol]], s=140, facecolors='none',
                   edgecolors='#34d399', linewidths=2.2, zorder=3,
                   label='pure proactive (B=None)')
        ax.scatter([fifo_agg['mean_wait']], [fifo_agg[ycol]],
                   s=140, marker='s', color='#f87171', zorder=3,
                   label='FIFO (reference)')
        ax.annotate('FIFO', (fifo_agg['mean_wait'], fifo_agg[ycol]),
                    textcoords='offset points', xytext=(6, -12), fontsize=8,
                    fontweight='bold', color='#b91c1c')
        ax.set_xlabel('Mean wait (ticks)')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    ax1.set_title('Wait-budget Pareto frontier: mean vs max wait')
    ax2.set_title('Mean wait vs fairness (Gini)')
    ax1.legend(fontsize=8)
    plt.suptitle(f'Bounded-fairness wait-budget sweep '
                 f'({NUM_RUNS} paired runs, {NUM_JOBS} jobs, '
                 f'{NUM_NODES}x{GPUS_PER_NODE} GPUs)', y=1.00)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'budget_pareto.png')
    plt.savefig(plot_path, dpi=160)

    print('\nSaved:', csv_path)
    print('Saved:', plot_path)

if __name__ == '__main__':
    main()

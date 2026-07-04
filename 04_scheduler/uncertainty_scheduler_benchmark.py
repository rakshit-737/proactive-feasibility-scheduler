import os
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths / config ───────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POINT_MODEL_PATH = os.path.join(PROJECT_ROOT, '03_models', 'wait_model_v2.pkl')
QUANT_MODEL_PATH = os.path.join(PROJECT_ROOT, '03_models', 'wait_model_quantile.pkl')
OUT_DIR = os.path.join(PROJECT_ROOT, '05_results', 'uncertainty')
os.makedirs(OUT_DIR, exist_ok=True)

SIM_TIME = 300
NUM_RUNS = 10               # paired runs per scenario
GPUS_PER_NODE = 4
BASE_JOBS = 110
ARRIVAL_WINDOW = 150        # in-dist arrival window (= SIM_TIME // 2, repo convention)

# in_dist matches the training regime; the shifted grid crosses arrival-load
# multipliers {0.5, 2.0} (job count scaled) with cluster sizes {4, 16} nodes.
SCENARIOS = [
    {'name': 'in_dist',        'nodes': 8,  'arrival_mult': 1.0},
    {'name': 'arr0.5_nodes4',  'nodes': 4,  'arrival_mult': 0.5},
    {'name': 'arr0.5_nodes16', 'nodes': 16, 'arrival_mult': 0.5},
    {'name': 'arr2.0_nodes4',  'nodes': 4,  'arrival_mult': 2.0},
    {'name': 'arr2.0_nodes16', 'nodes': 16, 'arrival_mult': 2.0},
]
POLICIES = ['FIFO', 'PROACTIVE', 'UCB', 'GUARDED']

# ── Load models ──────────────────────────────────────────────────────────────
with open(POINT_MODEL_PATH, 'rb') as f:
    point_bundle = pickle.load(f)
point_model = point_bundle['model']
FEATURES = point_bundle['features']

with open(QUANT_MODEL_PATH, 'rb') as f:
    q_bundle = pickle.load(f)
SPREAD_TAU = q_bundle['spread_tau']
print(f"Loaded wait_model_v2 (point) + wait_model_quantile "
      f"(holdout coverage={q_bundle['holdout_coverage']*100:.1f}%, spread_tau={SPREAD_TAU:.3f})")

def predict_quantiles(x_mat):
    """Return (n, 3) array [q10, q50, q90], sorted per row to prevent crossing."""
    if q_bundle['vector_alpha']:
        p = q_bundle['model'].predict(x_mat)
    else:
        p = np.column_stack([m.predict(x_mat) for m in q_bundle['models']])
    return np.sort(p, axis=1)

# ── Simulator (repo conventions: discrete ticks, dispatch-all-that-fit) ──────
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

def generate_jobs(n, arrival_window):
    jobs = []
    for i in range(n):
        jobs.append(Job(
            i,
            random.randint(0, arrival_window),
            random.randint(1, 8),
            random.randint(5, 20),
        ))
    return sorted(jobs, key=lambda j: j.arrival_time)

def feature_matrix(queue, cluster, running):
    """Batch feature extraction — matches generate_improved_dataset.py exactly."""
    tf = cluster.total_free_gpus()
    max_free = max(cluster.nodes)
    var_free = float(np.var(cluster.nodes))
    frag = float(np.std(cluster.nodes))
    q_gpus = sum(q.num_gpus for q in queue)
    q_press = q_gpus / (tf + 1)
    avg_free = tf / cluster.num_nodes
    rows = []
    for job in queue:
        rows.append([
            job.num_gpus, tf, len(queue), len(running),
            max_free, var_free,
            int(tf >= job.num_gpus),
            min(tf / (job.num_gpus + 1e-6), 1.0),
            frag,
            q_press,
            sum(1 for n in cluster.nodes if n >= job.num_gpus) / cluster.num_nodes,
            avg_free,
        ])
    return np.array(rows)

def order_queue(queue, cluster, running, policy, stats):
    """Return the dispatch order for this tick. `queue` itself stays in
    arrival (FIFO) order so the GUARDED fallback is a genuine FIFO tick."""
    if policy == 'FIFO' or not queue:
        return list(queue)

    x = feature_matrix(queue, cluster, running)
    if policy == 'PROACTIVE':
        scores = point_model.predict(x)
    else:
        q = predict_quantiles(x)
        q10, q50, q90 = q[:, 0], q[:, 1], q[:, 2]
        if policy == 'UCB':
            scores = q90
        else:  # GUARDED
            stats['decision_ticks'] += 1
            spread = (q90 - q10) / np.maximum(q50, 1.0)
            if float(np.mean(spread)) > SPREAD_TAU:
                stats['fallback_ticks'] += 1
                return list(queue)      # spread too wide -> trust FIFO this tick
            scores = q50

    scored = list(zip(scores, queue))
    scored.sort(key=lambda p: (p[0], p[1].arrival_time, p[1].job_id))
    return [j for _, j in scored]

def run_once(jobs_in, policy, num_nodes):
    cluster = Cluster(num_nodes, GPUS_PER_NODE)
    jobs = [Job(j.job_id, j.arrival_time, j.num_gpus, j.runtime) for j in jobs_in]
    queue, running, completed = [], [], []
    stats = {'decision_ticks': 0, 'fallback_ticks': 0}

    for t in range(SIM_TIME):
        for job in running[:]:
            if job.end_time == t:
                cluster.release(job)
                running.remove(job)
                completed.append(job)

        for job in jobs:
            if job.arrival_time == t:
                queue.append(job)

        ordered = order_queue(queue, cluster, running, policy, stats)

        # Dispatch ALL fitting jobs each tick (same throughput as FIFO baseline)
        for job in ordered:
            if cluster.total_free_gpus() >= job.num_gpus:
                if cluster.allocate(job, t):
                    running.append(job)
                    queue.remove(job)

    # Per-job waits. Jobs never started are right-censored at SIM_TIME
    # (wait >= SIM_TIME - arrival); censored values are used only for the
    # paired job-vs-FIFO failure comparison, not for mean/max wait.
    started = {j.job_id: j.start_time - j.arrival_time
               for j in jobs if j.start_time is not None}
    all_waits = {j.job_id: (j.start_time - j.arrival_time if j.start_time is not None
                            else SIM_TIME - j.arrival_time) for j in jobs}
    waits = list(started.values())
    return {
        'mean_wait': float(np.mean(waits)) if waits else float('nan'),
        'max_wait': float(np.max(waits)) if waits else float('nan'),
        'n_started': len(started),
        'job_waits': all_waits,
        'fallback_tick_pct': (100.0 * stats['fallback_ticks'] / stats['decision_ticks']
                              if stats['decision_ticks'] else 0.0),
    }

# ── Benchmark ────────────────────────────────────────────────────────────────
def main():
    rows = []
    for sc in SCENARIOS:
        n_jobs = int(round(BASE_JOBS * sc['arrival_mult']))
        print(f"\n--- Scenario {sc['name']} | nodes={sc['nodes']} jobs={n_jobs} ---")
        for run in range(NUM_RUNS):
            # Out-of-training seeds (training data uses 42+i).
            random.seed(7000 + run)
            np.random.seed(7000 + run)
            jobs = generate_jobs(n_jobs, ARRIVAL_WINDOW)

            results = {}
            for pol in POLICIES:
                results[pol] = run_once(jobs, pol, sc['nodes'])

            fifo_waits = results['FIFO']['job_waits']
            for pol in POLICIES:
                r = results[pol]
                # failure = this job waited strictly longer than under FIFO
                fails = sum(1 for jid, w in r['job_waits'].items() if w > fifo_waits[jid])
                rows.append({
                    'scenario': sc['name'],
                    'policy': pol,
                    'run': run + 1,
                    'mean_wait': r['mean_wait'],
                    'max_wait': r['max_wait'],
                    'failure_rate_pct': 100.0 * fails / len(r['job_waits']),
                    'fallback_tick_pct': r['fallback_tick_pct'] if pol == 'GUARDED' else float('nan'),
                    'n_started': r['n_started'],
                })
            print(f"  run {run+1:2d}: " + " | ".join(
                f"{p} {results[p]['mean_wait']:.2f}" for p in POLICIES))

    df = pd.DataFrame(rows)

    # Aggregate per scenario x policy (paired: same seeds -> same job sets)
    agg = df.groupby(['scenario', 'policy'], as_index=False).agg(
        mean_wait=('mean_wait', 'mean'),
        max_wait=('max_wait', 'mean'),
        failure_rate_pct=('failure_rate_pct', 'mean'),
        fallback_tick_pct=('fallback_tick_pct', 'mean'),
        n_started=('n_started', 'mean'),
    )
    fifo_ref = agg[agg['policy'] == 'FIFO'].set_index('scenario')['mean_wait']
    imps, notes = [], []
    for _, r in agg.iterrows():
        fw = fifo_ref[r['scenario']]
        if fw > 0:
            imps.append(100.0 * (fw - r['mean_wait']) / fw)
            notes.append('')
        else:
            imps.append(float('nan'))
            notes.append('degenerate: FIFO mean wait = 0 (no queueing pressure)')
    agg['improvement_vs_fifo_pct'] = imps
    agg['note'] = notes

    scen_order = [s['name'] for s in SCENARIOS]
    agg['scenario'] = pd.Categorical(agg['scenario'], categories=scen_order, ordered=True)
    agg['policy'] = pd.Categorical(agg['policy'], categories=POLICIES, ordered=True)
    agg = agg.sort_values(['scenario', 'policy'])
    agg = agg[['scenario', 'policy', 'mean_wait', 'improvement_vs_fifo_pct',
               'failure_rate_pct', 'max_wait', 'fallback_tick_pct', 'n_started', 'note']]

    csv_path = os.path.join(OUT_DIR, 'uncertainty_ood_benchmark.csv')
    agg.to_csv(csv_path, index=False, encoding='utf-8')

    print("\n=== Uncertainty-aware scheduling: scenario x policy summary ===")
    print(agg.to_string(index=False, formatters={
        'mean_wait': '{:.3f}'.format,
        'improvement_vs_fifo_pct': '{:.2f}'.format,
        'failure_rate_pct': '{:.2f}'.format,
        'max_wait': '{:.2f}'.format,
        'fallback_tick_pct': '{:.2f}'.format,
        'n_started': '{:.1f}'.format,
    }))

    # ── Plot: improvement% vs FIFO, grouped by scenario ──────────────────────
    # Colors are the first three slots of the validated categorical palette
    # (dataviz skill); sub-3:1 contrast slots are relieved by direct labels.
    SERIES = ['PROACTIVE', 'UCB', 'GUARDED']
    COLORS = {'PROACTIVE': '#2a78d6', 'UCB': '#1baf7a', 'GUARDED': '#eda100'}
    SURFACE, INK, MUTED, GRID, BASELINE = '#fcfcfb', '#0b0b0b', '#898781', '#e1e0d9', '#c3c2b7'

    piv = agg.pivot(index='scenario', columns='policy', values='improvement_vs_fifo_pct')
    piv = piv.reindex(scen_order)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(scen_order))
    width = 0.26

    for k, pol in enumerate(SERIES):
        vals = piv[pol].values.astype(float)
        pos = x + (k - 1) * width
        bars = ax.bar(pos, np.nan_to_num(vals), width=width * 0.92,
                      color=COLORS[pol], label=pol, zorder=3)
        for xi, v, b in zip(pos, vals, bars):
            if np.isnan(v):
                ax.text(xi, 0.5, 'n/a', ha='center', va='bottom',
                        fontsize=8, color=MUTED, zorder=4)
            else:
                off = 0.4 if v >= 0 else -0.4
                ax.text(xi, v + off, f'{v:.1f}%',
                        ha='center', va='bottom' if v >= 0 else 'top',
                        fontsize=8.5, color=INK, zorder=4)

    ax.axhline(0, color=BASELINE, linewidth=1.2, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(scen_order, fontsize=9, color=INK)
    ax.set_ylabel('Wait-time improvement vs FIFO (%)', fontsize=10, color=INK)
    ax.set_title('Uncertainty-aware scheduling under distribution shift '
                 f'({NUM_RUNS} paired runs/scenario; 0 = FIFO reference)',
                 fontsize=11, color=INK)
    ax.tick_params(colors=MUTED)
    ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color(BASELINE)
    ax.legend(frameon=False, fontsize=9, loc='best')
    # widen y-limits a touch so direct labels don't clip
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 2, hi + 3)
    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, 'uncertainty_summary.png')
    plt.savefig(png_path, dpi=160, facecolor=SURFACE)

    print('\nSaved:', csv_path)
    print('Saved:', png_path)

if __name__ == '__main__':
    main()

import os
import sys
import random
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from vizstyle import (figure, finish, save_both, PALETTE, color_of, label_of,
                      bar_ends, legend_roles)

MODEL_PATH = os.path.join(PROJECT_ROOT, '03_models', 'wait_model_v2.pkl')
OUT_DIR = os.path.join(PROJECT_ROOT, '05_results', 'scaling')
os.makedirs(OUT_DIR, exist_ok=True)

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)['model']

SIM_TIME = 300
RUNS = 8
SIZES = [
    {'nodes': 4, 'gpus_per_node': 2},
    {'nodes': 8, 'gpus_per_node': 4},
    {'nodes': 16, 'gpus_per_node': 8},
    {'nodes': 32, 'gpus_per_node': 8},
]

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
                use = min(free, need)
                self.nodes[i] -= use
                alloc.append((i, use))
                need -= use
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
    tf = cluster.total_free_gpus()
    return np.array([
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
    ]).reshape(1, -1)

def generate_jobs(capacity, n=110):
    jobs = []
    max_gpu = min(8, max(2, capacity // 8))
    for i in range(n):
        jobs.append(Job(i, random.randint(0, SIM_TIME // 2), random.randint(1, max_gpu), random.randint(5, 20)))
    return sorted(jobs, key=lambda j: j.arrival_time)

def run(jobs_in, nodes, gpn, proactive=False):
    cluster = Cluster(nodes, gpn)
    jobs = [Job(j.job_id, j.arrival_time, j.num_gpus, j.runtime) for j in jobs_in]
    queue, running, done = [], [], []
    cap = nodes * gpn
    util = []
    inference_total = 0.0
    scheduler_overhead = 0.0

    t0 = time.perf_counter()
    for t in range(SIM_TIME):
        for j in running[:]:
            if j.end_time == t:
                cluster.release(j)
                running.remove(j)
                done.append(j)

        for j in jobs:
            if j.arrival_time == t:
                queue.append(j)

        if queue and proactive:
            s0 = time.perf_counter()
            scored = []
            for j in queue:
                i0 = time.perf_counter()
                pred = float(model.predict(get_features(j, cluster, queue, running))[0])
                inference_total += (time.perf_counter() - i0)
                scored.append((pred, j))
            scored.sort(key=lambda x: x[0])
            queue = [j for _, j in scored]
            scheduler_overhead += (time.perf_counter() - s0)

        for j in queue[:]:
            if cluster.total_free_gpus() >= j.num_gpus and cluster.allocate(j, t):
                running.append(j)
                queue.remove(j)

        util.append((cap - cluster.total_free_gpus()) / cap)

    total_runtime = time.perf_counter() - t0
    waits = [j.start_time - j.arrival_time for j in done if j.start_time is not None]
    return {
        'mean_wait': float(np.mean(waits)) if waits else np.nan,
        'throughput': len(done) / SIM_TIME,
        'gpu_util': float(np.mean(util)),
        'scheduler_overhead_sec': scheduler_overhead,
        'inference_time_sec': inference_total,
        'total_sim_time_sec': total_runtime,
    }

def main():
    rows = []
    for cfg in SIZES:
        nodes = cfg['nodes']
        gpn = cfg['gpus_per_node']
        cap = nodes * gpn
        for run_id in range(RUNS):
            random.seed(4000 + run_id + nodes)
            np.random.seed(4000 + run_id + nodes)
            jobs = generate_jobs(cap)
            fifo = run(jobs, nodes, gpn, proactive=False)
            pro = run(jobs, nodes, gpn, proactive=True)
            rows.append({
                'nodes': nodes,
                'gpus': cap,
                'run': run_id + 1,
                'fifo_wait': fifo['mean_wait'],
                'proactive_wait': pro['mean_wait'],
                'wait_advantage_pct': (fifo['mean_wait'] - pro['mean_wait']) / max(fifo['mean_wait'], 1e-6) * 100,
                'fifo_throughput': fifo['throughput'],
                'proactive_throughput': pro['throughput'],
                'scheduler_overhead_sec': pro['scheduler_overhead_sec'],
                'inference_time_sec': pro['inference_time_sec'],
                'proactive_total_sim_time_sec': pro['total_sim_time_sec'],
            })
            print(f"nodes={nodes:2d} run={run_id+1:02d} wait_adv={rows[-1]['wait_advantage_pct']:+.2f}% overhead={pro['scheduler_overhead_sec']:.4f}s")

    df = pd.DataFrame(rows)
    summary = df.groupby(['nodes', 'gpus'], as_index=False).mean(numeric_only=True)
    out_csv = os.path.join(OUT_DIR, 'scaling_analysis.csv')
    summary.to_csv(out_csv, index=False)

    # ── Figure: scaling curves ────────────────────────────────────────────────
    # The old chart drew a percentage and a duration against ONE y-axis, so the
    # two units were silently made comparable. They are split into stacked
    # panels sharing the cluster-size axis. Both panels describe the SAME entity
    # (the proactive ML scheduler), so both wear that entity's colour.
    stem = os.path.join(OUT_DIR, 'scaling_curves')
    gpus = summary['gpus'].tolist()
    advantage = summary['wait_advantage_pct'].tolist()
    overhead = summary['scheduler_overhead_sec'].tolist()
    tick_labels = [f'{int(g)}\n({int(n)} nodes)'
                   for g, n in zip(gpus, summary['nodes'].tolist())]

    for mode in ('light', 'dark'):
        fig, (ax_gain, ax_cost) = figure(mode, figsize=(9, 6.8),
                                         nrows=2, ncols=1, sharex=True)
        p = PALETTE[mode]
        proactive = color_of('PROACTIVE', mode)

        ax_gain.plot(gpus, advantage, marker='o', color=proactive)
        ax_gain.axhline(0, color=p['axis'], linewidth=0.8, zorder=1)
        ax_gain.set_title(f'Wait-time advantage of {label_of("PROACTIVE")} over '
                          f'{label_of("FIFO")}', loc='left')
        ax_gain.set_ylabel('Mean wait reduction (%)')
        ax_gain.margins(y=0.22)
        ax_gain.annotate(f'{advantage[0]:+.1f}%', xy=(gpus[0], advantage[0]),
                         xytext=(8, 2), textcoords='offset points',
                         color=p['ink_2'], fontsize=9)
        ax_gain.annotate('no job waits at 128+ GPUs',
                         xy=(gpus[-1], advantage[-1]), xytext=(-6, 26),
                         textcoords='offset points', ha='right',
                         color=p['muted'], fontsize=9)

        ax_cost.plot(gpus, overhead, marker='o', color=proactive)
        ax_cost.set_title(f'Cost of the ranking step in {label_of("PROACTIVE")}',
                          loc='left')
        ax_cost.set_ylabel('Scheduler overhead (s per 300-tick run)')
        ax_cost.set_xlabel('Cluster size (total GPUs)')
        ax_cost.margins(y=0.22)
        ax_cost.annotate(f'{overhead[0]:.1f} s', xy=(gpus[0], overhead[0]),
                         xytext=(8, -2), textcoords='offset points',
                         color=p['ink_2'], fontsize=9)
        ax_cost.annotate(f'{overhead[-1]:.2f} s', xy=(gpus[-1], overhead[-1]),
                         xytext=(-6, 10), textcoords='offset points',
                         ha='right', color=p['ink_2'], fontsize=9)

        for ax in (ax_gain, ax_cost):
            ax.set_xscale('log', base=2)
            ax.set_xticks(gpus)
            ax.set_xticks([], minor=True)
            ax.set_xticklabels(tick_labels)
            ax.set_axisbelow(True)

        fig.tight_layout(rect=(0, 0.02, 1, 0.87))
        finish(fig, mode,
               title='Proactive scheduling across four cluster sizes',
               subtitle='Mean of 8 runs per size. The advantage and the ranking '
                        'cost both come from queue length, so both fade as the '
                        'cluster grows.',
               source='05_results/scaling/scaling_analysis.csv')
        out_plot = save_both(fig, stem, mode)
        if mode == 'light':
            light_plot = out_plot
        else:
            dark_plot = out_plot

    print('\nSaved:', out_csv)
    print('Saved:', light_plot)
    print('Saved:', dark_plot)

if __name__ == '__main__':
    main()

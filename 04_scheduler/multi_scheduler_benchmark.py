import os
import random
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

from sjf_scheduler import order_queue as order_sjf
from priority_scheduler import order_queue as order_priority
from neural_network_scheduler import order_queue as order_nn, build_feature_vector
from backfill_scheduler import easy_backfill_dispatch

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

with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)
wait_model = bundle['model']
FEATURES = bundle['features']

class Job:
    def __init__(self, job_id, arrival_time, num_gpus, runtime, priority_score):
        self.job_id = job_id
        self.arrival_time = arrival_time
        self.num_gpus = num_gpus
        self.runtime = runtime
        self.priority_score = priority_score
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
    ])

def gini(vals):
    x = np.array([v for v in vals if v >= 0], dtype=float)
    if len(x) == 0 or np.sum(x) == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)

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

def train_nn_predictor():
    df = pd.read_csv(DATA_PATH)
    x = df[FEATURES].values
    y = df['wait_time'].values
    nn = MLPRegressor(hidden_layer_sizes=(24, 12), max_iter=400, random_state=42)
    nn.fit(x, y)
    return nn

def rank_queue(queue, t, cluster, running, scheduler, nn_model):
    if scheduler in ('fifo', 'backfill', 'proactive_bf'):
        # backfill schedulers keep the queue in arrival order; the EASY
        # dispatch helper handles head reservation + backfill scan itself
        return queue
    if scheduler == 'sjf':
        return order_sjf(queue, t)
    if scheduler == 'priority':
        return order_priority(queue, t)
    if scheduler == 'proactive':
        scored = [(float(wait_model.predict(get_features(j, cluster, queue, running).reshape(1, -1))[0]), j) for j in queue]
        scored.sort(key=lambda x: (x[0], x[1].arrival_time, x[1].job_id))
        return [j for _, j in scored]
    return order_nn(queue, cluster, running, nn_model)

def run_once(jobs_in, scheduler, nn_model):
    cluster = Cluster(NUM_NODES, GPUS_PER_NODE)
    jobs = [Job(j.job_id, j.arrival_time, j.num_gpus, j.runtime, j.priority_score) for j in jobs_in]
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

        if scheduler in ('backfill', 'proactive_bf'):
            # EASY backfill: arrival-order head with a reservation guarantee
            # (anti-starvation); backfill jobs may never delay the head.
            bf_order = None
            hook = None
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
                                             pre_dispatch_hook=hook)
            for job in started:
                queue.remove(job)
                running.append(job)
        else:
            for job in queue[:]:
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

    waits = [j.start_time - j.arrival_time for j in completed if j.start_time is not None]
    return {
        'mean_wait': float(np.mean(waits)) if waits else float('nan'),
        'max_wait': float(np.max(waits)) if waits else float('nan'),
        'throughput': len(completed) / SIM_TIME,
        'fairness_gini': gini(waits),
        'gpu_util': float(np.mean(util)) if util else 0.0,
        # wait-prediction MAE at dispatch time; NaN for non-ML schedulers
        'pred_wait_mae': float(np.mean(mae_samples)) if mae_samples else float('nan'),
    }

def main():
    nn_model = train_nn_predictor()
    schedulers = ['fifo', 'sjf', 'priority', 'proactive', 'nn', 'backfill', 'proactive_bf']
    rows = []

    for run in range(NUM_RUNS):
        # per-run seed: paired workloads, every scheduler sees the same jobs
        # Eval seeds MUST differ from the model's training-data seeds (42+i in
        # generate_improved_dataset.py) or the benchmark evaluates on training
        # workloads. 1000+run also restores comparability with prior results.
        random.seed(1000 + run)
        np.random.seed(1000 + run)
        jobs = generate_jobs()
        for sch in schedulers:
            out = run_once(jobs, sch, nn_model)
            out['run'] = run + 1
            out['scheduler'] = sch.upper()
            rows.append(out)
            print(f"Run {run+1:02d} | {sch.upper():12s} wait={out['mean_wait']:.2f} throughput={out['throughput']:.3f}")

    df = pd.DataFrame(rows)
    summary = df.groupby('scheduler', as_index=False).agg({
        'pred_wait_mae': 'mean',
        'mean_wait': 'mean',
        'max_wait': 'mean',
        'fairness_gini': 'mean',
        'throughput': 'mean',
        'gpu_util': 'mean',
    }).sort_values('mean_wait')

    summary_path = os.path.join(OUT_DIR, 'multi_scheduler_benchmark.csv')
    summary.to_csv(summary_path, index=False)

    plt.figure(figsize=(11, 5))
    x = np.arange(len(summary))
    width = 0.35
    plt.bar(x - width / 2, summary['mean_wait'], width=width, label='Mean wait', color='#38bdf8')
    plt.bar(x + width / 2, summary['throughput'], width=width, label='Throughput', color='#34d399')
    plt.xticks(x, summary['scheduler'])
    plt.title('Scheduler benchmark (pred-wait MAE, wait, fairness, throughput in CSV)')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'scheduler_comparison.png')
    plt.savefig(plot_path, dpi=160)

    print('\n=== Multi-scheduler summary ===')
    print(summary.to_string(index=False, formatters={
        'pred_wait_mae': '{:.3f}'.format,
        'mean_wait': '{:.3f}'.format,
        'max_wait': '{:.3f}'.format,
        'fairness_gini': '{:.3f}'.format,
        'throughput': '{:.3f}'.format,
        'gpu_util': '{:.3f}'.format,
    }))
    print('Saved:', summary_path)
    print('Saved:', plot_path)

if __name__ == '__main__':
    main()

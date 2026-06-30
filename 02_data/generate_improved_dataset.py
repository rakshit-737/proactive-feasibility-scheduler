import random
import numpy as np
import pandas as pd

class Job:
    def __init__(self, job_id, arrival_time, num_gpus, runtime):
        self.job_id = job_id
        self.arrival_time = arrival_time
        self.num_gpus = num_gpus
        self.runtime = runtime
        self.start_time = None
        self.end_time = None
        self.allocated_nodes = []
        self.feature_snapshot = None

class Cluster:
    def __init__(self, num_nodes, gpus_per_node):
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.nodes = [gpus_per_node] * num_nodes

    def total_free_gpus(self):
        return sum(self.nodes)

    def allocate(self, job, current_time):
        required = job.num_gpus
        allocation = []
        for i in range(self.num_nodes):
            if required <= 0:
                break
            available = self.nodes[i]
            if available > 0:
                used = min(available, required)
                self.nodes[i] -= used
                allocation.append((i, used))
                required -= used
        if required == 0:
            job.start_time = current_time
            job.end_time = current_time + job.runtime
            job.allocated_nodes = allocation
            return True
        else:
            for node_id, used in allocation:
                self.nodes[node_id] += used
            return False

    def release(self, job):
        for node_id, used in job.allocated_nodes:
            self.nodes[node_id] += used

def generate_jobs(num_jobs, max_time):
    jobs = []
    for i in range(num_jobs):
        arrival_time = random.randint(0, max_time // 2)
        num_gpus = random.randint(1, 8)
        runtime = random.randint(5, 20)
        jobs.append(Job(i, arrival_time, num_gpus, runtime))
    return sorted(jobs, key=lambda x: x.arrival_time)

def extract_features(job, cluster, queue, running_jobs):
    """
    Principled, non-leaky cluster-state features.

    Removed: smaller_jobs_in_queue  (82% importance — direct proxy for FIFO
             queue rank, near data leakage in a FIFO scheduler)

    New features capture PHYSICAL cluster state:
      can_fit_now       — binary: can the job start immediately?
      gpu_fit_ratio     — how much of the job's GPU demand is currently available
      fragmentation     — std-dev of per-node free GPUs (high = harder to place)
      queue_pressure    — total GPUs demanded by waiting jobs vs total free
      node_availability — fraction of nodes that have enough GPUs for this job
      avg_free_per_node — mean free GPUs per node (global supply signal)
    """
    total_free    = cluster.total_free_gpus()
    max_free_node = max(cluster.nodes)
    variance_free = np.var(cluster.nodes)

    can_fit_now       = int(total_free >= job.num_gpus)
    gpu_fit_ratio     = min(total_free / (job.num_gpus + 1e-6), 1.0)
    fragmentation     = float(np.std(cluster.nodes))
    total_queued_gpus = sum(q.num_gpus for q in queue)
    queue_pressure    = total_queued_gpus / (total_free + 1)
    node_availability = sum(1 for n in cluster.nodes if n >= job.num_gpus) / cluster.num_nodes
    avg_free_per_node = total_free / cluster.num_nodes

    return {
        "job_gpu":           job.num_gpus,
        "total_free":        total_free,
        "queue_length":      len(queue),
        "running_jobs":      len(running_jobs),
        "max_free_node":     max_free_node,
        "variance_free":     variance_free,
        # new principled features
        "can_fit_now":       can_fit_now,
        "gpu_fit_ratio":     gpu_fit_ratio,
        "fragmentation":     fragmentation,
        "queue_pressure":    queue_pressure,
        "node_availability": node_availability,
        "avg_free_per_node": avg_free_per_node,
    }

def run_simulation():
    SIM_TIME     = 300
    NUM_NODES    = 8
    GPUS_PER_NODE = 4
    NUM_JOBS     = 110

    cluster       = Cluster(NUM_NODES, GPUS_PER_NODE)
    jobs          = generate_jobs(NUM_JOBS, SIM_TIME)
    queue         = []
    running_jobs  = []
    completed_jobs = []
    dataset       = []

    for t in range(SIM_TIME):
        # release finished jobs
        for job in running_jobs[:]:
            if job.end_time == t:
                cluster.release(job)
                running_jobs.remove(job)
                completed_jobs.append(job)

        # snapshot features at arrival time
        for job in jobs:
            if job.arrival_time == t:
                job.feature_snapshot = extract_features(job, cluster, queue, running_jobs)
                queue.append(job)

        # FIFO scheduling
        for job in queue[:]:
            if cluster.total_free_gpus() >= job.num_gpus:
                if cluster.allocate(job, t):
                    running_jobs.append(job)
                    queue.remove(job)

    # build dataset from completed jobs
    for job in completed_jobs:
        if job.feature_snapshot is not None and job.start_time is not None:
            row = job.feature_snapshot.copy()
            row["wait_time"] = job.start_time - job.arrival_time
            dataset.append(row)

    return dataset


# ── Run 20 simulations ──────────────────────────────────────────────────────
all_data = []
NUM_RUNS = 20

for i in range(NUM_RUNS):
    result = run_simulation()
    all_data.extend(result)
    print(f"Run {i+1}/{NUM_RUNS} — {len(result)} samples")

df = pd.DataFrame(all_data)
df.to_csv("improved_wait_dataset.csv", index=False)

print(f"\nTotal samples : {len(df)}")
print(f"Avg wait time : {df['wait_time'].mean():.2f}")
print(f"Wait time range: {df['wait_time'].min()} – {df['wait_time'].max()}")
print(f"Columns: {list(df.columns)}")

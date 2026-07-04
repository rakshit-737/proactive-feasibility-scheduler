# NOTE: Despite the file name, this script writes wait_dataset.csv — the LEGACY
# regression dataset consumed by 03_models/train_wait_model.py. The canonical
# dataset for the main pipeline is improved_wait_dataset.csv from
# generate_improved_dataset.py; shared-name feature columns here follow the
# same (canonical) formulas as that script.
import os
import random
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

def run_simulation():
    SIM_TIME = 300
    NUM_NODES = 8
    GPUS_PER_NODE = 4
    NUM_JOBS = 110
    TOTAL_GPUS = NUM_NODES * GPUS_PER_NODE

    cluster = Cluster(NUM_NODES, GPUS_PER_NODE)
    jobs = generate_jobs(NUM_JOBS, SIM_TIME)

    queue = []
    running_jobs = []
    completed_jobs = []
    dataset = []

    for t in range(SIM_TIME):
        for job in running_jobs[:]:
            if job.end_time == t:
                cluster.release(job)
                running_jobs.remove(job)
                completed_jobs.append(job)

        for job in jobs:
            if job.arrival_time == t:
                total_free = cluster.total_free_gpus()
                queue_length = len(queue)
                running_count = len(running_jobs)
                max_free_node = max(cluster.nodes)
                variance_free = np.var(cluster.nodes)

                # Principled features — no queue-rank leakage
                # (formulas match generate_improved_dataset.py exactly)
                can_fit_now      = 1 if total_free >= job.num_gpus else 0
                gpu_fit_ratio    = min(total_free / (job.num_gpus + 1e-6), 1.0)
                fragmentation    = float(np.std(cluster.nodes))
                queue_pressure   = sum(q.num_gpus for q in queue) / (total_free + 1)
                nodes_that_fit   = sum(1 for n in cluster.nodes if n >= job.num_gpus)
                node_availability = nodes_that_fit / NUM_NODES
                avg_free_per_node = total_free / NUM_NODES

                job.feature_snapshot = {
                    "job_gpu":           job.num_gpus,
                    "total_free":        total_free,
                    "queue_length":      queue_length,
                    "running_jobs":      running_count,
                    "max_free_node":     max_free_node,
                    "variance_free":     variance_free,
                    "can_fit_now":       can_fit_now,
                    "gpu_fit_ratio":     gpu_fit_ratio,
                    "fragmentation":     fragmentation,
                    "queue_pressure":    queue_pressure,
                    "node_availability": node_availability,
                    "avg_free_per_node": avg_free_per_node,
                }
                queue.append(job)

        for job in queue[:]:
            if cluster.total_free_gpus() >= job.num_gpus:
                success = cluster.allocate(job, t)
                if success:
                    running_jobs.append(job)
                    queue.remove(job)

    for job in completed_jobs:
        if job.feature_snapshot is not None:
            wait_time = job.start_time - job.arrival_time
            row = job.feature_snapshot.copy()
            row["wait_time"] = wait_time
            dataset.append(row)

    return dataset


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    all_data = []
    NUM_RUNS = 20

    for i in range(NUM_RUNS):
        # per-run seed: runs differ but the whole set is reproducible
        random.seed(42 + i)
        np.random.seed(42 + i)
        result = run_simulation()
        all_data.extend(result)
        print(f"Run {i+1}/{NUM_RUNS} — {len(result)} samples")

    df = pd.DataFrame(all_data)

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"\nDropped {before - len(df)} duplicate rows")

    output_path = os.path.join(PROJECT_ROOT, "02_data", "wait_dataset.csv")
    df.to_csv(output_path, index=False)

    print(f"Total samples: {len(df)}")
    print(f"Avg wait time: {df['wait_time'].mean():.2f}")
    print(f"Wait time range: {df['wait_time'].min()} to {df['wait_time'].max()}")
    print(f"Features: {[c for c in df.columns if c != 'wait_time']}")

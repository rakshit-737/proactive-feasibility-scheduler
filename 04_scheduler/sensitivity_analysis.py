import os
import random
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "03_models", "wait_model_v2.pkl")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "05_results", "sensitivity_analysis_results.csv")

SIM_TIME = 300
JOBS_PER_RUN = 110
RUNS_PER_SCENARIO = 10


with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
FEATURES = bundle["features"]


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
    def __init__(self, num_nodes, gpus_per_node=4):
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
        for node_id, used in allocation:
            self.nodes[node_id] += used
        return False

    def release(self, job):
        for node_id, used in job.allocated_nodes:
            self.nodes[node_id] += used


def get_features(job, cluster, queue, running_jobs):
    total_free = cluster.total_free_gpus()
    max_free_node = max(cluster.nodes)
    variance_free = np.var(cluster.nodes)

    return {
        "job_gpu": job.num_gpus,
        "total_free": total_free,
        "queue_length": len(queue),
        "running_jobs": len(running_jobs),
        "max_free_node": max_free_node,
        "variance_free": variance_free,
        "can_fit_now": int(total_free >= job.num_gpus),
        "gpu_fit_ratio": min(total_free / (job.num_gpus + 1e-6), 1.0),
        "fragmentation": float(np.std(cluster.nodes)),
        "queue_pressure": sum(q.num_gpus for q in queue) / (total_free + 1),
        "node_availability": sum(1 for n in cluster.nodes if n >= job.num_gpus) / cluster.num_nodes,
        "avg_free_per_node": total_free / cluster.num_nodes,
    }


def sample_arrival(pattern, max_t):
    if pattern == "uniform":
        return random.randint(0, max_t // 2)
    if pattern == "sparse":
        return random.randint(0, max_t - 1)
    if pattern == "dense":
        return random.randint(0, max_t // 3)

    block = random.randint(0, 5)
    if block % 2 == 0:
        return random.randint(block * (max_t // 6), block * (max_t // 6) + max(1, max_t // 12))
    return random.randint(block * (max_t // 6), min(max_t - 1, block * (max_t // 6) + max_t // 6 - 1))


def sample_runtime(runtime_profile):
    if runtime_profile == "short":
        return random.randint(3, 10)
    if runtime_profile == "long":
        return random.randint(12, 30)
    return random.randint(5, 20)


def generate_jobs(arrival_pattern, runtime_profile, num_jobs=JOBS_PER_RUN):
    jobs = []
    for i in range(num_jobs):
        jobs.append(
            Job(
                job_id=i,
                arrival_time=sample_arrival(arrival_pattern, SIM_TIME),
                num_gpus=random.randint(1, 8),
                runtime=sample_runtime(runtime_profile),
            )
        )
    return sorted(jobs, key=lambda x: x.arrival_time)


def run_fifo_collect_rows(num_nodes, arrival_pattern, runtime_profile):
    cluster = Cluster(num_nodes=num_nodes, gpus_per_node=4)
    jobs = generate_jobs(arrival_pattern, runtime_profile)

    queue = []
    running_jobs = []
    completed_jobs = []

    for t in range(SIM_TIME):
        for job in running_jobs[:]:
            if job.end_time == t:
                cluster.release(job)
                running_jobs.remove(job)
                completed_jobs.append(job)

        for job in jobs:
            if job.arrival_time == t:
                job.feature_snapshot = get_features(job, cluster, queue, running_jobs)
                queue.append(job)

        for job in queue[:]:
            if cluster.total_free_gpus() >= job.num_gpus and cluster.allocate(job, t):
                running_jobs.append(job)
                queue.remove(job)

    rows = []
    for job in completed_jobs:
        if job.feature_snapshot is not None and job.start_time is not None:
            row = job.feature_snapshot.copy()
            row["wait_time"] = job.start_time - job.arrival_time
            rows.append(row)
    return rows


def evaluate_scenario(name, num_nodes, arrival_pattern, runtime_profile):
    all_rows = []
    for i in range(RUNS_PER_SCENARIO):
        random.seed(5000 + i)
        np.random.seed(5000 + i)
        rows = run_fifo_collect_rows(num_nodes, arrival_pattern, runtime_profile)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    x = df[FEATURES]
    y = df["wait_time"]
    y_pred = model.predict(x)

    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return {
        "scenario": name,
        "num_nodes": num_nodes,
        "arrival_pattern": arrival_pattern,
        "runtime_profile": runtime_profile,
        "samples": len(df),
        "mae": mae,
        "r2": r2,
        "mean_wait": y.mean(),
    }


def main():
    scenarios = [
        ("OOD Nodes=6", 6, "uniform", "default"),
        ("OOD Nodes=10", 10, "uniform", "default"),
        ("OOD Nodes=12", 12, "uniform", "default"),
        ("OOD Runtime=Short", 8, "uniform", "short"),
        ("OOD Runtime=Long", 8, "uniform", "long"),
        ("OOD Arrival=Sparse", 8, "sparse", "default"),
        ("OOD Arrival=Dense", 8, "dense", "default"),
        ("OOD Arrival=Bursty", 8, "bursty", "default"),
    ]

    results = []
    for scenario in scenarios:
        row = evaluate_scenario(*scenario)
        results.append(row)
        print(f"{row['scenario']:<20} samples={row['samples']:<4d} MAE={row['mae']:.4f} R²={row['r2']:.4f}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Sensitivity Analysis (fixed model, no retraining) ===")
    print(result_df.to_string(index=False, formatters={
        "mae": "{:.4f}".format,
        "r2": "{:.4f}".format,
        "mean_wait": "{:.2f}".format,
    }))
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

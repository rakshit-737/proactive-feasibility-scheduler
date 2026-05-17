import os
import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NUM_RUNS = 20
JOBS_PER_RUN = 110

FEATURES = [
    "job_gpu", "total_free", "queue_length", "running_jobs",
    "max_free_node", "variance_free", "can_fit_now", "gpu_fit_ratio",
    "fragmentation", "queue_pressure", "node_availability", "avg_free_per_node"
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
        self.feature_snapshot = None


class Cluster:
    def __init__(self, num_nodes=8, gpus_per_node=4):
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


def extract_features(job, cluster, queue, running_jobs):
    total_free = cluster.total_free_gpus()
    max_free_node = max(cluster.nodes)
    variance_free = np.var(cluster.nodes)

    can_fit_now = int(total_free >= job.num_gpus)
    gpu_fit_ratio = min(total_free / (job.num_gpus + 1e-6), 1.0)
    fragmentation = float(np.std(cluster.nodes))
    total_queued_gpus = sum(q.num_gpus for q in queue)
    queue_pressure = total_queued_gpus / (total_free + 1)
    node_availability = sum(1 for n in cluster.nodes if n >= job.num_gpus) / cluster.num_nodes
    avg_free_per_node = total_free / cluster.num_nodes

    return {
        "job_gpu": job.num_gpus,
        "total_free": total_free,
        "queue_length": len(queue),
        "running_jobs": len(running_jobs),
        "max_free_node": max_free_node,
        "variance_free": variance_free,
        "can_fit_now": can_fit_now,
        "gpu_fit_ratio": gpu_fit_ratio,
        "fragmentation": fragmentation,
        "queue_pressure": queue_pressure,
        "node_availability": node_availability,
        "avg_free_per_node": avg_free_per_node,
    }


def sample_arrival(profile, sim_time):
    if profile == "low":
        return random.randint(0, sim_time // 3)
    if profile == "high":
        return random.randint(0, sim_time // 2)

    block = random.randint(0, 5)
    if block % 2 == 0:
        return random.randint(block * (sim_time // 6), block * (sim_time // 6) + max(1, sim_time // 12))
    return random.randint(block * (sim_time // 6), min(sim_time - 1, block * (sim_time // 6) + sim_time // 6 - 1))


def generate_jobs(profile, num_jobs, sim_time):
    jobs = []
    for i in range(num_jobs):
        if profile == "low":
            num_gpus = random.randint(1, 2) if i % 10 != 0 else random.randint(3, 4)
            runtime = random.randint(6, 12)
        elif profile == "high":
            num_gpus = random.randint(4, 8)
            runtime = random.randint(10, 26)
        else:
            num_gpus = random.randint(1, 8)
            runtime = random.randint(5, 24)

        arrival_time = sample_arrival(profile, sim_time)
        jobs.append(Job(i, arrival_time, num_gpus, runtime))
    return sorted(jobs, key=lambda x: x.arrival_time)


def run_simulation(profile):
    sim_time = 360
    cluster = Cluster(num_nodes=8, gpus_per_node=4)
    jobs = generate_jobs(profile, JOBS_PER_RUN, sim_time)

    queue = []
    running_jobs = []
    completed_jobs = []

    t = 0
    while t < 1200 and (t < sim_time or jobs or queue or running_jobs):
        for job in running_jobs[:]:
            if job.end_time == t:
                cluster.release(job)
                running_jobs.remove(job)
                completed_jobs.append(job)

        arrived_now = [job for job in jobs if job.arrival_time == t]
        for job in arrived_now:
            job.feature_snapshot = extract_features(job, cluster, queue, running_jobs)
            queue.append(job)
        jobs = [job for job in jobs if job.arrival_time != t]

        for job in queue[:]:
            if cluster.total_free_gpus() >= job.num_gpus and cluster.allocate(job, t):
                running_jobs.append(job)
                queue.remove(job)

        t += 1

    dataset = []
    for job in completed_jobs:
        if job.feature_snapshot is not None and job.start_time is not None:
            row = job.feature_snapshot.copy()
            row["wait_time"] = job.start_time - job.arrival_time
            dataset.append(row)

    return dataset


def generate_profile_dataset(profile_name, file_name):
    all_rows = []
    for run in range(NUM_RUNS):
        rows = run_simulation(profile_name)
        all_rows.extend(rows)
        print(f"{profile_name:<6} run {run + 1:02d}/{NUM_RUNS} -> {len(rows)} samples")

    df = pd.DataFrame(all_rows)
    output_path = os.path.join(PROJECT_ROOT, "02_data", file_name)
    df.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print(f"Rows: {len(df)} | Mean wait: {df['wait_time'].mean():.2f} | Range: {df['wait_time'].min()}-{df['wait_time'].max()}\n")
    return output_path


def evaluate_xgboost(dataset_path, profile_name):
    df = pd.read_csv(dataset_path)
    x = df[FEATURES]
    y = df["wait_time"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    return {"profile": profile_name, "samples": len(df), "mae": mae, "r2": r2}


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    low_path = generate_profile_dataset("low", "improved_wait_dataset_low_contention.csv")
    high_path = generate_profile_dataset("high", "improved_wait_dataset_high_contention.csv")
    bursty_path = generate_profile_dataset("bursty", "improved_wait_dataset_mixed_bursty.csv")

    profile_results = [
        evaluate_xgboost(low_path, "low_contention"),
        evaluate_xgboost(high_path, "high_contention"),
        evaluate_xgboost(bursty_path, "mixed_bursty"),
    ]
    results_df = pd.DataFrame(profile_results)
    results_output_path = os.path.join(PROJECT_ROOT, "05_results", "xgboost_load_profile_performance.csv")
    results_df.to_csv(results_output_path, index=False)
    print("=== XGBoost performance across load profiles ===")
    print(results_df.to_string(index=False, formatters={"mae": "{:.4f}".format, "r2": "{:.4f}".format}))
    print(f"\nSaved: {results_output_path}")

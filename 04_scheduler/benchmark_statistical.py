import os
import random
import pickle
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "03_models", "wait_model_v2.pkl")
RESULTS_CSV = os.path.join(PROJECT_ROOT, "05_results", "benchmark_statistical_results.csv")
SUMMARY_CSV = os.path.join(PROJECT_ROOT, "05_results", "benchmark_statistical_summary.csv")

SIM_TIME = 300
NUM_NODES = 8
GPUS_PER_NODE = 4
TOTAL_CAPACITY = NUM_NODES * GPUS_PER_NODE
NUM_RUNS = 40


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
        for node_id, used in allocation:
            self.nodes[node_id] += used
        return False

    def release(self, job):
        for node_id, used in job.allocated_nodes:
            self.nodes[node_id] += used


def generate_jobs(num_jobs=110, max_time=300):
    jobs = [Job(i, random.randint(0, max_time // 2), random.randint(1, 8), random.randint(5, 20)) for i in range(num_jobs)]
    return sorted(jobs, key=lambda x: x.arrival_time)


def get_features(job, cluster, queue, running_jobs):
    total_free = cluster.total_free_gpus()
    max_free_node = max(cluster.nodes)
    variance_free = np.var(cluster.nodes)

    return [
        job.num_gpus,
        total_free,
        len(queue),
        len(running_jobs),
        max_free_node,
        variance_free,
        int(total_free >= job.num_gpus),
        min(total_free / (job.num_gpus + 1e-6), 1.0),
        float(np.std(cluster.nodes)),
        sum(q.num_gpus for q in queue) / (total_free + 1),
        sum(1 for n in cluster.nodes if n >= job.num_gpus) / cluster.num_nodes,
        total_free / cluster.num_nodes,
    ]


def run_scheduler(jobs_input, proactive=False):
    cluster = Cluster(NUM_NODES, GPUS_PER_NODE)
    jobs = [Job(j.job_id, j.arrival_time, j.num_gpus, j.runtime) for j in jobs_input]

    queue = []
    running_jobs = []
    completed_jobs = []
    gpu_usage = []

    for t in range(SIM_TIME):
        for job in running_jobs[:]:
            if job.end_time == t:
                cluster.release(job)
                running_jobs.remove(job)
                completed_jobs.append(job)

        for job in jobs:
            if job.arrival_time == t:
                queue.append(job)

        if proactive and queue:
            scored = []
            for job in queue:
                predicted_wait = model.predict([get_features(job, cluster, queue, running_jobs)])[0]
                scored.append((predicted_wait, job))
            scored.sort(key=lambda x: x[0])
            queue = [job for _, job in scored]

        for job in queue[:]:
            if cluster.total_free_gpus() >= job.num_gpus and cluster.allocate(job, t):
                running_jobs.append(job)
                queue.remove(job)

        gpu_usage.append((TOTAL_CAPACITY - cluster.total_free_gpus()) / TOTAL_CAPACITY)

    wait_times = [j.start_time - j.arrival_time for j in completed_jobs if j.start_time is not None]
    return float(np.mean(wait_times)), float(np.mean(gpu_usage)), len(completed_jobs)


def confidence_interval_95(values):
    arr = np.array(values, dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=1)
    sem = std / np.sqrt(len(arr))
    t_critical = stats.t.ppf(0.975, df=len(arr) - 1)
    margin = t_critical * sem
    return mean, std, mean - margin, mean + margin


def main():
    rows = []
    for run in range(NUM_RUNS):
        random.seed(1000 + run)
        jobs = generate_jobs()

        baseline_wait, baseline_util, baseline_done = run_scheduler(jobs, proactive=False)
        proactive_wait, proactive_util, proactive_done = run_scheduler(jobs, proactive=True)

        improvement_pct = (baseline_wait - proactive_wait) / baseline_wait * 100
        rows.append({
            "run": run + 1,
            "baseline_wait": baseline_wait,
            "proactive_wait": proactive_wait,
            "baseline_util": baseline_util,
            "proactive_util": proactive_util,
            "baseline_completed": baseline_done,
            "proactive_completed": proactive_done,
            "improvement_pct": improvement_pct,
            "wait_diff": baseline_wait - proactive_wait,
        })
        print(f"Run {run + 1:02d}/{NUM_RUNS} | baseline={baseline_wait:.2f} proactive={proactive_wait:.2f} improvement={improvement_pct:+.2f}%")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_CSV, index=False)

    wait_diff = results_df["wait_diff"].to_numpy()
    improv_pct = results_df["improvement_pct"].to_numpy()

    ttest = stats.ttest_rel(results_df["baseline_wait"], results_df["proactive_wait"])
    wilcoxon_note = ""
    try:
        wilcoxon = stats.wilcoxon(results_df["baseline_wait"], results_df["proactive_wait"], zero_method="wilcox")
        wilcoxon_p = float(wilcoxon.pvalue)
    except ValueError as exc:
        wilcoxon_p = np.nan
        wilcoxon_note = str(exc)

    diff_mean, diff_std, diff_ci_low, diff_ci_high = confidence_interval_95(wait_diff)
    pct_mean, pct_std, pct_ci_low, pct_ci_high = confidence_interval_95(improv_pct)

    summary_df = pd.DataFrame([
        {"metric": "runs", "value": NUM_RUNS},
        {"metric": "baseline_wait_mean", "value": results_df['baseline_wait'].mean()},
        {"metric": "proactive_wait_mean", "value": results_df['proactive_wait'].mean()},
        {"metric": "mean_wait_difference", "value": diff_mean},
        {"metric": "std_wait_difference", "value": diff_std},
        {"metric": "mean_improvement_pct", "value": pct_mean},
        {"metric": "std_improvement_pct", "value": pct_std},
        {"metric": "ttest_pvalue", "value": float(ttest.pvalue)},
        {"metric": "wilcoxon_pvalue", "value": wilcoxon_p},
        {"metric": "wait_diff_ci95_low", "value": diff_ci_low},
        {"metric": "wait_diff_ci95_high", "value": diff_ci_high},
        {"metric": "improvement_ci95_low", "value": pct_ci_low},
        {"metric": "improvement_ci95_high", "value": pct_ci_high},
    ])
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print("\n=== Statistical Benchmark Summary ===")
    print(f"Runs: {NUM_RUNS}")
    print(f"Baseline wait mean:  {results_df['baseline_wait'].mean():.3f}")
    print(f"Proactive wait mean: {results_df['proactive_wait'].mean():.3f}")
    print(f"Mean difference (B-P): {diff_mean:.3f} ± {diff_std:.3f}")
    print(f"Paired t-test p-value: {ttest.pvalue:.6g}")
    print(f"Wilcoxon p-value: {wilcoxon_p:.6g}" if not np.isnan(wilcoxon_p) else "Wilcoxon p-value: n/a")
    if wilcoxon_note:
        print(f"Wilcoxon warning: {wilcoxon_note}")
    print(f"95% CI (difference): [{diff_ci_low:.3f}, {diff_ci_high:.3f}]")
    print(f"95% CI (improvement %): [{pct_ci_low:.3f}%, {pct_ci_high:.3f}%]")
    print(f"Saved: {RESULTS_CSV}")
    print(f"Saved: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()

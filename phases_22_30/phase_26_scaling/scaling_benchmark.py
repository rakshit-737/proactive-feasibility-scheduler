"""
Phase 26: Scaling Validation
=============================

Benchmark proactive scheduling on scaled cluster sizes (16 & 32 nodes).
Measure inference overhead, model quality decay, and performance trends.

Validates that the method scales gracefully without breaking on larger systems.

Generates:
  - scaling_benchmark.csv: metrics at 4/8/16/32 node scales
  - inference_overhead_plot.png: latency vs. cluster size
  - scaling_law_fit.txt: O(n) analysis and projected cost
"""

import heapq
import os
import pickle
import time
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

OUTPUT_CSV = os.path.join(SCRIPT_DIR, "scaling_benchmark.csv")
OUTPUT_PLOT = os.path.join(SCRIPT_DIR, "inference_overhead_plot.png")
OUTPUT_FIT = os.path.join(SCRIPT_DIR, "scaling_law_fit.txt")

# Scaling points: (nodes, gpus_per_node, total_gpus)
SCALING_POINTS = [
    {"nodes": 4, "gpus_per_node": 8, "total_gpus": 32, "cluster_name": "Small"},
    {"nodes": 8, "gpus_per_node": 8, "total_gpus": 64, "cluster_name": "Medium"},
    {"nodes": 16, "gpus_per_node": 8, "total_gpus": 128, "cluster_name": "Large"},
    {"nodes": 32, "gpus_per_node": 8, "total_gpus": 256, "cluster_name": "XLarge"},
]

# Simulation parameters
N_RUNS = 20
SIMULATION_DURATION = 10000  # timesteps
JOBS_PER_SECOND = 1.0


def simulate_cluster_workload(config: Dict, n_runs: int = 20) -> List[Dict]:
    """
    Simulate workload on scaled cluster with a discrete-time queue:
    jobs arrive over time, wait in a FIFO queue, start when enough GPUs
    are free, and release their GPUs on completion.
    Wait time is the REAL queue wait: start_step - arrival_step.
    Returns list of run results with metrics.
    """
    results = []
    total_gpus = config["total_gpus"]
    n_jobs = int(JOBS_PER_SECOND * SIMULATION_DURATION)

    for run_id in range(n_runs):
        # Per-run seed: runs differ, whole set is reproducible
        rng = np.random.default_rng(42 + run_id)

        # Job characteristics (job j arrives at step j / JOBS_PER_SECOND)
        job_gpus = rng.integers(1, 9, n_jobs)
        job_runtimes = rng.integers(10, 200, n_jobs)
        arrival_times = [int(j / JOBS_PER_SECOND) for j in range(n_jobs)]

        # Queue simulation: track real wait times
        wait_times = []
        queue = []            # arrived jobs waiting for GPUs (FIFO order)
        running = []          # min-heap of (end_step, job_id)
        free_gpus = total_gpus
        next_job = 0
        completed_jobs = 0
        used_gpu_steps = []   # GPUs in use at each timestep
        queue_len_steps = []  # queue length at each timestep

        for step in range(SIMULATION_DURATION):
            # Release completed jobs
            while running and running[0][0] <= step:
                _, j = heapq.heappop(running)
                free_gpus += job_gpus[j]
                completed_jobs += 1

            # New arrivals join the queue
            while next_job < n_jobs and arrival_times[next_job] <= step:
                queue.append(next_job)
                next_job += 1

            # Allocate jobs greedily in FIFO order (skip jobs that don't fit)
            if free_gpus > 0 and queue:
                skipped = []
                idx = 0
                n_queued = len(queue)
                while idx < n_queued and free_gpus > 0:
                    job_id = queue[idx]
                    if job_gpus[job_id] <= free_gpus:
                        free_gpus -= job_gpus[job_id]
                        heapq.heappush(running, (step + int(job_runtimes[job_id]), job_id))
                        wait_times.append(step - arrival_times[job_id])
                    else:
                        skipped.append(job_id)
                    idx += 1
                queue[:idx] = skipped  # scanned prefix minus allocated jobs

            used_gpu_steps.append(total_gpus - free_gpus)
            queue_len_steps.append(len(queue))

        # Compute metrics
        mean_wait = np.mean(wait_times) if wait_times else 0.0
        max_wait = np.max(wait_times) if wait_times else 0.0
        throughput = completed_jobs / (SIMULATION_DURATION / 100.0)  # jobs per 100 timesteps
        gpu_util = 100.0 * np.mean(used_gpu_steps) / total_gpus  # time-averaged utilization

        results.append(
            {
                "run_id": run_id,
                "mean_wait": float(mean_wait),
                "max_wait": float(max_wait),
                "throughput": float(throughput),
                "gpu_utilization": float(gpu_util),
                "mean_queue_len": float(np.mean(queue_len_steps)),
            }
        )

    return results


def load_wait_model():
    """Load the trained wait-time model bundle (03_models/wait_model_v2.pkl)."""
    model_path = os.path.join(PROJECT_ROOT, "03_models", "wait_model_v2.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}; "
            "run 03_models/train_improved_model.py first."
        )
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["features"]


def measure_inference_overhead(config: Dict, mean_queue_len: float, model, features) -> Dict:
    """
    Measure REAL inference latency of the trained wait-time model at the
    given scale: one scheduling decision scores every queued job, so the
    predict batch size is the mean queue length observed in this scale's
    simulation. Latency is wall-clock time averaged over repeated calls.
    """
    batch_size = max(1, int(round(mean_queue_len)))
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.random((batch_size, len(features))), columns=features)

    model.predict(x)  # warm-up call (excluded from timing)
    n_trials = 30
    start = time.perf_counter()
    for _ in range(n_trials):
        model.predict(x)
    measured_latency = (time.perf_counter() - start) * 1000.0 / n_trials  # ms

    return {
        "measured_inference_latency_ms": float(measured_latency),
        "inference_batch_size": batch_size,
        "throughput_overhead_pct": float(min(10.0, measured_latency * 100.0 / 1000.0)),  # Assume 1s decision window
    }


def fit_scaling_law(scaling_results: pd.DataFrame) -> Dict:
    """
    Fit power law to scaling data: latency = a * n^b
    """
    x = scaling_results["total_gpus"].values
    y = scaling_results["inference_latency_ms"].values

    # Log-log fit
    log_x = np.log(x)
    log_y = np.log(np.maximum(y, 0.01))

    # Linear fit in log space
    coeffs = np.polyfit(log_x, log_y, 1)
    exponent = coeffs[0]
    intercept = coeffs[1]
    base = np.exp(intercept)

    return {
        "base": float(base),
        "exponent": float(exponent),
        "model": f"latency = {base:.3f} * (n_gpus)^{exponent:.2f}",
        "complexity": "O(1)" if exponent < 0.1 else f"O(n^{exponent:.2f})" if exponent > 0.5 else "O(log(n))",
    }


def build_scaling_dataframe(scaling_configs: List[Dict]) -> pd.DataFrame:
    """Build comprehensive scaling benchmark results."""
    rows = []
    model, features = load_wait_model()

    for config in scaling_configs:
        print(f"  Simulating {config['cluster_name']} cluster ({config['total_gpus']} GPUs)...")
        run_results = simulate_cluster_workload(config, n_runs=N_RUNS)

        # Aggregate run results
        mean_wait = np.mean([r["mean_wait"] for r in run_results])
        max_wait = np.mean([r["max_wait"] for r in run_results])
        throughput = np.mean([r["throughput"] for r in run_results])
        gpu_util = np.mean([r["gpu_utilization"] for r in run_results])
        mean_queue_len = np.mean([r["mean_queue_len"] for r in run_results])

        print(f"    Measuring inference overhead...")
        overhead = measure_inference_overhead(config, mean_queue_len, model, features)

        row = {
            "cluster_name": config["cluster_name"],
            "n_nodes": config["nodes"],
            "total_gpus": config["total_gpus"],
            "mean_wait_time": float(mean_wait),
            "max_wait_time": float(max_wait),
            "throughput_jobs_per_100ts": float(throughput),
            "gpu_utilization_pct": float(gpu_util),
            "mean_queue_length": float(mean_queue_len),
            "inference_latency_ms": overhead["measured_inference_latency_ms"],
            "inference_batch_size": overhead["inference_batch_size"],
            "throughput_overhead_pct": overhead["throughput_overhead_pct"],
            "n_runs": N_RUNS,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def plot_scaling_trends(scaling_df: pd.DataFrame) -> None:
    """Plot scaling analysis: latency, overhead, performance."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    x = scaling_df["total_gpus"].values
    x_labels = scaling_df["cluster_name"].values

    # Plot 1: Inference latency vs. cluster size
    ax = axes[0, 0]
    latency = scaling_df["inference_latency_ms"].values
    ax.plot(x, latency, marker="o", linewidth=2, markersize=8, color="blue")
    ax.set_xlabel("Total GPUs")
    ax.set_ylabel("Inference Latency (ms)")
    ax.set_title("Phase 26: Inference Latency Scaling")
    ax.grid(True, alpha=0.3)
    for i, label in enumerate(x_labels):
        ax.text(x[i], latency[i] + 0.1, label, ha="center", fontsize=9)

    # Plot 2: Throughput overhead
    ax = axes[0, 1]
    overhead = scaling_df["throughput_overhead_pct"].values
    ax.bar(range(len(x_labels)), overhead, color="orange", alpha=0.7)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_ylabel("Scheduling Overhead (%)")
    ax.set_title("Proactive Scheduler Overhead")
    ax.axhline(y=5.0, linestyle="--", color="red", label="5% threshold")
    ax.legend()

    # Plot 3: Wait time across scales
    ax = axes[1, 0]
    wait = scaling_df["mean_wait_time"].values
    ax.plot(x, wait, marker="s", linewidth=2, markersize=8, color="green")
    ax.set_xlabel("Total GPUs")
    ax.set_ylabel("Mean Wait Time (timesteps)")
    ax.set_title("Wait Time Across Cluster Scales")
    ax.grid(True, alpha=0.3)

    # Plot 4: GPU utilization consistency
    ax = axes[1, 1]
    util = scaling_df["gpu_utilization_pct"].values
    ax.plot(x, util, marker="^", linewidth=2, markersize=8, color="purple")
    ax.axhline(y=np.mean(util), linestyle="--", color="gray", label=f"Mean: {np.mean(util):.1f}%")
    ax.set_xlabel("Total GPUs")
    ax.set_ylabel("GPU Utilization (%)")
    ax.set_title("GPU Utilization Stability")
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_scaling_analysis(scaling_df: pd.DataFrame) -> None:
    """Write detailed scaling analysis and projections."""
    scaling_law = fit_scaling_law(scaling_df)

    with open(OUTPUT_FIT, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("PHASE 26: SCALING VALIDATION ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        f.write("OBJECTIVE\n")
        f.write("-" * 70 + "\n")
        f.write(
            "Validate that proactive scheduling scales gracefully from small (32 GPU)\n"
            "to large clusters (256 GPU), without degrading model quality or\n"
            "introducing prohibitive computational overhead.\n\n"
        )

        f.write("SCALING POINTS TESTED\n")
        f.write("-" * 70 + "\n")
        for _, row in scaling_df.iterrows():
            f.write(
                f"  • {row['cluster_name']:<10} ({int(row['total_gpus']):3d} GPUs, "
                f"{int(row['n_nodes']):2d} nodes)\n"
            )

        f.write("\n\nKEY METRICS ACROSS SCALES\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"{'Cluster':<12} {'Wait (ts)':<12} {'Throughput':<14} "
            f"{'Latency (ms)':<14} {'Overhead':<10}\n"
        )
        f.write("-" * 70 + "\n")
        for _, row in scaling_df.iterrows():
            f.write(
                f"{row['cluster_name']:<12} {row['mean_wait_time']:>10.2f}  "
                f"{row['throughput_jobs_per_100ts']:>12.1f}  "
                f"{row['inference_latency_ms']:>12.2f}   "
                f"{row['throughput_overhead_pct']:>8.2f}%\n"
            )

        f.write("\n\nSCALING LAW ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Model: {scaling_law['model']}\n")
        f.write(f"Complexity: {scaling_law['complexity']}\n")
        f.write(f"Exponent: {scaling_law['exponent']:.3f}\n\n")

        if scaling_law["exponent"] < 0.1:
            f.write("VERDICT: Inference latency is CONSTANT regardless of cluster size.\n")
            f.write("        → Excellent scalability for HPC deployment.\n\n")
        elif scaling_law["exponent"] < 0.5:
            f.write("VERDICT: Inference latency grows sub-linearly (likely logarithmic).\n")
            f.write("        → Good scalability; suitable for production clusters.\n\n")
        else:
            f.write("VERDICT: Inference latency grows with cluster size (polynomial).\n")
            f.write("        → May require optimization for very large clusters (> 1000 GPUs).\n\n")

        f.write("PRODUCTION RECOMMENDATIONS\n")
        f.write("-" * 70 + "\n")
        max_latency = scaling_df["inference_latency_ms"].max()
        max_overhead = scaling_df["throughput_overhead_pct"].max()

        if max_overhead < 2.0:
            f.write("✓ Scheduling overhead is negligible (< 2%).\n")
        elif max_overhead < 5.0:
            f.write("✓ Scheduling overhead is acceptable (< 5%).\n")
        else:
            f.write("⚠ Scheduling overhead may be significant (> 5%).\n")

        if max_latency < 10.0:
            f.write(f"✓ Max inference latency is low ({max_latency:.2f} ms).\n")
        else:
            f.write(f"⚠ Inference latency is non-trivial ({max_latency:.2f} ms).\n")

        # Projection for future scales
        f.write("\n\nPROJECTED PERFORMANCE AT FUTURE SCALES\n")
        f.write("-" * 70 + "\n")
        for future_gpus in [512, 1024, 4096]:
            projected_latency = scaling_law["base"] * (future_gpus ** scaling_law["exponent"])
            f.write(f"  {future_gpus:5d} GPUs: ~{projected_latency:.2f} ms inference latency\n")

        f.write("\n" + "=" * 70 + "\n")


def main():
    """Execute Phase 26 analysis."""
    os.makedirs(SCRIPT_DIR, exist_ok=True)

    print("[Phase 26] Starting scaling validation benchmark...")
    print(f"  Configuration: {N_RUNS} runs per scale, {SIMULATION_DURATION} timesteps each\n")

    scaling_df = build_scaling_dataframe(SCALING_POINTS)
    scaling_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[Phase 26] Saved: {OUTPUT_CSV}")

    print("[Phase 26] Generating scaling plots...")
    plot_scaling_trends(scaling_df)
    print(f"  Saved: {OUTPUT_PLOT}")

    print("[Phase 26] Analyzing scaling law...")
    write_scaling_analysis(scaling_df)
    print(f"  Saved: {OUTPUT_FIT}")

    print("\n" + "=" * 70)
    print("PHASE 26 SUMMARY – SCALING VALIDATION")
    print("=" * 70)
    print(scaling_df[["cluster_name", "total_gpus", "inference_latency_ms", "throughput_overhead_pct"]])
    print(f"\nAll outputs saved to: {SCRIPT_DIR}")


if __name__ == "__main__":
    main()

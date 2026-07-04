"""
Phase 23 OOD robustness analysis.

Evaluates the trained wait-time predictor (03_models/wait_model_v2.pkl) across
72 domain-shift scenarios (6 arrival-rate multipliers x 4 cluster sizes x
3 runtime distributions). Every scenario is a REAL discrete-time simulation
(same Job/Cluster/FIFO mechanics and feature definitions as
02_data/generate_improved_dataset.py); the reported R2/MAPE are computed by
scoring the actual trained model on (features, observed wait) pairs collected
from those simulations, and improvement_pct is measured by re-running each
scenario with the model-guided proactive queue ordering vs FIFO.

There is NO heuristic fallback: if the model file is missing the script fails
loudly rather than silently substituting synthetic predictions.
"""

import os
import pickle
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

matplotlib.use("Agg")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

OUTPUT_CSV = os.path.join(SCRIPT_DIR, "ood_failure_modes.csv")
OUTPUT_HEATMAP = os.path.join(SCRIPT_DIR, "ood_heatmap.png")

ARRIVAL_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
CLUSTER_SIZES = [4, 8, 16, 32]  # total GPUs; nodes = size / GPUS_PER_NODE
JOB_DISTRIBUTIONS = ["short_heavy", "balanced", "long_heavy"]
HIGH_RISK_IMPROVEMENT_THRESHOLD = -2.0
GPUS_PER_NODE = 4

# Training configuration (generate_improved_dataset.py): 110 jobs arriving in
# [0, SIM_TIME // 2] on 8 nodes x 4 GPUs, runtimes uniform 5-20.
SIM_TIME = 300
BASE_NUM_JOBS = 110
RUNS_PER_SCENARIO = 3
BASE_SEED = 42

FEATURE_COLUMNS = [
    "job_gpu",
    "total_free",
    "queue_length",
    "running_jobs",
    "max_free_node",
    "variance_free",
    "can_fit_now",
    "gpu_fit_ratio",
    "fragmentation",
    "queue_pressure",
    "node_availability",
    "avg_free_per_node",
]

BASELINE_KEYS = ["baseline_wait", "fifo_avg_wait", "fifo_wait", "fifo_mean_wait"]
PROACTIVE_KEYS = ["proactive_wait", "proactive_avg_wait", "proactive_wait_time", "proactive_mean_wait"]
IMPROVEMENT_KEYS = ["improvement_pct", "improvement_percent", "pct_improvement"]

MODEL_PATH = os.path.join(PROJECT_ROOT, "03_models", "wait_model_v2.pkl")

DATA_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "05_results", "benchmarks", "40_run_benchmark.pkl"),
    os.path.join(PROJECT_ROOT, "05_results", "benchmark_statistical_results.pkl"),
    os.path.join(PROJECT_ROOT, "05_results", "benchmark_statistical_results.csv"),
]


# Load Phase 21 model
def load_proactive_model():
    """Load the trained XGBoost model. Fails loudly if the artifact is missing."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Run 03_models/train_improved_model.py first; Phase 23 refuses to "
            "substitute a heuristic for the real model."
        )
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    features = list(bundle.get("features", FEATURE_COLUMNS))
    if features != FEATURE_COLUMNS:
        raise ValueError(
            f"Model feature schema {features} does not match the canonical "
            f"12-feature schema {FEATURE_COLUMNS}."
        )
    return model, features, MODEL_PATH


def _first_present(df, keys):
    for key in keys:
        if key in df.columns:
            return key
    return None


# Load Phase 09 benchmark data (training distribution reference)
def load_baseline_data():
    """Load 40-run benchmark data to establish the in-distribution baseline."""
    for path in DATA_CANDIDATES:
        if not os.path.exists(path):
            continue
        if path.endswith(".pkl"):
            with open(path, "rb") as f:
                raw = pickle.load(f)
            df = pd.DataFrame(raw if isinstance(raw, list) else raw.get("results", raw))
        else:
            df = pd.read_csv(path)

        base_col = _first_present(df, BASELINE_KEYS)
        pro_col = _first_present(df, PROACTIVE_KEYS)
        imp_col = _first_present(df, IMPROVEMENT_KEYS)
        if base_col is None or pro_col is None:
            continue

        out = pd.DataFrame(
            {
                "baseline_wait": pd.to_numeric(df[base_col], errors="coerce"),
                "proactive_wait": pd.to_numeric(df[pro_col], errors="coerce"),
            }
        ).dropna()

        if imp_col is not None:
            out["improvement_pct"] = pd.to_numeric(df[imp_col], errors="coerce")
        else:
            out["improvement_pct"] = (
                (out["baseline_wait"] - out["proactive_wait"]) / out["baseline_wait"] * 100.0
            )

        out = out.replace([np.inf, -np.inf], np.nan).dropna()
        if not out.empty:
            return out, path

    raise FileNotFoundError(
        "No benchmark artifact found among: "
        + ", ".join(DATA_CANDIDATES)
        + ". Run the Phase 09 benchmark (05_results/benchmark_and_plot.py) first; "
        "Phase 23 refuses to fabricate baseline numbers."
    )


# Simulation core — identical mechanics/formulas to generate_improved_dataset.py
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


def extract_features(job, cluster, queue, running_jobs):
    """Canonical 12-feature schema — must match generate_improved_dataset.py EXACTLY."""
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


# Generate OOD scenarios
def generate_ood_scenarios():
    """
    Generate domain shifts:
    - arrival_rates: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    - cluster_sizes: [4, 8, 16, 32]
    - job_distributions: ['short_heavy', 'balanced', 'long_heavy']
    Returns: list of (scenario_name, params) tuples
    """
    scenarios = []
    for arrival_rate in ARRIVAL_MULTIPLIERS:
        for cluster_size in CLUSTER_SIZES:
            for job_dist in JOB_DISTRIBUTIONS:
                name = f"arr_{arrival_rate:.2f}_cluster_{cluster_size}_{job_dist}"
                scenarios.append(
                    (
                        name,
                        {
                            "scenario": name,
                            "arrival_rate_multiplier": arrival_rate,
                            "cluster_size": cluster_size,
                            "job_dist_type": job_dist,
                        },
                    )
                )
    return scenarios


def _sample_runtime(job_dist_type):
    """Shifted runtime profiles; 'balanced' matches the training distribution (5-20)."""
    r = random.random()
    if job_dist_type == "short_heavy":
        if r < 0.72:
            return random.randint(2, 8)
        if r < 0.95:
            return random.randint(8, 18)
        return random.randint(18, 40)
    if job_dist_type == "long_heavy":
        if r < 0.12:
            return random.randint(2, 8)
        if r < 0.45:
            return random.randint(8, 18)
        return random.randint(18, 40)
    return random.randint(5, 20)


def generate_shifted_jobs(params):
    """Draw a workload under the scenario's domain shift (uses the seeded RNG)."""
    num_jobs = max(1, round(BASE_NUM_JOBS * params["arrival_rate_multiplier"]))
    max_gpu = min(8, params["cluster_size"])  # infeasible requests cannot exist
    jobs = []
    for i in range(num_jobs):
        arrival_time = random.randint(0, SIM_TIME // 2)
        num_gpus = random.randint(1, max_gpu)
        runtime = _sample_runtime(params["job_dist_type"])
        jobs.append(Job(i, arrival_time, num_gpus, runtime))
    return sorted(jobs, key=lambda x: x.arrival_time)


def _clone_jobs(jobs):
    return [Job(j.job_id, j.arrival_time, j.num_gpus, j.runtime) for j in jobs]


def run_fifo(jobs_input, num_nodes):
    """FIFO simulation (training-data mechanics). Returns per-job (features, wait)
    rows for completed jobs plus their wait times and the completed count."""
    cluster = Cluster(num_nodes, GPUS_PER_NODE)
    jobs = _clone_jobs(jobs_input)
    queue, running_jobs, completed_jobs = [], [], []

    for t in range(SIM_TIME):
        for job in running_jobs[:]:
            if job.end_time == t:
                cluster.release(job)
                running_jobs.remove(job)
                completed_jobs.append(job)
        for job in jobs:
            if job.arrival_time == t:
                job.feature_snapshot = extract_features(job, cluster, queue, running_jobs)
                queue.append(job)
        for job in queue[:]:
            if cluster.total_free_gpus() >= job.num_gpus:
                if cluster.allocate(job, t):
                    running_jobs.append(job)
                    queue.remove(job)

    rows, waits = [], []
    for job in completed_jobs:
        if job.feature_snapshot is not None and job.start_time is not None:
            row = job.feature_snapshot.copy()
            row["wait_time"] = job.start_time - job.arrival_time
            rows.append(row)
            waits.append(row["wait_time"])
    return rows, waits, len(completed_jobs)


def run_proactive(jobs_input, num_nodes, model, model_features):
    """Model-guided simulation: each tick the queue is re-ordered by the trained
    model's predicted wait (ascending), then all fitting jobs dispatch — the same
    policy as 04_scheduler/proactive_Schedule_v2.py, with batched predictions."""
    cluster = Cluster(num_nodes, GPUS_PER_NODE)
    jobs = _clone_jobs(jobs_input)
    queue, running_jobs, completed_jobs = [], [], []

    for t in range(SIM_TIME):
        for job in running_jobs[:]:
            if job.end_time == t:
                cluster.release(job)
                running_jobs.remove(job)
                completed_jobs.append(job)
        for job in jobs:
            if job.arrival_time == t:
                queue.append(job)

        # Re-rank only when a dispatch is possible this tick — ordering has no
        # effect otherwise, and skipping keeps the 72-scenario sweep fast.
        free = cluster.total_free_gpus()
        if len(queue) > 1 and any(j.num_gpus <= free for j in queue):
            feats = np.asarray(
                [
                    [snap[c] for c in model_features]
                    for snap in (extract_features(job, cluster, queue, running_jobs) for job in queue)
                ],
                dtype=float,
            )
            predicted = model.predict(feats)
            order = np.argsort(predicted, kind="stable")
            queue = [queue[i] for i in order]

        for job in queue[:]:
            if cluster.total_free_gpus() >= job.num_gpus:
                if cluster.allocate(job, t):
                    running_jobs.append(job)
                    queue.remove(job)

    waits = [j.start_time - j.arrival_time for j in completed_jobs if j.start_time is not None]
    return waits, len(completed_jobs)


def _compute_mape(y_true, y_pred):
    """MAPE over jobs with actual wait >= 1 (zero-wait jobs make MAPE undefined)."""
    mask = y_true >= 1.0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100.0)


# Run OOD evaluation
def evaluate_ood_scenario(model, model_features, scenario_params, baseline_mean_improvement, scenario_index):
    """
    For each scenario:
    1. Simulate the shifted workload (real FIFO simulation, seeded per run)
    2. Score the trained model on the observed (features, wait) pairs -> R2, MAPE
    3. Re-run the scenario with model-guided ordering -> improvement vs FIFO
    4. Measure the fraction of submitted jobs left unfinished (failure rate)
    Returns: dict with metrics
    """
    num_nodes = max(1, scenario_params["cluster_size"] // GPUS_PER_NODE)

    rows = []
    fifo_waits, proactive_waits = [], []
    submitted = completed_proactive = 0
    for run in range(RUNS_PER_SCENARIO):
        # Deterministic but distinct seed per (scenario, run).
        seed = BASE_SEED + scenario_index * RUNS_PER_SCENARIO + run
        random.seed(seed)
        np.random.seed(seed)

        jobs = generate_shifted_jobs(scenario_params)
        submitted += len(jobs)

        run_rows, run_waits, _ = run_fifo(jobs, num_nodes)
        rows.extend(run_rows)
        fifo_waits.extend(run_waits)

        pro_waits, pro_completed = run_proactive(jobs, num_nodes, model, model_features)
        proactive_waits.extend(pro_waits)
        completed_proactive += pro_completed

    sample_df = pd.DataFrame(rows)
    y_true = sample_df["wait_time"].to_numpy(dtype=float)
    y_pred = np.asarray(model.predict(sample_df[model_features]), dtype=float)

    # R2 is undefined for a constant target (e.g. every job starts instantly).
    if len(y_true) < 2 or np.var(y_true) == 0.0:
        r2 = float("nan")
    else:
        r2 = float(r2_score(y_true, y_pred))
    mape = _compute_mape(y_true, y_pred)

    fifo_mean = float(np.mean(fifo_waits)) if fifo_waits else float("nan")
    proactive_mean = float(np.mean(proactive_waits)) if proactive_waits else float("nan")
    if fifo_waits and fifo_mean > 0:
        improvement_pct = (fifo_mean - proactive_mean) / fifo_mean * 100.0
    else:
        improvement_pct = 0.0  # no measurable queueing under this shift

    # Jobs still unfinished at the simulation horizon under the proactive policy.
    failure_rate = float(100.0 * (1.0 - completed_proactive / submitted)) if submitted else float("nan")

    arrival = scenario_params["arrival_rate_multiplier"]
    cluster_size = scenario_params["cluster_size"]
    return {
        "scenario": scenario_params["scenario"],
        "arrival_rate_multiplier": arrival,
        "cluster_size": cluster_size,
        "job_dist_type": scenario_params["job_dist_type"],
        "r2_score": r2,
        "mape": mape,
        "improvement_pct": float(improvement_pct),
        "improvement_degradation_pct": float(baseline_mean_improvement - improvement_pct),
        "failure_rate_pct": failure_rate,
        "n_samples": int(len(y_true)),
    }


# Failure mode detection
def detect_failure_mode(improvement_pct, r2_value, mape, arrival_rate_multiplier, cluster_size):
    """
    Categorize failures:
    - NONE: improvement > 2%
    - MODEL_OVERCONFIDENCE: improvement < -2% but R² > 0.8
    - SATURATION: improvement < -2% at high arrival rates
    - DISTRIBUTION_MISMATCH: high MAPE
    Returns: failure_mode_str, risk_level
    """
    # Per Phase 23 requirement, scenarios below -2% improvement are flagged as HIGH_RISK.
    if improvement_pct < HIGH_RISK_IMPROVEMENT_THRESHOLD and r2_value > 0.8:
        return "MODEL_OVERCONFIDENCE", "HIGH_RISK"

    if (
        improvement_pct < HIGH_RISK_IMPROVEMENT_THRESHOLD
        and arrival_rate_multiplier >= 1.5
        and cluster_size <= 8
    ):
        return "SATURATION", "HIGH_RISK"

    if mape >= 35.0:
        return (
            "DISTRIBUTION_MISMATCH",
            "HIGH_RISK" if improvement_pct < HIGH_RISK_IMPROVEMENT_THRESHOLD else "MEDIUM_RISK",
        )

    if arrival_rate_multiplier >= 1.5 and mape >= 25.0:
        return "ARRIVAL_SPIKE", "MEDIUM_RISK"

    if improvement_pct < HIGH_RISK_IMPROVEMENT_THRESHOLD:
        return "GENERAL_REGRESSION", "HIGH_RISK"

    if improvement_pct > 2.0 and r2_value >= 0.5 and mape <= 30.0:
        return "NONE", "LOW_RISK"

    return "MARGINAL_GAIN", "MEDIUM_RISK"


# Visualization
def plot_ood_heatmap(results_df):
    """
    Create 2D heatmap: rows=arrival_rates, cols=cluster_sizes, values=improvement_pct
    Save to ood_heatmap.png
    """
    heat = (
        results_df.groupby(["arrival_rate_multiplier", "cluster_size"], as_index=False)["improvement_pct"]
        .mean()
        .pivot(index="arrival_rate_multiplier", columns="cluster_size", values="improvement_pct")
        .reindex(index=ARRIVAL_MULTIPLIERS, columns=CLUSTER_SIZES)
    )
    r2_heat = (
        results_df.groupby(["arrival_rate_multiplier", "cluster_size"], as_index=False)["r2_score"]
        .mean()
        .pivot(index="arrival_rate_multiplier", columns="cluster_size", values="r2_score")
        .reindex(index=ARRIVAL_MULTIPLIERS, columns=CLUSTER_SIZES)
    )

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    im = ax.imshow(heat.to_numpy(), cmap="RdYlGn", aspect="auto", origin="upper")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Improvement (%) vs FIFO", rotation=90)

    ax.set_xticks(range(len(CLUSTER_SIZES)))
    ax.set_xticklabels(CLUSTER_SIZES)
    ax.set_yticks(range(len(ARRIVAL_MULTIPLIERS)))
    ax.set_yticklabels([f"{v:.2f}x" for v in ARRIVAL_MULTIPLIERS])
    ax.set_xlabel("Cluster size (GPUs)")
    ax.set_ylabel("Arrival rate multiplier")
    ax.set_title("Phase 23 OOD Robustness: Improvement Heatmap with R² Contours")

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.iloc[i, j]
            if np.isnan(value):
                continue
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", fontsize=9, color="black")
            if value < 0:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.1))

    # Only draw contour levels that actually fall inside the observed R2 range.
    r2_grid = r2_heat.to_numpy(dtype=float)
    finite = r2_grid[np.isfinite(r2_grid)]
    levels = [lv for lv in (0.3, 0.5, 0.7) if finite.size and finite.min() < lv < finite.max()]
    if levels:
        xx, yy = np.meshgrid(range(len(CLUSTER_SIZES)), range(len(ARRIVAL_MULTIPLIERS)))
        cs = ax.contour(xx, yy, r2_grid, levels=levels, colors="black", linewidths=1.0)
        ax.clabel(cs, inline=True, fmt="R²=%.1f", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_HEATMAP, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    """Execute Phase 23 analysis."""
    os.makedirs(SCRIPT_DIR, exist_ok=True)

    model, model_features, model_source = load_proactive_model()
    baseline_df, baseline_source = load_baseline_data()
    baseline_mean_improvement = float(baseline_df["improvement_pct"].mean())

    scenarios = generate_ood_scenarios()

    results = []
    for scenario_index, (scenario_name, params) in enumerate(scenarios):
        result = evaluate_ood_scenario(
            model, model_features, params, baseline_mean_improvement, scenario_index
        )
        mode, risk = detect_failure_mode(
            result["improvement_pct"],
            result["r2_score"],
            result["mape"],
            result["arrival_rate_multiplier"],
            result["cluster_size"],
        )
        result["failure_mode"] = mode
        result["risk_level"] = risk
        result["scenario"] = scenario_name
        results.append(result)
        print(
            f"[{scenario_index + 1:2d}/{len(scenarios)}] {scenario_name}: "
            f"R2={result['r2_score']:.3f} MAPE={result['mape']:.1f}% "
            f"improvement={result['improvement_pct']:.2f}% ({risk})"
        )

    results_df = pd.DataFrame(results)
    ordered_columns = [
        "scenario",
        "arrival_rate_multiplier",
        "cluster_size",
        "job_dist_type",
        "r2_score",
        "mape",
        "improvement_pct",
        "failure_mode",
        "risk_level",
        "improvement_degradation_pct",
        "failure_rate_pct",
        "n_samples",
    ]
    results_df = results_df[ordered_columns]

    results_df.to_csv(OUTPUT_CSV, index=False)
    plot_ood_heatmap(results_df)

    high_risk = results_df[results_df["risk_level"] == "HIGH_RISK"]
    dangerous = results_df[results_df["improvement_pct"] <= -5.0]

    print(f"Model source: {model_source}")
    print(f"Baseline source: {baseline_source}")
    print(f"Total scenarios: {len(results_df)}")
    print(f"High-risk scenarios: {len(high_risk)}")
    print(f"Dangerous zones (<= -5%): {len(dangerous)}")
    print(f"Average improvement across OOD: {results_df['improvement_pct'].mean():.2f}%")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_HEATMAP}")


if __name__ == "__main__":
    main()

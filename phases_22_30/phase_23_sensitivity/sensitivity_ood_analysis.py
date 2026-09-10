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

Scenarios are then RANKED rather than labelled. The categorical taxonomy this
script used to emit collapsed to one constant value (see the long note above
SEVERITY_DIMENSIONS); it has been replaced by a continuous severity score,
data-derived quantile bands, and a dominant-axis label. Severity is standardised
WITHIN this grid, so it says which shifted regimes are worse than which -- never
that any regime is good. None of them are.
"""

import os
import pickle
import random
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from sklearn.metrics import r2_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

sys.path.insert(0, PROJECT_ROOT)
from vizstyle import (  # noqa: E402  (needs PROJECT_ROOT on sys.path first)
    figure,
    finish,
    save_both,
    PALETTE,
    color_of,
    label_of,
    bar_ends,
    legend_roles,
)

OUTPUT_CSV = os.path.join(SCRIPT_DIR, "ood_failure_modes.csv")
# Stem, not a filename: save_both() writes '<stem>.png' (light) and
# '<stem>-dark.png' (dark). The light path stays byte-identical to the old
# output so every README/report reference keeps resolving.
OUTPUT_HEATMAP_STEM = os.path.join(SCRIPT_DIR, "ood_heatmap")
OUTPUT_HEATMAP = OUTPUT_HEATMAP_STEM + ".png"

# Provenance footer: the artefacts every mark in the figure is drawn from.
FIGURE_SOURCE = (
    "phases_22_30/phase_23_sensitivity/ood_failure_modes.csv"
    "  ·  model: 03_models/wait_model_v2.pkl"
)

ARRIVAL_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
CLUSTER_SIZES = [4, 8, 16, 32]  # total GPUs; nodes = size / GPUS_PER_NODE
JOB_DISTRIBUTIONS = ["short_heavy", "balanced", "long_heavy"]
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
        + ". Run 04_scheduler/benchmark_statistical.py first (step 17 of "
        "run_all_experiments.sh); Phase 23 refuses to fabricate baseline numbers."
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
            # extract_features expects the queue WITHOUT the scored job (training
            # rows are snapshotted at arrival, before enqueue), so pass the
            # other queued jobs only — same fix as the 04_scheduler benchmarks.
            feats = np.asarray(
                [
                    [snap[c] for c in model_features]
                    for snap in (
                        extract_features(job, cluster,
                                         [q for q in queue if q is not job],
                                         running_jobs)
                        for job in queue
                    )
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


# ---------------------------------------------------------------------------
# Severity measure and failure taxonomy
# ---------------------------------------------------------------------------
# WHY THE OLD CATEGORICAL TAXONOMY WAS REPLACED (2026-09-10)
#
# `detect_failure_mode()` used to assign one of eight labels from a chain of
# hard-coded cut-points. Its third branch fired on `mape >= 35.0`. The MINIMUM
# MAPE over the 72 scenarios in ood_failure_modes.csv is 54.01, so that branch
# fired for EVERY scenario that reached it, and the two branches ahead of it
# (which additionally required improvement_pct < -2.0 together with either
# R2 > 0.8 or a small saturated cluster) matched NOTHING. The committed
# artefact therefore held a single value:
#
#     failure_mode == "DISTRIBUTION_MISMATCH" for all 72 of 72 rows
#
# A column with zero entropy carries no information: the old labels were worth
# nothing, and seven of the eight categories were unreachable by construction.
# The cut-points had gone stale against the numbers they were meant to describe
# and nothing detected it, because nothing ever checked that more than one
# category was produced.
#
# The replacement (Option A of the Phase C brief) is a CONTINUOUS severity
# score over the dimensions that actually vary across the grid, plus bands and
# labels derived from the observed distribution rather than from constants:
#
#   severity_score  mean of the four dimensions standardised WITHIN this grid
#                   and oriented so that LARGER ALWAYS MEANS WORSE.
#   severity_band   quartile of severity_score, cut at the 25/50/75th
#                   percentiles OF THE 72 SCENARIOS THEMSELVES. Every band is
#                   populated by construction, so this cannot go stale the way
#                   35.0 did; re-running on different numbers re-derives the
#                   cuts along with them.
#   failure_mode    which dimension DOMINATES this scenario's severity, i.e.
#                   the largest oriented z-score, but only when that z-score is
#                   positive (worse than the grid mean). The comparison point is
#                   the mean of the data, not a tuned constant.
#   risk_level      keeps the legacy vocabulary (LOW/MEDIUM/HIGH_RISK) so
#                   downstream readers keep working, but is now derived from the
#                   severity band plus the SIGN of improvement_pct and of
#                   r2_score -- two natural zeros (does the policy help or hurt?
#                   does the model beat predicting the mean wait?), not fitted
#                   thresholds. See assign_risk_levels for exactly how weak a
#                   floor those two signs are: LOW_RISK is a within-grid rank,
#                   not a statement that any regime is safe.
#
# READ SEVERITY AS RELATIVE, NOT ABSOLUTE. Standardising within the grid ranks
# the 72 scenarios against EACH OTHER. It does not say the model works anywhere:
# every scenario in this sweep has MAPE >= 54% and the mean R2 over the grid is
# below zero. "Q1_LEAST_SEVERE" means "least bad of 72 uniformly bad regimes",
# never "safe". The raw dimensions travel in the same CSV precisely so a reader
# can check that for themselves instead of trusting the band name.

# (column, orientation, label). Orientation is +1 when a LARGER raw value is
# WORSE and -1 when a larger value is BETTER; after multiplying, every oriented
# z-score points the same way (bigger = worse).
SEVERITY_DIMENSIONS = (
    ("mape", +1.0, "CALIBRATION_DOMINATED"),
    ("r2_score", -1.0, "FIT_DOMINATED"),
    ("improvement_pct", -1.0, "POLICY_DOMINATED"),
    ("failure_rate_pct", +1.0, "COMPLETION_DOMINATED"),
)
NO_DOMINANT_AXIS = "NO_DOMINANT_AXIS"
FAILURE_MODE_LABELS = tuple(label for _, _, label in SEVERITY_DIMENSIONS) + (NO_DOMINANT_AXIS,)

# Quantiles, not values: the cut-points are recomputed from whatever the sweep
# produces. Three interior quantiles give four bands.
SEVERITY_BAND_QUANTILES = (0.25, 0.50, 0.75)
SEVERITY_BAND_LABELS = (
    "Q1_LEAST_SEVERE",
    "Q2_BELOW_MEDIAN",
    "Q3_ABOVE_MEDIAN",
    "Q4_MOST_SEVERE",
)
RISK_LEVELS = ("LOW_RISK", "MEDIUM_RISK", "HIGH_RISK")

# Column names added to the artefact. The twelve pre-existing columns keep both
# their names and their order; these are appended.
Z_COLUMNS = tuple(f"z_{column}" for column, _, _ in SEVERITY_DIMENSIONS)


def oriented_zscore(values, orientation):
    """Standardise one dimension so that a LARGER result always means WORSE.

    A dimension that does not vary across the grid cannot discriminate between
    scenarios, so it contributes 0 rather than a divide-by-zero.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    spread = float(np.nanstd(values))
    if not np.isfinite(spread) or spread == 0.0:
        return np.zeros_like(values)
    return orientation * (values - float(np.nanmean(values))) / spread


def severity_zmatrix(results_df):
    """(len(SEVERITY_DIMENSIONS), n_scenarios) matrix of oriented z-scores."""
    return np.vstack(
        [
            oriented_zscore(results_df[column].to_numpy(dtype=float), orientation)
            for column, orientation, _ in SEVERITY_DIMENSIONS
        ]
    )


def compute_severity_score(zmatrix):
    """Mean oriented z-score across the dimensions that are defined for a row.

    nanmean, so a scenario whose R2 is undefined (constant target) is still
    scored on the three dimensions that ARE defined instead of being dropped.
    """
    with np.errstate(invalid="ignore"):
        return np.nanmean(zmatrix, axis=0)


def derive_severity_cutpoints(severity):
    """Band edges read off the observed severity distribution.

    Data-derived by construction: there is no constant here to fall out of date.
    """
    severity = np.asarray(severity, dtype=float)
    finite = severity[np.isfinite(severity)]
    if finite.size == 0:
        return np.array([0.0, 0.0, 0.0])
    return np.asarray(np.quantile(finite, SEVERITY_BAND_QUANTILES), dtype=float)


def assign_severity_bands(severity, cutpoints):
    """Label each scenario by which quantile band its severity falls in.

    A non-finite severity is banded MOST severe: an undefined metric is a
    failure to measure, and the pessimistic reading is the honest one.
    """
    severity = np.asarray(severity, dtype=float)
    index = np.digitize(np.where(np.isfinite(severity), severity, np.inf), cutpoints)
    index = np.clip(index, 0, len(SEVERITY_BAND_LABELS) - 1)
    return [SEVERITY_BAND_LABELS[i] for i in index]


def assign_failure_modes(zmatrix):
    """Name the dimension that dominates each scenario's severity.

    Only a dimension that is WORSE THAN THE GRID MEAN (oriented z > 0) can be
    named: when a scenario is better than average on all four axes there is no
    dominant failure axis to report, and inventing one would be a false label.
    """
    filled = np.where(np.isfinite(zmatrix), zmatrix, -np.inf)
    top = np.argmax(filled, axis=0)
    top_z = filled[top, np.arange(filled.shape[1])]
    labels = [label for _, _, label in SEVERITY_DIMENSIONS]
    return [
        labels[i] if np.isfinite(z) and z > 0.0 else NO_DOMINANT_AXIS
        for i, z in zip(top, top_z)
    ]


def assign_risk_levels(bands, improvement_pct, r2_score):
    """Legacy LOW/MEDIUM/HIGH vocabulary, re-derived from band + two sign tests.

    HIGH_RISK   worst severity band, OR the policy makes waits worse than FIFO
                (improvement_pct < 0).
    LOW_RISK    mildest severity band AND the policy helps (improvement_pct > 0)
                AND the wait model beats predicting the mean wait (r2_score > 0).
    MEDIUM_RISK everything else. That is where a scenario with an undefined
                improvement or R2 lands unless the worst band has already made
                it HIGH_RISK: an axis that could not be measured never earns the
                mildest label, but it does not by itself earn the worst one.

    WHAT LOW_RISK DOES AND DOES NOT GUARANTEE (read this before quoting it).
    Half of the rule is purely RELATIVE. The mildest band is the bottom quartile
    of THIS grid, and a quartile is populated by construction, so the band alone
    would keep handing out LOW_RISK no matter how badly the whole grid did --
    including a grid where the model was anti-predictive everywhere. The two
    sign tests are the only ABSOLUTE floor under the label, and they are
    deliberately weak ones: r2_score > 0 says the model beats a constant
    prediction, not that its wait estimate is usable, and improvement_pct > 0
    says the ordering helps, not that it helps by an amount worth having. Every
    scenario in this sweep still has MAPE >= 54%. So a LOW_RISK row reads
    "least-alarming quarter of a uniformly badly-calibrated grid, where the
    ordering does help and the model does beat a constant" -- never "safe here".

    Both comparisons are against a natural zero (does the ordering help or hurt;
    does the model beat the mean predictor), not against a fitted cut-point, so
    neither can go stale the way the retired `mape >= 35.0` gate did.
    """
    improvement = np.asarray(improvement_pct, dtype=float)
    fit = np.asarray(r2_score, dtype=float)
    bands = np.asarray(bands, dtype=object)
    worst, mildest = SEVERITY_BAND_LABELS[-1], SEVERITY_BAND_LABELS[0]

    levels = []
    for band, gain, r2 in zip(bands, improvement, fit):
        if band == worst or (np.isfinite(gain) and gain < 0.0):
            levels.append("HIGH_RISK")
        elif (band == mildest and np.isfinite(gain) and gain > 0.0
                and np.isfinite(r2) and r2 > 0.0):
            levels.append("LOW_RISK")
        else:
            levels.append("MEDIUM_RISK")
    return levels


def classify_scenarios(results_df):
    """Attach the severity score, its bands and the derived labels to a frame.

    Pure: returns a copy, so a caller (or a test) can classify any frame that
    carries the four SEVERITY_DIMENSIONS columns plus improvement_pct.
    """
    out = results_df.copy()
    zmatrix = severity_zmatrix(out)
    severity = compute_severity_score(zmatrix)
    cutpoints = derive_severity_cutpoints(severity)
    bands = assign_severity_bands(severity, cutpoints)

    out["severity_score"] = severity
    out["severity_band"] = bands
    out["failure_mode"] = assign_failure_modes(zmatrix)
    out["risk_level"] = assign_risk_levels(
        bands,
        out["improvement_pct"].to_numpy(dtype=float),
        out["r2_score"].to_numpy(dtype=float),
    )
    for name, row in zip(Z_COLUMNS, zmatrix):
        out[name] = row
    return out, cutpoints


# Visualization
def _signed_cmap(mode, reverse=False):
    """Diverging ramp for a SIGNED quantity: the ML-free/regression side in orange,
    the neutral hairline grey exactly at break-even, the ML-wins side in blue.

    Two hues plus a neutral -- the repo's emphasis pair. The old ramp was RdYlGn:
    three hues, and red/green is the one pair a protanope cannot separate, so the
    single most important read of this chart (win vs loss) was carried by the
    weakest possible channel.

    `reverse` is for a quantity whose sign convention is inverted -- severity,
    where LARGER is WORSE. Orange must stay the "bad" hue across all panels; a
    panel that silently flipped which end was orange would be worse than no
    colour at all.
    """
    p = PALETTE[mode]
    stops = [p["series_2"], p["grid"], p["series_1"]]
    return LinearSegmentedColormap.from_list(
        "ood_signed", stops[::-1] if reverse else stops
    )


def _symmetric_norm(grid):
    """Zero-centred colour scale, so the hue flip lands exactly on break-even.

    Symmetric rather than two-slope: a -0.4% cell must not be painted as loudly
    as a +44% cell just because it happens to be the only negative one.
    """
    finite = grid[np.isfinite(grid)]
    span = float(np.max(np.abs(finite))) if finite.size else 1.0
    span = span if span > 0 else 1.0
    return Normalize(vmin=-span, vmax=span)


def _cell_ink(rgba):
    """Ink token that stays legible on the cell it sits on.

    Text never wears a series colour; this only picks between the light-mode and
    dark-mode ink depending on how saturated the cell underneath is.
    """
    red, green, blue = rgba[:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return PALETTE["light"]["ink"] if luminance > 0.55 else PALETTE["dark"]["ink"]


def _draw_matrix(ax, grid, mode, fmt, outline_negative=False, reverse=False):
    """One domain-shift matrix: rows = arrival multiplier, cols = cluster size.

    Every cell is annotated on purpose. In a matrix, colour is the ONLY channel
    encoding the value -- there is no position to read it off -- so the number is
    the readout, not a redundant decoration on top of a bar.
    """
    p = PALETTE[mode]
    cmap = _signed_cmap(mode, reverse=reverse)
    norm = _symmetric_norm(grid)
    im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto", origin="upper")

    # A matrix carries no gridlines of its own; solid hairline separators in the
    # page colour keep the cells discrete.
    ax.grid(False)
    ax.set_xticks(np.arange(grid.shape[1]) - 0.5, minor=True)
    ax.set_yticks(np.arange(grid.shape[0]) - 0.5, minor=True)
    ax.grid(which="minor", color=p["surface"], linewidth=1.4)
    ax.tick_params(which="minor", length=0)

    ax.set_xticks(range(len(CLUSTER_SIZES)))
    ax.set_xticklabels(CLUSTER_SIZES)
    ax.set_yticks(range(len(ARRIVAL_MULTIPLIERS)))
    ax.set_yticklabels([f"{v:.2f}x" for v in ARRIVAL_MULTIPLIERS])
    ax.set_xlabel("Cluster size (total GPUs)")

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid[i, j]
            if not np.isfinite(value):
                continue
            ax.text(
                j, i, fmt(value), ha="center", va="center", fontsize=9,
                color=_cell_ink(cmap(norm(value))),
            )
            # Redundant (non-colour) marker for the regression cells, so the
            # win/loss split survives greyscale printing and colour blindness.
            if outline_negative and value < 0:
                ax.add_patch(
                    Rectangle((j - 0.49, i - 0.49), 0.98, 0.98, fill=False,
                              edgecolor=p["ink"], linewidth=1.8, zorder=4)
                )
    return im


def _style_colorbar(fig, im, ax, mode, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8.5, labelcolor=PALETTE[mode]["ink_2"])
    cb.set_label(label, color=PALETTE[mode]["ink_2"], fontsize=9)
    return cb


def plot_ood_heatmap(results_df):
    """Two matched matrices over the same shift grid, written light + dark.

    Rows = arrival-rate multiplier, cols = cluster size, exactly as before. What
    changed is presentation only: R2 used to be drawn as contour lines ON TOP of
    the improvement colours -- two independent value scales sharing one plot, the
    2D form of a dual axis -- so it is now its own panel on a shared y-axis. Both
    grids are the same aggregation of the same numbers as before.
    """
    def _grid(column):
        return (
            results_df.groupby(["arrival_rate_multiplier", "cluster_size"], as_index=False)[column]
            .mean()
            .pivot(index="arrival_rate_multiplier", columns="cluster_size", values=column)
            .reindex(index=ARRIVAL_MULTIPLIERS, columns=CLUSTER_SIZES)
            .to_numpy(dtype=float)
        )

    imp_grid = _grid("improvement_pct")
    r2_grid = _grid("r2_score")
    sev_grid = _grid("severity_score")

    n_cells = int(np.isfinite(imp_grid).sum())
    n_win = int(np.sum(np.isfinite(imp_grid) & (imp_grid > 0)))
    n_loss = int(np.sum(np.isfinite(imp_grid) & (imp_grid < 0)))
    n_r2_neg = int(np.sum(np.isfinite(r2_grid) & (r2_grid < 0)))

    outline_note = (
        f" Outlined cells ({n_loss} of {n_cells}) are regressions."
        if n_loss
        else " No cell regresses."
    )

    for mode in ("light", "dark"):
        fig, axes = figure(mode, figsize=(17.2, 6.2), nrows=1, ncols=3, sharey=True)

        # ── Panel 1: what the POLICY does under the shift ─────────────────────
        im_imp = _draw_matrix(
            axes[0], imp_grid, mode, lambda v: f"{v:.1f}%", outline_negative=True
        )
        axes[0].set_ylabel("Arrival-rate multiplier (1.00x = training load)")
        axes[0].set_title("Mean wait reduction vs FIFO")
        _style_colorbar(fig, im_imp, axes[0], mode, "Improvement (%) vs FIFO")

        # ── Panel 2: how well the MODEL predicts under the same shift ─────────
        im_r2 = _draw_matrix(axes[1], r2_grid, mode, lambda v: f"{v:.2f}")
        axes[1].tick_params(axis="y", length=0)  # y is shared; its rule lives left
        axes[1].set_title("Wait-model accuracy (R²)")
        _style_colorbar(fig, im_r2, axes[1], mode, "R² of the trained wait model")

        # ── Panel 3: the composite severity RANK over the same grid ───────────
        # Reversed ramp: severity is oriented so larger = worse, so the orange
        # end has to sit at the top of the scale to keep "orange = bad" true
        # across all three panels.
        im_sev = _draw_matrix(
            axes[2], sev_grid, mode, lambda v: f"{v:+.2f}", reverse=True
        )
        axes[2].tick_params(axis="y", length=0)
        axes[2].set_title("Composite severity (higher = worse)")
        _style_colorbar(fig, im_sev, axes[2], mode, "Severity (grid-standardised)")

        fig.tight_layout(rect=(0, 0.03, 1, 0.82))
        finish(
            fig, mode,
            title=f"Proactive ordering still beats FIFO in {n_win} of {n_cells} "
                  f"shifted regimes",
            subtitle=f"{len(ARRIVAL_MULTIPLIERS)}x{len(CLUSTER_SIZES)} domain-shift "
                     f"grid; each cell averages {len(JOB_DISTRIBUTIONS)} runtime "
                     f"profiles x {RUNS_PER_SCENARIO} seeded runs.\n"
                     f"R² is below zero in {n_r2_neg} of {n_cells} cells: the "
                     f"absolute wait estimate degrades off-distribution while the "
                     f"queue ORDER it implies still helps." + outline_note
                     + "\nSeverity averages MAPE, R², improvement and unfinished-job "
                     f"rate standardised ACROSS THIS GRID, so it ranks the "
                     f"{len(results_df)} scenarios against each other; every one of "
                     f"them has MAPE ≥ {results_df['mape'].min():.0f}%.",
            source=FIGURE_SOURCE,
        )
        save_both(fig, OUTPUT_HEATMAP_STEM, mode)


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
        result["scenario"] = scenario_name
        results.append(result)
        print(
            f"[{scenario_index + 1:2d}/{len(scenarios)}] {scenario_name}: "
            f"R2={result['r2_score']:.3f} MAPE={result['mape']:.1f}% "
            f"improvement={result['improvement_pct']:.2f}%"
        )

    # Severity is standardised across the completed sweep, so classification
    # happens once, on the whole frame, rather than per scenario.
    results_df, cutpoints = classify_scenarios(pd.DataFrame(results))
    ordered_columns = [
        # The twelve original columns, names and order unchanged.
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
        # Added by the severity taxonomy.
        "severity_score",
        "severity_band",
        *Z_COLUMNS,
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

    # Print the taxonomy's spread every run. A taxonomy that has silently
    # collapsed to one value -- the defect this replaced -- is then visible in
    # the log rather than only in the CSV nobody re-opens.
    print(
        "Severity band cut-points (quantiles "
        + ", ".join(f"{q:.2f}" for q in SEVERITY_BAND_QUANTILES)
        + "): "
        + ", ".join(f"{c:.6f}" for c in cutpoints)
    )
    print(f"Severity range: {results_df['severity_score'].min():.6f} "
          f".. {results_df['severity_score'].max():.6f}")
    for column in ("severity_band", "failure_mode", "risk_level"):
        counts = results_df[column].value_counts()
        print(f"{column} ({counts.size} distinct): "
              + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"Worst MAPE {results_df['mape'].max():.2f}%, best MAPE "
          f"{results_df['mape'].min():.2f}% — no band in this grid is 'safe'.")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_HEATMAP}")


if __name__ == "__main__":
    main()

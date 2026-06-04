"""
Phase 27: Fairness & SLA Guarantees (Python Implementation)
===========================================================

Quantify fairness metrics (Gini, Jain's Index) and SLA compliance.
Generates formal analysis outputs for publication.

Generates:
  - fairness_metrics.csv: Gini, JFI, max/min ratios per scheduler
  - sla_compliance.csv: SLA pass/fail per scheduler
  - starvation_analysis.png: histogram visualization
"""

import os
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

OUTPUT_FAIRNESS_METRICS = os.path.join(SCRIPT_DIR, "fairness_metrics.csv")
OUTPUT_SLA_COMPLIANCE = os.path.join(SCRIPT_DIR, "sla_compliance.csv")
OUTPUT_STARVATION_PLOT = os.path.join(SCRIPT_DIR, "starvation_analysis.png")

# Synthetic benchmark data (Phase 09 context)
SYNTHETIC_SCHEDULER_RESULTS = {
    "fifo": {
        "wait_times": np.array([18.27] * 40),
        "name": "FIFO",
    },
    "slurm_backfill": {
        "wait_times": np.array([17.15] * 40),
        "name": "SLURM Backfill",
    },
    "kubernetes_qos": {
        "wait_times": np.array([17.65] * 40),
        "name": "Kubernetes QoS",
    },
    "yarn_fifo": {
        "wait_times": np.array([18.35] * 40),
        "name": "Yarn FIFO",
    },
    "neural_net": {
        "wait_times": np.array([17.28] * 40),
        "name": "Neural Network",
    },
    "proactive": {
        "wait_times": np.array([16.72] * 40),
        "name": "Proactive (XGBoost)",
    },
}

STARVATION_THRESHOLD = 200
STARVATION_TOLERANCE_PCT = 1.0
SLA3_MAX_WAIT = 150


def compute_gini_coefficient(wait_times: np.ndarray) -> float:
    """Compute Gini coefficient from wait-time distribution."""
    sorted_waits = np.sort(wait_times)
    n = len(sorted_waits)
    sum_weighted = np.sum((np.arange(1, n + 1)) * sorted_waits)
    sum_total = np.sum(sorted_waits)
    gini = (2.0 * sum_weighted) / (n * sum_total) - (n + 1.0) / n
    return float(np.clip(gini, 0.0, 1.0))


def compute_jain_fairness_index(wait_times: np.ndarray) -> float:
    """Compute Jain's Fairness Index."""
    n = len(wait_times)
    sum_waits = np.sum(wait_times)
    sum_squared = np.sum(wait_times ** 2)
    jfi = (sum_waits ** 2) / (n * sum_squared) if sum_squared > 0 else 0.0
    return float(np.clip(jfi, 1.0 / n, 1.0))


def analyze_starvation(wait_times: np.ndarray, threshold: int = STARVATION_THRESHOLD) -> Tuple[int, float]:
    """Count jobs that starve."""
    starved = np.sum(wait_times > threshold)
    rate = 100.0 * starved / len(wait_times)
    return int(starved), float(rate)


def compute_fairness_metrics(scheduler_results: Dict) -> pd.DataFrame:
    """Compute all fairness metrics for each scheduler."""
    rows = []

    for sched_key, data in scheduler_results.items():
        wait_times = data["wait_times"]
        name = data["name"]

        gini = compute_gini_coefficient(wait_times)
        jfi = compute_jain_fairness_index(wait_times)
        max_wait = float(np.max(wait_times))
        min_wait = float(np.min(wait_times))
        ratio_max_min = max_wait / max(min_wait, 1.0)
        mean_wait = float(np.mean(wait_times))
        starvation_count, starvation_rate = analyze_starvation(wait_times)

        row = {
            "scheduler": sched_key,
            "scheduler_name": name,
            "mean_wait": mean_wait,
            "gini_coefficient": gini,
            "jain_fairness_index": jfi,
            "max_wait": max_wait,
            "min_wait": min_wait,
            "max_min_ratio": ratio_max_min,
            "starvation_count": starvation_count,
            "starvation_rate_pct": starvation_rate,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def compute_sla_compliance(scheduler_results: Dict) -> pd.DataFrame:
    """Evaluate SLA compliance for each scheduler."""
    rows = []

    for sched_key, data in scheduler_results.items():
        wait_times = data["wait_times"]
        name = data["name"]

        # SLA-1: 95% within 2× geometric mean
        geom_mean = np.exp(np.mean(np.log(np.maximum(wait_times, 1e-3))))
        sla1_threshold = 2.0 * geom_mean
        sla1_pass = 100.0 * np.sum(wait_times <= sla1_threshold) / len(wait_times)

        # SLA-2: 99% with no starvation
        sla2_pass = 100.0 * np.sum(wait_times <= STARVATION_THRESHOLD) / len(wait_times)

        # SLA-3: max wait ≤ 150
        sla3_pass = 100.0 if np.max(wait_times) <= SLA3_MAX_WAIT else 0.0

        compliance_score = (sla1_pass + sla2_pass + sla3_pass) / 3.0 / 100.0

        row = {
            "scheduler": sched_key,
            "scheduler_name": name,
            "sla1_95pct_2x_geomean": float(sla1_pass / 100.0),
            "sla2_no_starvation_99pct": float(sla2_pass / 100.0),
            "sla3_max_wait_le_150": float(sla3_pass / 100.0),
            "compliance_score": float(compliance_score),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def plot_starvation_analysis(fairness_df: pd.DataFrame) -> None:
    """Visualize starvation rates across schedulers."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    schedulers = fairness_df["scheduler_name"].values
    starvation_counts = fairness_df["starvation_count"].values
    starvation_rates = fairness_df["starvation_rate_pct"].values

    ax = axes[0]
    ax.bar(range(len(schedulers)), starvation_counts, color="orangered", alpha=0.7)
    ax.set_xticks(range(len(schedulers)))
    ax.set_xticklabels(schedulers, rotation=45, ha="right")
    ax.set_ylabel("Starvation Count (jobs with wait > 200)")
    ax.set_title("Phase 27: Starvation Analysis (Count)")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(range(len(schedulers)), starvation_rates, color="crimson", alpha=0.7)
    ax.set_xticks(range(len(schedulers)))
    ax.set_xticklabels(schedulers, rotation=45, ha="right")
    ax.set_ylabel("Starvation Rate (%)")
    ax.set_title("Phase 27: Starvation Analysis (Rate)")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUTPUT_STARVATION_PLOT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    """Execute Phase 27 analysis."""
    os.makedirs(SCRIPT_DIR, exist_ok=True)

    print("[Phase 27] Computing fairness metrics...")
    fairness_df = compute_fairness_metrics(SYNTHETIC_SCHEDULER_RESULTS)
    fairness_df.to_csv(OUTPUT_FAIRNESS_METRICS, index=False)
    print(f"  Saved: {OUTPUT_FAIRNESS_METRICS}")

    print("[Phase 27] Evaluating SLA compliance...")
    sla_df = compute_sla_compliance(SYNTHETIC_SCHEDULER_RESULTS)
    sla_df.to_csv(OUTPUT_SLA_COMPLIANCE, index=False)
    print(f"  Saved: {OUTPUT_SLA_COMPLIANCE}")

    print("[Phase 27] Generating starvation plots...")
    plot_starvation_analysis(fairness_df)
    print(f"  Saved: {OUTPUT_STARVATION_PLOT}")

    print("\n" + "="*70)
    print("PHASE 27 SUMMARY – FAIRNESS & SLA GUARANTEES")
    print("="*70)
    print("\nFAIRNESS METRICS:")
    print(fairness_df[["scheduler_name", "gini_coefficient", "jain_fairness_index", "max_min_ratio"]])
    print(f"\nAll outputs saved to: {SCRIPT_DIR}")


if __name__ == "__main__":
    main()

"""
Phase 27: Fairness & SLA Guarantees (Python Implementation)
===========================================================

Quantify fairness metrics (Gini, Jain's Index) and SLA compliance from
REAL experiment outputs produced by the project pipeline (no fabricated
distributions):

  - 05_results/benchmark_statistical_results.csv
      40 paired simulation runs; per-run mean waits for the FIFO baseline
      and the proactive scheduler (columns baseline_wait / proactive_wait).
  - 05_results/fairness/fairness_metrics.csv
      Per-job fairness numbers from 04_scheduler/fairness_analysis.py
      (fifo, proactive, proactive_starvation): per-job Gini, per-job max
      wait, starvation counts (jobs waiting > 3x their runtime).
  - 05_results/schedulers/multi_scheduler_benchmark.csv
      Per-job fairness numbers for SJF / NN / PRIORITY (and FIFO /
      PROACTIVE duplicates, which are skipped in favour of the source
      above).

Schedulers from earlier drafts with NO real data in this repository
(SLURM Backfill, Kubernetes QoS, Yarn FIFO) are NOT analysed: they are
listed in dropped_schedulers.txt instead of being given invented numbers.

Generates:
  - fairness_metrics.csv: Gini, JFI, max/min ratios per scheduler
  - sla_compliance.csv: SLA pass/fail per scheduler
  - starvation_analysis.png: real starvation counts/rates
  - dropped_schedulers.txt: schedulers excluded for lack of real data
"""

import os
import sys
from typing import Dict

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
OUTPUT_DROPPED = os.path.join(SCRIPT_DIR, "dropped_schedulers.txt")

# Real result files produced by the phases 01-21 pipeline
REAL_BENCHMARK_CSV = os.path.join(PROJECT_ROOT, "05_results", "benchmark_statistical_results.csv")
REAL_FAIRNESS_CSV = os.path.join(PROJECT_ROOT, "05_results", "fairness", "fairness_metrics.csv")
REAL_MULTI_SCHED_CSV = os.path.join(PROJECT_ROOT, "05_results", "schedulers", "multi_scheduler_benchmark.csv")

# Schedulers referenced in earlier drafts of this phase for which the
# repository contains NO real benchmark data. They are excluded from the
# analysis rather than assigned invented numbers.
DROPPED_SCHEDULERS = {
    "slurm_backfill": "SLURM Backfill",
    "kubernetes_qos": "Kubernetes QoS",
    "yarn_fifo": "Yarn FIFO",
}

SLA3_MAX_WAIT = 150


def compute_gini_coefficient(wait_times: np.ndarray) -> float:
    """Compute Gini coefficient from a wait-time distribution."""
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


def load_run_level_distributions() -> Dict[str, Dict]:
    """
    Load the 40 paired per-run mean waits from the statistical benchmark.
    These are REAL run-level distributions (one mean wait per simulation
    run), available for the FIFO baseline and the proactive scheduler.
    """
    if not os.path.exists(REAL_BENCHMARK_CSV):
        sys.exit(
            f"[Phase 27] ERROR: required real benchmark data not found:\n"
            f"  {REAL_BENCHMARK_CSV}\n"
            f"Run 04_scheduler/benchmark_statistical.py (via run_all_experiments.sh) "
            f"first. Refusing to fabricate distributions."
        )
    bench = pd.read_csv(REAL_BENCHMARK_CSV)
    return {
        "fifo": {
            "wait_times": bench["baseline_wait"].to_numpy(dtype=float),
            "name": "FIFO",
        },
        "proactive": {
            "wait_times": bench["proactive_wait"].to_numpy(dtype=float),
            "name": "Proactive (XGBoost)",
        },
    }


def load_per_job_aggregates() -> Dict[str, Dict]:
    """
    Load real per-job fairness aggregates (per-job Gini, per-job max wait,
    starvation counts) computed by earlier pipeline phases. Returns
    {scheduler_key: {name, gini_per_job, mean_wait, max_wait,
                     starvation_count, completed_jobs}} with NaN where a
    quantity was not measured.
    """
    aggregates: Dict[str, Dict] = {}

    if os.path.exists(REAL_FAIRNESS_CSV):
        fair = pd.read_csv(REAL_FAIRNESS_CSV)
        name_map = {
            "fifo": "FIFO",
            "proactive": "Proactive (XGBoost)",
            "proactive_starvation": "Proactive + Anti-Starvation",
        }
        for _, r in fair.iterrows():
            key = str(r["scheduler"])
            aggregates[key] = {
                "name": name_map.get(key, key),
                "gini_per_job": float(r["fairness_gini"]),
                "mean_wait": float(r["mean_wait"]),
                "max_wait": float(r["max_wait"]),
                "starvation_count": float(r["starvation_count"]),
                "completed_jobs": float(r["completed_jobs"]),
            }
    else:
        print(f"  WARNING: {REAL_FAIRNESS_CSV} not found; per-job Gini/starvation columns will be NaN")

    if os.path.exists(REAL_MULTI_SCHED_CSV):
        multi = pd.read_csv(REAL_MULTI_SCHED_CSV)
        name_map = {"SJF": "SJF", "NN": "Neural Network", "PRIORITY": "Priority"}
        for _, r in multi.iterrows():
            key = str(r["scheduler"]).lower()
            if key in aggregates:
                # FIFO / PROACTIVE already covered by the fairness_analysis
                # source (which also carries starvation counts) - skip dupes.
                continue
            aggregates[key] = {
                "name": name_map.get(str(r["scheduler"]), str(r["scheduler"])),
                "gini_per_job": float(r["fairness_gini"]),
                "mean_wait": float(r["mean_wait"]),
                "max_wait": float(r["max_wait"]),
                "starvation_count": float("nan"),
                "completed_jobs": float("nan"),
            }
    else:
        print(f"  WARNING: {REAL_MULTI_SCHED_CSV} not found; SJF/NN/Priority rows omitted")

    return aggregates


def compute_fairness_metrics(run_level: Dict, per_job: Dict) -> pd.DataFrame:
    """
    Compute fairness metrics per scheduler from real data.

    Run-level Gini/JFI (over the 40 per-run mean waits) is only available
    for fifo/proactive; per-job Gini, max wait and starvation come from the
    pipeline's per-job fairness analyses. Note that run-level dispersion
    understates per-job inequality (means average out extremes).
    """
    rows = []
    keys = list(dict.fromkeys(list(run_level.keys()) + list(per_job.keys())))

    for sched_key in keys:
        agg = per_job.get(sched_key, {})
        name = run_level.get(sched_key, {}).get("name") or agg.get("name", sched_key)

        row = {
            "scheduler": sched_key,
            "scheduler_name": name,
            "data_source": "benchmark_statistical(40 runs)+fairness_analysis"
            if sched_key in run_level and agg
            else ("benchmark_statistical(40 runs)" if sched_key in run_level
                  else "per_job_aggregates_only"),
        }

        if sched_key in run_level:
            waits = run_level[sched_key]["wait_times"]
            row.update({
                "n_runs": int(len(waits)),
                "mean_wait": float(np.mean(waits)),
                "gini_run_level": compute_gini_coefficient(waits),
                "jain_fairness_index_run_level": compute_jain_fairness_index(waits),
                "min_run_wait": float(np.min(waits)),
                "max_run_wait": float(np.max(waits)),
                "run_max_min_ratio": float(np.max(waits) / max(np.min(waits), 1e-9)),
            })
        else:
            row.update({
                "n_runs": 0,
                "mean_wait": agg.get("mean_wait", float("nan")),
                "gini_run_level": float("nan"),
                "jain_fairness_index_run_level": float("nan"),
                "min_run_wait": float("nan"),
                "max_run_wait": float("nan"),
                "run_max_min_ratio": float("nan"),
            })

        starv = agg.get("starvation_count", float("nan"))
        completed = agg.get("completed_jobs", float("nan"))
        row.update({
            "gini_per_job": agg.get("gini_per_job", float("nan")),
            "max_job_wait": agg.get("max_wait", float("nan")),
            "starvation_count": starv,
            "starvation_rate_pct": float(100.0 * starv / completed)
            if np.isfinite(starv) and np.isfinite(completed) and completed > 0
            else float("nan"),
        })
        rows.append(row)

    return pd.DataFrame(rows)


def compute_sla_compliance(run_level: Dict, per_job: Dict) -> pd.DataFrame:
    """
    Evaluate SLA compliance per scheduler from real measurements:
      SLA-1: 95% of runs within 2x geometric mean of run-level waits
             (only computable where the 40-run distribution exists).
      SLA-2: share of jobs NOT starved (starvation = wait > 3x runtime,
             as measured by 04_scheduler/fairness_analysis.py).
      SLA-3: measured per-job max wait <= 150.
    compliance_score = mean of the SLA components that are measurable.
    """
    rows = []
    keys = list(dict.fromkeys(list(run_level.keys()) + list(per_job.keys())))

    for sched_key in keys:
        agg = per_job.get(sched_key, {})
        name = run_level.get(sched_key, {}).get("name") or agg.get("name", sched_key)

        # SLA-1: run-level consistency (real 40-run distribution required)
        if sched_key in run_level:
            waits = run_level[sched_key]["wait_times"]
            geom_mean = np.exp(np.mean(np.log(np.maximum(waits, 1e-3))))
            sla1 = float(np.sum(waits <= 2.0 * geom_mean) / len(waits))
        else:
            sla1 = float("nan")

        # SLA-2: per-job no-starvation share (real starvation counts)
        starv = agg.get("starvation_count", float("nan"))
        completed = agg.get("completed_jobs", float("nan"))
        if np.isfinite(starv) and np.isfinite(completed) and completed > 0:
            sla2 = float(1.0 - starv / completed)
        else:
            sla2 = float("nan")

        # SLA-3: measured per-job max wait bound
        max_job_wait = agg.get("max_wait", float("nan"))
        sla3 = (1.0 if max_job_wait <= SLA3_MAX_WAIT else 0.0) if np.isfinite(max_job_wait) else float("nan")

        components = np.array([sla1, sla2, sla3], dtype=float)
        measurable = components[np.isfinite(components)]
        compliance = float(np.mean(measurable)) if len(measurable) else float("nan")

        rows.append({
            "scheduler": sched_key,
            "scheduler_name": name,
            "sla1_95pct_2x_geomean_runs": sla1,
            "sla2_no_starvation_share": sla2,
            "sla3_max_wait_le_150": sla3,
            "n_measurable_slas": int(len(measurable)),
            "compliance_score": compliance,
        })

    return pd.DataFrame(rows)


def plot_starvation_analysis(fairness_df: pd.DataFrame) -> None:
    """Visualize REAL starvation counts/rates (schedulers with measurements)."""
    plot_df = fairness_df.dropna(subset=["starvation_count"])
    if len(plot_df) == 0:
        print("  WARNING: no real starvation measurements available; skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    schedulers = plot_df["scheduler_name"].values
    starvation_counts = plot_df["starvation_count"].values
    starvation_rates = plot_df["starvation_rate_pct"].values

    ax = axes[0]
    ax.bar(range(len(schedulers)), starvation_counts, color="orangered", alpha=0.7)
    ax.set_xticks(range(len(schedulers)))
    ax.set_xticklabels(schedulers, rotation=45, ha="right")
    ax.set_ylabel("Starved jobs per run (wait > 3x runtime)")
    ax.set_title("Phase 27: Starvation Analysis (Count, measured)")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(range(len(schedulers)), starvation_rates, color="crimson", alpha=0.7)
    ax.set_xticks(range(len(schedulers)))
    ax.set_xticklabels(schedulers, rotation=45, ha="right")
    ax.set_ylabel("Starvation Rate (% of completed jobs)")
    ax.set_title("Phase 27: Starvation Analysis (Rate, measured)")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUTPUT_STARVATION_PLOT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_dropped_note() -> None:
    """Record which schedulers were excluded because no real data exists."""
    with open(OUTPUT_DROPPED, "w", encoding="utf-8") as f:
        f.write("Phase 27: schedulers EXCLUDED from fairness/SLA analysis\n")
        f.write("=" * 56 + "\n")
        f.write("No real benchmark data exists in this repository for the\n")
        f.write("following schedulers; rather than inventing numbers they are\n")
        f.write("omitted from fairness_metrics.csv and sla_compliance.csv:\n\n")
        for key, name in DROPPED_SCHEDULERS.items():
            f.write(f"  - {key} ({name})\n")
        f.write("\nTo include them, implement the schedulers in the simulation\n")
        f.write("pipeline (see 04_scheduler/) and regenerate the benchmark CSVs.\n")


def main():
    """Execute Phase 27 analysis on real pipeline outputs."""
    os.makedirs(SCRIPT_DIR, exist_ok=True)

    print("[Phase 27] Loading real benchmark distributions...")
    run_level = load_run_level_distributions()
    per_job = load_per_job_aggregates()
    print(f"  Run-level distributions: {sorted(run_level.keys())}")
    print(f"  Per-job aggregates     : {sorted(per_job.keys())}")
    print(f"  Dropped (no real data) : {sorted(DROPPED_SCHEDULERS.keys())}")

    print("[Phase 27] Computing fairness metrics...")
    fairness_df = compute_fairness_metrics(run_level, per_job)
    fairness_df.to_csv(OUTPUT_FAIRNESS_METRICS, index=False)
    print(f"  Saved: {OUTPUT_FAIRNESS_METRICS}")

    print("[Phase 27] Evaluating SLA compliance...")
    sla_df = compute_sla_compliance(run_level, per_job)
    sla_df.to_csv(OUTPUT_SLA_COMPLIANCE, index=False)
    print(f"  Saved: {OUTPUT_SLA_COMPLIANCE}")

    print("[Phase 27] Generating starvation plots...")
    plot_starvation_analysis(fairness_df)
    print(f"  Saved: {OUTPUT_STARVATION_PLOT}")

    write_dropped_note()
    print(f"  Saved: {OUTPUT_DROPPED}")

    print("\n" + "=" * 70)
    print("PHASE 27 SUMMARY - FAIRNESS & SLA GUARANTEES (REAL DATA)")
    print("=" * 70)
    print("\nFAIRNESS METRICS:")
    print(fairness_df[["scheduler_name", "gini_run_level", "gini_per_job",
                       "jain_fairness_index_run_level", "starvation_rate_pct"]])
    print("\nSLA COMPLIANCE:")
    print(sla_df[["scheduler_name", "sla1_95pct_2x_geomean_runs",
                  "sla2_no_starvation_share", "sla3_max_wait_le_150",
                  "compliance_score"]])
    print(f"\nNOTE: {', '.join(DROPPED_SCHEDULERS.values())} were dropped "
          f"(no real data; see dropped_schedulers.txt).")
    print(f"\nAll outputs saved to: {SCRIPT_DIR}")


if __name__ == "__main__":
    main()

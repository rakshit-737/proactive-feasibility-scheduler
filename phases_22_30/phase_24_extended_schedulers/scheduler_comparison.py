"""
Phase 24: Extended Scheduler Comparison
========================================

Compare the proactive scheduler against the baselines actually implemented
and benchmarked in this project (04_scheduler/multi_scheduler_benchmark.py):
  - FIFO (arrival-order baseline)
  - SJF (shortest-job-first heuristic)
  - Priority (priority-score heuristic)
  - Neural Network scheduler (MLP-based, Phase 14)
  - Proactive (XGBoost wait-model ordering, Phase 09)

All numbers are read from REAL benchmark outputs:
  - 05_results/schedulers/multi_scheduler_benchmark.csv (15-run multi-scheduler
    benchmark, per-scheduler means)
  - 05_results/benchmark_statistical_results.csv (40-run paired FIFO-vs-proactive
    benchmark, per-run rows) for the statistical significance test.
No metric in the outputs is hardcoded; if the input CSVs are missing the
script exits with instructions to regenerate them.

Generates:
  - baseline_comparison.csv: side-by-side metrics across schedulers
  - scheduler_heatmap.png: wait time, fairness, throughput comparison
  - novelty_claim.txt: quantitative positioning derived from the computed numbers
"""

import os
import pickle
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

OUTPUT_CSV = os.path.join(SCRIPT_DIR, "baseline_comparison.csv")
OUTPUT_HEATMAP = os.path.join(SCRIPT_DIR, "scheduler_heatmap.png")
OUTPUT_CLAIM = os.path.join(SCRIPT_DIR, "novelty_claim.txt")

# Scheduler names and characteristics (keys match the 'scheduler' column of
# 05_results/schedulers/multi_scheduler_benchmark.csv)
SCHEDULERS = {
    "FIFO": {
        "name": "FCFS + first-fit (liberal dispatch)",
        "type": "baseline",
        "reference": "04_scheduler/multi_scheduler_benchmark.py (any fitting job starts; label kept for continuity)",
    },
    "FIFO_STRICT": {
        "name": "FIFO (head-blocking)",
        "type": "baseline",
        "reference": "04_scheduler/multi_scheduler_benchmark.py (textbook FIFO, blocked head blocks queue)",
    },
    "SJF": {
        "name": "Shortest Job First (oracle)",
        "type": "heuristic_baseline",
        "reference": "04_scheduler/sjf_scheduler.py (true runtimes)",
    },
    "SJF_EST": {
        "name": "SJF (user estimates)",
        "type": "heuristic_baseline",
        "reference": "04_scheduler/sjf_scheduler.py (f-model estimates, C=5)",
    },
    "PRIORITY": {
        "name": "Priority",
        "type": "heuristic_baseline",
        "reference": "04_scheduler/priority_scheduler.py",
    },
    "HRRN": {
        "name": "Highest Response Ratio Next",
        "type": "heuristic_baseline",
        "reference": "04_scheduler/hrrn_scheduler.py (starvation-aware)",
    },
    "SMALLEST": {
        "name": "Smallest requested size first (no ML)",
        "type": "degeneracy_control",
        "reference": "04_scheduler/size_scheduler.py (the ML-free control implied "
                     "by ranking_degeneracy.py: the wait model's per-job features "
                     "are all functions of requested size)",
    },
    "BACKFILL": {
        "name": "EASY backfill (oracle)",
        "type": "heuristic_baseline",
        "reference": "04_scheduler/backfill_scheduler.py (true runtimes)",
    },
    "BACKFILL_EST": {
        "name": "EASY backfill (user estimates)",
        "type": "heuristic_baseline",
        "reference": "04_scheduler/backfill_scheduler.py (f-model estimates, C=5)",
    },
    "CONS_BF": {
        "name": "Conservative backfill",
        "type": "heuristic_baseline",
        "reference": "04_scheduler/backfill_scheduler.py (reservation per queued job)",
    },
    "SRPT": {
        "name": "Preemptive SRPT (+1 tick overhead)",
        "type": "heuristic_baseline",
        "reference": "04_scheduler/multi_scheduler_benchmark.py (run_once_preemptive)",
    },
    "PROACTIVE_BF": {
        "name": "EASY backfill + predicted-wait scan",
        "type": "ml_hybrid",
        "reference": "04_scheduler/multi_scheduler_benchmark.py (proactive_bf)",
    },
    "NN": {
        "name": "Neural Network (MLP)",
        "type": "ml_baseline",
        "reference": "04_scheduler/neural_network_scheduler.py (Phase 14)",
    },
    "PROACTIVE": {
        "name": "Proactive (XGBoost)",
        "type": "proposed",
        "reference": "Phase 09 proactive feasibility scheduler",
    },
}

# Metrics available in the real multi-scheduler benchmark summary
METRICS = [
    "mean_wait",
    "max_wait",
    "fairness_gini",
    "throughput",
    "gpu_util",
]

# Wait-prediction error column: the benchmark renamed 'mae' to 'pred_wait_mae'
# (it is only defined for the predictor-driven schedulers); accept either.
PRED_MAE_CANDIDATES = ("pred_wait_mae", "mae")

MULTI_SCHEDULER_CSV = os.path.join(
    PROJECT_ROOT, "05_results", "schedulers", "multi_scheduler_benchmark.csv"
)

BENCHMARK_DATA_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "05_results", "benchmarks", "40_run_benchmark.pkl"),
    os.path.join(PROJECT_ROOT, "05_results", "benchmark_statistical_results.pkl"),
    os.path.join(PROJECT_ROOT, "05_results", "benchmark_statistical_results.csv"),
]


def _load_paired_benchmark() -> Tuple[Optional[pd.DataFrame], str]:
    """Load the Phase 09 40-run paired FIFO-vs-proactive benchmark (real data)."""
    for path in BENCHMARK_DATA_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            if path.endswith(".pkl"):
                with open(path, "rb") as f:
                    raw = pickle.load(f)
                if isinstance(raw, list):
                    df = pd.DataFrame(raw)
                elif isinstance(raw, dict) and "results" in raw:
                    df = pd.DataFrame(raw["results"])
                else:
                    df = pd.DataFrame(raw)
            else:
                df = pd.read_csv(path)
            if {"baseline_wait", "proactive_wait"}.issubset(df.columns):
                return df, path
        except Exception:
            continue
    return None, "missing"


def _compute_paired_stats(bench_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute FIFO-vs-proactive statistics from the per-run paired benchmark:
    mean waits, mean per-run improvement, paired t-test, 95% CI on the wait
    difference. Everything is computed from the loaded data.
    """
    baseline = bench_df["baseline_wait"].to_numpy(dtype=float)
    proactive = bench_df["proactive_wait"].to_numpy(dtype=float)
    diff = baseline - proactive
    n = len(diff)

    t_stat, p_value = stats.ttest_rel(baseline, proactive)
    se = diff.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, n - 1)

    if "improvement_pct" in bench_df.columns:
        mean_improvement_pct = float(bench_df["improvement_pct"].mean())
    else:
        mean_improvement_pct = float(np.mean(diff / baseline * 100.0))

    return {
        "n_runs": n,
        "baseline_wait_mean": float(baseline.mean()),
        "proactive_wait_mean": float(proactive.mean()),
        "mean_wait_diff": float(diff.mean()),
        "wait_diff_ci95_low": float(diff.mean() - t_crit * se),
        "wait_diff_ci95_high": float(diff.mean() + t_crit * se),
        "mean_improvement_pct": mean_improvement_pct,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def _load_scheduler_results() -> pd.DataFrame:
    """
    Load the real per-scheduler benchmark summary. There is no synthetic
    fallback: if the CSV is missing, exit with instructions to regenerate it.
    """
    if not os.path.exists(MULTI_SCHEDULER_CSV):
        raise SystemExit(
            "[Phase 24] ERROR: real benchmark results not found at\n"
            f"  {MULTI_SCHEDULER_CSV}\n"
            "Run `python 04_scheduler/multi_scheduler_benchmark.py` first to "
            "generate them (hardcoded/fabricated results are not used)."
        )

    df = pd.read_csv(MULTI_SCHEDULER_CSV)
    missing_cols = [c for c in ["scheduler"] + METRICS if c not in df.columns]
    if missing_cols:
        raise SystemExit(
            f"[Phase 24] ERROR: {MULTI_SCHEDULER_CSV} is missing columns: {missing_cols}"
        )
    for required in ("FIFO", "PROACTIVE"):
        if required not in set(df["scheduler"]):
            raise SystemExit(
                f"[Phase 24] ERROR: scheduler '{required}' not present in {MULTI_SCHEDULER_CSV}"
            )
    return df


def _compute_improvements_vs_fifo(summary_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute mean improvements for each scheduler vs. FIFO baseline
    from the real benchmark summary.
    """
    fifo = summary_df[summary_df["scheduler"] == "FIFO"].iloc[0]

    improvements = {}
    for _, row in summary_df.iterrows():
        wait_improvement_pct = (fifo["mean_wait"] - row["mean_wait"]) / fifo["mean_wait"] * 100.0
        throughput_improvement_pct = (row["throughput"] - fifo["throughput"]) / fifo["throughput"] * 100.0
        util_improvement_pct = (row["gpu_util"] - fifo["gpu_util"]) / fifo["gpu_util"] * 100.0
        gini_improvement_pct = (fifo["fairness_gini"] - row["fairness_gini"]) / fifo["fairness_gini"] * 100.0

        improvements[row["scheduler"]] = {
            "wait_improvement_pct": float(wait_improvement_pct),
            "throughput_improvement_pct": float(throughput_improvement_pct),
            "util_improvement_pct": float(util_improvement_pct),
            "fairness_improvement_pct": float(gini_improvement_pct),
        }

    return improvements


def build_comparison_dataframe(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the comparison CSV from the real per-scheduler benchmark summary.
    """
    rows = []

    pred_mae_col = next(
        (c for c in PRED_MAE_CANDIDATES if c in summary_df.columns), None
    )

    for _, r in summary_df.iterrows():
        sched_key = r["scheduler"]
        sched_info = SCHEDULERS.get(
            sched_key, {"name": sched_key, "type": "unknown", "reference": "n/a"}
        )
        # pred_wait_mae is only defined for predictor-driven schedulers
        # (PROACTIVE, NN); others carry NaN.
        pred_mae = float(r[pred_mae_col]) if pred_mae_col is not None else float("nan")
        row = {
            "scheduler": sched_key,
            "scheduler_name": sched_info["name"],
            "scheduler_type": sched_info["type"],
            "reference": sched_info["reference"],
            "mean_wait": float(r["mean_wait"]),
            "max_wait": float(r["max_wait"]),
            "fairness_gini": float(r["fairness_gini"]),
            "throughput_jobs_per_ts": float(r["throughput"]),
            "gpu_util": float(r["gpu_util"]),
            "predictor_mae": pred_mae,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def plot_comparison_heatmap(comparison_df: pd.DataFrame) -> None:
    """
    Create side-by-side heatmap comparing schedulers across normalized metrics.
    """
    # Normalize each metric to [0, 1] scale (lower=better for wait/gini, higher=better for throughput/util)
    metrics_to_plot = [
        "mean_wait",
        "throughput_jobs_per_ts",
        "gpu_util",
        "fairness_gini",
    ]
    normalize_invert = {
        "mean_wait": True,  # Lower is better
        "throughput_jobs_per_ts": False,  # Higher is better
        "gpu_util": False,  # Higher is better
        "fairness_gini": True,  # Lower is better (more fair)
    }

    data_for_heat = comparison_df[["scheduler_name"] + metrics_to_plot].set_index("scheduler_name")

    # Normalize
    for col in metrics_to_plot:
        col_min = data_for_heat[col].min()
        col_max = data_for_heat[col].max()
        if col_max == col_min:
            # Metric identical across schedulers (e.g., all jobs complete):
            # neutral score instead of a 0/0 division
            data_for_heat[col] = 0.5
            continue
        data_for_heat[col] = (data_for_heat[col] - col_min) / (col_max - col_min)

        # Invert if needed
        if normalize_invert[col]:
            data_for_heat[col] = 1.0 - data_for_heat[col]

    # Rename for display
    data_for_heat.columns = [
        "Wait Time",
        "Throughput",
        "GPU Util",
        "Fairness (1-Gini)",
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data_for_heat.T.to_numpy(), cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized Performance (↑ better)", rotation=90)

    ax.set_xticks(range(len(data_for_heat)))
    ax.set_xticklabels(data_for_heat.index, rotation=45, ha="right")
    ax.set_yticks(range(len(data_for_heat.columns)))
    ax.set_yticklabels(data_for_heat.columns)
    ax.set_title("Phase 24: Scheduler Comparison Heatmap (Normalized Performance)")

    # Annotate with actual values
    for i in range(len(data_for_heat)):
        for j in range(len(data_for_heat.columns)):
            value = data_for_heat.iloc[i, j]
            text = ax.text(i, j, f"{value:.2f}", ha="center", va="center", color="black", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_HEATMAP, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_novelty_claim(
    comparison_df: pd.DataFrame,
    improvements: Dict[str, Dict[str, float]],
    paired_stats: Optional[Dict[str, float]],
) -> None:
    """
    Write a positioning statement in which EVERY quantitative claim is derived
    from the numbers computed above (real benchmark CSVs), including honest
    negative findings where the proactive scheduler does not win.
    """
    proactive_row = comparison_df[comparison_df["scheduler"] == "PROACTIVE"].iloc[0]
    fifo_row = comparison_df[comparison_df["scheduler"] == "FIFO"].iloc[0]

    with open(OUTPUT_CLAIM, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("PHASE 24: EXTENDED SCHEDULER COMPARISON – POSITIONING STATEMENT\n")
        f.write("=" * 70 + "\n\n")

        f.write("OBJECTIVE\n")
        f.write("-" * 70 + "\n")
        f.write(
            "Position the proactive feasibility scheduler against the baselines\n"
            "implemented and benchmarked in this project. All figures below are\n"
            "computed from the real benchmark outputs (05_results/); no numbers\n"
            "are hardcoded. External systems (e.g., SLURM backfill, Kubernetes\n"
            "QoS) are NOT simulated here, so no claims are made about them.\n\n"
        )

        f.write("DATA SOURCES\n")
        f.write("-" * 70 + "\n")
        f.write(f"  • {MULTI_SCHEDULER_CSV}\n")
        f.write("    (15-run multi-scheduler benchmark, per-scheduler means)\n")
        if paired_stats is not None:
            f.write("  • 05_results/benchmark_statistical_results.csv\n")
            f.write(f"    ({paired_stats['n_runs']}-run paired FIFO-vs-proactive benchmark)\n")
        f.write("\n")

        f.write("SCHEDULERS COMPARED\n")
        f.write("-" * 70 + "\n")
        for _, row in comparison_df.iterrows():
            f.write(f"  • {row['scheduler_name']:<30} ({row['scheduler_type']})\n")
            f.write(f"    Reference: {row['reference']}\n")
            f.write(f"    Mean wait: {row['mean_wait']:.2f} ts | Max wait: {row['max_wait']:.1f} ts | Gini: {row['fairness_gini']:.3f}\n\n")

        f.write("KEY FINDINGS\n")
        f.write("-" * 70 + "\n")

        proactive_wait_imp = improvements["PROACTIVE"]["wait_improvement_pct"]
        f.write("\n1. MEAN WAIT vs. FIFO (multi-scheduler benchmark)\n")
        f.write(f"   Proactive: {proactive_row['mean_wait']:.2f} ts (FIFO: {fifo_row['mean_wait']:.2f} ts)\n")
        f.write(f"   Change: {proactive_wait_imp:+.2f}% "
                f"({'reduction' if proactive_wait_imp > 0 else 'increase'} in mean wait)\n")

        if paired_stats is not None:
            significant = paired_stats["p_value"] < 0.05
            f.write(f"\n   Paired {paired_stats['n_runs']}-run FIFO-vs-proactive benchmark:\n")
            f.write(f"   Mean wait {paired_stats['baseline_wait_mean']:.2f} -> "
                    f"{paired_stats['proactive_wait_mean']:.2f} ts "
                    f"(mean per-run improvement {paired_stats['mean_improvement_pct']:.2f}%)\n")
            f.write(f"   Wait difference 95% CI: [{paired_stats['wait_diff_ci95_low']:.2f}, "
                    f"{paired_stats['wait_diff_ci95_high']:.2f}] ts\n")
            f.write(f"   Paired t-test: t = {paired_stats['t_stat']:.2f}, p = {paired_stats['p_value']:.2e} "
                    f"({'statistically significant at p < 0.05' if significant else 'NOT statistically significant at p < 0.05'})\n\n")
        else:
            f.write("\n   (Paired 40-run benchmark data not found; no significance test performed.)\n\n")

        f.write("2. RANKING BY MEAN-WAIT IMPROVEMENT vs. FIFO\n")
        sorted_by_wait = sorted(improvements.items(), key=lambda x: x[1]["wait_improvement_pct"], reverse=True)
        for rank, (sched_name, impr) in enumerate(sorted_by_wait, 1):
            sched_label = SCHEDULERS.get(sched_name, {"name": sched_name})["name"]
            f.write(f"   {rank}. {sched_label:<25} {impr['wait_improvement_pct']:+6.2f}%\n")

        f.write("\n3. FAIRNESS (Gini coefficient of wait times, lower is fairer)\n")
        proactive_gini = proactive_row["fairness_gini"]
        fifo_gini = fifo_row["fairness_gini"]
        fairness_imp = improvements["PROACTIVE"]["fairness_improvement_pct"]
        f.write(f"   Proactive: {proactive_gini:.3f} (FIFO: {fifo_gini:.3f})\n")
        if fairness_imp >= 0:
            f.write(f"   Proactive is {fairness_imp:.2f}% fairer than FIFO on this metric.\n\n")
        else:
            f.write(f"   Proactive is {-fairness_imp:.2f}% LESS fair than FIFO on this metric\n")
            f.write("   (a wait-time/fairness trade-off of reordering the queue).\n\n")

        f.write("4. THROUGHPUT & GPU UTILIZATION\n")
        f.write(f"   Proactive throughput: {proactive_row['throughput_jobs_per_ts']:.3f} jobs/ts "
                f"(FIFO: {fifo_row['throughput_jobs_per_ts']:.3f} jobs/ts, "
                f"{improvements['PROACTIVE']['throughput_improvement_pct']:+.2f}%)\n")
        f.write(f"   Proactive GPU utilization: {proactive_row['gpu_util'] * 100:.1f}% "
                f"(FIFO: {fifo_row['gpu_util'] * 100:.1f}%, "
                f"{improvements['PROACTIVE']['util_improvement_pct']:+.2f}%)\n\n")

        f.write("POSITIONING vs. IMPLEMENTED BASELINES\n")
        f.write("-" * 70 + "\n")
        beats_wait = []
        loses_wait = []
        for sched_name, impr in improvements.items():
            if sched_name in ("PROACTIVE", "FIFO"):
                continue  # FIFO-vs-proactive already covered in finding 1
            sched_label = SCHEDULERS.get(sched_name, {"name": sched_name})["name"]
            their_imp = impr["wait_improvement_pct"]
            if proactive_wait_imp > their_imp:
                beats_wait.append(sched_label)
                f.write(f"\n✓ Proactive achieves a larger mean-wait reduction vs. FIFO than {sched_label}\n")
            else:
                loses_wait.append(sched_label)
                f.write(f"\n✗ Proactive does NOT beat {sched_label} on mean-wait reduction vs. FIFO\n")
            f.write(f"  - Proactive {proactive_wait_imp:+.2f}% vs. {sched_label} {their_imp:+.2f}%\n")

        nn_rows = comparison_df[comparison_df["scheduler"] == "NN"]
        if len(nn_rows):
            nn_row = nn_rows.iloc[0]
            f.write("\nModel-quality note vs. the NN baseline (same feature space):\n")
            f.write(f"  - Wait-prediction MAE: Proactive {proactive_row['predictor_mae']:.2f} "
                    f"vs. NN {nn_row['predictor_mae']:.2f} "
                    f"({'lower' if proactive_row['predictor_mae'] < nn_row['predictor_mae'] else 'higher'} is the proactive model)\n")
            f.write("  - XGBoost additionally offers faster inference and SHAP interpretability.\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("CONCLUSION\n")
        f.write("=" * 70 + "\n")
        if paired_stats is not None and paired_stats["p_value"] < 0.05 and paired_stats["mean_improvement_pct"] > 0:
            f.write(
                f"In the paired {paired_stats['n_runs']}-run benchmark, the proactive scheduler reduces\n"
                f"mean wait vs. FIFO by {paired_stats['mean_improvement_pct']:.2f}% on average "
                f"(p = {paired_stats['p_value']:.2e}).\n"
            )
        if beats_wait:
            f.write(
                "In the multi-scheduler benchmark it outperforms "
                + ", ".join(beats_wait)
                + " on mean-wait reduction.\n"
            )
        if loses_wait:
            f.write(
                "However, it is outperformed on mean wait by "
                + ", ".join(loses_wait)
                + ",\nand its wait-time Gini coefficient is "
                + ("higher (less fair) than FIFO's" if fairness_imp < 0 else "lower (fairer) than FIFO's")
                + ".\n"
            )
        f.write(
            "These comparisons cover only the schedulers implemented in this\n"
            "project's simulator; claims about external production schedulers\n"
            "(SLURM, Kubernetes, Yarn) would require dedicated implementations.\n"
        )


def main():
    """Execute Phase 24 analysis."""
    os.makedirs(SCRIPT_DIR, exist_ok=True)

    print("[Phase 24] Loading paired FIFO-vs-proactive benchmark...")
    bench_df, benchmark_source = _load_paired_benchmark()
    print(f"  Benchmark source: {benchmark_source}")
    paired_stats = _compute_paired_stats(bench_df) if bench_df is not None else None

    print("[Phase 24] Loading multi-scheduler benchmark results...")
    summary_df = _load_scheduler_results()
    print(f"  Schedulers: {list(summary_df['scheduler'])}")

    print("[Phase 24] Computing improvements vs. FIFO...")
    improvements = _compute_improvements_vs_fifo(summary_df)

    print("[Phase 24] Building comparison dataframe...")
    comparison_df = build_comparison_dataframe(summary_df)
    comparison_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved: {OUTPUT_CSV}")

    print("[Phase 24] Generating heatmap...")
    plot_comparison_heatmap(comparison_df)
    print(f"  Saved: {OUTPUT_HEATMAP}")

    print("[Phase 24] Writing positioning statement...")
    write_novelty_claim(comparison_df, improvements, paired_stats)
    print(f"  Saved: {OUTPUT_CLAIM}")

    print("\n" + "=" * 70)
    print("PHASE 24 SUMMARY")
    print("=" * 70)
    print(comparison_df[["scheduler_name", "mean_wait", "throughput_jobs_per_ts", "fairness_gini"]])
    print(f"\nAll outputs saved to: {SCRIPT_DIR}")


if __name__ == "__main__":
    main()

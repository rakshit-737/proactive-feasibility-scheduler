import os
import pickle
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

matplotlib.use("Agg")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

OUTPUT_CSV = os.path.join(SCRIPT_DIR, "stats_summary.csv")
OUTPUT_PLOT = os.path.join(SCRIPT_DIR, "ci_plots.png")

DATA_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "05_results", "benchmarks", "40_run_benchmark.pkl"),
    os.path.join(PROJECT_ROOT, "05_results", "benchmark_statistical_results.pkl"),
    os.path.join(PROJECT_ROOT, "05_results", "benchmark_statistical_results.csv"),
]

BASELINE_KEYS = ["baseline_wait", "fifo_avg_wait", "fifo_wait", "fifo_mean_wait"]
PROACTIVE_KEYS = ["proactive_wait", "proactive_avg_wait", "proactive_wait_time", "proactive_mean_wait"]
IMPROVEMENT_KEYS = ["improvement_pct", "improvement_percent", "pct_improvement"]
# Fixed seed keeps bootstrap CI values reproducible across runs.
BOOTSTRAP_SEED = 42


def _first_present(df, keys):
    for key in keys:
        if key in df.columns:
            return key
    return None


def _to_dataframe(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.copy()

    if isinstance(obj, dict):
        if "results" in obj and isinstance(obj["results"], (list, tuple)):
            return pd.DataFrame(obj["results"])
        return pd.DataFrame(list(obj.values()))

    if isinstance(obj, (list, tuple)):
        return pd.DataFrame(obj)

    raise ValueError("Unsupported benchmark data format.")


def load_benchmark_data():
    chosen = None
    for path in DATA_CANDIDATES:
        if os.path.exists(path):
            chosen = path
            break

    if chosen is None:
        raise FileNotFoundError(
            "No benchmark artifact found. Expected one of:\n- " + "\n- ".join(DATA_CANDIDATES)
        )

    if chosen.endswith(".pkl"):
        with open(chosen, "rb") as f:
            raw = pickle.load(f)
        df = _to_dataframe(raw)
    else:
        df = pd.read_csv(chosen)

    baseline_col = _first_present(df, BASELINE_KEYS)
    proactive_col = _first_present(df, PROACTIVE_KEYS)
    improvement_col = _first_present(df, IMPROVEMENT_KEYS)

    if baseline_col is None or proactive_col is None:
        raise KeyError(
            f"Could not identify required wait columns. Found: {list(df.columns)}"
        )

    result = pd.DataFrame(
        {
            "baseline_wait": pd.to_numeric(df[baseline_col], errors="coerce"),
            "proactive_wait": pd.to_numeric(df[proactive_col], errors="coerce"),
        }
    ).dropna()

    if improvement_col is not None:
        result["improvement_pct"] = pd.to_numeric(df[improvement_col], errors="coerce")
    else:
        result["improvement_pct"] = (
            (result["baseline_wait"] - result["proactive_wait"]) / result["baseline_wait"] * 100.0
        )

    result["improvement_pct"] = result["improvement_pct"].replace([np.inf, -np.inf], np.nan)
    result = result.dropna()

    optional_pairs = {}
    if "baseline_util" in df.columns and "proactive_util" in df.columns:
        optional_pairs["gpu_utilization"] = (
            pd.to_numeric(df["baseline_util"], errors="coerce"),
            pd.to_numeric(df["proactive_util"], errors="coerce"),
        )
    if "baseline_completed" in df.columns and "proactive_completed" in df.columns:
        optional_pairs["completed_jobs"] = (
            pd.to_numeric(df["baseline_completed"], errors="coerce"),
            pd.to_numeric(df["proactive_completed"], errors="coerce"),
        )

    return result, optional_pairs, chosen


def bootstrap_mean_ci(values, n_bootstrap=10000, ci=0.95, seed=BOOTSTRAP_SEED):
    """Return percentile bootstrap CI bounds for the sample mean."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot bootstrap empty data.")

    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    means = values[sample_idx].mean(axis=1)

    alpha = 1.0 - ci
    lower = np.percentile(means, 100 * (alpha / 2.0))
    upper = np.percentile(means, 100 * (1.0 - alpha / 2.0))
    return float(lower), float(upper)


def benjamini_hochberg(pvalues):
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        bh = ranked[i] * m / rank
        prev = min(prev, bh)
        adjusted[i] = prev

    out = np.empty(m, dtype=float)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def paired_tests(df, optional_pairs):
    """Run two-tailed paired t-tests for wait time and optional paired metrics."""
    tests = [
        (
            "wait_time",
            pd.to_numeric(df["baseline_wait"], errors="coerce").to_numpy(),
            pd.to_numeric(df["proactive_wait"], errors="coerce").to_numpy(),
        )
    ]
    for metric, (baseline, proactive) in optional_pairs.items():
        mask = ~(baseline.isna() | proactive.isna())
        tests.append((metric, baseline[mask].to_numpy(), proactive[mask].to_numpy()))

    stats_rows = []
    raw_pvalues = []
    testable_rows = []
    for metric, baseline, proactive in tests:
        diff = baseline - proactive
        # Guard the degenerate case: identical paired columns (zero-variance
        # differences) make ttest_rel return NaN; flag as not applicable
        # instead of masquerading as a real "not significant" result.
        if diff.size < 2 or np.allclose(np.var(diff), 0.0):
            warnings.warn(
                f"Paired t-test not applicable for '{metric}': zero-variance paired differences.",
                RuntimeWarning,
                stacklevel=2,
            )
            stats_rows.append(
                {
                    "metric": metric,
                    "t_stat": np.nan,
                    "p_value_raw": np.nan,
                    "p_value_bh": np.nan,
                    "significant_0_05": False,
                    "note": "n/a (zero variance)",
                }
            )
            continue
        t_stat, p_raw = stats.ttest_rel(baseline, proactive, alternative="two-sided")
        row = {"metric": metric, "t_stat": float(t_stat), "p_value_raw": float(p_raw), "note": ""}
        stats_rows.append(row)
        testable_rows.append(row)
        raw_pvalues.append(float(p_raw))

    if raw_pvalues:
        corrected = benjamini_hochberg(raw_pvalues)
        for row, p_corr in zip(testable_rows, corrected):
            row["p_value_bh"] = float(p_corr)
            row["significant_0_05"] = bool(p_corr < 0.05)

    return stats_rows


def build_summary(df, test_rows):
    """Assemble CSV-ready summary rows with CI bounds and corrected p-values."""
    fifo_ci_low, fifo_ci_high = bootstrap_mean_ci(df["baseline_wait"].to_numpy())
    proactive_ci_low, proactive_ci_high = bootstrap_mean_ci(df["proactive_wait"].to_numpy())
    improve_ci_low, improve_ci_high = bootstrap_mean_ci(df["improvement_pct"].to_numpy())

    wait_test = next(row for row in test_rows if row["metric"] == "wait_time")

    rows = [
        {
            "metric": "fifo_mean_wait",
            "mean": float(df["baseline_wait"].mean()),
            "ci_lower": fifo_ci_low,
            "ci_upper": fifo_ci_high,
            "n": int(df.shape[0]),
            "t_stat": np.nan,
            "p_value_raw": wait_test["p_value_raw"],
            "p_value_bh": wait_test["p_value_bh"],
            "significant_0_05": wait_test["significant_0_05"],
        },
        {
            "metric": "proactive_mean_wait",
            "mean": float(df["proactive_wait"].mean()),
            "ci_lower": proactive_ci_low,
            "ci_upper": proactive_ci_high,
            "n": int(df.shape[0]),
            "t_stat": np.nan,
            "p_value_raw": wait_test["p_value_raw"],
            "p_value_bh": wait_test["p_value_bh"],
            "significant_0_05": wait_test["significant_0_05"],
        },
        {
            "metric": "wait_improvement_pct",
            "mean": float(df["improvement_pct"].mean()),
            "ci_lower": improve_ci_low,
            "ci_upper": improve_ci_high,
            "n": int(df.shape[0]),
            "t_stat": wait_test["t_stat"],
            "p_value_raw": wait_test["p_value_raw"],
            "p_value_bh": wait_test["p_value_bh"],
            "significant_0_05": wait_test["significant_0_05"],
        },
    ]
    for row in rows:
        row["note"] = wait_test.get("note", "")

    for row in test_rows:
        if row["metric"] == "wait_time":
            continue
        rows.append(
            {
                "metric": f"paired_ttest_{row['metric']}",
                "mean": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "n": int(df.shape[0]),
                "t_stat": row["t_stat"],
                "p_value_raw": row["p_value_raw"],
                "p_value_bh": row["p_value_bh"],
                "significant_0_05": row["significant_0_05"],
                "note": row.get("note", ""),
            }
        )

    return pd.DataFrame(rows)


def plot_confidence_intervals(summary_df):
    """Create and save Phase 22 CI forest-style plots for key wait metrics."""
    sns.set_theme(style="whitegrid")
    wait_rows = summary_df[summary_df["metric"].isin(["fifo_mean_wait", "proactive_mean_wait"])].copy()
    imp_row = summary_df[summary_df["metric"] == "wait_improvement_pct"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [3, 2]})

    y_positions = np.arange(wait_rows.shape[0])
    means = wait_rows["mean"].to_numpy()
    lower = wait_rows["ci_lower"].to_numpy()
    upper = wait_rows["ci_upper"].to_numpy()
    xerr = np.vstack((means - lower, upper - means))

    axes[0].errorbar(means, y_positions, xerr=xerr, fmt="o", capsize=5, color="#1f77b4")
    metric_labels = [label.replace("_", " ").title() for label in wait_rows["metric"]]
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(metric_labels)
    axes[0].set_xlabel("Mean wait time (timesteps)")
    axes[0].set_title("95% Bootstrap CI (Wait Time)")

    mean_imp = imp_row["mean"]
    ci_low = imp_row["ci_lower"]
    ci_high = imp_row["ci_upper"]
    axes[1].errorbar(
        [mean_imp],
        [0],
        xerr=[[mean_imp - ci_low], [ci_high - mean_imp]],
        fmt="o",
        capsize=5,
        color="#2ca02c",
    )
    axes[1].axvline(0.0, linestyle="--", color="black", linewidth=1)
    axes[1].set_yticks([0])
    axes[1].set_yticklabels(["Improvement"])
    axes[1].set_xlabel("Percent (%)")
    axes[1].set_title("95% Bootstrap CI (Wait Improvement)")

    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    """Execute Phase 22 analysis and write CSV/plot outputs to this directory."""
    os.makedirs(SCRIPT_DIR, exist_ok=True)
    df, optional_pairs, source_path = load_benchmark_data()
    test_rows = paired_tests(df, optional_pairs)
    summary_df = build_summary(df, test_rows)
    summary_df.to_csv(OUTPUT_CSV, index=False)
    plot_confidence_intervals(summary_df)

    wait_row = summary_df.loc[summary_df["metric"] == "wait_improvement_pct"].iloc[0]
    print(f"Loaded benchmark artifact: {source_path}")
    print(f"Runs analyzed: {int(wait_row['n'])}")
    print(f"Wait improvement mean: {wait_row['mean']:.3f}%")
    print(f"95% CI: [{wait_row['ci_lower']:.3f}%, {wait_row['ci_upper']:.3f}%]")
    print(f"Paired t-test p-value (raw): {wait_row['p_value_raw']:.6g}")
    print(f"Paired t-test p-value (BH-corrected): {wait_row['p_value_bh']:.6g}")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()

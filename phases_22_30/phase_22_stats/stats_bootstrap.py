import os
import pickle
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")


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

OUTPUT_CSV = os.path.join(SCRIPT_DIR, "stats_summary.csv")
# Stem, not a filename: save_both() writes '<stem>.png' (light) and
# '<stem>-dark.png' (dark). The light path is byte-identical to the old output.
OUTPUT_PLOT_STEM = os.path.join(SCRIPT_DIR, "ci_plots")
OUTPUT_PLOT = OUTPUT_PLOT_STEM + ".png"

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


# Which scheduler each summary row describes. Colour follows the ENTITY, so the
# baseline stays grey and the proposed method stays blue in every panel.
METRIC_POLICY = {"fifo_mean_wait": "FIFO", "proactive_mean_wait": "PROACTIVE"}


def _repo_rel(path):
    """Repo-relative, forward-slashed path for the provenance footer."""
    try:
        return os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")
    except ValueError:  # different drive on Windows
        return os.path.basename(path)


def plot_confidence_intervals(summary_df, source_path=None):
    """Create and save Phase 22 CI forest-style plots for key wait metrics."""
    wait_rows = summary_df[summary_df["metric"].isin(["fifo_mean_wait", "proactive_mean_wait"])].copy()
    imp_row = summary_df[summary_df["metric"] == "wait_improvement_pct"].iloc[0]

    means = wait_rows["mean"].to_numpy()
    lower = wait_rows["ci_lower"].to_numpy()
    upper = wait_rows["ci_upper"].to_numpy()
    policies = [METRIC_POLICY.get(m, m) for m in wait_rows["metric"]]
    y_positions = np.arange(wait_rows.shape[0])

    mean_imp = imp_row["mean"]
    ci_low = imp_row["ci_lower"]
    ci_high = imp_row["ci_upper"]
    n_runs = int(imp_row["n"])

    source = _repo_rel(OUTPUT_CSV)
    if source_path is not None:
        source += f"  ·  benchmark runs: {_repo_rel(source_path)}"

    for mode in ("light", "dark"):
        ink = PALETTE[mode]["ink_2"]
        muted = PALETTE[mode]["muted"]

        fig, axes = figure(mode, figsize=(11, 4.8), nrows=1, ncols=2,
                           gridspec_kw={"width_ratios": [3, 2]})

        # ── Panel 1: mean wait per policy, with bootstrap CI ──────────────────
        for y, policy, mu, lo, hi in zip(y_positions, policies, means, lower, upper):
            axes[0].errorbar(
                [mu], [y],
                xerr=[[mu - lo], [hi - mu]],
                fmt="o", capsize=4, elinewidth=1.6, capthick=1.6, markersize=7,
                color=color_of(policy, mode), label=label_of(policy),
            )
        axes[0].set_yticks(y_positions)
        axes[0].set_yticklabels([label_of(p) for p in policies])
        axes[0].set_ylim(y_positions.max() + 0.95, y_positions.min() - 0.55)
        axes[0].set_xlabel("Mean wait time (timesteps)")
        axes[0].set_title("Mean wait time, 95% bootstrap CI")
        bar_ends(axes[0], "h")
        legend_roles(axes[0], mode, roles=("ml", "baseline"), loc="lower right")

        # ── Panel 2: the paired difference the study is actually about ────────
        # One series colour only (blue = the ML method's effect); no second hue.
        axes[1].axvline(0.0, color=muted, linewidth=1.0, zorder=1)
        axes[1].errorbar(
            [mean_imp], [0],
            xerr=[[mean_imp - ci_low], [ci_high - mean_imp]],
            fmt="o", capsize=4, elinewidth=1.6, capthick=1.6, markersize=7,
            color=PALETTE[mode]["series_1"], zorder=3,
        )
        # Direct-label the single estimate; no per-point labels anywhere else.
        axes[1].annotate(
            f"{mean_imp:.1f}%  [{ci_low:.1f}, {ci_high:.1f}]",
            xy=(mean_imp, 0), xytext=(0, 14), textcoords="offset points",
            ha="center", va="bottom", color=ink, fontsize=9.5,
        )
        axes[1].set_yticks([0])
        axes[1].set_yticklabels(["Wait reduction"])
        axes[1].set_ylim(0.9, -0.9)
        axes[1].set_xlabel("Reduction vs FCFS baseline (%)")
        axes[1].set_title("Paired improvement, 95% bootstrap CI")
        bar_ends(axes[1], "h")
        axes[1].annotate(
            "0 = no change", xy=(0.0, 0.62), xytext=(4, 0),
            textcoords="offset points", ha="left", va="center",
            color=muted, fontsize=8.5,
        )

        fig.tight_layout(rect=(0, 0.03, 1, 0.85))
        finish(
            fig, mode,
            title=f"Proactive scheduling cuts mean wait {mean_imp:.1f}% "
                  f"(95% CI {ci_low:.1f}-{ci_high:.1f}%)",
            subtitle=f"Paired across {n_runs} benchmark runs · "
                     f"10,000-resample percentile bootstrap of the mean",
            source=source,
        )
        save_both(fig, OUTPUT_PLOT_STEM, mode)


def main():
    """Execute Phase 22 analysis and write CSV/plot outputs to this directory."""
    os.makedirs(SCRIPT_DIR, exist_ok=True)
    df, optional_pairs, source_path = load_benchmark_data()
    test_rows = paired_tests(df, optional_pairs)
    summary_df = build_summary(df, test_rows)
    summary_df.to_csv(OUTPUT_CSV, index=False)
    plot_confidence_intervals(summary_df, source_path=source_path)

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

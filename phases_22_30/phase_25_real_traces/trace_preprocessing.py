"""
Phase 25: Real Trace Integration (Part 1)
==========================================

Inventory and plot the workload traces that are actually on disk.

NOTE: no real LANL or Alibaba trace ships with this repository. The only trace
this script reads is the synthetic LANL-schema proxy
02_data/synthetic_proxy_lanl_schema_trace.csv, labelled 'synthetic_proxy_trace'
in every output. The two REAL traces the project evaluates (LANL CM-5 and
SDSC SP2, shipped as .swf.gz) are handled by 02_data/build_real_trace_datasets.py
and 02_data/real_trace_validation.py, not here.

v3.6 removed three things from this file:
  - the _map_lanl and _map_alibaba mappers, which SYNTHESISED the wait-time
    target from a hard-coded linear formula while their trace candidates were
    tagged source_type="real". No file they read has ever existed in this
    repository, so the code path was dead as well as dishonest.
  - the LANL-SJC-2014, Alibaba-2018 and LANL-Theta candidates they served.
  - estimate_cross_trace_mae and its cross_trace_mae.csv, which scored a
    hand-written heuristic whose coefficients were near-identical to the
    generator's, so the reported error was circular. Its only row was the
    synthetic proxy, with mape_pct pinned at the 200.0 saturation value.

Generates:
  - trace_inventory.csv: metadata for every trace actually loaded
  - real_vs_synthetic_comparison.png (+ -dark.png): distribution alignment plots
"""

import os
import sys
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

sys.path.insert(0, PROJECT_ROOT)
from vizstyle import (  # noqa: E402,F401  (needs PROJECT_ROOT on sys.path first)
    figure,
    finish,
    save_both,
    PALETTE,
    color_of,
    label_of,
    bar_ends,
    legend_roles,
)

OUTPUT_INVENTORY = os.path.join(SCRIPT_DIR, "trace_inventory.csv")
# Stem, not a filename: save_both() writes '<stem>.png' (light) and
# '<stem>-dark.png' (dark). The light path is byte-identical to the old output.
OUTPUT_PLOT_STEM = os.path.join(SCRIPT_DIR, "real_vs_synthetic_comparison")
OUTPUT_PLOT = OUTPUT_PLOT_STEM + ".png"

# ---------------------------------------------------------------------------
# WHERE THE PROJECT'S REAL-TRACE EVIDENCE ACTUALLY COMES FROM
# ---------------------------------------------------------------------------
# Not from this file. The two real traces the project evaluates are committed
# as gzipped Standard Workload Format logs:
#     02_data/LANL-CM5-1994-4.1-cln.swf.gz
#     02_data/SDSC-SP2-1998-4.2-cln.swf.gz
# and are consumed by 02_data/build_real_trace_datasets.py (chronological
# replay of the RECORDED schedule to reconstruct cluster state at each submit
# instant), 02_data/real_trace_validation.py, and
# 04_scheduler/trace_driven_benchmark.py. Those are the scripts behind every
# real-trace number in RESULTS.md and the manuscript.
#
# This file only inventories and plots the synthetic LANL-schema proxy. It used
# to advertise ingestion of LANL-SJC-2014, LANL-Theta and Alibaba-2018 through
# mappers that fabricated the wait-time target; v3.6 removed them. Adding a new
# real trace means extending build_real_trace_datasets.py, not this file.
#
# Further traces, if you want them: Parallel Workloads Archive,
# https://www.cs.huji.ac.il/labs/parallel/workload/ (SWF, directly usable by
# build_real_trace_datasets.py); Alibaba cluster traces,
# https://github.com/alibaba/clusterdata (not SWF -- needs a converter that
# does not exist here).
# ---------------------------------------------------------------------------

# Trace source candidates. Only the synthetic proxy is listed: the three
# real-trace entries that used to sit here (LANL-SJC-2014, Alibaba-2018,
# LANL-Theta) named files that have never existed in this repository and were
# served by mappers that synthesised the wait-time target. See the module
# docstring. The project's real-trace evidence comes from the two committed
# .swf.gz traces via 02_data/build_real_trace_datasets.py.
TRACE_CANDIDATES = [
    {
        "name": "synthetic_proxy_trace",
        "type": "LANL_SAMPLE",
        "path": os.path.join(PROJECT_ROOT, "02_data",
                             "synthetic_proxy_lanl_schema_trace.csv"),
        "expected_rows": 2_000,
        "source_type": "synthetic_proxy",
    },
]

# Presentation only: trace name -> the repo-relative file the chart is drawn
# from, used for the provenance footer. The fallback trace has no file on disk.
TRACE_SOURCE_PATH = {
    c["name"]: os.path.relpath(c["path"], PROJECT_ROOT).replace(os.sep, "/")
    for c in TRACE_CANDIDATES
}
TRACE_SOURCE_PATH["synthetic_fallback"] = "generated in-script (no trace file on disk)"

# Presentation only: source_type slug -> the phrase a reader can parse.
SOURCE_TYPE_LABEL = {
    "real": "real trace",
    "synthetic_proxy": "synthetic proxy",
    "synthetic_fallback": "synthetic fallback",
}

# Synthetic feature template (from Phase 01–21)
SYNTHETIC_FEATURES = [
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


class RealTraceLoader:
    """Load and map real workload traces to synthetic feature space."""

    def __init__(self, trace_path: str, trace_type: str):
        self.trace_path = trace_path
        self.trace_type = trace_type
        self.raw_df = None
        self.mapped_df = None

    def load(self) -> bool:
        """Attempt to load trace file."""
        if not os.path.exists(self.trace_path):
            return False
        try:
            self.raw_df = pd.read_csv(self.trace_path, low_memory=False)
            return True
        except Exception:
            return False

    def map_to_synthetic_features(self, sample_size: int = 10_000) -> pd.DataFrame:
        """
        Map real trace attributes to synthetic feature space.
        This is a heuristic mapping; actual implementation requires domain knowledge.
        """
        if self.raw_df is None or len(self.raw_df) == 0:
            return None

        # Sample for efficiency
        df_sample = self.raw_df.sample(min(sample_size, len(self.raw_df)), random_state=42)

        features = {}

        if self.trace_type == "LANL_SAMPLE":
            features = self._map_lanl_sample(df_sample)

        self.mapped_df = pd.DataFrame(features)
        return self.mapped_df

    def _map_lanl_sample(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Map the synthetic LANL-schema proxy
        (02_data/synthetic_proxy_lanl_schema_trace.csv; columns arrival_time,
        wait_time, runtime, num_gpus).

        job_gpu and the wait_time TARGET come from the trace file itself. The 11
        cluster-state features are not logged in any SWF-style trace, so they are
        drawn from seeded distributions here -- which is why every output of this
        script is labelled synthetic_proxy and why no claim in the paper rests on
        it.
        """
        n = len(df)
        rng = np.random.default_rng(42)

        job_gpu = pd.to_numeric(df["num_gpus"], errors="coerce").fillna(1).clip(1, 8).to_numpy(dtype=float)

        total_free = np.clip(rng.normal(32.0, 12.0, n), 0.0, 256.0)
        queue_length = np.clip(rng.poisson(5.0, n), 0.0, 50.0).astype(float)
        running_jobs = np.clip(rng.poisson(8.0, n), 0.0, 40.0).astype(float)

        node_count = 64.0
        max_free_node = np.clip(total_free / (node_count / 4.0) + rng.normal(0.0, 0.5, n), 0.0, 4.0)
        variance_free = np.clip(rng.gamma(2.0, 1.0, n), 0.0, None)
        can_fit_now = (total_free >= job_gpu).astype(float)
        gpu_fit_ratio = np.clip(total_free / np.maximum(job_gpu, 1e-3), 0.0, 1.0)
        fragmentation = np.clip(rng.normal(0.75, 0.2, n), 0.0, None)
        queue_pressure = np.clip((queue_length + 0.5 * running_jobs) / np.maximum(total_free + 1.0, 1.0), 0.0, None)
        node_availability = np.clip(total_free / 256.0 + rng.normal(0.0, 0.05, n), 0.0, 1.0)
        avg_free_per_node = total_free / (node_count / 4.0)

        # Target: the trace's own wait_time column (NOT synthesized)
        wait_time = pd.to_numeric(df["wait_time"], errors="coerce").fillna(0).clip(lower=1.0).to_numpy(dtype=float)

        return {
            "job_gpu": job_gpu,
            "total_free": total_free,
            "queue_length": queue_length,
            "running_jobs": running_jobs,
            "max_free_node": max_free_node,
            "variance_free": variance_free,
            "can_fit_now": can_fit_now,
            "gpu_fit_ratio": gpu_fit_ratio,
            "fragmentation": fragmentation,
            "queue_pressure": queue_pressure,
            "node_availability": node_availability,
            "avg_free_per_node": avg_free_per_node,
            "wait_time": wait_time,
        }

    def get_stats(self) -> Dict:
        """Return summary statistics."""
        if self.mapped_df is None:
            return {}
        return {
            "n_samples": len(self.mapped_df),
            "wait_time_mean": float(self.mapped_df["wait_time"].mean()),
            "wait_time_std": float(self.mapped_df["wait_time"].std()),
            "wait_time_min": float(self.mapped_df["wait_time"].min()),
            "wait_time_max": float(self.mapped_df["wait_time"].max()),
            "gpu_request_mean": float(self.mapped_df["job_gpu"].mean()),
            "gpu_request_std": float(self.mapped_df["job_gpu"].std()),
        }


def _print_no_real_trace_warning() -> None:
    """Loud, unmissable warning that no real trace data is being used."""
    print("!" * 78)
    print("!! WARNING: NO REAL TRACES FOUND -- Phase 25 is NOT validating on real data !!")
    print("!! None of 02_data/lanl_sjc_2014.csv, 02_data/alibaba_2018.csv,")
    print("!! 02_data/lanl_theta.csv exist. All outputs are labelled")
    print("!! 'synthetic_proxy_trace' / 'synthetic_fallback' accordingly.")
    print("!! To obtain real traces:")
    print("!!   - LANL/HPC (SWF): https://www.cs.huji.ac.il/labs/parallel/workload/")
    print("!!     -> convert to 02_data/lanl_sjc_2014.csv or 02_data/lanl_theta.csv")
    print("!!   - Alibaba: https://github.com/alibaba/clusterdata (cluster-trace-v2018)")
    print("!!     -> convert to 02_data/alibaba_2018.csv")
    print("!! See the comment block above TRACE_CANDIDATES in this script.")
    print("!" * 78)


def load_traces() -> List[Tuple[str, pd.DataFrame, Dict, str]]:
    """Load all available traces and map to synthetic features."""
    results = []
    real_loaded = False

    for candidate in TRACE_CANDIDATES:
        loader = RealTraceLoader(candidate["path"], candidate["type"])
        if not loader.load():
            print(f"  Skipped {candidate['name']}: file not found ({candidate['path']})")
            continue

        mapped = loader.map_to_synthetic_features()
        if mapped is None or len(mapped) == 0:
            print(f"  Skipped {candidate['name']}: mapping failed")
            continue

        stats = loader.get_stats()
        results.append((candidate["name"], mapped, stats, candidate["source_type"]))
        if candidate["source_type"] == "real":
            real_loaded = True
        print(f"  Loaded {candidate['name']}: {stats['n_samples']} samples "
              f"[{candidate['source_type']}]")

    if not real_loaded:
        _print_no_real_trace_warning()

    # Fallback: synthesize a representative trace (only if nothing loaded)
    if len(results) == 0:
        rng = np.random.default_rng(42)
        synthetic_loader = RealTraceLoader("synthetic", "LANL")
        synthetic_df = pd.DataFrame(
            {
                "used_time": rng.integers(10, 500, 50_000),
                "requested_cores": rng.integers(4, 256, 50_000),
            }
        )
        synthetic_loader.raw_df = synthetic_df
        mapped = synthetic_loader.map_to_synthetic_features(sample_size=50_000)
        stats = synthetic_loader.get_stats()
        results.append(("synthetic_fallback", mapped, stats, "synthetic_fallback"))
        print(f"  Loaded synthetic_fallback: {stats['n_samples']} samples "
              f"[synthetic_fallback -- even the wait_time target is generated]")

    return results


def build_trace_inventory(trace_results: List[Tuple[str, pd.DataFrame, Dict, str]]) -> pd.DataFrame:
    """Build inventory CSV summarizing all traces."""
    rows = []
    for trace_name, df, stats, source_type in trace_results:
        row = {
            "trace_name": trace_name,
            "source_type": source_type,
            "n_samples": stats.get("n_samples", len(df)),
            "wait_time_mean": stats.get("wait_time_mean", df["wait_time"].mean() if "wait_time" in df else 0.0),
            "wait_time_std": stats.get("wait_time_std", df["wait_time"].std() if "wait_time" in df else 0.0),
            "gpu_request_mean": stats.get("gpu_request_mean", df["job_gpu"].mean() if "job_gpu" in df else 0.0),
            "gpu_request_std": stats.get("gpu_request_std", df["job_gpu"].std() if "job_gpu" in df else 0.0),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _trace_legend_label(name: str, source_type: str, n_rows: int) -> str:
    """Legend/subtitle identity for one trace. Presentation only."""
    kind = SOURCE_TYPE_LABEL.get(source_type, source_type)
    return f"{name} - {kind} (n = {n_rows:,})"


def _trace_color(index: int, mode: str) -> str:
    """Colour follows the TRACE, never the panel: trace i keeps colour i in all
    four panels. Only two hues exist (blue, orange); any further trace recedes
    to neutral ink rather than inventing a third hue."""
    keys = ("series_1", "series_2", "ink_2", "muted")
    return PALETTE[mode][keys[min(index, len(keys) - 1)]]


def plot_real_vs_synthetic(trace_results: List[Tuple[str, pd.DataFrame, Dict, str]]) -> None:
    """Create comparison plots between loaded traces (real or synthetic).

    Presentation only -- nothing here touches the mapped features, the decile
    binning or any statistic. The four panels share one colour assignment keyed
    on the trace, so a trace that appears in every panel is the same colour in
    every panel.
    """
    # Identity of each trace, computed once and reused by all four panels.
    labels = [
        _trace_legend_label(name, source_type, len(df))
        for name, df, _, source_type in trace_results
    ]
    markers = ("o", "s", "^", "D")
    multi = len(trace_results) > 1

    sources = " | ".join(
        dict.fromkeys(TRACE_SOURCE_PATH.get(name, name) for name, _, _, _ in trace_results)
    )
    if any(source_type == "real" for _, _, _, source_type in trace_results):
        subtitle = "Traces loaded: " + "; ".join(labels)
    else:
        subtitle = (
            "No real LANL/Alibaba trace is on disk, so every series below is the "
            "synthetic LANL-schema proxy: " + "; ".join(labels) + "."
        )

    for mode in ("light", "dark"):
        p = PALETTE[mode]
        fig, axes = figure(mode, figsize=(12, 8.6), nrows=2, ncols=2)

        # Panel 1: wait-time distribution -----------------------------------
        ax = axes[0, 0]
        for i, (name, df, _, _) in enumerate(trace_results):
            wait = df["wait_time"].to_numpy()
            color = _trace_color(i, mode)
            # Overlapping translucent fills would blend into a third hue, so
            # only a single trace is filled; further traces are drawn as outlines.
            if multi:
                ax.hist(wait, bins=50, histtype="step", linewidth=1.6,
                        color=color, label=labels[i])
            else:
                ax.hist(wait, bins=50, color=color, label=labels[i])
                # One direct label, not a value on every bar (rule 7).
                mean_wait = float(np.mean(wait))
                ax.axvline(mean_wait, color=p["ink_2"], linewidth=1.0)
                ax.text(mean_wait, ax.get_ylim()[1] * 0.96,
                        f"  mean {mean_wait:.1f}", color=p["ink_2"],
                        fontsize=9, ha="left", va="top")
        ax.set_xlabel("Wait time (timesteps)")
        ax.set_ylabel("Number of jobs")
        ax.set_title("Wait-time distribution")

        # Panel 2: GPU request distribution ---------------------------------
        ax = axes[0, 1]
        for i, (name, df, _, _) in enumerate(trace_results):
            gpu = df["job_gpu"].to_numpy()
            color = _trace_color(i, mode)
            if multi:
                ax.hist(gpu, bins=8, histtype="step", linewidth=1.6,
                        color=color, label=labels[i])
            else:
                ax.hist(gpu, bins=8, color=color, label=labels[i])
        ax.set_xlabel("GPUs requested per job")
        ax.set_ylabel("Number of jobs")
        ax.set_title("Job-size distribution")

        # Panels 3 and 4: mean wait across deciles of a cluster-state feature.
        # Same binning as before; only the tick labels change (deciles are named
        # 1..10 rather than left as raw positional indices).
        decile_panels = (
            (axes[1, 0], "queue_pressure", "Queue-pressure decile (1 = least pressure)",
             "Wait time vs. queue pressure"),
            (axes[1, 1], "node_availability", "Node-availability decile (1 = least free)",
             "Wait time vs. node availability"),
        )
        for ax, column, xlabel, panel_title in decile_panels:
            n_bins = 0
            for i, (name, df, _, _) in enumerate(trace_results):
                bins = pd.qcut(df[column], q=10, duplicates="drop")
                wait_by_bin = df.groupby(bins)["wait_time"].mean()
                n_bins = max(n_bins, len(wait_by_bin))
                ax.plot(range(len(wait_by_bin)), wait_by_bin,
                        marker=markers[min(i, len(markers) - 1)],
                        color=_trace_color(i, mode), label=labels[i])
            ax.set_xticks(range(n_bins))
            ax.set_xticklabels([str(b + 1) for b in range(n_bins)])
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Mean wait time (timesteps)")
            ax.set_title(panel_title)

        # A legend only earns its space when >= 2 series are drawn; with a
        # single trace the identity lives in the subtitle instead.
        if multi:
            axes[0, 0].legend(loc="upper right")

        fig.tight_layout(rect=(0, 0.03, 1, 0.87))
        finish(
            fig,
            mode,
            title="Phase 25: what the loaded workload traces look like",
            subtitle=subtitle,
            source=sources,
        )
        save_both(fig, OUTPUT_PLOT_STEM, mode)


def main():
    """Execute Phase 25 analysis."""
    os.makedirs(SCRIPT_DIR, exist_ok=True)

    print("[Phase 25] Loading traces (real if available, else labelled synthetic)...")
    trace_results = load_traces()

    print("[Phase 25] Building trace inventory...")
    inventory_df = build_trace_inventory(trace_results)
    inventory_df.to_csv(OUTPUT_INVENTORY, index=False)
    print(f"  Saved: {OUTPUT_INVENTORY}")

    print("[Phase 25] Generating comparison plots...")
    plot_real_vs_synthetic(trace_results)
    print(f"  Saved: {OUTPUT_PLOT}")


    print("\n" + "=" * 70)
    print("PHASE 25 SUMMARY – REAL TRACE INTEGRATION")
    print("=" * 70)
    print("\nTrace Inventory:")
    print(inventory_df)
    print(f"\nAll outputs saved to: {SCRIPT_DIR}")


if __name__ == "__main__":
    main()

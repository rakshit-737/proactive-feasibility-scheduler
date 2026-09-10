#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# Resolve a Python interpreter. A caller that wants a specific interpreter --
# an activated virtualenv, or tools/verify_artifacts.py, which exports
# PY=sys.executable -- sets PY and we honour it. Without the override, bare
# `python3` on Windows resolves to the Microsoft Store shim rather than to an
# activated venv, so the pipeline would silently run against the wrong
# environment.
PY="${PY:-$(command -v python3 || command -v python)}"

# Force UTF-8 stdio on every Python step: several scripts print Unicode
# (importance bars, R-squared symbols) which crashes under Windows' default
# cp1252 console encoding when output is piped or redirected.
export PYTHONUTF8=1

# SMOKE=1 selects the reduced configuration of the two long steps. A smoke run
# writes REDUCED numbers and its outputs must never be committed;
# tools/verify_artifacts.py --smoke therefore runs it inside a scratch copy.
SMOKE_DEG="${SMOKE:+--quick}"
SMOKE_TDB="${SMOKE:+--smoke}"

TOTAL=20

# ---------------------------------------------------------------------------
# Step 0 regenerates the dataset and the v2 model so the pipeline works on a
# fresh checkout. Every downstream script depends on
#   02_data/improved_wait_dataset.csv  and  03_models/wait_model_v2.pkl
# which were previously assumed to already exist.
# ---------------------------------------------------------------------------
echo "[1/$TOTAL] Generate dataset and train wait_model_v2"
( cd 02_data && "$PY" generate_improved_dataset.py )
"$PY" 03_models/train_improved_model.py

echo "[2/$TOTAL] Phase-01 minimal cluster simulation (02_data/dataset.csv)"
"$PY" 01_simulation/dataset_minimal_gpu_cluster.py

echo "[3/$TOTAL] Quantile model (q10/q50/q90) for the uncertainty study"
"$PY" 03_models/train_quantile_model.py

echo "[4/$TOTAL] Ablation study"
"$PY" 03_models/ablation_study.py

echo "[5/$TOTAL] Fairness analysis"
"$PY" 04_scheduler/fairness_analysis.py

echo "[6/$TOTAL] Fairness wait-budget sweep (Pareto frontier)"
"$PY" 04_scheduler/fairness_budget_sweep.py

echo "[7/$TOTAL] Synthetic scheduler benchmark (14 schedulers + significance + TOST)"
"$PY" 04_scheduler/multi_scheduler_benchmark.py

echo "[8/$TOTAL] Runtime-estimate sensitivity sweep (SJF/backfill vs estimate quality)"
( cd 04_scheduler && "$PY" estimate_sensitivity.py )

# ---------------------------------------------------------------------------
# v3.4 headline experiments. Step 9 tests whether the learned wait model
# contributes anything to queue ORDERING beyond the job's requested size;
# step 10 re-asks the whole scheduler comparison on real traces with the real
# user runtime estimates the traces ship with. Both are prerequisites for the
# claims in the manuscript, so they run before the supporting analyses.
# ---------------------------------------------------------------------------
echo "[9/$TOTAL] Ranking-degeneracy diagnostic (is the ML score a function of size?)"
( cd 04_scheduler && "$PY" ranking_degeneracy.py $SMOKE_DEG )

echo "[10/$TOTAL] Trace-driven scheduler benchmark (real SWF traces, real user estimates)"
( cd 04_scheduler && "$PY" trace_driven_benchmark.py $SMOKE_TDB )

echo "[11/$TOTAL] Real-trace datasets (SWF replay) and real-trace validation"
"$PY" 02_data/build_real_trace_datasets.py
"$PY" 02_data/real_trace_validation.py

echo "[12/$TOTAL] SHAP explainability"
"$PY" 03_models/explainability_shap.py

echo "[13/$TOTAL] Synthetic-proxy trace: out-of-distribution check (NOT real trace data)"
"$PY" 02_data/load_real_traces.py
"$PY" 02_data/synthetic_vs_real_comparison.py

echo "[14/$TOTAL] Scaling analysis"
"$PY" 04_scheduler/scaling_analysis.py

echo "[15/$TOTAL] Uncertainty-aware scheduling benchmark (quantile intervals, OOD)"
"$PY" 04_scheduler/uncertainty_scheduler_benchmark.py

echo "[16/$TOTAL] Online learning and concept drift"
"$PY" 03_models/online_learning.py
"$PY" 03_models/concept_drift_detection.py

echo "[17/$TOTAL] Baseline statistical benchmark refresh"
"$PY" 04_scheduler/benchmark_statistical.py

echo "[18/$TOTAL] ROI analysis"
"$PY" 05_results/roi_analysis.py

echo "[19/$TOTAL] Multi-model comparison (Table 1)"
"$PY" 03_models/compare_multiple_models.py

# ---------------------------------------------------------------------------
# Phases 22-27, formerly phases_22_30/run_all_experiments_v2.sh. They read
# 05_results/benchmark_statistical_results.csv,
# 05_results/schedulers/multi_scheduler_benchmark.csv and
# 05_results/fairness/fairness_metrics.csv, so they must follow steps 5, 7
# and 17. Folding them in gives the repository ONE entry point, so that
# "run_all_experiments.sh regenerates every result" is a true statement.
# ---------------------------------------------------------------------------
echo "[20/$TOTAL] Phases 22-27: bootstrap CIs, OOD sensitivity, scheduler landscape, traces, scaling, fairness/SLA"
"$PY" phases_22_30/phase_22_stats/stats_bootstrap.py
"$PY" phases_22_30/phase_23_sensitivity/sensitivity_ood_analysis.py
"$PY" phases_22_30/phase_24_extended_schedulers/scheduler_comparison.py
"$PY" phases_22_30/phase_25_real_traces/trace_preprocessing.py
"$PY" phases_22_30/phase_26_scaling/scaling_benchmark.py
"$PY" phases_22_30/phase_27_fairness/fairness_sla_analysis.py

echo
echo "Done. Verify the regenerated artefacts against the committed ones with:"
echo "    $PY tools/verify_artifacts.py --quick"
echo "Dashboard: streamlit run dashboard.py"

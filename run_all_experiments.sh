#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# Resolve a Python interpreter (python3 preferred, fall back to python).
PY="$(command -v python3 || command -v python)"

# Force UTF-8 stdio on every Python step: several scripts print Unicode
# (importance bars, R-squared symbols) which crashes under Windows' default
# cp1252 console encoding when output is piped or redirected.
export PYTHONUTF8=1

# ---------------------------------------------------------------------------
# Step 0 regenerates the dataset and the v2 model so the pipeline works on a
# fresh checkout. Every downstream script depends on
#   02_data/improved_wait_dataset.csv  and  03_models/wait_model_v2.pkl
# which were previously assumed to already exist.
# ---------------------------------------------------------------------------
echo "[0/14] Generate dataset and train wait_model_v2"
( cd 02_data && "$PY" generate_improved_dataset.py )
"$PY" 03_models/train_improved_model.py

echo "[1/14] Ablation study"
"$PY" 03_models/ablation_study.py

echo "[2/14] Fairness analysis"
"$PY" 04_scheduler/fairness_analysis.py

echo "[3/14] Synthetic scheduler benchmark (14 schedulers + significance + TOST)"
"$PY" 04_scheduler/multi_scheduler_benchmark.py

echo "[3b/14] Runtime-estimate sensitivity sweep (SJF/backfill vs estimate quality)"
( cd 04_scheduler && "$PY" estimate_sensitivity.py )

# ---------------------------------------------------------------------------
# v3.4 headline experiments. Step 4 tests whether the learned wait model
# contributes anything to queue ORDERING beyond the job's requested size;
# step 5 re-asks the whole scheduler comparison on real traces with the real
# user runtime estimates the traces ship with. Both are prerequisites for the
# claims in the manuscript, so they run before the supporting analyses.
# ---------------------------------------------------------------------------
echo "[4/14] Ranking-degeneracy diagnostic (is the ML score a function of size?)"
( cd 04_scheduler && "$PY" ranking_degeneracy.py )

echo "[5/14] Trace-driven scheduler benchmark (real SWF traces, real user estimates)"
( cd 04_scheduler && "$PY" trace_driven_benchmark.py )

echo "[6/14] SHAP explainability"
"$PY" 03_models/explainability_shap.py

echo "[7/14] Real trace loading and synthetic-vs-real validation"
"$PY" 02_data/load_real_traces.py
"$PY" 02_data/synthetic_vs_real_comparison.py

echo "[8/14] Scaling analysis"
"$PY" 04_scheduler/scaling_analysis.py

echo "[9/14] Online learning and concept drift"
"$PY" 03_models/online_learning.py
"$PY" 03_models/concept_drift_detection.py

echo "[10/14] Baseline statistical benchmark refresh"
"$PY" 04_scheduler/benchmark_statistical.py

echo "[11/14] ROI analysis"
"$PY" 05_results/roi_analysis.py

echo "[12/14] Multi-model comparison (Table 1)"
"$PY" 03_models/compare_multiple_models.py

echo "[13/14] Done. Launch dashboard with: streamlit run dashboard.py"

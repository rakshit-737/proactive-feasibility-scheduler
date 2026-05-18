#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "[1/10] Ablation study"
python 03_models/ablation_study.py

echo "[2/10] Fairness analysis"
python 04_scheduler/fairness_analysis.py

echo "[3/10] Scheduler baselines benchmark"
python 04_scheduler/multi_scheduler_benchmark.py

echo "[4/10] SHAP explainability"
python 03_models/explainability_shap.py

echo "[5/10] Real trace loading and synthetic-vs-real validation"
python 02_data/load_real_traces.py
python 02_data/synthetic_vs_real_comparison.py

echo "[6/10] Scaling analysis"
python 04_scheduler/scaling_analysis.py

echo "[7/10] Online learning and concept drift"
python 03_models/online_learning.py
python 03_models/concept_drift_detection.py

echo "[8/10] ROI analysis"
python 05_results/roi_analysis.py

echo "[9/10] Baseline statistical benchmark refresh"
python 04_scheduler/benchmark_statistical.py

echo "[10/10] Done. Launch dashboard with: streamlit run dashboard.py"

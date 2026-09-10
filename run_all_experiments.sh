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

# SMOKE=1 selects the reduced configuration of the long steps. A smoke run
# writes REDUCED numbers and its outputs must never be committed;
# tools/verify_artifacts.py --smoke therefore runs it inside a scratch copy.
SMOKE_DEG="${SMOKE:+--quick}"
SMOKE_TDB="${SMOKE:+--smoke}"
SMOKE_POWER="${SMOKE:+--replicates 2000}"

TOTAL=22

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

# ---------------------------------------------------------------------------
# Reads step 10's per-window CSV and asks whether the equivalence tests printed
# there could have concluded anything at all. On LANL the published 20-window
# TOST had ~0.5% power at the observed effect, so its failure to certify
# equivalence is INCONCLUSIVE, not evidence of a difference. This step also
# counts the disjoint windows each trace can actually supply, so a required n
# can be compared against what the trace physically has. It changes no
# published number and does NOT raise the benchmark's window count.
# ---------------------------------------------------------------------------
echo "[11/$TOTAL] Power of the trace-driven equivalence tests (how many windows would be needed?)"
"$PY" 04_scheduler/tost_power.py $SMOKE_POWER

echo "[12/$TOTAL] Real-trace datasets (SWF replay) and real-trace validation"
"$PY" 02_data/build_real_trace_datasets.py
"$PY" 02_data/real_trace_validation.py

echo "[13/$TOTAL] SHAP explainability"
"$PY" 03_models/explainability_shap.py

echo "[14/$TOTAL] Synthetic-proxy trace: out-of-distribution check (NOT real trace data)"
"$PY" 02_data/load_real_traces.py
"$PY" 02_data/synthetic_vs_real_comparison.py

echo "[15/$TOTAL] Scaling analysis"
"$PY" 04_scheduler/scaling_analysis.py

echo "[16/$TOTAL] Uncertainty-aware scheduling benchmark (quantile intervals, OOD)"
"$PY" 04_scheduler/uncertainty_scheduler_benchmark.py

# Audits step 16's own headline. Under load a policy's published mean wait is
# taken over the jobs it managed to START, and different policies strand
# different jobs, so the improvement column can compare two different job
# populations. This step re-runs the same simulations (same 7000+run seeds) and
# reports the paired common-set comparison and each policy's started fraction
# beside the published number. It reads the two trained models (steps 1 and 3)
# and must follow step 16, whose committed CSV it reproduces cell for cell.
echo "[17/$TOTAL] Censoring / selection-bias audit of the uncertainty benchmark"
"$PY" 04_scheduler/censoring_analysis.py

echo "[18/$TOTAL] Online learning and concept drift"
"$PY" 03_models/online_learning.py
"$PY" 03_models/concept_drift_detection.py

echo "[19/$TOTAL] Baseline statistical benchmark refresh"
"$PY" 04_scheduler/benchmark_statistical.py

echo "[20/$TOTAL] Multi-model comparison (Table 1)"
"$PY" 03_models/compare_multiple_models.py

# ---------------------------------------------------------------------------
# The published model score comes from a uniformly random 80/20 row split, but
# the 2200 rows are 20 simulation runs of 110 jobs: a random split puts rows of
# the SAME run -- and adjacent instants of it -- on both sides. This step
# re-scores the identical model configuration under a run-wise grouped split
# and a within-run chronological split, plus constant baselines, so the
# generalisation number is reported next to the optimistic one. It reads only
# 02_data/improved_wait_dataset.csv (step 1) and nothing reads its output, so
# it sits with the other model-evaluation steps.
# ---------------------------------------------------------------------------
echo "[21/$TOTAL] Honest split comparison (random vs run-wise vs chronological)"
"$PY" 03_models/evaluate_splits.py

# ---------------------------------------------------------------------------
# Phases 22-27, formerly phases_22_30/run_all_experiments_v2.sh. They read
# 05_results/benchmark_statistical_results.csv,
# 05_results/schedulers/multi_scheduler_benchmark.csv and
# 05_results/fairness/fairness_metrics.csv, so they must follow steps 5, 7
# and 19. Folding them in gives the repository ONE entry point, so that
# "run_all_experiments.sh regenerates every result" is a true statement.
# ---------------------------------------------------------------------------
echo "[22/$TOTAL] Phases 22-27: bootstrap CIs, OOD sensitivity, scheduler landscape, traces, scaling, fairness/SLA"
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

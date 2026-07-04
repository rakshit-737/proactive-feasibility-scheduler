#!/bin/bash
# Phase 29: Complete Reproducibility Script
# Runs all phases 01–30 end-to-end for artifact evaluation

set -e  # Exit on error

# Force UTF-8 stdio on every Python step (Windows defaults to cp1252, which
# crashes scripts that print Unicode symbols when output is piped/tee'd).
export PYTHONUTF8=1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${PROJECT_ROOT}/run_all_experiments_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Proactive Feasibility Scheduler"
echo "Complete Reproducibility Run (Phases 01–30)"
echo "=========================================="
echo "Log: ${LOG_FILE}"
echo ""

exec &> >(tee -a "${LOG_FILE}")

# Phase 01–09: Core simulation & model training (root pipeline)
echo "[Phases 01–09] Running core simulation pipeline..."
if [ -f "${PROJECT_ROOT}/../run_all_experiments.sh" ]; then
    bash "${PROJECT_ROOT}/../run_all_experiments.sh"
else
    echo "WARNING: Phases 01-09 pipeline (run_all_experiments.sh) not found - skipping"
fi

# Phase 22: Statistical rigor
echo "[Phase 22] Computing bootstrap confidence intervals..."
if [ -f "${PROJECT_ROOT}/phase_22_stats/stats_bootstrap.py" ]; then
    python "${PROJECT_ROOT}/phase_22_stats/stats_bootstrap.py"
fi

# Phase 23: OOD sensitivity
echo "[Phase 23] Analyzing OOD robustness..."
if [ -f "${PROJECT_ROOT}/phase_23_sensitivity/sensitivity_ood_analysis.py" ]; then
    python "${PROJECT_ROOT}/phase_23_sensitivity/sensitivity_ood_analysis.py"
fi

# Phase 24: Scheduler comparison
echo "[Phase 24] Comparing against published baselines..."
if [ -f "${PROJECT_ROOT}/phase_24_extended_schedulers/scheduler_comparison.py" ]; then
    python "${PROJECT_ROOT}/phase_24_extended_schedulers/scheduler_comparison.py"
fi

# Phase 25: Real traces
echo "[Phase 25] Ingesting and validating real traces..."
if [ -f "${PROJECT_ROOT}/phase_25_real_traces/trace_preprocessing.py" ]; then
    python "${PROJECT_ROOT}/phase_25_real_traces/trace_preprocessing.py"
fi

# Phase 26: Scaling
echo "[Phase 26] Benchmarking on scaled clusters..."
if [ -f "${PROJECT_ROOT}/phase_26_scaling/scaling_benchmark.py" ]; then
    python "${PROJECT_ROOT}/phase_26_scaling/scaling_benchmark.py"
fi

# Phase 27: Fairness
echo "[Phase 27] Evaluating fairness and SLA compliance..."
if [ -f "${PROJECT_ROOT}/phase_27_fairness/fairness_sla_analysis.py" ]; then
    python "${PROJECT_ROOT}/phase_27_fairness/fairness_sla_analysis.py"
fi

echo ""
echo "=========================================="
echo "REPRODUCIBILITY RUN COMPLETE"
echo "=========================================="
echo "Outputs:"
echo "  - Log file: ${LOG_FILE}"
echo "  - Results directory: ${PROJECT_ROOT}/../05_results/"
echo "  - Phases 22–30 outputs: ${PROJECT_ROOT}/"
echo ""

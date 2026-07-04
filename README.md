# Proactive Feasibility Scheduler

Simulation-driven proactive GPU job scheduling with ML-based wait-time prediction.

## Quick start
```bash
pip install -r requirements.txt
bash run_all_experiments.sh
```
The pipeline now bootstraps itself: step 0 of `run_all_experiments.sh` regenerates `02_data/improved_wait_dataset.csv` and trains `03_models/wait_model_v2.pkl` before any analysis runs, so a fresh checkout works end-to-end. `requirements.txt` includes every dependency the scripts import (including `lightgbm` and `catboost`, used by the multi-model comparison).

## Key features
- 30-phase research pipeline (simulation, ML, benchmarking, robustness, explainability, ROI, statistics, OOD, fairness/SLA, deployment)
- Ablation analysis over 12 features
- Fairness and starvation-prevention analysis
- 5-scheduler benchmark (FIFO, SJF, Priority, Proactive, NN)
- Proxy-trace validation (SWF parser ready for real traces), scaling analysis, concept drift adaptation
- Reproducibility kit (shell script + Docker + requirements)
- Interactive dashboard (`dashboard.py`) and GitHub Pages docs (`docs/`)

## Headline results
All figures below come from a proper 20% holdout / 5-fold CV (model) and a seeded 40-run paired benchmark (scheduler). They are deliberately the honest numbers, not in-sample ones, and are fully reproducible via `bash run_all_experiments.sh` (the pipeline is now seeded end-to-end).

| Metric | Value | Notes |
|---|---|---|
| Wait-time model quality | **R² ≈ 0.84, MAE ≈ 4.69** (20% holdout) | 5-fold CV MAE 4.74 ± 0.42. Never quote in-sample numbers as model quality. |
| Mean wait-time reduction | **7.7% ± 8.9%** vs FIFO | 40-run paired benchmark, paired t-test p = 1.4e-06, bootstrap 95% CI [5.0%, 10.3%] |
| GPU utilisation | **Unchanged** (≈64%) | Improvement is from queue ordering only |
| Tail latency (max wait) | **Worse: ~58 → 125 ts** | Trade-off: proactive reordering increases tail latency |
| Fairness (Gini of waits) | **Worse: 0.53 → 0.80** | Mean-wait gain comes at a fairness cost; anti-starvation variant recovers to 0.69 |
| Proxy-trace transfer | **R² ≈ 0.02** | Evaluated on a **synthetic LANL-schema proxy** (no real trace is committed); real-trace transfer is unvalidated |
| OOD robustness | **Mean R² < 0** across 72 shifted scenarios | Model must be retrained per deployment regime; FIFO fallback recommended |

**Honest summary:** the proactive scheduler delivers a small but statistically significant mean-wait reduction at no utilisation cost, but trades off tail latency and fairness, degrades out-of-distribution, and has not yet been validated on real traces (the LANL-style trace on disk is a synthetic proxy). See `RESULTS.md`, `docs/project_report.html`, and `docs/research_progress.html` for detail.

## Results folders
- `05_results/models`
- `05_results/schedulers`
- `05_results/scaling`
- `05_results/fairness`
- `05_results/shap`
- `05_results/traces`
- `05_results/roi`

## Documentation
- **Full project documentation (start here): `docs/index.html`**
- Research progress: `docs/research_progress.html`
- Project report: `docs/project_report.html`
- Methodology: `METHODOLOGY.md`
- Results summary: `RESULTS.md`
- Reproducibility: `README_REPRODUCIBILITY.md`
- Deployment guide: `DEPLOYMENT.md`

## Citation
If you use this work, cite the repository and include the phase/version context (v3.1, phases 01–30, post-audit July 2026).

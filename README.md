# Proactive Feasibility Scheduler

Simulation-driven proactive GPU job scheduling with ML-based wait-time prediction.

## Quick start
```bash
pip install -r requirements.txt
bash run_all_experiments.sh
```
<<<<<<< HEAD
The pipeline now bootstraps itself: step 0 of `run_all_experiments.sh` regenerates `02_data/improved_wait_dataset.csv` and trains `03_models/wait_model_v2.pkl` before any analysis runs, so a fresh checkout works end-to-end. `requirements.txt` includes every dependency the scripts import (including `lightgbm` and `catboost`, used by the multi-model comparison).
=======
>>>>>>> d420ccc5fc239b03e840287805338721fb55585d

## Key features
- 21-phase research pipeline (simulation, ML, benchmarking, robustness, explainability, ROI)
- Ablation analysis over 12 features
- Fairness and starvation-prevention analysis
- 5-scheduler benchmark (FIFO, SJF, Priority, Proactive, NN)
- Real-trace validation, scaling analysis, concept drift adaptation
- Reproducibility kit (shell script + Docker + requirements)
- Interactive dashboard (`dashboard.py`) and GitHub Pages docs (`docs/`)

<<<<<<< HEAD
## Headline results
All figures below come from a proper 20% holdout / 5-fold CV (model) and a 40-run paired benchmark (scheduler). They are deliberately the honest numbers, not in-sample ones.

| Metric | Value | Notes |
|---|---|---|
| Wait-time model quality | **R² ≈ 0.83, MAE ≈ 5.35** (20% holdout) | 5-fold CV MAE 5.68 ± 0.71. In-sample reads R² ≈ 0.96 — do **not** quote that as model quality. |
| Mean wait-time reduction | **7.5% ± 9.0%** vs FIFO | 40-run paired benchmark, paired t-test p = 1.10e-06 |
| GPU utilisation | **Unchanged** (≈64%) | Improvement is from queue ordering only |
| Tail latency (max wait) | **Worse: ~58 → 128 ts** | Trade-off: proactive reordering increases tail latency |
| Fairness (Gini of waits) | **Worse: 0.51 → 0.79** | Mean-wait gain comes at a fairness cost |
| Real-trace transfer (LANL) | **R² ≈ −0.02** | Synthetic-trained model does not generalise without retraining |

**Honest summary:** the proactive scheduler delivers a small but statistically significant mean-wait reduction at no utilisation cost, but trades off tail latency and fairness, and does not yet transfer to real traces. See `RESULTS.md`, `project_report.html`, and `research_progress.html` for detail.

=======
>>>>>>> d420ccc5fc239b03e840287805338721fb55585d
## Results folders
- `05_results/models`
- `05_results/schedulers`
- `05_results/scaling`
- `05_results/fairness`
- `05_results/shap`
- `05_results/traces`
- `05_results/roi`

## Documentation
- Research progress: `research_progress.html`
- Project report: `project_report.html`
- Methodology: `METHODOLOGY.md`
- Results summary: `RESULTS.md`
- Reproducibility: `README_REPRODUCIBILITY.md`
- GitHub Pages entry: `docs/index.html`

## Citation
If you use this work, cite the repository and include the phase/version context (v3.0, phases 01–21).

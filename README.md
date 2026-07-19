<div align="center">

# Proactive Feasibility Scheduler

**Simulation-driven proactive GPU job scheduling with ML-based wait-time prediction.**

[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/rakshit-737/proactive-feasibility-scheduler/actions/workflows/ci.yml/badge.svg)](https://github.com/rakshit-737/proactive-feasibility-scheduler/actions/workflows/ci.yml)
[![Results: reproducible](https://img.shields.io/badge/results-reproducible-brightgreen.svg)](README_REPRODUCIBILITY.md)

[Methodology](METHODOLOGY.md) ·
[Results](RESULTS.md) ·
[Reproducibility](README_REPRODUCIBILITY.md) ·
[Deployment](DEPLOYMENT.md) ·
[Documentation](docs/index.html)

</div>

---

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
- **Real-trace validation (v3.2)**: two Parallel Workloads Archive traces (LANL CM-5, SDSC SP2) evaluated end-to-end
- EASY-backfill baseline + predicted-wait hybrid; bounded-fairness wait-budget (Pareto-swept); uncertainty-aware scheduling study
- Scaling analysis, online learning, concept drift adaptation
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
| Real-trace transfer, zero-shot | **R² ≈ 0** (both traces) | Synthetic-trained model does not transfer to LANL CM-5 or SDSC SP2 — quantified on real data (v3.2) |
| Real-trace, **retrained** | **R²(log) 0.49** on SDSC SP2 | Chronological holdout, vs −0.69 median baseline; LANL CM-5 (interactive machine) only 0.10 — signal is machine-dependent |
| Backfill baseline (EASY) | **Gini 0.365, max 55 ts** | Reservation guarantee costs ~45% mean wait vs dispatch-all FIFO but is the fairest policy in the study (v3.2) |
| Fairness budget B | **B=60: +7.1% wait gain, max 81 ts** | Tunable Pareto dial between pure proactive (+13.2%, max 136) and FIFO (v3.2) |
| OOD robustness | **Mean R² < 0** across 72 shifted scenarios | Retrain per regime; interval-width guards tested and **not** reliable (68% coverage) — use the drift trigger |

**Honest summary:** the proactive scheduler delivers a small but statistically significant mean-wait reduction at no utilisation cost. The costs are quantified: tail latency and per-job fairness worsen (now tunable via the wait-budget B), prediction collapses out-of-distribution, and zero-shot sim-to-real transfer fails — but retraining on a real batch trace recovers R²(log) 0.49 (SDSC SP2), making "retrain per deployment" an evidence-backed protocol rather than a caveat. See `RESULTS.md`, `docs/index.html`, `docs/project_report.html`, and `docs/research_progress.html` for detail.

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

If you use this work, please cite it. GitHub's **"Cite this repository"** button (shown on
the repository sidebar) reads the machine-readable [`CITATION.cff`](CITATION.cff). A BibTeX
entry:

```bibtex
@software{rameshbabu_proactive_feasibility_scheduler_2026,
  author  = {Rameshbabu, Rakshit},
  title   = {Proactive Feasibility Scheduler: Simulation-Driven GPU Job
             Scheduling with ML-Based Wait-Time Prediction},
  version = {3.2},
  year    = {2026},
  url     = {https://github.com/rakshit-737/proactive-feasibility-scheduler}
}
```

Please include the version context (v3.2, phases 01–30 + research extensions, July 2026).
For a permanently archived, DOI-backed snapshot, enable the GitHub–Zenodo integration and
publish a release, then add the resulting DOI to [`CITATION.cff`](CITATION.cff) and this
section.

## License

Released under the [MIT License](LICENSE) © 2026 Rakshit Rameshbabu.

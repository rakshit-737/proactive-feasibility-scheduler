<div align="center">

# Proactive Feasibility Scheduler

**An evaluation study of ML-based job scheduling — and a negative result.**

[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/rakshit-737/proactive-feasibility-scheduler/actions/workflows/ci.yml/badge.svg)](https://github.com/rakshit-737/proactive-feasibility-scheduler/actions/workflows/ci.yml)
[![Results: reproducible](https://img.shields.io/badge/results-reproducible-brightgreen.svg)](README_REPRODUCIBILITY.md)

[Methodology](METHODOLOGY.md) ·
[Results](RESULTS.md) ·
[Manuscript](phases_22_30/phase_28_manuscript/manuscript.tex) ·
[Reproducibility](README_REPRODUCIBILITY.md) ·
[Deployment](DEPLOYMENT.md) ·
[Documentation](docs/explanation.html)

</div>

---

> **Headline (v3.4).** Using a learned wait-time regressor to order a scheduling
> queue is *structurally degenerate*. At any dispatch instant every queued job
> sees the same cluster, so only the job's own features differ — and in the
> standard cluster-state feature set every one of those is a deterministic
> function of the job's requested size. The learned score is therefore a
> function of requested size alone. Verified over **25,306 real dispatch
> instants with zero counterexamples**: two equally-sized queued jobs *never*
> receive different scores, and 8 of the 12 features vary across the queue in
> **0.0%** of instants. Consequently a one-line `sort by requested size`
> is statistically **equivalent** to the full XGBoost pipeline (paired TOST
> p = 2e-16), an MLP over the same features reproduces that sort *bit-identically*,
> and the synthetic 7.7% gain over FCFS does not replicate on real traces.
>
> See [`04_scheduler/ranking_degeneracy.py`](04_scheduler/ranking_degeneracy.py)
> and [the manuscript](phases_22_30/phase_28_manuscript/manuscript.tex).

## Quick start
```bash
pip install -r requirements.txt
bash run_all_experiments.sh
```
The pipeline now bootstraps itself: step 0 of `run_all_experiments.sh` regenerates `02_data/improved_wait_dataset.csv` and trains `03_models/wait_model_v2.pkl` before any analysis runs, so a fresh checkout works end-to-end. `requirements.txt` includes every dependency the scripts import (including `lightgbm` and `catboost`, used by the multi-model comparison).

## Key features
- **Ranking-degeneracy diagnostic (v3.4)**: instruments real dispatch decisions to test whether a learned wait-time score can distinguish co-queued jobs at all, and recovers the size→priority lookup table the model collapses to
- **Trace-driven scheduler benchmark (v3.4)**: event-driven, second-exact replay of two real Parallel Workloads Archive traces through 12 policies, using the **real user runtime estimates the traces contain** instead of a simulated estimate model
- **Equivalence testing (v3.4)**: paired TOST throughout, so "these two policies perform the same" is a positive finding rather than a failure to reject
- **14-scheduler synthetic benchmark**: FCFS/first-fit, strict FIFO, SJF (oracle / f-model / modal), Priority, HRRN, **Smallest-first (the ML-free control)**, canonical EASY backfill (oracle + estimates), conservative backfill, preemptive SRPT, Proactive, NN, predicted-wait backfill hybrid — Holm-adjusted pairwise significance plus an estimate-quality sweep
- 30-phase research pipeline (simulation, ML, benchmarking, robustness, explainability, ROI, statistics, OOD, fairness/SLA, deployment)
- Bounded-fairness wait-budget (Pareto-swept); uncertainty-aware scheduling study
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
| Classical-baseline landscape (v3.3) | **Any runtime signal beats Proactive on mean wait** | SJF-oracle 12.34 / SJF-est 13.32 / SJF-modal 14.06 / SRPT 14.02 / HRRN 15.55 vs Proactive 16.10 ts — but at 2× worse tails (SJF max 146, Gini 0.79 vs HRRN 65 / 0.54) |
| **Ranking degeneracy (v3.4)** | **0 counterexamples in 25,306 dispatch instants** | Two equally-sized co-queued jobs never get different scores. 8/12 features vary across the queue in 0.0% of instants; a ~9-job queue gets only 2.3–3.1 distinct priority levels; in 18–26% of instants all scores tie and the policy silently *is* FCFS |
| **ML-free control (v3.4)** | **`sort by requested size` ≡ XGBoost pipeline** | Synthetic: −0.18%, paired TOST p=2e-16, diff CI [−0.14,+0.08] ts vs ±1.61 margin. SDSC SP2: −0.05%, TOST p=1.8e-12. The MLP baseline reproduces the size sort **bit-identically** on all 20 runs |
| **Trace-driven benchmark (v3.4)** | **The synthetic gain does not replicate** | 20 paired 7-day windows/trace at load ≈0.70. Proactive vs FCFS: −20.4% on SDSC (p=0.042) but **−4.5%, p=0.48 on LANL**. SJF on *real* user estimates beats Proactive by 20.2% (SDSC, Holm p=0.009) and 15.3% (LANL) |
| **Real estimate error (v3.4)** | **The f-model understates it badly** | Real: SDSC median 6.9× over-estimate, 0.1% under; LANL median 1.5× but **36.3% under-estimates** — which the over-estimate-only f-model cannot produce. Cost to EASY vs perfect estimates: +6.2% (SDSC) but **+74% (LANL, p=0.025)**, against "near-insensitive" under the f-model |
| Backfill baseline (canonical EASY, v3.3) | **+11.8% mean wait vs FCFS/first-fit, −26% vs strict FIFO** | Reservation price re-measured after implementing the full two-condition EASY rule (v3.2's stricter variant overstated it at ~45%); Gini 0.45, max 52 ts |
| Fairness budget B | **B=60: +7.1% wait gain, max 81 ts** | Tunable Pareto dial between pure proactive (+13.2%, max 136) and FIFO (v3.2) |
| OOD robustness | **Mean R² < 0** across 72 shifted scenarios | Retrain per regime; interval-width guards tested and **not** reliable (68% coverage) — use the drift trigger |

**Honest summary (v3.4).** The v3.3 study established that any runtime signal beats the proactive scheduler on mean wait, leaving it a claimed niche in the *zero-runtime-information* regime. v3.4 removes that niche. The learned score cannot distinguish two co-queued jobs by anything except requested size — this is a property of the feature set, provable by construction and confirmed with zero counterexamples over 25,306 real dispatch decisions — so the policy is a per-instant lookup table from size to priority. An ML-free size sort is statistically equivalent to it, an MLP over the same features *is* that sort, and on real traces its advantage over plain FCFS is machine-dependent and insignificant on LANL CM-5. The measured improvement was evidence about size-based ordering, not about learning.

The constructive takeaways: (1) the **non-degeneracy condition** — a wait-time feature set can only produce a meaningful ranking if it contains a per-job attribute that is *not* a function of size given the state (a runtime estimate, user history, partition identity, dependency structure); (2) report the **ML-free control the feature set implies**, not FIFO; (3) use **equivalence tests** — with difference tests alone, the p=0.65 size-sort comparison reads as "no significant difference" and gets dropped instead of being recognised as the finding. See `RESULTS.md`, the [manuscript](phases_22_30/phase_28_manuscript/manuscript.tex), and `docs/explanation.html`.

## Results folders
- `05_results/degeneracy` — ranking-degeneracy diagnostic (v3.4)
- `05_results/trace_schedulers` — trace-driven benchmark on real SWF traces (v3.4)
- `05_results/models`
- `05_results/schedulers`
- `05_results/scaling`
- `05_results/fairness`
- `05_results/shap`
- `05_results/traces`
- `05_results/roi`

## Documentation
- **Full project documentation (start here): `docs/explanation.html`**
- Research progress: `docs/research_progress.html`
- Project report: `docs/project_report.html`
- Methodology: `METHODOLOGY.md`
- Results summary: `RESULTS.md`
- Reproducibility: `README_REPRODUCIBILITY.md`
- Deployment guide: `DEPLOYMENT.md`

## Reproducing the v3.4 headline results
```bash
cd 04_scheduler
python ranking_degeneracy.py        # the degeneracy result (25,306 instants)
python trace_driven_benchmark.py    # 12 policies x 2 real traces x 20 windows
python multi_scheduler_benchmark.py # 14-scheduler synthetic study + TOST
```
Or `bash run_all_experiments.sh` for the full seeded pipeline (steps 4 and 5).

## Citation

If you use this work, please cite it. GitHub's **"Cite this repository"** button (shown on
the repository sidebar) reads the machine-readable [`CITATION.cff`](CITATION.cff). A BibTeX
entry:

```bibtex
@software{rameshbabu_proactive_feasibility_scheduler_2026,
  author  = {Rameshbabu, Rakshit},
  title   = {Proactive Feasibility Scheduler: An Evaluation Study of ML-Based
             GPU Job Scheduling},
  version = {3.4},
  year    = {2026},
  url     = {https://github.com/rakshit-737/proactive-feasibility-scheduler}
}
```

Please include the version context (v3.4, phases 01–30 + research extensions, July 2026).
For a permanently archived, DOI-backed snapshot, enable the GitHub–Zenodo integration and
publish a release, then add the resulting DOI to [`CITATION.cff`](CITATION.cff) and this
section.

## License

Released under the [MIT License](LICENSE) © 2026 Rakshit Rameshbabu.

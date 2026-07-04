# Changelog

## v3.1 — July 2026 · Full audit, repair & regeneration

This release is the result of a complete project audit: every file was read, every
claim re-verified against the code and data, and every script executed. A pre-repair
snapshot of the project was kept locally (outside the repo).

### 🔬 Scientific-integrity fixes
- **Trace relabelled honestly** — `02_data/lanl_trace_sample.csv` was proven to be the
  loader's synthetic fallback (row-for-row reproduction), not a real LANL trace. All
  code, results, and docs now label it `synthetic_proxy_trace`; the loader falls back
  loudly and documents how to obtain real traces. Real-trace validation is tracked as
  the open Phase 25.
- **Split-before-fit** — `synthetic_vs_real_comparison.py` fit the model on all data
  *before* splitting, inflating the "holdout" to R² ≈ 0.96. Fixed; honest holdout is
  R² 0.837 / MAE 4.69 ts.
- **Fabricated results removed** — Phases 23 (OOD), 24 (scheduler landscape),
  26 (scaling), and 27 (fairness/SLA) previously reported hardcoded constants or a
  silent heuristic stand-in for the model. All four now compute from real data and
  real simulations; baselines without real implementations (SLURM/K8s/Yarn rows) were
  dropped rather than invented.
- **Manuscript corrected** — `manuscript.tex` now compiles (inputenc fix), all
  citations resolve, and every number matches the regenerated artifacts, including
  the fairness/tail-latency trade-off and the unvalidated sim-to-real transfer.
- **v1 scheduler dispatch bug** — a stray `break` limited the v1 proactive experiment
  to one job per tick; fixed (v2 semantics: dispatch everything that fits).
- **Degenerate profile fixed** — the low-contention dataset produced all-zero waits
  (trivial R² = 1.0); it now has real variance, plus a zero-variance guard.

### 🔧 Engineering fixes
- **Determinism** — dataset generators and benchmarks are seeded (global 42,
  per-run 42+i); two runs produce byte-identical outputs.
- **Windows support** — pipelines export `PYTHONUTF8=1`; all text writes specify
  UTF-8 (previously three scripts crashed on cp1252 consoles).
- **Path independence** — eight scripts that only worked from a specific working
  directory now resolve paths from their own location.
- **Pipeline order** — the 40-run statistical benchmark now runs *before* the ROI
  analysis that consumes it; the ROI parser reads the file's actual (long) format.
- **SWF parser** — off-by-one in the Standard Workload Format field mapping fixed
  (requested processors was read from the requested-time column).
- **Dependencies** — `requirements.txt` re-pinned to versions verified working on
  Python 3.14 (adds previously-missing seaborn/joblib; shap/lightgbm/catboost/
  streamlit/plotly now installable and used by the pipeline).
- **Archive hygiene** — `07_archive/` scripts carry ARCHIVED headers naming their
  successors and fail fast with clear messages instead of cryptic errors.

### 📊 Regenerated results (canonical)
- Model: holdout R² 0.837 / MAE 4.69 ts (5-fold CV 4.74 ± 0.42; tuned 4.57 / 0.850).
- Scheduler: 7.71% mean-wait reduction vs FIFO (40 seeded paired runs; bootstrap
  95% CI [4.95%, 10.34%]; paired t-test p = 1.4e-06; utilisation unchanged ≈64%).
- Trade-offs: max wait ~58 → ~125 ts; per-job Gini 0.526 → 0.796 (anti-starvation
  variant: 0.688 / 87 ts).
- OOD: mean R² −0.31 across 72 shifted scenarios → retrain per regime + FIFO fallback.
- Proxy-trace transfer: R² ≈ 0.015 (real-trace validation open).

### 📚 New & reorganised documentation
- **`docs/index.html`** — complete interactive project documentation: every concept
  explained from scratch, 9 data charts, architecture diagrams, a **live in-browser
  simulator** running a distilled 40-tree export of the real XGBoost model
  (distilled holdout R² 0.842), per-file repository guide, audit changelog, future
  roadmap, glossary. Light/dark themes, mobile navigation, accessible table views.
- `docs/project_report.html` and `docs/research_progress.html` — moved into `docs/`
  and updated to the regenerated numbers (July 2026 banners link to the main docs).
- `DEPLOYMENT.md` — new operations runbook (rollout modes, monitoring thresholds,
  drift response, rollback).
- README / RESULTS / METHODOLOGY / phases_22_30 summaries — synced to regenerated
  artifacts; stale paths and version references fixed.
- Removed from the tree: stray `Microsoft.VisualStudio.Services.VSIXPackage`,
  `__pycache__/`, `catboost_info/` (also gitignored).

### How to publish this update to GitHub
From your existing clone of the repository:

```bash
# 1. copy the contents of this folder over your clone (or use it directly)
# 2. review what changed
git status
git diff --stat

# 3. commit and push
git add -A
git commit -m "v3.1: full audit & repair — honest regenerated results, seeded pipeline, interactive docs"
git push
```

If this folder itself should become the repo (fresh start):

```bash
git init
git add -A
git commit -m "v3.1: full audit & repair — honest regenerated results, seeded pipeline, interactive docs"
git remote add origin https://github.com/<you>/proactive-feasibility-scheduler.git
git push -u origin main --force   # only if you intend to replace history
```

Tip: enable **GitHub Pages → deploy from branch → /docs** to serve
`docs/index.html` as the project site.

---

## v3.0 — May 2026
Phases 01–21 research pipeline (simulation, features, model, schedulers,
benchmarks, explainability, ROI) plus phases 22–30 scaffolding.

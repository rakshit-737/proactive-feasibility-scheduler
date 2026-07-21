# Changelog

## v3.4 — July 2026 · Ranking degeneracy, trace-driven re-evaluation, and a reframed manuscript

v3.3 established that any runtime signal beats the proactive scheduler on mean
wait, leaving it a claimed niche in the *zero-runtime-information* regime. v3.4
tests that niche and removes it, and moves the scheduler evaluation off the
synthetic generator onto real workloads.

### The headline: the ML score cannot distinguish co-queued jobs
At a fixed dispatch instant every queued job observes the same cluster, so only
per-job features can differ — and in this feature set each of those
(`can_fit_now`, `gpu_fit_ratio`, `node_availability`, `queue_pressure`) is a
deterministic function of the requested size given the state. The predicted
score is therefore `g_S(size)` and the ranking is a permutation of the size
order; the remaining **eight** features describe only the cluster, so they shift
all scores equally and **cannot reorder anything**.

- **NEW `04_scheduler/ranking_degeneracy.py`** — instruments real dispatch
  decisions via a `RANK_OBSERVER` hook added to both benchmarks (no simulator
  duplication). Over **25,306 instants in three settings: zero counterexamples**
  — two equally-sized co-queued jobs never receive different scores. 8 of the 12
  synthetic features vary across the queue in **0.0%** of instants. A ~9-job
  queue receives only 2.3–3.1 distinct priority levels, and in 18–26% of instants
  every score ties, so the policy silently *is* FCFS. Recovers and plots the
  size→priority lookup table (monotone in 52–67% of instants).
- **NEW `04_scheduler/size_scheduler.py`** — `SMALLEST` / `SMALLEST_FIRST`: sort
  by requested size, no model. The ML-free control the feature set implies.
- **Result**: statistically **equivalent** to the XGBoost pipeline — synthetic
  −0.18%, paired TOST p=2e-16, diff CI [−0.14,+0.08] ts vs a ±1.61 margin; SDSC
  SP2 −0.05%, TOST p=1.8e-12. The **MLP baseline is bit-identical to the size
  sort on every metric of all 20 runs**. On LANL the verdict is **inconclusive**,
  not "different": the size sort is 14.4% worse on mean wait, but that does not
  survive Holm correction on the t-test (p=0.25; Wilcoxon disagrees at p=0.015),
  shrinks to +7.3% at p=0.60 on bounded slowdown, and TOST cannot certify
  equivalence either (p=0.77). On that machine the ML scheduler also fails to
  beat plain FCFS (p=0.48).
- **Retraction within this release.** A draft of this entry explained the LANL
  result by its learned size table being less monotone. That is wrong: the
  per-instant monotone fraction is 51.9% on SDSC — where equivalence *does* hold
  — against 61.5% on LANL, so monotonicity does not track where the equivalence
  fires. LANL is reported as underpowered, not mechanistically explained. The
  jagged SDSC curve in `size_priority_table.png` is likewise mostly sampling
  noise (51 distinct sizes vs 6 on LANL, 8 synthetic), not ranking instability.

### Trace-driven scheduler benchmark on real workloads
- **NEW `04_scheduler/trace_driven_benchmark.py`** — event-driven, second-exact
  replay of LANL CM-5 and SDSC SP2 through **12 policies**, 20 paired windows per
  trace (3-day warm-up not measured + 7 measured days), offered load ≈0.70,
  windows spaced evenly rather than selected by load. Capacity-only allocator
  (SWF records no per-node placement). The ML scheduler is given its best case:
  a model **retrained on the same machine's earlier data** (chronological 60%
  split, no leakage).
- Uses the **real user runtime estimates the traces contain** (SWF field 9)
  instead of the simulated f-model. Missing estimates fall back to the trace
  median, deliberately *not* the true runtime, so estimate-driven policies get no
  free oracle.
- **The synthetic gain does not replicate**: Proactive vs FCFS is −20.4% on SDSC
  (p=0.042) but **−4.5%, p=0.48 on LANL**. SJF on real user estimates beats
  Proactive by 20.2% (SDSC, Holm p=0.009) and 15.3% (LANL). `PROACTIVE_EST` —
  the model retrained *with* the estimate as a feature — still loses to plain SJF
  on both machines (+7.9% / +17.5%): the pipeline's ceiling is the sort it is
  trying to learn.

### Real estimate error vs the f-model
- Measured, not simulated: SDSC median 6.91× over-estimate with 0.1%
  under-estimates; LANL median 1.51× but **36.3% under-estimates** — which the
  over-estimate-only f-model (`est = runtime·U(1,C)`) **cannot produce at all**.
- v3.3 concluded from the f-model that EASY is "near-insensitive to estimate
  quality". On the traces real estimate error costs EASY **+6.2% on SDSC but
  +74% on LANL** (p=0.025) versus perfect estimates: under-estimates break the
  reservation guarantee, and over-estimate-only noise cannot reveal it. The
  f-model should not be the sole estimate-error model in backfill studies.

### Methodology hardening
- **Equivalence testing** (`simstats.tost_equivalence`, `equivalence_table`):
  claims of *sameness* now use paired TOST with a 10%-of-reference margin,
  reporting both one-sided p-values and the 90% CI of the difference. Without
  it, the p=0.65 size-sort comparison reads as "no significant difference" and
  gets discarded instead of recognised as the result.
- **Mean bounded slowdown** `max(turnaround/max(runtime,τ),1)` promoted to a
  first-class reported metric everywhere (it was computed but never surfaced).
- **NEW `04_scheduler/simstats.py`** — shared Gini / Holm / paired-significance /
  TOST helpers, so the synthetic and trace studies apply identical statistical
  machinery; removes the duplicated copies from `multi_scheduler_benchmark.py`.
- `hrrn_scheduler.py` gains `order_queue_estimated` (deployable HRRN on user
  estimates), mirroring `sjf_scheduler.py`.
- Synthetic benchmark now batches PROACTIVE's per-queue predictions into one
  call — **verified bit-identical** to the previous per-job loop (max change in
  `mean_wait` across all pre-existing schedulers: 0.0).
- `run_all_experiments.sh` extended to 14 steps (adds the degeneracy diagnostic
  and the trace-driven benchmark); phase 24 positioning statement regenerated
  with the SMALLEST control.

### Manuscript reframed
`phases_22_30/phase_28_manuscript/manuscript.tex` rewritten from "we propose an
ML scheduler that beats FIFO" to what the evidence supports: **"Ranking
Degeneracy: A Learned Wait-Time Model Schedules No Better Than Sorting by
Requested Size"** — an evaluation study and a negative result, with a stated
non-degeneracy condition for feature sets and two methodological
recommendations. Also fixes two long-standing manuscript defects: a stale
in-sample `R² ≈ 0.937` presented as holdout quality, and a claim that runtimes
were "sampled from real distributions (LANL, Alibaba traces)" when the synthetic
generator uses `randint(5,20)`.

### Known limitation (stated, not hidden)
Simulator fidelity against recorded waits is good on LANL (38.8 vs 33.2 min mean;
the simulator is slightly pessimistic) but permissive on SDSC (174.6 vs 630.9 min
mean, recorded **median** 19.3 min) — SDSC's mean is dominated by a tail our
capacity-only model cannot reproduce, since SWF records no placement or site
policy. SDSC results read as "a correctly-loaded 128-processor machine driven by
real distributions", not a replay of SDSC's queue. The degeneracy result does not
depend on this: it is a property of the feature vector and holds in all three
substrates.

## v3.3 — July 2026 · Classical-baseline landscape: estimated SJF, canonical EASY, conservative BF, HRRN, preemptive SRPT

Response to faculty review: "add EASY/reservation backfill, SJF with perfect or
estimated lengths, and preemption/starvation-aware policies — do they explain
the improvement?" The 20-run paired benchmark now covers **13 schedulers**
with Holm-adjusted pairwise significance (t + Wilcoxon) against both
PROACTIVE and FCFS (`05_results/schedulers/multi_scheduler_{runs,benchmark,significance}.csv`).

### New classical baselines
- **SJF-EST** — SJF on user estimates (f-model, Mu'alem & Feitelson 2001:
  est = runtime·f, f~U(1,C), C=5) and **SJF-MODAL** (estimates rounded up to a
  {10, 20}-tick menu — the rank-destroying regime of Tsafrir & Feitelson 2005).
- **Canonical EASY** — added the missing second backfill condition (a candidate
  fitting in the shadow-time surplus may start even if it outlives the shadow);
  `04_scheduler/backfill_scheduler.py` now takes `est_runtime_of` so EASY also
  runs on estimates (**EASY-EST**). Verified: 300-trial property test that a
  reserved head is never delayed.
- **Conservative backfill (CONS-BF)** — a reservation for EVERY queued job via
  a future free-capacity profile (helper property-tested against a brute-force
  oracle on 3000 random instances).
- **HRRN** — highest response ratio next (starvation-aware aging built into the
  dispatch ratio).
- **Preemptive SRPT** — shortest-remaining-processing-time with a 1-tick
  checkpoint penalty per preemption (runtime oracle; ~41.5 preemptions/run).
- **Strict FIFO** — textbook head-blocking FIFO, exposing that the historical
  'FIFO' baseline is FCFS + unrestricted first-fit dispatch (label kept, now
  documented; strict FIFO is the honest EASY reference).
- **Estimate-quality sweep** — `04_scheduler/estimate_sensitivity.py` sweeps
  f-model C ∈ {1,2,3,5,10} + modal estimates with coupled noise and per-C
  paired CIs vs PROACTIVE (`05_results/schedulers/estimate_sensitivity*.csv/png`).

### Headline findings (honest)
- **Any runtime signal beats the wait-prediction scheduler on mean wait**:
  SJF-oracle 12.34 < SJF-EST 13.32 < SRPT 14.02 < HRRN 15.55 < PROACTIVE 16.10
  ts; even modal SJF (two estimate classes) reaches 14.06. Runtime-ordered
  policies pay in tails (SJF max 146 ts / Gini 0.79; SRPT 151 / 0.81); HRRN is
  the classical all-rounder (15.55 / max 65 / Gini 0.54, beating PROACTIVE's
  tail by 2×). PROACTIVE's defensible niche: **zero-runtime-information**
  (−6.5% vs FCFS, Holm p=0.0012, no estimates required).
- **Canonical EASY re-prices the reservation guarantee**: +11.8% mean wait vs
  FCFS/first-fit (v3.2's non-canonical variant overstated it as ~45%), −26% vs
  strict FIFO, and near-insensitive to estimate quality (19.0–19.8 ts across
  perfect/f-model/modal). Hybrid predicted-wait backfill still ties plain EASY.

### Metric & statistics fixes (from a 13-agent adversarial review; 7 confirmed)
- **Unified wait = turnaround − runtime** for all schedulers: bit-identical for
  the non-preemptive ones, but closes SRPT's time-to-first-dispatch artifact
  that hid ~29% of its queueing delay (SRPT mean wait 10.66 → honest 14.02,
  which flips its ranking vs SJF).
- New metrics for every scheduler: p95 wait, mean turnaround, mean bounded
  slowdown, preemption count.
- Holm correction no longer converts NaN p-values into finite ones; Wilcoxon
  p-values now get their own Holm column instead of being reported raw next to
  adjusted t-tests; HRRN docstring no longer overclaims "cannot starve".
- Pre-existing schedulers (FIFO/FCFS, SJF, Priority, Proactive, NN) verified
  **bit-identical** to v3.2 outputs; estimate RNG uses a dedicated stream
  (seed 20000+run) so workload pairing is untouched.

---

## v3.2 — July 2026 · Research extensions: real traces, backfill, fairness budget, uncertainty

The four highest-leverage items from the v3.1 future-scope roadmap, implemented,
run at full scale, and documented. Every result below is seeded and regenerable.

### ① Real-trace validation (Phase 25 closed)
- Downloaded and committed two genuine Parallel Workloads Archive traces
  (`02_data/LANL-CM5-1994-4.1-cln.swf.gz`, 1,024 procs, 122k jobs;
  `02_data/SDSC-SP2-1998-4.2-cln.swf.gz`, 128 nodes, 54k jobs).
- New pipeline: `02_data/build_real_trace_datasets.py` (chronological replay of the
  recorded schedule reconstructs 8 honest cluster-state features per arrival) →
  `02_data/real_trace_validation.py` (transfer + retrain + baselines).
- **Zero-shot transfer of the synthetic model ≈ 0 on both traces** (best-case
  protocol with rescaled features and affine calibration).
- **Retrained on the trace: SDSC SP2 R²(log) = 0.494** (vs −0.694 median baseline)
  — half the log-wait variance of a real batch supercomputer explained; LANL CM-5
  only 0.101 (interactive machine, median wait ~4 s) — machine-dependent signal,
  reported honestly.

### ② EASY-backfill baseline (+ hybrid)
- `04_scheduler/backfill_scheduler.py` (EASY with head reservation, perfect runtime
  estimates = strongest baseline) + integration into the now-7-scheduler benchmark.
- Finding: this simulator's "FIFO" already backfills without reservations, so
  EASY's guarantee **costs ~45% mean wait but is the fairest policy in the study**
  (Gini 0.365, max wait ~55 ts) — the reservation price, measured.
- The predicted-wait backfill hybrid ties plain EASY — honest null result.

### ③ Bounded-fairness wait budget
- `04_scheduler/fairness_budget_sweep.py`: hard escalation budget B swept over
  9 values × 20 paired runs → a smooth mean-wait ↔ tail-latency Pareto frontier
  (`05_results/fairness/budget_sweep.csv` + `budget_pareto.png`).
- **B=60 keeps +7.1% of the +13.2% unbounded gain while capping max wait at 81 ts
  (vs 136) and Gini at 0.69.** B≤30 is slightly worse than FIFO (churn) — reported.

### ④ Uncertainty-aware scheduling (honest negative)
- `03_models/train_quantile_model.py` (q10/q50/q90 XGBoost; holdout q50 MAE 4.61;
  **interval coverage 68% vs 80% nominal — under-dispersed, reported as-is**) +
  `04_scheduler/uncertainty_scheduler_benchmark.py` (UCB + spread-guarded FIFO
  fallback across 5 scenarios).
- Parity with the point model everywhere; the guard fails to fire (0.2% of ticks)
  in the one regime where reordering hurts. Interval width is **not** a reliable
  OOD alarm here; the rolling-MAE drift trigger remains the deployment mechanism.

### Fixes & hygiene
- Eliminated train/eval seed leakage in three benchmarks (evaluation now uses
  out-of-training seed ranges 1000+/5000+/7000+); PROACTIVE's edge over FIFO on
  fresh seeds: 6.5% (vs 7.7% near-training).
- Fixed a unit-scale bug in the real-trace transfer mapping (raw processor counts
  were fed to a model trained on a 32-GPU world).
- All three HTML docs, README, RESULTS, DEPLOYMENT, roadmap/status docs, and the
  manuscript updated with the v3.2 results (including the negatives).

---

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
- **`docs/index.html`** (since renamed to **`docs/explanation.html`**) — complete
  interactive project documentation: every concept
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

Tip: enable **GitHub Pages → deploy from branch → /docs** to serve the HTML
documentation as the project site. (The entry page was `docs/index.html` at the
time of this release; it is now `docs/explanation.html`.)

---

## v3.0 — May 2026
Phases 01–21 research pipeline (simulation, features, model, schedulers,
benchmarks, explainability, ROI) plus phases 22–30 scaffolding.

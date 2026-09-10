# Phases 22–30 Completion Summary

> **Provenance note (2026-07):** this summary was rewritten after a full project audit.
> Earlier versions of this document reported results from scripts that hardcoded
> synthetic constants (Phases 23, 24, 26, 27) and quoted numbers that contradicted
> the measured data. All phase scripts now compute their outputs from real data or
> real simulations, the pipeline is seeded end-to-end, and every number below comes
> from a regenerated artifact on disk.
>
> **Second pass (2026-09, v3.6):** a hostile-review hardening pass found that several numbers
> in *this* file had never been re-read from disk and still dated to v3.1. Phase 24's scheduler
> table, Phase 26's scaling verdict, Phase 27's fairness figures and the Phase 25 trace metadata
> are corrected below, and the Phase 26 complexity-class verdict is **withdrawn outright**.
> Corrections are stated plainly rather than quietly deleted.

## Status Overview

| Phase | Title | Key File(s) | Status |
|-------|-------|-------------|--------|
| 22 | Statistical Rigor | `phase_22_stats/stats_bootstrap.py` | ✅ Complete (real 40-run data) |
| 23 | OOD Sensitivity | `phase_23_sensitivity/sensitivity_ood_analysis.py` | ✅ Complete (real model, real shifted sims) |
| 24 | Extended Schedulers | `phase_24_extended_schedulers/scheduler_comparison.py` | ✅ Complete (real benchmark CSVs; SLURM/K8s/Yarn baselines **dropped** — no real implementations existed) |
| 25 | Real Traces | `phase_25_real_traces/trace_preprocessing.py`, `02_data/real_trace_validation.py` | ✅ Complete (v3.2: two real PWA traces evaluated — LANL CM-5, SDSC SP2) |
| 26 | Scaling Validation | `phase_26_scaling/scaling_benchmark.py`, `scaling_measurements.txt` | ⚠️ Measurements stand; the **complexity-class verdict is withdrawn** (v3.6) |
| 27 | Fairness & SLA | `phase_27_fairness/fairness_sla_analysis.py` | ✅ Complete (real per-run/per-job data; schedulers without real data dropped) |
| 28 | Manuscript | `phase_28_manuscript/manuscript.tex` | ✅ Draft compiles; numbers synced to regenerated results |
| 29 | Reproducibility | `../run_all_experiments.sh` (20 steps; `run_all_experiments_v2.sh` is now a forwarding shim) | ✅ Complete (seeded, UTF-8-safe, path-independent) |
| 30 | Deployment Guide | `../DEPLOYMENT.md` | ✅ Complete |

---

## Phase-by-Phase Results (regenerated, honest)

### Phase 22: Statistical Rigor
- Wait improvement: **7.90%**, bootstrap 95% CI **[4.91%, 10.67%]**, paired t-test p = 2.0e-06 (BH-corrected), Wilcoxon p = 2.7e-06, n = 40 seeded paired runs.
- **Label discipline (v3.6):** the interval above is a genuine 10,000-resample *percentile
  bootstrap* of the mean (`phase_22_stats/stats_bootstrap.py` → `stats_summary.csv`), and it is
  the only real bootstrap in the repository. The wider **[4.88%, 10.91%]** interval quoted
  elsewhere in the project comes from `04_scheduler/benchmark_statistical.py`, which computes a
  **Student-t** interval — it must not be called a bootstrap. Same 40 runs, two different
  estimators; both are reported, neither is swapped for the other.
- GPU utilisation and completed-jobs comparisons are zero-variance and are reported as **"n/a (zero variance)"** — not as significance.

### Phase 23: OOD Sensitivity (72 scenarios)
- Scores the **actual trained model** (`wait_model_v2.pkl`) on freshly simulated shifted workloads.
- **Prediction quality collapses OOD**: mean R² ≈ **−0.31** (range −2.32 … +0.85).
- Scheduling improvement is erratic OOD: −10% … +53%, mean failure rate ≈ 32%.
- Conclusion: retrain per deployment regime; keep a FIFO fallback.

### Phase 24: Extended Scheduler Comparison
- Compares only schedulers with **real implementations**, from the regenerated multi-scheduler benchmark. The comparison has since grown to the full 14-scheduler landscape (`phase_24_extended_schedulers/baseline_comparison.csv`); the five originally compared here were FIFO, SJF, static priority, NN and Proactive.
- SLURM-backfill / Kubernetes-QoS / Yarn-FIFO rows were **removed**: no real implementations existed and their previous numbers were fabricated constants. A **real EASY-backfill baseline was implemented in v3.2** (`04_scheduler/backfill_scheduler.py`; see the v3.2 Extensions section below).
- Result (mean wait, `05_results/schedulers/multi_scheduler_benchmark.csv`): SJF 12.34 ts
  (needs known runtimes) < PROACTIVE 15.95 < NN 16.07 < STATIC_PRIORITY 16.53 < FIFO 17.22.
  Of these five, FIFO leads on fairness (per-job Gini 0.516) and tail latency (max 54.65 ts) —
  though across the full 14-scheduler landscape the backfill family is fairer still
  (see the v3.2 Extensions section).
  *(Corrected in v3.6: this line previously read SJF 12.00 / NN 15.69 / Proactive 15.71 /
  Priority 16.34 / FIFO 16.65, Gini 0.518, max 53.9. `git log -S` traces every one of those to
  commit 69c733d (v3.1); none matched any current artifact. They are replaced, not deleted.)*
- **NN is not a distinct policy.** NN and SMALLEST (the ML-free "smallest requested size first"
  control) both come in at exactly 16.074545 ts and are bit-identical on all 20 runs — the
  neural baseline reproduces size-ordering.
- **"Priority" is relabelled STATIC_PRIORITY** (mean wait unchanged at 16.53 ts). The old
  "Priority + aging" label was false: the key expands to
  `(priority_score + 0.03*arrival_time) - 0.03*current_time`, and the `current_time` term is a
  common additive shift at any instant, so it cancels pairwise and the induced order is
  time-invariant. **A waiting job can never overtake, so this policy's "aging" prevents nothing
  — any claim that it mitigates starvation is withdrawn.** HRRN is the repository's genuinely
  aging baseline.

### Phase 25: Trace Integration — **COMPLETE (v3.2)**
- Two real Parallel Workloads Archive traces downloaded, committed
  (`02_data/LANL-CM5-1994-4.1-cln.swf.gz`, `02_data/SDSC-SP2-1998-4.2-cln.swf.gz`)
  and evaluated via `02_data/build_real_trace_datasets.py` + `02_data/real_trace_validation.py`.
- Method: chronological replay of each recorded schedule reconstructs 8 honestly-derivable
  cluster-state features at every arrival. Transfer of the synthetic model tested via a
  rescaled 12-feature mapping with best-case affine calibration; retraining uses a
  chronological 80/20 split with target log1p(wait minutes).
- **SDSC SP2** — *trace metadata, from the SWF header of the committed
  `02_data/SDSC-SP2-1998-4.2-cln.swf.gz`, not a measured result of this project:*
  `MaxNodes: 128`, `MaxProcs: 128`, log spanning Apr 1998 – Apr 2000 (the earlier "1998" named
  only the start year). 42,117 evaluated arrivals; heavy-tailed batch queue, p90 wait 54,170 s
  ≈ 15 h (measured over `02_data/real_trace_dataset_sdsc.csv`).
  Transfer R² = 0.072 (MAE 452 min); **retrained R²(log) = 0.494**,
  MAE 622 min vs median baseline R² −0.694 / 684 min and rolling-mean R² 0.016 / 1009 min.
  Cluster-state features explain about half the log-wait variance on a real batch
  supercomputer once retrained.
- **LANL CM-5** — *trace metadata, from the SWF header of the committed
  `02_data/LANL-CM5-1994-4.1-cln.swf.gz`:* `MaxProcs: 1024`, log spanning Oct 1994 – Sep 1996.
  122,055 usable jobs; **median wait 2 s** (p90 77 s) over `02_data/real_trace_dataset_lanl.csv`
  — largely interactive. *(Corrected in v3.6: this line said "median wait ~4 s", which matched
  no artifact.)* Transfer R² ≈ −0.001 (MAE 41 min); retrained R²(log) = 0.101,
  MAE 33.7 min vs median baseline R² −0.106 / 33.5 min. Weak-signal machine —
  honest partial negative.
- Conclusion: **zero-shot sim-to-real transfer ≈ 0 on both traces** (now quantified on
  real data); retraining recovers strong signal where queueing dominates; signal strength
  is machine-dependent. "Retrain per deployment" is now evidence-backed, not a caveat.

### Phase 26: Scaling Validation
**What survives.** Real discrete-time simulation at 32–256 GPUs under saturation (10,000
submitted jobs, `phase_26_scaling/scaling_benchmark.csv`):

- Utilisation stayed **above 99.69%** at every scale — minimum 99.69416% at 256 GPUs.
  *(Corrected in v3.6: the old bound ">99.7%" is false as written, by 0.006 of a point.)*
- **Scheduling overhead stayed under 5% of throughput** at every measured scale (peak 1.55%).

**⚠️ Withdrawn (v3.6): the complexity-class verdict and every projection.** Earlier versions of
this file and of `phase_26_scaling/scaling_law_fit.txt` reported **"O(1) / latency is CONSTANT"**
and **"excellent scalability"**, and extrapolated to 512, 1024 and 4096 GPUs. All of that is
retracted, for three reasons worth stating once:

1. The fitted exponent was **−0.234**, taken from **four non-monotone wall-clock points**.
2. The classifier's test was **one-sided**, so a *negative* exponent was labelled "constant" by
   fall-through rather than by evidence.
3. The `inference_latency_ms` column is **wall-clock**: on a repeat run of this repository the
   timing columns moved by up to **84%** while every non-timing column was bit-identical.

Four noisy, non-monotone points cannot identify a scaling law. The file is now
`phase_26_scaling/scaling_measurements.txt`; it infers no complexity class and makes no
projection past the largest measured cluster.

- Inference latency is therefore reported as **machine-dependent, not as a property of the
  scheduler**. The range recorded in the current artifact is **10.24–15.49 ms** across the four
  points. *(The old "10–48 ms" was quoted from a superseded run whose own low end was 9.60 ms,
  so it was wrong at the low end even against the run it came from.)*
- Complementary moderate-load study (Phase 17, `05_results/scaling/scaling_analysis.csv`):
  proactive advantage is **14.5%** on a small contended cluster (14.5219% at 4 nodes / 8 GPUs),
  falls to 5.1% at 8 nodes / 32 GPUs, and is exactly 0% at 16 and 32 nodes once capacity removes
  queueing. *(Corrected in v3.6 from 14.4%.)*

### Phase 27: Fairness & SLA
- Computed from the real 40-run benchmark + per-job fairness data. Schedulers without real distributions are listed with `data_source = per_job_aggregates_only` or dropped (noted in `dropped_schedulers.txt`).
- **Proactive worsens per-job Gini (0.53 → 0.79)** and max wait (58 → ~123 ts); the
  anti-starvation variant recovers to Gini 0.69 / max ~88 ts.
  Exact values (`phase_27_fairness/fairness_metrics.csv`): per-job Gini 0.52619 / 0.79363 /
  0.69210; max job wait 57.85 / 122.65 / 88.15 ts.
  *(Corrected in v3.6: this line read 0.80 / 125 / 87. Those came from a **stale fork** — the
  phase-27 copy of `fairness_metrics.csv` had been committed before the benchmark reached 14
  schedulers, and its producer iterates every row of that benchmark. Regenerated, the phase-27
  copy and `05_results/fairness/fairness_metrics.csv` now **agree**; the two files were never
  measuring different things, one was simply old.)*
- **Starvation now has exactly one definition** across the repository: a job is starved when its
  wait exceeds **3× its own runtime** (`04_scheduler/fairness_analysis.py`, and phase 27's
  SLA-2). `04_scheduler/fairness_budget_sweep.py` previously used "3× the run's *mean* wait";
  that definition has been removed, and any prose still describing it is describing a rule the
  repository no longer computes. On the surviving definition: 22.25 (FIFO) / 18.35 (Proactive) /
  30.20 (anti-starvation) starved jobs per run.
- SLA compliance: FIFO 0.933, Proactive 0.945, anti-starvation 0.863. Run-level Jain: 0.918 vs 0.922.

#### ⚠️ Withdrawn claim: "proactive ≥ FIFO on fairness"

> **This is the only place in the repository that records this withdrawal, so it is kept
> deliberately and kept prominent. Do not delete it.**
>
> The claim is false and is retracted. The honest claim is a **quantified trade-off with a
> partial mitigation**: proactive buys its ~7.9% mean-wait improvement by paying in tail wait
> (57.85 → 122.65 ts) and per-job inequality (Gini 0.52619 → 0.79363). The anti-starvation
> variant recovers part of the fairness (Gini 0.69210, max 88.15 ts) but raises the starvation
> count (22.25 → 30.20 jobs/run) and lowers SLA compliance (0.9326 → 0.8627).
> **There is no configuration in this study where proactive dominates FIFO on fairness.**

### Phase 28: Manuscript
- `manuscript.tex` compiles (inputenc fixed, all citations resolve, booktabs rules correct) and now reports the regenerated numbers above, including the fairness trade-off, OOD collapse, and (v3.2) the real-trace validation, backfill baselines, wait-budget frontier, and the uncertainty-guard negative result.

### Phase 29: Reproducibility
- **One command now regenerates every result**, and as of v3.6 that sentence is literally true:
  `run_all_experiments.sh` at the repository root runs all **20 steps**, having absorbed the
  quantile model, the fairness budget sweep, the real-trace dataset build and validation, the
  uncertainty benchmark, the phase-01 simulation, and phases 22–27.
  `phases_22_30/run_all_experiments_v2.sh` is now a **forwarding shim** kept for backwards
  compatibility, not a second entry point. Seeded, cwd-independent, UTF-8-safe on Windows.
- Verify a regenerated tree against the committed artifacts with
  `python tools/verify_artifacts.py` (`--quick` / `--smoke`).
- `requirements.txt` pins the exact versions verified working on Python 3.14.

### Phase 30: Deployment Guide
- `DEPLOYMENT.md` (project root) covers advisory-mode rollout, monitoring thresholds, drift-triggered retraining, FIFO fallback, and an honest ROI framing.

---

## v3.2 Extensions (July 2026)

Four research extensions, all regenerated from artifacts on disk.

### 1. Real-trace validation (closes Phase 25)
- Two real PWA traces (LANL CM-5, SDSC SP2) replayed and evaluated
  (`05_results/traces/real_trace_validation.csv`).
- Zero-shot sim-to-real transfer ≈ 0 on both (LANL R² ≈ −0.001; SDSC R² = 0.072).
- Retrained on-trace: **R²(log) = 0.494 on SDSC SP2** (vs median baseline −0.694) —
  strong signal on a heavy-tailed batch machine; **0.101 on LANL CM-5** — weak-signal,
  largely interactive machine (honest partial negative). Details in the Phase 25 section.

### 2. Seven-scheduler benchmark with real backfill baselines
- `04_scheduler/backfill_scheduler.py` adds EASY backfilling (hard reservation for the
  head job, perfect runtime estimates = strongest-baseline setting) and a
  PROACTIVE_BF hybrid (predicted-wait backfill order).
- 20 paired runs on out-of-training seeds 1000+i
  (`05_results/schedulers/multi_scheduler_benchmark.csv`), mean wait / max wait / Gini
  (current regeneration; the benchmark now spans 14 schedulers — subset shown):
  SJF 12.34/145.7/0.785 · NN 16.07/136.4/0.797 · PROACTIVE 15.95/128.8/0.793 ·
  STATIC_PRIORITY 16.53/131.7/0.743 · FIFO 17.22/54.7/0.516 · PROACTIVE_BF 19.20/53.8/0.479 ·
  BACKFILL(EASY) 19.25/51.6/0.451. Utilisation 0.637 and throughput identical across all.
- **Semantics caveat**: the benchmark's "FIFO" dispatches every fitting job each tick,
  i.e. it is already unrestricted no-reservation backfilling. EASY's head-job reservation
  (canonical two-condition rule, v3.3) costs +11.8% mean wait but delivers among the
  **best fairness in the whole study** (Gini 0.451, max wait ~52 ts). The hybrid ties
  EASY — an honest null result.
- PROACTIVE beats FIFO by 7.4% on these fresh seeds (vs 7.9% near-training — good
  generalization).

### 3. Bounded-fairness wait-budget sweep
- `04_scheduler/fairness_budget_sweep.py`, 20 paired runs, seeds 5000+i, FIFO reference
  mean 22.4 ts (`05_results/fairness/budget_sweep.csv`, `budget_pareto.png`).
- Budget B (ts) → mean-wait gain vs FIFO / max wait / Gini:
  B=0: −0.9%/62/0.49 · B=10: −0.6%/66/0.50 · B=20: −1.4%/64/0.52 · B=30: −0.1%/64/0.56 ·
  B=40: +1.9%/68/0.60 · B=60: +7.4%/81/0.68 · B=80: +10.2%/96/0.74 ·
  B=120: +12.7%/125/0.78 · B=∞ (pure proactive): +13.1%/138/0.79.
- A smooth, tunable Pareto frontier: **B=60 keeps over half the mean-wait gain while
  capping max wait at 81 ts** (vs 138 unbounded). Tiny budgets (0–30) are slightly
  *worse* than FIFO (escalation churn without freedom) — reported honestly.

### 4. Uncertainty-aware scheduling (quantile XGBoost)
- `03_models/train_quantile_model.py` (q10/q50/q90 → `wait_model_quantile.pkl`),
  `04_scheduler/uncertainty_scheduler_benchmark.py`; 10 paired runs/scenario, seeds 7000+i
  (`05_results/uncertainty/uncertainty_ood_benchmark.csv`, `uncertainty_summary.png`).
- Quantile holdout: q50 MAE 4.61 (point model 4.69); [q10,q90] empirical coverage **68%
  vs 80% nominal — under-dispersed**, reported honestly. Spread trigger τ = p95 of
  in-distribution relative spread.
- In-distribution: PROACTIVE +6.6%, UCB(q90) +6.8%, GUARDED +6.5% — parity, no cost.
- Overload (2× arrivals, 4 nodes): all smart policies +38.6 to +39.0% (caveat:
  among-started-jobs under horizon censoring; smart policies also *started* more jobs,
  131 vs 114).
- Light-load small cluster (0.5× arrivals, 4 nodes): **all smart policies negative**
  (−2.0 to −4.8% vs FIFO) and the spread guard fires on only 0.5% of ticks — the
  under-dispersed intervals fail to detect this regime. **Honest negative result:
  interval width is not a reliable OOD alarm here**; the drift-triggered FIFO fallback
  (Phase 19 rolling-MAE) remains the deployment mechanism.

---

## Quality Gates

- [x] Phase 22: headline claims carry CIs and corrected p-values
- [x] Phase 23: failure modes measured with the real model and discussed in the manuscript
- [x] Phase 24: comparison uses only real scheduler implementations
- [x] Phase 25: cross-dataset R² on a **real** trace — zero-shot transfer R² ≈ 0 on both PWA traces; retrained R²(log) 0.494 (SDSC SP2) / 0.101 (LANL CM-5)
- [x] Phase 26: overhead < 5% demonstrated at 256 GPUs (peak 1.55%); **no scaling law claimed,
      no projection beyond 256 GPUs** — the O(1) verdict is withdrawn
- [x] Phase 27: fairness trade-off quantified (not claimed away); starvation analysed on a
      single repository-wide definition (wait > 3× the job's own runtime)
- [x] Phase 28: manuscript numbers match regenerated artifacts
- [x] Phase 29: one-command reproducibility verified on this machine — a single 20-step
      `run_all_experiments.sh` regenerates every result, checkable with `tools/verify_artifacts.py`
- [x] Phase 30: deployment guide includes fallback procedures

---

## Next Steps

1. **Compile & iterate the manuscript**:
   ```bash
   cd phases_22_30/phase_28_manuscript && pdflatex manuscript.tex && pdflatex manuscript.tex
   ```
2. **Tag a release** (v3.6) — real-trace validation and the backfill baseline landed in v3.2,
   and the v3.6 hardening pass has since retracted the scaling verdict, resolved the fairness
   fork and corrected the labelling above. `CITATION.cff` is already stamped `version: "3.6"`.

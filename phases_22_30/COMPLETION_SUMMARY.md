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
>
> **Third pass — Phase C (2026-09):** a methodology-hardening pass aimed at the *protocols*
> behind the numbers rather than the numbers themselves. Six studies; five of them **weaken**
> something this file used to claim, and one study (the ROI analysis) is **deleted outright** as a
> category error. The headline model accuracy falls, the OOD failure taxonomy turns out to have
> been a constant column, and the ablation acquires intervals that put nine of twelve features'
> contributions inside the noise. **None of this touches the central negative result**: ranking
> degeneracy is a statement about the *functional form* of the score — at a fixed dispatch instant
> every queued job sees the same cluster state, so the score is a function of requested size alone
> — and it holds at any accuracy. Full detail in the **Phase C Hardening Pass** section below;
> the per-phase entries above it are corrected in place.

## Status Overview

| Phase | Title | Key File(s) | Status |
|-------|-------|-------------|--------|
| 22 | Statistical Rigor | `phase_22_stats/stats_bootstrap.py` | ✅ Complete (real 40-run data) |
| 23 | OOD Sensitivity | `phase_23_sensitivity/sensitivity_ood_analysis.py` | ✅ Complete (real model, real shifted sims); **failure taxonomy re-derived in Phase C** — the old one was a constant |
| 24 | Extended Schedulers | `phase_24_extended_schedulers/scheduler_comparison.py` | ✅ Complete (real benchmark CSVs; SLURM/K8s/Yarn baselines **dropped** — no real implementations existed) |
| 25 | Real Traces | `phase_25_real_traces/trace_preprocessing.py`, `02_data/real_trace_validation.py` | ✅ Complete (v3.2: two real PWA traces evaluated — LANL CM-5, SDSC SP2) |
| 26 | Scaling Validation | `phase_26_scaling/scaling_benchmark.py`, `scaling_measurements.txt` | ⚠️ Measurements stand; the **complexity-class verdict is withdrawn** (v3.6) |
| 27 | Fairness & SLA | `phase_27_fairness/fairness_sla_analysis.py` | ✅ Complete (real per-run/per-job data; schedulers without real data dropped) |
| 28 | Manuscript | `phase_28_manuscript/manuscript.tex` | ✅ Draft compiles; numbers synced to regenerated results |
| 29 | Reproducibility | `../run_all_experiments.sh` (**22 steps** as of Phase C; `run_all_experiments_v2.sh` is now a forwarding shim) | ✅ Complete (seeded, UTF-8-safe, path-independent) |
| 30 | Deployment Guide | `../DEPLOYMENT.md` | ✅ Complete; the **ROI framing is withdrawn** (Phase C7) |

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
- Scheduling improvement is erratic OOD: −10% … +53%, mean failure rate 31.5%.
- **⚠️ Corrected in Phase C: the failure taxonomy was a constant, not a measurement.** Every one
  of the 72 scenarios in `phase_23_sensitivity/ood_failure_modes.csv` used to be labelled
  `DISTRIBUTION_MISMATCH` — zero entropy across the whole grid. The cause was a gate ordering
  bug, not a property of the data: the classifier tested `mape >= 35.0` ahead of most branches,
  and the **minimum MAPE over the 72 scenarios is 54.01**, so that branch fired 72 times out of
  72 and seven of the eight categories were unreachable dead code. `risk_level` was equally
  degenerate at 65 MEDIUM / 7 HIGH / 0 LOW. Any earlier sentence in this repository describing
  "the OOD failure taxonomy" was describing a column that said the same thing 72 times.
- **The re-derived taxonomy (Phase C5)** replaces the threshold cascade with a continuous
  severity score standardised across the grid, four data-derived quantile bands of 18 scenarios
  each (`Q1_LEAST_SEVERE` / `Q2_BELOW_MEDIAN` / `Q3_ABOVE_MEDIAN` / `Q4_MOST_SEVERE`), and a
  dominant-axis label. `failure_mode` now spans five values — COMPLETION_DOMINATED 22,
  POLICY_DOMINATED 21, FIT_DOMINATED 14, NO_DOMINANT_AXIS 8, CALIBRATION_DOMINATED 7 — and
  `risk_level` three: MEDIUM 37 / HIGH 24 / LOW 11. The file grew from 9 to **18 columns**.
- **Read `LOW_RISK` correctly.** Severity is standardised *within this grid of shifted regimes*,
  so `Q1_LEAST_SEVERE` / LOW means "least severe among 72 shifted regimes", never "safe". None of
  these regimes is good: the mean R² across all 72 is **−0.310**, i.e. negative.
- The simulation, the scenarios and the seeds are **unchanged**. Only the classification of the
  results changed; no result was re-run to obtain a nicer label.
- Conclusion (unchanged, and now supported by a taxonomy that discriminates): retrain per
  deployment regime; keep a FIFO fallback.

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
- SLA compliance: FIFO 0.933, Proactive 0.945, anti-starvation 0.863. Run-level Jain: 0.918 (FIFO) vs 0.924 (proactive).

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
  `run_all_experiments.sh` at the repository root runs all **22 steps** (`TOTAL=22`), having
  absorbed the quantile model, the fairness budget sweep, the real-trace dataset build and
  validation, the uncertainty benchmark, the phase-01 simulation, and phases 22–27.
  *(Corrected in Phase C from 20. Phase C added `03_models/evaluate_splits.py`,
  `04_scheduler/tost_power.py` and `04_scheduler/censoring_analysis.py` to the pipeline and
  removed the ROI step; the net is +2.)*
  `phases_22_30/run_all_experiments_v2.sh` is now a **forwarding shim** kept for backwards
  compatibility, not a second entry point. Seeded, cwd-independent, UTF-8-safe on Windows.
- Verify a regenerated tree against the committed artifacts with
  `python tools/verify_artifacts.py` (`--quick` / `--smoke`).
- `requirements.txt` pins the exact versions verified working on Python 3.14.

### Phase 30: Deployment Guide
- `DEPLOYMENT.md` (project root) covers advisory-mode rollout, monitoring thresholds,
  drift-triggered retraining and FIFO fallback.
- **⚠️ Withdrawn in Phase C: the "honest ROI framing" this line used to advertise.** The ROI
  study is deleted, not caveated — see Phase C7. The dollar figure was a conversion of a
  wait-time percentage into GPU-hours saved, and the repository's own 40-run benchmark records
  **`baseline_util == proactive_util` in 40 of 40 runs** (mean 0.641177, identical to six decimal
  places) with the same 110 jobs completing under both policies. Reordering a queue changes
  *when* jobs start, not how many GPU-hours they consume; the quantity being monetised was
  measured at zero. A deployment guide should quote the wait-time result and the operational
  procedures, and no money.

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
  BACKFILL(EASY) 19.25/51.6/0.451. Throughput is identical across all 14 (0.3667). Utilisation is 0.6375 for 13 of them; preemptive SRPT is the exception at 0.6545.
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
  *(Phase C context: the 4.69 point-model figure is the **random row split**, which Phase C1
  shows to be optimistic. Both MAEs here are quoted on that same split, so the ~0.08 gap between
  them is a like-for-like comparison and stands; the absolute level does not. See Phase C1.)*
- In-distribution: PROACTIVE +6.6%, UCB(q90) +6.8%, GUARDED +6.5% — parity, no cost.
- Overload (2× arrivals, 4 nodes): all smart policies +38.6 to +39.0% among started jobs;
  smart policies also *started* more jobs (131.3 PROACTIVE vs 113.7 FIFO, means over 10 runs).
  **Phase C4 measured the direction of that selection effect and it runs the other way from the
  usual worry**: on the common set the FIFO→PROACTIVE improvement is 42.27%, i.e. **3.3 pp
  larger** than the 39.01% published here, so horizon censoring *understated* the learned
  policies rather than flattering them. Detail and its limits in Phase C4 below.
- Light-load small cluster (0.5× arrivals, 4 nodes): **all smart policies negative**
  (−2.0 to −4.8% vs FIFO) and the spread guard fires on only 0.5% of ticks — the
  under-dispersed intervals fail to detect this regime. **Honest negative result:
  interval width is not a reliable OOD alarm here**; the drift-triggered FIFO fallback
  (Phase 19 rolling-MAE) remains the deployment mechanism.

---

## Phase C Hardening Pass (September 2026)

Six studies aimed at the *protocols* behind the numbers. Five weaken a claim, one deletes a study.
Every figure below was read out of the named artifact on disk. The governing rule for this pass
was **weaken a claim freely, strengthen one never**.

**Read this first.** Phase C corrects how well the model predicts and how the ablation, the
explanations and the equivalence tests were computed. It does **not** touch the project's central
contribution. Ranking degeneracy is a statement about the **functional form** of the score — at a
fixed dispatch instant every queued job sees the same cluster state, so the score reduces to a
function of requested size alone — and that is true whatever the model's R² turns out to be. A
more accurate model would be degenerate in exactly the same way. Do not read C1 as a softening of
the degeneracy result.

### C1. The model evaluation was optimistic — new headline accuracy

`05_results/models/evaluation_splits.csv` (`03_models/evaluate_splits.py`).

The published **R² 0.837 / MAE 4.69** came from `train_test_split(test_size=0.2, random_state=42)`
over 2,200 rows that are **20 simulation runs of 110 jobs**. Rows from one run share a cluster
trajectory, so a random *row* split puts near-duplicates on both sides of the fence. The split was
measuring interpolation within a trajectory, not generalisation to a new one.

| split | folds | R² | MAE |
|-------|-------|-----|-----|
| random (old protocol) | 1 | 0.836840 | 4.6935 |
| **run-wise (NEW HEADLINE)** | 5 | **0.810910 ± 0.020672** | **4.8970 ± 0.4492** |
| leave-one-run-out | 20 | 0.793486 ± 0.054176 | 4.8207 ± 1.0156 |
| chronological | 1 | 0.725147 | 7.2403 |

- **The headline model accuracy is now R² 0.81 (0.810910 ± 0.020672), MAE 4.90**, from
  `GroupKFold(n_splits=5)` on `run_id`. This is **lower than the 0.837 previously reported**, and
  it is lower because it is the only split that answers the question that matters: does the model
  work on a cluster trajectory it has not seen? The old number is not retracted as *wrong
  arithmetic* — it reproduces exactly, at 0.836840 — it is retracted as *the wrong question*.
- **The gap is not a data-volume artefact.** Five folds were chosen precisely so the training set
  stays at **1,760 rows**, the same size as the random split's, with 440 held out. Same rows, same
  count, different partition; the whole difference is the grouping.
- **Deployment claims must quote the chronological MAE of 7.24**, not 4.69. Training on each run's
  early arrivals and testing on its late ones — the order a deployed model actually sees — costs
  **54% more error than this repository used to advertise**.
- **Scale, so the reader can judge "0.81".** Against a constant predictor fitted on the same
  folds, run-wise mean-constant scores R² −0.019958 and median-constant −0.088624 (MAE ≈ 13.0–13.5).
  The model beats the constant by about **64% on MAE**. It is a real but ordinary regressor, not a
  dressed-up mean — and not the oracle the 0.837 implied.
- **Why five folds and not twenty.** Leave-one-run-out per-fold R² spans **0.6899 to 0.8744**. A
  single grouped hold-out would have been unquotable: which run you held out would have decided
  the headline. The 20-fold mean (0.793486 ± 0.054176) is reported as a sensitivity check on the
  fold count, not as the headline.
- Chronological constant baselines, for completeness: mean-constant R² −0.702853 (MAE 19.2303),
  median-constant −1.175099.

### C2. SHAP is now computed on held-out data — and the conclusion survives

`05_results/shap/shap_provenance.csv`, `shap_explained_rows.csv` (`03_models/explainability_shap.py`).

The explanations used to cover 400 rows sampled from the **full** dataset with the **full** dataset
as background; roughly 320 of those background rows were rows the model had been fitted on, which
moves the base value the attributions are measured against. They are now drawn only from the
**440-row held-out test split**, with the background taken from that split: `rows_from_training = 0`,
`rows_from_test = 400`, `background_rows_from_training = 0`. Crucially, the provenance is
**measured, not declared** — the row set is fingerprinted by a sha256 of the actual explained row
indices (`explained_index_sha256`) and the labels themselves are written out, so a reader can
recompute it against an independently reconstructed split rather than trust the script's own
account of its intentions.

**The ordering is essentially unchanged.** `job_gpu` still dominates at mean |SHAP| **7.3393**,
carrying **35.4%** of all attribution mass and **76.2%** of the mass carried by the four
job-dependent features. The only rank movement anywhere in the table is a **7/8 swap between
`running_jobs` and `variance_free`, which differed by 0.0003** — noise.

That non-result is the point. The degeneracy story previously rested on explanations a reviewer
could have dismissed as partly in-sample; it now rests on held-out ones, and it did not move.
This is the one Phase C study that **confirms** rather than weakens.

### C3. The LANL equivalence row is inconclusive, and cannot be settled on that trace

`05_results/trace_schedulers/tost_power.csv` (`04_scheduler/tost_power.py`), 20,000 Monte-Carlo
replicates, seed **31337** (outside every protected seed family), margin 0.10 × the reference mean
exactly as `simstats.tost_equivalence` computes it.

LANL CM-5, **SMALLEST_FIRST vs PROACTIVE, mean wait**:

| quantity | value |
|----------|-------|
| observed paired difference | 320.02 s |
| equivalence margin (10%) | 222.93 s |
| achieved power | **0.47%** (0.00465; bootstrap variant 5.8%) |
| n for 80% power, *equivalence* | **not achievable at any n** |
| n for 80% power, *difference* | 29 windows; **50** after Holm over the family of 11 |
| disjoint 7-day windows LANL supplies | **28** |

- **This is stronger than "underpowered".** A test with a **0.47% chance of certifying
  equivalence** returning "not equivalent" carries no information: it was always going to return
  that. The LANL row must be reported **INCONCLUSIVE**, and never as "different".
- **"More windows would settle it" is also false**, and the honest sentence has to say so. The
  observed difference (320.02 s) **exceeds the margin** (222.93 s), so TOST power falls toward
  zero as n grows and **no sample size can certify equivalence there at 10%**. Establishing a
  *difference* instead would need 29 windows, or **50 after Holm correction over the family of 11
  tests** — and the trace supplies **28**. On disjoint windows, LANL is not settleable either way.
- **The equivalence claim rests on SDSC, and there it rests solidly.** Same pair, same metric:
  achieved power **1.000**, **3** disjoint windows would have sufficed, **29** available.
- Two limits stated once and kept: power computed from an observed effect is **post-hoc** and is
  an *estimate*, not a design parameter; and the window supply is a property of the trace, not a
  budget this project can increase.

### C4. The censoring bias runs the other way, and is confined to one scenario

`05_results/uncertainty/censoring_analysis.csv` (`04_scheduler/censoring_analysis.py`).

- **Four of five scenarios have nothing to correct.** **24 of the 30** pair-rows have a selection
  gap of **exactly 0.0** — every job starts under every policy, so the common set *is* the full
  set and the published improvement is unbiased. All censoring lives in **`arr2.0_nodes4`**.
- **Where it exists, it ran against the learned policies.** All six non-zero gaps are **negative**,
  from **−0.38 to −3.25 percentage points**, meaning the common-set improvement is *larger* than
  the published one. The published statistic **understated** the learned policies by up to
  **3.3 pp** (FIFO vs PROACTIVE: 39.01% published → **42.27%** on the common set). Nothing
  "disappears on the common set".
- **It is a two-way exchange, not a one-way rescue.** Started-set decomposition for FIFO vs
  PROACTIVE, means over 10 runs: **common 91.9 jobs, FIFO-only 21.8, PROACTIVE-only 39.4**. Each
  policy starts jobs the other does not.
- **The concern is not "refuted" and is not described that way here.** The selection effect is
  real; it is confined to one of five scenarios; and it points in the direction that penalised the
  learned policies. What the common set buys is a clean verdict on the **~92 shared jobs**. The
  **~61 exchanged jobs** are *described* by the decomposition above but are **not adjudicated** by
  it — no estimator in this study rules on them.
- One further caveat carried from the artifact itself: `improvement_pct_lowerbound` divides two
  lower-bound means, so it is **not** a bound on the improvement (bounding each of two means bounds
  neither the sign nor the size of their ratio). It is descriptive only.

### C5. The OOD failure taxonomy was a constant and is now a ranking

`phases_22_30/phase_23_sensitivity/ood_failure_modes.csv` — 72 rows, now **18 columns**.
Full statement in the **Phase 23** section above; in brief: all 72 scenarios were labelled
`DISTRIBUTION_MISMATCH` because a `mape >= 35.0` gate sat ahead of most branches and the minimum
MAPE over the grid is **54.01**; seven of eight categories were unreachable. The replacement is a
standardised severity score, four 18-scenario quantile bands, and a dominant-axis label —
`failure_mode` spans five values (22 / 21 / 14 / 8 / 7) and `risk_level` three (37 / 24 / 11).
`LOW_RISK` means *least severe among 72 shifted regimes*, never *safe*: the mean R² across the
grid is **−0.310**. Simulation, scenarios and seeds unchanged; only the classification changed.

### C6. The ablation now has intervals, and two features are the same variable twice

`05_results/models/feature_collinearity.csv` and `05_results/models/ablation_study_results.csv`
(byte-identical duplicate at `05_results/ablation_study_results.csv` — one artifact under two names).

**Measured collinearity.** `total_free` and `avg_free_per_node` correlate at **1.000000** with
**infinite VIF**: in the synthetic generator one *is* the other divided by a constant node count.
They are the same variable twice. `fragmentation` and `variance_free` correlate at **0.950720**
(VIF **66.8** and **34.1**). A single-feature ablation cannot say anything about either member of
such a pair — dropping one leaves the information intact in the other, so a near-zero drop there
is **arithmetic, not evidence**.

**Intervals.** Each ablation is re-fit over **20 leave-one-run-out folds** on `run_id`, with drops
paired within fold and a Student-t 95% interval taken over the 20 paired differences. The honest
baseline falls with everything else: **R² 0.836840** (single fixed row split) →
**0.793486 [0.767472, 0.819500]** leave-one-run-out.

**Only three of twelve drops are distinguishable from zero:**

| feature | R² drop | 95% CI |
|---------|---------|--------|
| `job_gpu` | 0.2051 | [0.1565, 0.2538] |
| `queue_length` | 0.0217 | [0.0081, 0.0353] |
| `queue_pressure` | 0.0158 | [0.0051, 0.0264] |

The other nine intervals span zero.

- **State the null correctly.** An interval crossing zero means the drop is **not distinguishable
  from zero at this sample size**. It does **not** establish that the feature adds nothing.
  Accepting a null from a failure to reject it is exactly the error this project criticises
  elsewhere — it is why the repository uses TOST for equivalence rather than resting on a large
  p-value. The same standard applies to its own ablation.
- **`queue_pressure` surviving is consistent with the degeneracy result, not in tension with it.**
  v3.5 established that `queue_pressure` is itself a deterministic function of requested size
  given the cluster state. A feature that carries size information *should* contribute, and that
  it does is another way of seeing why the score collapses to a size ordering.

### C7. The ROI study is deleted

Not caveated, not re-estimated — **deleted**: `05_results/roi_analysis.py`, `05_results/roi/`, the
pipeline step and the dashboard panel are all gone.

The study converted a wait-time percentage into GPU-hours saved and priced them at roughly
**$80k/yr, ~90% return**. The repository's own 40-run benchmark
(`05_results/benchmark_statistical_results.csv`) records **`baseline_util == proactive_util` in
40 of 40 runs** — mean **0.641177**, identical to six decimal places — with the same 110 jobs
completing under both policies. The cluster performs the same compute either way. **Reordering a
queue changes *when* jobs start, not how many GPU-hours they consume.** The quantity being
monetised was measured at zero.

This is a **category error, not an uncertain assumption**, and the distinction decides the remedy.
Widening the error bars on a number whose true value is zero still reports a saving; the only
correct action is removal. Every ROI claim, dollar figure and percentage return is withdrawn
repository-wide.

**What survives untouched:** the wait-time reduction itself — **7.9% vs FIFO**, Student-t 95% CI
[4.88, 10.91] over 40 paired runs — remains a real, measured result. What is withdrawn is solely
the claim that it converts into money.

---

## Quality Gates

- [x] Phase 22: headline claims carry CIs and corrected p-values
- [x] Phase 23: failure modes measured with the real model and discussed in the manuscript;
      the taxonomy **discriminates** — 5 failure modes and 3 risk levels over 72 scenarios, where
      it previously emitted one constant label 72 times (Phase C5)
- [x] Phase 24: comparison uses only real scheduler implementations
- [x] Phase 25: cross-dataset R² on a **real** trace — zero-shot transfer R² ≈ 0 on both PWA traces; retrained R²(log) 0.494 (SDSC SP2) / 0.101 (LANL CM-5)
- [x] Phase 26: overhead < 5% demonstrated at 256 GPUs (peak 1.55%); **no scaling law claimed,
      no projection beyond 256 GPUs** — the O(1) verdict is withdrawn
- [x] Phase 27: fairness trade-off quantified (not claimed away); starvation analysed on a
      single repository-wide definition (wait > 3× the job's own runtime)
- [x] Phase 28: manuscript numbers match regenerated artifacts
- [x] Phase 29: one-command reproducibility verified on this machine — a single **22-step**
      `run_all_experiments.sh` regenerates every result, checkable with `tools/verify_artifacts.py`
- [x] Phase 30: deployment guide includes fallback procedures
- [x] **Phase C1**: the headline accuracy comes from a split that holds out whole runs
      (run-wise R² 0.810910 ± 0.020672), the deployment figure from a chronological split
      (MAE 7.2403), and both are compared against same-fold constant baselines
- [x] **Phase C2**: SHAP explanations are computed on held-out rows only
      (`rows_from_training = 0`), and the provenance is fingerprinted rather than self-reported
- [x] **Phase C3**: the LANL equivalence row is reported **INCONCLUSIVE** with its achieved power
      (0.47%) attached, and is stated to be **unsettleable** on that trace's 28 disjoint windows
- [x] **Phase C4**: censoring is quantified per scenario (24 of 30 pair-rows have a zero gap), its
      direction is stated (against the learned policies), and the ~61 exchanged jobs are named as
      described-but-not-adjudicated
- [x] **Phase C6**: every ablation drop carries a 95% interval; the two exactly-collinear features
      are named; nine drops are reported as *not distinguishable from zero*, never as *zero*
- [x] **Phase C7**: no ROI claim, dollar figure or percentage return remains in the study

---

## Next Steps

1. **Compile & iterate the manuscript**:
   ```bash
   cd phases_22_30/phase_28_manuscript && pdflatex manuscript.tex && pdflatex manuscript.tex
   ```
2. **Tag a release** (v3.6) — real-trace validation and the backfill baseline landed in v3.2,
   and the v3.6 hardening pass has since retracted the scaling verdict, resolved the fairness
   fork and corrected the labelling above. `CITATION.cff` is currently stamped `version: "3.6"`.
   Note that the **Phase C** pass recorded above landed on top of v3.6 and is not yet reflected in
   that stamp: it lowers the headline model accuracy, re-derives the OOD taxonomy, adds intervals
   to the ablation and deletes the ROI study, so the tag should be cut after deciding how to
   version it.
3. **Propagate Phase C outward.** Any surviving text anywhere in the repository that quotes
   R² 0.837 or MAE 4.69 as the headline, describes the OOD failure taxonomy, calls the LANL
   equivalence row "different", or mentions ROI, dollars or a percentage return is now stale
   against the artifacts. The corrections are recorded in the Phase C section above.

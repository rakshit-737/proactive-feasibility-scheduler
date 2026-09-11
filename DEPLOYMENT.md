# Deployment & Operations Guide (Phase 30)

A runbook for HPC/cloud operators evaluating the proactive feasibility scheduler.
Everything here is grounded in the measured results of this repository — including
the limitations. Read the **Honest preconditions** section before deploying anything.

---

## 1. Honest preconditions — read first

The measured evidence supports deployment **only** under all of the following:

| Precondition | Why (measured evidence) |
|---|---|
| The cluster is **contended** (jobs actually queue) | The wait-time advantage is 14.5% on small contended clusters and **0%** when capacity removes queueing (`05_results/scaling/scaling_analysis.csv`) |
| The model is **retrained on your own workload** | Validated on two **real** Parallel Workloads Archive traces (v3.2): zero-shot transfer of the synthetic model is R² ≈ 0 on both (LANL CM-5 −0.001, SDSC SP2 0.072), but retraining on the trace itself recovers R²(log) = 0.494 on SDSC SP2 (vs −0.694 for a median baseline); OOD mean R² < 0 across 72 shifted scenarios (`05_results/traces/real_trace_validation.csv`, `phases_22_30/phase_23_sensitivity/`; both are pipeline steps — `bash run_all_experiments.sh` regenerates them) |
| **Tail latency and fairness are monitored** | Mean-wait gain costs tail latency (max wait ~58 → ~123 ts) and per-job Gini (0.53 → 0.79) (`05_results/fairness/fairness_metrics.csv`) |
| A **FIFO fallback** is wired in | OOD failure rate averages ≈ 32%; drift must trigger fallback + retraining (Phase 19/23) |
| Your accuracy expectation is sized on the **chronological** split | Held-out MAE is 4.69 only under a random split of *rows* drawn from the same 20 simulation runs, where near-duplicate instants of one cluster trajectory land on both sides. Hold out whole runs and it is 4.90 (R² 0.811 ± 0.021); hold out later arrivals — which is what deployment does — and it is **7.24** (R² 0.725), 54% more error than the random split reports (`05_results/models/evaluation_splits.csv`, a step of `run_all_experiments.sh`) |
| A **plain size-sort** has been ruled out on your own workload | On the SDSC SP2 trace the learned proactive policy and a fixed SMALLEST_FIRST sort are statistically *equivalent* in mean wait — TOST over 20 disjoint 7-day windows, observed difference −4.44 s against an 870.17 s margin, p = 1.8e-12, achieved power 1.000. On LANL CM-5 the same comparison is **inconclusive** (below). This is a property of the score's functional form, not of the model's accuracy: at a fixed dispatch instant every queued job sees the same cluster state, so the score varies only with requested size — 7 of the 12 features show 0.0% variation across the queue, over 45,432 measured dispatch instants with 0 violations. If a size-sort matches the model on your trace, it is the cheaper thing to run (`05_results/trace_schedulers/tost_power.csv`) |

If any precondition fails, run FIFO (or your incumbent scheduler) — the measured
data does not support the switch.

**On the LANL row, specifically.** The equivalence test there did *not* find a
difference; it found nothing at all. Observed paired difference 320.02 s against a
222.93 s margin, achieved power **0.47%** (0.00465; 5.8% under the bootstrap
variant). A test with a 0.47% chance of certifying equivalence returning "not
equivalent" is not evidence, and it should be read as **inconclusive**, never as
"different". Nor is it a matter of collecting more data: the observed difference
*exceeds* the margin, so no sample size can certify equivalence at 10%, and
establishing a *difference* would need 29 disjoint windows (50 after Holm over the
family of 11 tests) while the trace supplies 28. It is not settleable with disjoint
windows on LANL at all. The equivalence claim rests on SDSC, where 3 windows would
have sufficed and 29 are available. Two limits apply to both rows: power computed
from an observed effect is post-hoc and is an estimate, and the supply of windows is
a property of the trace, not something a longer run recovers.

## 2. Deployment modes

Roll out in this order; never skip advisory mode.

1. **Shadow mode** — the model predicts wait times for every queued job; predictions
   are logged, the incumbent scheduler still decides. Collect ≥ 2–4 weeks of
   (features, actual wait) pairs. This doubles as training data.
2. **Advisory mode** — the proactive ordering is computed and shown to operators
   (dashboard column "suggested order"), decisions stay manual.
3. **Active mode with guardrails** — proactive ordering drives dispatch, with:
   - anti-starvation bumping ON (see `04_scheduler/fairness_analysis.py`,
     `proactive_starvation` variant: recovers Gini 0.79 → 0.69, max wait 123 → 88 ts),
   - automatic FIFO fallback on drift trigger (below).

## 3. Retraining pipeline

```bash
# 1. Export your accounting log to SWF (sacct/scontrol for SLURM) and place it at:
#    02_data/<your_cluster>.swf
# 2. Parse + featurize:
python 02_data/load_real_traces.py          # SWF parser (field mapping documented in-file)
# 3. Retrain:
python 03_models/train_improved_model.py    # writes 03_models/wait_model_v2.pkl
# 4. Validate before promoting:
python 03_models/evaluate_splits.py              # random / run-wise / chronological R² and MAE
python 04_scheduler/benchmark_statistical.py     # paired benchmark on your workload
python phases_22_30/phase_23_sensitivity/sensitivity_ood_analysis.py  # OOD check
```

Promote the new model only if holdout R² ≥ 0.7 **on your own trace** and the paired
benchmark shows a statistically significant improvement (the script reports the
paired t-test and a **Student-t** 95% CI — `stats.t.ppf`, not a bootstrap). If you
want a percentile bootstrap over the same runs, that is
`phases_22_30/phase_22_stats/stats_bootstrap.py`.

**The R² ≥ 0.7 gate no longer means what it used to, so state which split it is
measured on.** It was written when this repository's headline was R² 0.837 from a
single random split of rows, and it sat 0.137 below the number it was screening.
Measured honestly on the same data, the numbers it screens are R²
**0.810910 ± 0.020672** run-wise (`GroupKFold` on `run_id`, 5 folds — the headline),
0.793486 ± 0.054176 leave-one-run-out over 20 folds, and **0.725147** chronological.
The gate now has about 0.025 of headroom on the split that matches deployment order,
and individual leave-one-run-out folds span 0.6899 to 0.8744 — some of them fall
*below* it. So read the gate as: **R² ≥ 0.7 on a split that holds out whole runs, or
whole time periods, and never on a random split of rows.** A random row split of a
trace whose rows share a cluster trajectory will clear 0.7 for a model that fails the
grouped test; that is exactly how the 0.837 figure this guide used to lean on was
produced. Five run-wise folds are used so the training set stays at 1760 rows, the
same size as the random split — the gap is not a matter of less training data.

**Reading the OOD output.** `phases_22_30/phase_23_sensitivity/ood_failure_modes.csv`
scores 72 shifted scenarios. Until v3.6 every one of the 72 rows was labelled
`DISTRIBUTION_MISMATCH`: the classifier tested `mape >= 35.0` ahead of most other
branches and the smallest MAPE anywhere in the grid is 54.01, so that single branch
fired 72 times out of 72 and the column carried no information. It now reports a
continuous severity score standardised across the grid, four data-derived quantile
bands of 18 scenarios each, and a dominant-axis label: `failure_mode` spans five
values (COMPLETION_DOMINATED 22, POLICY_DOMINATED 21, FIT_DOMINATED 14,
NO_DOMINANT_AXIS 8, CALIBRATION_DOMINATED 7) and `risk_level` three (MEDIUM_RISK 37,
HIGH_RISK 24, LOW_RISK 11). The simulation, the scenarios and the seeds are
unchanged; only the classification of the results changed.

Read those bands as a **ranking within a set of regimes that are all bad**, not as a
safety rating. Mean R² across the 72 scenarios is −0.310. Even the
`Q1_LEAST_SEVERE` band averages R² 0.288 with a 12.4% failure rate — under the
promote gate above — while `Q4_MOST_SEVERE` averages R² −1.045 and 49.6%.
**`LOW_RISK` means "least severe among 72 shifted regimes", never "safe".** A
scenario landing in that band is still one your model should not be serving without
retraining; the band tells you which shift to investigate first, not which shift to
tolerate.

## 4. Monitoring & drift response

Emit these metrics per scheduling interval (Prometheus/Grafana or equivalent):

| Metric | Source | Alert threshold (from measured behavior) |
|---|---|---|
| Rolling prediction MAE (30-job trigger window; the curve logged to CSV uses an 80-job window) | compare predicted vs realized waits | > 0.55 × the standard deviation of the training-set wait times → **fallback to FIFO + retrain** (this is exactly what `03_models/concept_drift_detection.py` implements) |
| p95 / max wait | queue accounting | sustained rise vs FIFO baseline week → tighten anti-starvation threshold |
| Per-job wait Gini (daily) | queue accounting | > 0.7 → tighten anti-starvation threshold |
| Starvation count (wait > 3× **the job's own runtime**) | queue accounting | any sustained increase → investigate |
| Scheduler decision latency | wrap the predict call | > 5% of dispatch interval → batch predictions. Measured scheduling overhead stayed under 5% of throughput at every scale tested (peak 1.55% at 64 GPUs); per-decision latency was 10.2–15.5 ms at 32–256 GPUs on one machine (`phases_22_30/phase_26_scaling/scaling_measurements.txt`) — that is a wall-clock number whose timing columns moved by up to 84% between runs of the same benchmark, so budget for your own hardware rather than these figures |
| Fraction of arrivals that actually **start** within the window, per policy | queue accounting | any gap between policies means your wait-time comparison is running over different sets of jobs. Measured: in the one heavily-loaded small-cluster scenario out of five (`arr2.0_nodes4`), 51.7% of jobs start under FIFO against 59.7% under proactive; restricted to the 91.9 jobs that start under both, proactive's mean-wait improvement is **3.3 pp larger** than the all-jobs figure — the naive comparison understated the learned policy rather than flattering it. The remaining jobs are exchanged, not rescued (21.8 FIFO-only against 39.4 proactive-only), and neither statistic adjudicates them (`05_results/uncertainty/censoring_analysis.csv`) |

Size the rolling-MAE expectation on the chronological number, not the random-split
one. On this repository's own data a model scoring MAE 4.69 under a random row split
scores **7.24** once the hold-out is later arrivals; a constant predictor on that
same chronological split scores 19.23 (`05_results/models/evaluation_splits.csv`).
So the model is a real regressor and not a dressed-up mean — run-wise it is 4.90
against a constant predictor's 13.45 on the same folds, roughly 64% less error — but
an operator who calibrates alerting to 4.69 will page on ordinary behaviour.

Drift response, in order: (1) automatic FIFO fallback, (2) retrain on the last
2–4 weeks of data, (3) shadow-validate, (4) re-promote.

**Do not rely on prediction-interval width as an OOD alarm.** We tested this in
v3.2 with a quantile model (q10/q50/q90): the intervals are under-dispersed
(68% empirical coverage vs 80% nominal), and in a light-load small-cluster
regime — where every smart policy was *worse* than FIFO (−2.0 to −4.8%) — the
spread guard fired on only 0.5% of ticks
(`05_results/uncertainty/uncertainty_ood_benchmark.csv`, regenerated by
`run_all_experiments.sh` along with everything else cited here). The rolling-MAE drift
trigger above (Phase 19) remains the canonical fallback mechanism.

## 5. Configuration knobs

| Knob | Where | Default | Effect |
|---|---|---|---|
| Anti-starvation threshold | `04_scheduler/fairness_analysis.py` (inline bump test; the named constant `STARVATION_RUNTIME_MULTIPLE` lives in `04_scheduler/fairness_budget_sweep.py`) | 3× **the job's own runtime** | Lower = fairer tails, less mean-wait gain |
| Wait budget B | `04_scheduler/fairness_budget_sweep.py` (hard per-job wait cap; job escalates to head-of-queue at B) | 60 ts | Tunable Pareto knob: B=60 keeps ~half the mean-wait gain (+7.4% of the +13.1% unbounded) while capping max wait at 81 ts vs 138 unbounded; B ≤ 30 is slightly *worse* than FIFO (escalation churn), B=∞ = pure proactive (`05_results/fairness/budget_sweep.csv`, a step of `run_all_experiments.sh`) |
| Reordering interval | scheduler loop | every tick | Longer intervals cut inference cost, delay adaptation |
| Drift window / threshold | `03_models/concept_drift_detection.py` | rolling 30 for the trigger test (80 for the logged MAE curve), threshold 0.55 × std(training waits) | Smaller window = faster fallback, more false alarms |
| Model refresh cadence | ops calendar | 3–6 months, or on drift trigger | Stale models decay with workload drift |

## 6. What this does not buy you

**There is no cost-benefit projection here any more, and the one that used to be here
is withdrawn.** It converted the wait-time reduction into GPU-hours and priced them.
That was a category error, not an uncertain assumption: reordering a queue changes
*when* jobs start, not how much compute they consume. Across this repository's own
40-run paired benchmark, GPU utilisation is identical under both policies in **40 of
40 runs** — mean 0.641177 either way, equal to six decimal places — and the same 110
jobs complete under both (`05_results/benchmark_statistical_results.csv`). The
quantity being monetised was measured at zero, so widening the error bars on the
dollar figure would not have rescued it; it would still have reported a saving.
`05_results/roi_analysis.py`, `05_results/roi/`, the pipeline step and the dashboard
panel are all deleted.

What survives is the measured result itself: a **7.897862%** mean-wait reduction
against FIFO in-distribution, Student-t 95% CI [4.8824, 10.9133] over 40 paired runs.
Queue latency is what your users feel and it is worth something on a contended
cluster — but this repository has not measured what, and you should not quote a
currency figure it does not support.

## 7. Container deployment

```bash
docker build -t proactive-scheduler .
docker run --rm -v "$PWD/05_results:/app/05_results" proactive-scheduler   # full pipeline
streamlit run dashboard.py                                                  # results dashboard
```

Outside Docker the same work is one command: `bash run_all_experiments.sh` runs all
22 steps and regenerates every result cited in this guide — including the three
studies added in v3.6 that this guide now leans on: `03_models/evaluate_splits.py`
(the split comparison behind the promote gate), `04_scheduler/tost_power.py` (the
equivalence power analysis) and `04_scheduler/censoring_analysis.py` (the started-job
decomposition). Check the output against the committed artefacts with
`python tools/verify_artifacts.py --quick` (`--smoke` for a faster pass).

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `UnicodeEncodeError` on Windows consoles | cp1252 default encoding | pipelines export `PYTHONUTF8=1`; set it in your shell for ad-hoc runs |
| `FileNotFoundError` for CSV/PKL | running a script from the wrong cwd (legacy) | all active scripts are now cwd-independent; re-pull if you see this |
| Dashboard shows "Run pipeline to generate…" | results not yet generated | `bash run_all_experiments.sh` |
| Model predicts constant/absurd waits | feature schema mismatch | retrain with `03_models/train_improved_model.py`; the pre-v2 model files were deleted in v3.6, so the only models the v2 12-feature pipeline loads are `03_models/wait_model_v2.pkl` and `wait_model_quantile.pkl` — any other `.pkl` you have kept is stale, do not point the scheduler at it |
| Proactive shows no benefit | cluster not contended | expected — see Honest preconditions |

## 9. Rollback

FIFO fallback is a configuration flip, not a redeploy: the queue is already
maintained in arrival order; disabling reordering restores FIFO semantics
immediately. Keep the previous `wait_model_v2.pkl` alongside any promoted model
for instant model rollback.

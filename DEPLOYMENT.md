# Deployment & Operations Guide (Phase 30)

A runbook for HPC/cloud operators evaluating the proactive feasibility scheduler.
Everything here is grounded in the measured results of this repository — including
the limitations. Read the **Honest preconditions** section before deploying anything.

---

## 1. Honest preconditions — read first

The measured evidence supports deployment **only** under all of the following:

| Precondition | Why (measured evidence) |
|---|---|
| The cluster is **contended** (jobs actually queue) | The wait-time advantage is 14.4% on small contended clusters and **0%** when capacity removes queueing (`05_results/scaling/scaling_analysis.csv`) |
| The model is **retrained on your own workload** | Validated on two **real** Parallel Workloads Archive traces (v3.2): zero-shot transfer of the synthetic model is R² ≈ 0 on both (LANL CM-5 −0.001, SDSC SP2 0.072), but retraining on the trace itself recovers R²(log) = 0.494 on SDSC SP2 (vs −0.694 for a median baseline); OOD mean R² < 0 across 72 shifted scenarios (`05_results/traces/real_trace_validation.csv`, `phases_22_30/phase_23_sensitivity/`) |
| **Tail latency and fairness are monitored** | Mean-wait gain costs tail latency (max wait ~58 → ~125 ts) and per-job Gini (0.53 → 0.80) (`05_results/fairness/`) |
| A **FIFO fallback** is wired in | OOD failure rate averages ≈ 32%; drift must trigger fallback + retraining (Phase 19/23) |

If any precondition fails, run FIFO (or your incumbent scheduler) — the measured
data does not support the switch.

## 2. Deployment modes

Roll out in this order; never skip advisory mode.

1. **Shadow mode** — the model predicts wait times for every queued job; predictions
   are logged, the incumbent scheduler still decides. Collect ≥ 2–4 weeks of
   (features, actual wait) pairs. This doubles as training data.
2. **Advisory mode** — the proactive ordering is computed and shown to operators
   (dashboard column "suggested order"), decisions stay manual.
3. **Active mode with guardrails** — proactive ordering drives dispatch, with:
   - anti-starvation bumping ON (see `04_scheduler/fairness_analysis.py`,
     `proactive_starvation` variant: recovers Gini 0.80 → 0.69, max wait 125 → 87 ts),
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
python 04_scheduler/benchmark_statistical.py     # paired benchmark on your workload
python phases_22_30/phase_23_sensitivity/sensitivity_ood_analysis.py  # OOD check
```

Promote the new model only if holdout R² ≥ 0.7 **on your own trace** and the paired
benchmark shows a statistically significant improvement (the script reports the
paired t-test and bootstrap CI).

## 4. Monitoring & drift response

Emit these metrics per scheduling interval (Prometheus/Grafana or equivalent):

| Metric | Source | Alert threshold (from measured behavior) |
|---|---|---|
| Rolling prediction MAE (window 50–100 jobs) | compare predicted vs realized waits | > 1.5× training-holdout MAE → **fallback to FIFO + retrain** (this is the Phase 19 trigger) |
| p95 / max wait | queue accounting | sustained rise vs FIFO baseline week → tighten anti-starvation threshold |
| Per-job wait Gini (daily) | queue accounting | > 0.7 → tighten anti-starvation threshold |
| Starvation count (wait > 3× mean) | queue accounting | any sustained increase → investigate |
| Scheduler decision latency | wrap the predict call | > 5% of dispatch interval → batch predictions (measured: 10–48 ms per batched decision at 32–256 GPUs) |

Drift response, in order: (1) automatic FIFO fallback, (2) retrain on the last
2–4 weeks of data, (3) shadow-validate, (4) re-promote.

**Do not rely on prediction-interval width as an OOD alarm.** We tested this in
v3.2 with a quantile model (q10/q50/q90): the intervals are under-dispersed
(68% empirical coverage vs 80% nominal), and in a light-load small-cluster
regime — where every smart policy was *worse* than FIFO (−4.3 to −5.0%) — the
spread guard fired on only 0.2% of ticks
(`05_results/uncertainty/uncertainty_ood_benchmark.csv`). The rolling-MAE drift
trigger above (Phase 19) remains the canonical fallback mechanism.

## 5. Configuration knobs

| Knob | Where | Default | Effect |
|---|---|---|---|
| Anti-starvation threshold | `fairness_analysis.py` (`STARVATION_THRESHOLD`-style bump age) | 3× mean wait | Lower = fairer tails, less mean-wait gain |
| Wait budget B | `04_scheduler/fairness_budget_sweep.py` (hard per-job wait cap; job escalates to head-of-queue at B) | 60 ts | Tunable Pareto knob: B=60 keeps ~half the mean-wait gain (+7.1% of the +13.2% unbounded) while capping max wait at 81 ts vs 136 unbounded; B ≤ 30 is slightly *worse* than FIFO (escalation churn), B=∞ = pure proactive (`05_results/fairness/budget_sweep.csv`) |
| Reordering interval | scheduler loop | every tick | Longer intervals cut inference cost, delay adaptation |
| Drift window / multiplier | `concept_drift_detection.py` | rolling 50, 1.5× | Smaller window = faster fallback, more false alarms |
| Model refresh cadence | ops calendar | 3–6 months, or on drift trigger | Stale models decay with workload drift |

## 6. Cost-benefit framing

`05_results/roi/cost_benefit_analysis.csv` projects ≈ $78k/yr savings (86%
first-year ROI) for a 180k-job/yr cluster **under strong assumptions** (linear
wait→GPU-hour conversion, $2.20/GPU-hr cloud rate, flat energy pricing). Treat it
as an order-of-magnitude illustration, not a promise; recompute with your own
rates via `05_results/roi_analysis.py` after your own paired benchmark.

## 7. Container deployment

```bash
docker build -t proactive-scheduler .
docker run --rm -v "$PWD/05_results:/app/05_results" proactive-scheduler   # full pipeline
streamlit run dashboard.py                                                  # results dashboard
```

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `UnicodeEncodeError` on Windows consoles | cp1252 default encoding | pipelines export `PYTHONUTF8=1`; set it in your shell for ad-hoc runs |
| `FileNotFoundError` for CSV/PKL | running a script from the wrong cwd (legacy) | all active scripts are now cwd-independent; re-pull if you see this |
| Dashboard shows "Run pipeline to generate…" | results not yet generated | `bash run_all_experiments.sh` |
| Model predicts constant/absurd waits | feature schema mismatch | retrain — never mix `wait_model.pkl` (legacy) with the v2 12-feature pipeline |
| Proactive shows no benefit | cluster not contended | expected — see Honest preconditions |

## 9. Rollback

FIFO fallback is a configuration flip, not a redeploy: the queue is already
maintained in arrival order; disabling reordering restores FIFO semantics
immediately. Keep the previous `wait_model_v2.pkl` alongside any promoted model
for instant model rollback.

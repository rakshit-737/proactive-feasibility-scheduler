# Phases 22–30 Completion Summary

> **Provenance note (2026-07):** this summary was rewritten after a full project audit.
> Earlier versions of this document reported results from scripts that hardcoded
> synthetic constants (Phases 23, 24, 26, 27) and quoted numbers that contradicted
> the measured data. All phase scripts now compute their outputs from real data or
> real simulations, the pipeline is seeded end-to-end, and every number below comes
> from a regenerated artifact on disk.

## Status Overview

| Phase | Title | Key File(s) | Status |
|-------|-------|-------------|--------|
| 22 | Statistical Rigor | `phase_22_stats/stats_bootstrap.py` | ✅ Complete (real 40-run data) |
| 23 | OOD Sensitivity | `phase_23_sensitivity/sensitivity_ood_analysis.py` | ✅ Complete (real model, real shifted sims) |
| 24 | Extended Schedulers | `phase_24_extended_schedulers/scheduler_comparison.py` | ✅ Complete (real benchmark CSVs; SLURM/K8s/Yarn baselines **dropped** — no real implementations existed) |
| 25 | Real Traces | `phase_25_real_traces/trace_preprocessing.py` | ⚠️ Scaffolding complete; **no real trace evaluated** — outputs labelled `synthetic_proxy_trace` |
| 26 | Scaling Validation | `phase_26_scaling/scaling_benchmark.py` | ✅ Complete (real discrete-time simulation, real model inference timing) |
| 27 | Fairness & SLA | `phase_27_fairness/fairness_sla_analysis.py` | ✅ Complete (real per-run/per-job data; schedulers without real data dropped) |
| 28 | Manuscript | `phase_28_manuscript/manuscript.tex` | ✅ Draft compiles; numbers synced to regenerated results |
| 29 | Reproducibility | `run_all_experiments_v2.sh` | ✅ Complete (seeded, UTF-8-safe, path-independent) |
| 30 | Deployment Guide | `../DEPLOYMENT.md` | ✅ Complete |

---

## Phase-by-Phase Results (regenerated, honest)

### Phase 22: Statistical Rigor
- Wait improvement: **7.71%**, bootstrap 95% CI **[4.95%, 10.34%]**, paired t-test p = 1.4e-06 (BH-corrected), Wilcoxon p = 7.1e-06, n = 40 seeded paired runs.
- GPU utilisation and completed-jobs comparisons are zero-variance and are reported as **"n/a (zero variance)"** — not as significance.

### Phase 23: OOD Sensitivity (72 scenarios)
- Scores the **actual trained model** (`wait_model_v2.pkl`) on freshly simulated shifted workloads.
- **Prediction quality collapses OOD**: mean R² ≈ **−0.31** (range −2.32 … +0.85).
- Scheduling improvement is erratic OOD: −14% … +50%, mean failure rate ≈ 32%.
- Conclusion: retrain per deployment regime; keep a FIFO fallback.

### Phase 24: Extended Scheduler Comparison
- Compares the five schedulers with **real implementations** (FIFO, SJF, Priority, NN, Proactive) from the regenerated multi-scheduler benchmark.
- SLURM-backfill / Kubernetes-QoS / Yarn-FIFO rows were **removed**: no real implementations existed and their previous numbers were fabricated constants. Implementing a real backfill baseline is future work (see README/roadmap).
- Result: SJF 12.00 ts (needs known runtimes) < NN 15.69 ≈ Proactive 15.71 < Priority 16.34 < FIFO 16.65; FIFO dominates fairness (Gini 0.518) and tail latency (max 53.9 ts).

### Phase 25: Trace Integration — **open item**
- SWF parser and feature mapper work; the loader falls back **loudly** to a synthetic proxy when no real trace file exists (which is the current state).
- Cross-trace error vs the proxy: MAE ≈ 18.3 ts. Real-trace validation requires downloading a trace (instructions in the script header and `02_data/load_real_traces.py`).

### Phase 26: Scaling Validation
- Real discrete-time simulation at 32–256 GPUs under saturation (10,000 queued jobs): utilisation > 99.7% at all scales, batched inference 10–48 ms per decision, **scheduling overhead < 5%** of throughput.
- Complementary moderate-load study (Phase 17): proactive advantage is 14.4% on small contended clusters and → 0% when capacity removes queueing.

### Phase 27: Fairness & SLA
- Computed from the real 40-run benchmark + per-job fairness data. Schedulers without real distributions are listed with `data_source = per_job_aggregates_only` or dropped (noted in `dropped_schedulers.txt`).
- **Proactive worsens per-job Gini (0.53 → 0.80)** and max wait (58 → 125 ts); anti-starvation variant recovers to Gini 0.69 / max 87 ts.
- SLA compliance: FIFO 0.933, Proactive 0.945, anti-starvation 0.863. Run-level Jain: 0.918 vs 0.922.
- The earlier claim "proactive ≥ FIFO on fairness" is **withdrawn**; the honest claim is a quantified trade-off with a partial mitigation.

### Phase 28: Manuscript
- `manuscript.tex` compiles (inputenc fixed, all citations resolve, booktabs rules correct) and now reports the regenerated numbers above, including the fairness trade-off, OOD collapse, and the unvalidated real-trace transfer.

### Phase 29: Reproducibility
- `run_all_experiments.sh` (root) and `run_all_experiments_v2.sh` (phases 22–27) run end-to-end on a fresh checkout: seeded, cwd-independent, UTF-8-safe on Windows.
- `requirements.txt` pins the exact versions verified working on Python 3.14.

### Phase 30: Deployment Guide
- `DEPLOYMENT.md` (project root) covers advisory-mode rollout, monitoring thresholds, drift-triggered retraining, FIFO fallback, and an honest ROI framing.

---

## Quality Gates

- [x] Phase 22: headline claims carry CIs and corrected p-values
- [x] Phase 23: failure modes measured with the real model and discussed in the manuscript
- [x] Phase 24: comparison uses only real scheduler implementations
- [ ] Phase 25: cross-dataset R² on a **real** trace — *open; proxy only*
- [x] Phase 26: overhead < 5% demonstrated at 256 GPUs
- [x] Phase 27: fairness trade-off quantified (not claimed away); starvation analysed
- [x] Phase 28: manuscript numbers match regenerated artifacts
- [x] Phase 29: one-command reproducibility verified on this machine
- [x] Phase 30: deployment guide includes fallback procedures

---

## Next Steps

1. **Real-trace validation (Phase 25 completion)** — download an SWF trace
   (Parallel Workloads Archive) → `02_data/`, rerun `load_real_traces.py`,
   `synthetic_vs_real_comparison.py`, and `trace_preprocessing.py`; retrain per
   trace and update RESULTS.md.
2. **Real backfill baseline (Phase 24 completion)** — implement conservative or
   EASY backfilling in the simulator; it is the strongest practical competitor.
3. **Compile & iterate the manuscript**:
   ```bash
   cd phases_22_30/phase_28_manuscript && pdflatex manuscript.tex && pdflatex manuscript.tex
   ```
4. **Tag a release** once real-trace validation lands.

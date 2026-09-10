# Phases 22–30 Roadmap: Advanced Validation & Publication

## Executive Summary
This roadmap extends the proactive feasibility scheduler from core research (Phases 1–21) to publication-ready artifacts (Phases 22–30). Each phase has discrete deliverables, effort estimates, and clear advancement goals.

This is a **planning document**, written before the work. Several of the promises below were not delivered as written. Rather than quietly editing the plan to match the outcome, each such cell now states what was planned, what was actually produced, and — where the promise was abandoned — why.

---

## Phase Breakdown

| Phase | Title | Goal (1–2 sent) | Key Deliverables | Effort | How It Advances | Dependencies |
|-------|-------|-----------------|------------------|--------|-----------------|--------------|
| 22 | Statistical Rigor & Confidence Intervals | *Planned*: 95% CI on **all** metrics (wait, fairness, speedup) via bootstrap and paired t-tests. **Partly delivered**: `stats_summary.csv` carries CIs for three *wait* metrics only (FIFO mean wait, proactive mean wait, wait-improvement %). No fairness CI and no speedup CI exists anywhere in the repository. | `stats_bootstrap.py` (percentile bootstrap; wait-improvement CI [4.9102%, 10.6714%] over the same 40 runs), `stats_summary.csv`, updated `RESULTS.md` | 2–3 days | Removes p-hacking concerns; strengthens publication claim | Phase 09 (40-run data) |
| 23 | Sensitivity & OOD Robustness | Quantify model degradation under realistic domain shifts (arrival rates, job size distributions, cluster sizes). | `sensitivity_ood_analysis.py`, `ood_failure_modes.csv`, OOD R² heatmap (the planned `sensitivity_report.csv` was never produced under that name) | 2 days | Honest about limitations; shows when proactive doesn't help | Phase 11 (existing OOD) |
| **24** | **Extended Scheduler Comparison** | **_Planned_: compare proactive vs. published baselines (SLURM backfill heuristics, Kubernetes QoS, Yarn FIFO) via simulation. NOT DELIVERED AS PLANNED: none of the three was ever implemented, and all three were dropped — the reason is recorded in `phase_27_fairness/dropped_schedulers.txt`. What exists instead is a benchmark of 14 schedulers actually implemented in this repository (FIFO variants, SJF/SJF_EST, SRPT, HRRN, static priority, EASY and conservative backfill, SMALLEST, NN, proactive variants).** | **`baseline_comparison.csv`, scheduler heatmap, `05_results/schedulers/multi_scheduler_benchmark.csv` (14 schedulers). No 5-scheduler benchmark plot with SLURM / K8s / Yarn exists.** | **2–3 days** | **Positions work in landscape; claims novelty rigorously** | **Phase 14 (existing schedulers)** |
| **25** | **Real Trace Integration (1)** | **_Planned_: ingest LANL/Alibaba traces; map job attributes; retrain model on real data; measure cross-dataset R². PARTLY DELIVERED, DIFFERENTLY: two real PWA traces (SDSC SP2 1998, LANL CM-5 1994) were ingested and models retrained on them. No Alibaba trace has ever existed in this repository — only scaffolding did, and that scaffolding has been removed.** | **`trace_preprocessing.py`, `05_results/traces/real_trace_validation.csv`. `cross_trace_mae.csv` is DELETED: it held a single row for the synthetic proxy with `mape_pct` pinned at the 200.0 saturation value, and it scored a heuristic whose coefficients nearly matched the generator's, so the error it reported was circular. No trace-specific model checkpoints were retained.** | **3–4 days** | **Validates sim-to-real assumption; grounds in practice** | **Phase 16 (trace code stub exists)** |
| **26** | **Scaling Validation** | **_Planned_: benchmark proactive on 16 and 32-node clusters; measure inference overhead and model quality decay; fit a scaling law. MEASUREMENTS DELIVERED, VERDICT WITHDRAWN (v3.6): the latency columns are wall-clock and machine-dependent, the four measured points are non-monotone, and a fitted exponent over them identifies nothing. No complexity class is claimed and no projection past the largest measured cluster is made.** | **`scaling_benchmark.csv`, overhead plot, `scaling_measurements.txt` (renamed from `scaling_law_fit.txt`; it no longer fits a law). What survives: scheduling overhead stayed under 5% of throughput at every measured scale, and utilisation stayed above 99.69%.** | **2 days** | **Shows method doesn't break at scale; production-relevant** | **Phase 17 (partial)** |
| **27** | **Fairness & SLA Guarantees** | **_Planned_: formalize wait-time bounds (percentile, starvation count); prove proactive ≥ FIFO on fairness metrics. THE "PROVE PROACTIVE ≥ FIFO" GOAL IS WITHDRAWN (withdrawal recorded at `COMPLETION_SUMMARY.md:72`) and is contradicted by every artefact: per-job Gini rises from 0.526 under FIFO to 0.794 under proactive — materially _less_ fair. The honest finding is a quantified fairness cost with a partial mitigation: the anti-starvation variant recovers to Gini 0.692. A job counts as starved when its wait exceeds 3× its own runtime.** | **SLA metrics table (`sla_compliance.csv`), `fairness_metrics.csv`. `fairness_formal_analysis.md` and the promised proof sketch are DELETED — no formal proof was ever produced.** | **1–2 days** | **Addresses practical concerns (ops don't want unfair schedulers)** | **Phase 13 (fairness stub)** |
| **28** | **Manuscript Draft** | **Write 8–12 page conference/journal submission (problem, method, experiments, results, threats).** | **`manuscript.pdf` (or LaTeX), reference list, author guidelines checklist** | **5–7 days** | **Publication-ready artifact** | **Phases 22–27** |
| **29** | **Reproducibility & Release** | **Package repo for artifact evaluation; Docker image, one-line `run_all_experiments.sh`, versioned outputs, zenodo DOI. DELIVERED EXCEPT THE DOI: the root pipeline is now a single entry point of 20 steps that regenerates every result. No Zenodo DOI has been obtained.** | **`Dockerfile`, `run_all_experiments.sh` (one command, 20 steps), reproducibility README, `tools/verify_artifacts.py`. DOI: NOT DONE.** | **2–3 days** | **Enables others to validate and extend** | **All prior** |
| **30** | **Deployment & Operations Guide** | **Write runbook for HPC ops teams: config tuning, monitoring, fallback to FIFO, cost projections.** | **`DEPLOYMENT.md`, config template, monitoring dashboard spec, ROI calculator** | **2–3 days** | **Bridges research to practice; shows real-world impact pathway** | **All prior** |

*The **Effort** column holds pre-work planning estimates. Nothing in it was measured against time actually spent, and it should not be read as a measurement.*

---

## Status Tracking

### Completed (✅)
- Phase 22: ✅ Statistical bootstrap analysis (stats_bootstrap.py, CI plots) — but the CIs cover three wait metrics only; the promised fairness and speedup CIs were never produced
- Phase 23: ✅ OOD sensitivity analysis (real model on real shifted sims; ood_heatmap, failure modes)
- Phase 24: ✅ Extended scheduler comparison (real implementations only — SLURM/K8s/Yarn baselines dropped, see `phase_27_fairness/dropped_schedulers.txt` and COMPLETION_SUMMARY.md; the benchmark now carries 14 schedulers, including real EASY and conservative backfill)
- Phase 25: ✅ Real trace integration (v3.2: two real PWA traces evaluated — zero-shot transfer R² ≈ 0 on both; retrained R²(log) 0.494 on SDSC SP2, 0.101 on LANL CM-5; signal is machine-dependent). No Alibaba trace was ever ingested; `cross_trace_mae.csv` has been deleted as circular
- Phase 28: ✅ Manuscript draft (numbers synced to regenerated artifacts)
- Phase 29: ✅ Reproducibility (seeded, cwd-independent, UTF-8-safe; one entry point — `run_all_experiments.sh`, 20 steps, regenerates every result — with `phases_22_30/run_all_experiments_v2.sh` reduced to a forwarding shim, verified by `python tools/verify_artifacts.py`)
- Phase 30: ✅ Deployment guide (../DEPLOYMENT.md)

### Completed with a withdrawn claim (⚠️)
- Phase 26: ⚠️ Scaling validation — the simulation and the timing measurements stand, but the scalability *verdict* is withdrawn (v3.6). The "O(1) / constant latency" class and the 512 / 1024 / 4096-GPU projections came from fitting an exponent to four non-monotone wall-clock points, and are retracted. `scaling_law_fit.txt` is now `scaling_measurements.txt`
- Phase 27: ⚠️ Fairness & SLA analysis (real per-run/per-job data) — the phase goal "prove proactive ≥ FIFO on fairness" is **withdrawn**. Proactive is *less* fair: per-job Gini 0.526 (FIFO) → 0.794, max job wait 57.85 → 122.65 ts. The anti-starvation variant partly mitigates: Gini 0.692, max wait 88.15 ts. SLA compliance: FIFO 0.9326, proactive 0.9444, anti-starvation 0.8627 (`phase_27_fairness/sla_compliance.csv`)

### Open (⚠️)
- No Zenodo DOI has been obtained; the versioned release remains uncited.
- The SLURM-backfill, Kubernetes-QoS and Yarn-FIFO baselines promised in Phase 24 were never implemented and are not planned.
- The formal fairness proof promised in Phase 27 does not exist; fairness is measured, not proved.
- No scaling law is fitted or claimed; identifying one would need repeated timings on controlled hardware, not four points from one machine.
- No fairness or speedup confidence interval exists (Phase 22 promised both).

---

## Cross-Phase Dependencies

```
Phase 22 (Stats) ──────┐
Phase 23 (OOD) ────┐   │
                   ├─→ Phase 24–27 (validation)
Phase 14 (Sched) ──┤       ↓
Phase 16 (Traces) ─┤    Phase 28 (Manuscript)
Phase 13 (Fair) ───┤       ↓
                   └─→ Phase 29 (Release)
                       Phase 30 (Deployment)
```

---

## Key Quality Gates

1. **Phase 24**: Scheduler comparison must show proactive advantage is **not artifact** of narrow baseline set. — *Gate applied, and the scheduler fails it.* Widening the baseline set removes the advantage: in the 14-scheduler benchmark proactive's 15.948 ts mean wait is beaten by SJF (12.340), SJF_EST (13.318), SRPT (14.023) and HRRN (15.554). The advantage is over FIFO (17.216) and the size heuristics, not over the field.
2. **Phase 25**: Cross-dataset R² must be documented; sim-to-real gap quantified. — *Met* (`05_results/traces/real_trace_validation.csv`).
3. **Phase 26**: ~~Scaling must prove linear or sub-linear inference cost.~~ — *Gate retired as unanswerable with the data collected.* Four non-monotone wall-clock latency points from one machine cannot identify a complexity class, and no class is claimed. What is reported instead: scheduling overhead under 5% of throughput and utilisation above 99.69% at every scale measured.
4. **Phase 27**: ~~Fairness proof must address starvation risk formally.~~ — *Not met.* No formal proof was produced and `fairness_formal_analysis.md` has been deleted. Starvation is measured rather than proved, under one definition — wait exceeding 3× the job's own runtime — giving per-run counts of FIFO 22.25, proactive 18.35, anti-starvation 30.20.
5. **Phase 28**: Manuscript must include failure-mode discussion (Phase 23 output).
6. **Phase 29**: Reproducibility script must run end-to-end on clean environment. — *Met.* `run_all_experiments.sh` is one command covering 20 steps and regenerates every result; check the outputs with `python tools/verify_artifacts.py` (`--quick` / `--smoke`).
7. **Phase 30**: Operations guide must include fallback procedures.

---

## Artifact Outputs Summary

### Code & Scripts
- `phases_22_30/phase_24_extended_schedulers/scheduler_comparison.py`
- `phases_22_30/phase_25_real_traces/trace_preprocessing.py`
- `phases_22_30/phase_26_scaling/scaling_benchmark.py`
- `phases_22_30/phase_27_fairness/fairness_sla_analysis.py`
- `run_all_experiments.sh` (project root) — the single 20-step entry point; `phases_22_30/run_all_experiments_v2.sh` is now only a forwarding shim to it

### Documents
- `phases_22_30/phase_28_manuscript/manuscript.tex`
- ~~`phases_22_30/phase_27_fairness/fairness_formal_analysis.md`~~ — deleted; the formal fairness analysis it promised was never written
- `DEPLOYMENT.md` (project root)
- `README_REPRODUCIBILITY.md` (project root)

### Data Artifacts
- CSV reports (comparison, traces, scaling, fairness metrics)
- Heatmaps, forest plots, violin plots
- ~~Model checkpoints (trace-specific)~~ — not produced; the retrained per-trace models were evaluated and their scores recorded in `05_results/traces/real_trace_validation.csv`, but no per-trace checkpoint was kept

---

## Timeline

- **Week 1**: Phase 24–26 (scheduler comparison, trace integration, scaling)
- **Week 2**: Phase 27–28 (fairness, manuscript draft)
- **Week 3**: Phase 29–30 (release, deployment guide)

**Total effort**: ~3 weeks (intensive research sprints) — a planning estimate made before the work, not a measurement of time spent.

---

## Success Criteria

- [x] All phases produce publication-quality tables/plots
- [x] Phase 23 failure modes explicitly discussed in Phase 28 manuscript
- [x] Phase 29 reproducibility passes fresh environment test (verified on this machine, July 2026)
- [ ] Phase 30 operations guide reviewed by HPC domain expert
- [ ] DOI assigned to versioned release — **not done**; no Zenodo DOI has ever been obtained, and none is pending


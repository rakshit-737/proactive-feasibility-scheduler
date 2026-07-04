# Phases 22–30 Roadmap: Advanced Validation & Publication

## Executive Summary
This roadmap extends the proactive feasibility scheduler from core research (Phases 1–21) to publication-ready artifacts (Phases 22–30). Each phase has discrete deliverables, effort estimates, and clear advancement goals.

---

## Phase Breakdown

| Phase | Title | Goal (1–2 sent) | Key Deliverables | Effort | How It Advances | Dependencies |
|-------|-------|-----------------|------------------|--------|-----------------|--------------|
| 22 | Statistical Rigor & Confidence Intervals | Compute 95% CI on all metrics (wait, fairness, speedup) via bootstrap and paired t-tests. | `stats_bootstrap.py`, CI tables (CSV), updated `RESULTS.md` | 2–3 days | Removes p-hacking concerns; strengthens publication claim | Phase 09 (40-run data) |
| 23 | Sensitivity & OOD Robustness | Quantify model degradation under realistic domain shifts (arrival rates, job size distributions, cluster sizes). | `sensitivity_report.csv`, failure mode matrices, OOD R² heatmap | 2 days | Honest about limitations; shows when proactive doesn't help | Phase 11 (existing OOD) |
| **24** | **Extended Scheduler Comparison** | **Compare proactive vs. published baselines (SLURM backfill heuristics, Kubernetes QoS, Yarn FIFO) via simulation.** | **`baseline_comparison.csv`, 5-scheduler benchmark plots** | **2–3 days** | **Positions work in landscape; claims novelty rigorously** | **Phase 14 (existing schedulers)** |
| **25** | **Real Trace Integration (1)** | **Ingest LANL/Alibaba traces; map job attributes; retrain model on real data; measure cross-dataset R².** | **`real_traces_preprocessing.py`, `cross_trace_mae.csv`, trace-specific models** | **3–4 days** | **Validates sim-to-real assumption; grounds in practice** | **Phase 16 (trace code stub exists)** |
| **26** | **Scaling Validation** | **Benchmark proactive on 16 and 32-node clusters; measure inference overhead and model quality decay.** | **`scaling_benchmark.csv`, overhead plot, scaling law fit (O(n) analysis)** | **2 days** | **Shows method doesn't break at scale; production-relevant** | **Phase 17 (partial)** |
| **27** | **Fairness & SLA Guarantees** | **Formalize wait-time bounds (percentile, starvation count); prove proactive ≥ FIFO on fairness metrics.** | **`fairness_formal_analysis.md`, SLA metrics table, proof sketch** | **1–2 days** | **Addresses practical concerns (ops don't want unfair schedulers)** | **Phase 13 (fairness stub)** |
| **28** | **Manuscript Draft** | **Write 8–12 page conference/journal submission (problem, method, experiments, results, threats).** | **`manuscript.pdf` (or LaTeX), reference list, author guidelines checklist** | **5–7 days** | **Publication-ready artifact** | **Phases 22–27** |
| **29** | **Reproducibility & Release** | **Package repo for artifact evaluation; Docker image, one-line `run_all_experiments.sh`, versioned outputs, zenodo DOI.** | **`Dockerfile`, versioned `run_all_experiments.sh`, reproducibility README, DOI** | **2–3 days** | **Enables others to validate and extend** | **All prior** |
| **30** | **Deployment & Operations Guide** | **Write runbook for HPC ops teams: config tuning, monitoring, fallback to FIFO, cost projections.** | **`DEPLOYMENT.md`, config template, monitoring dashboard spec, ROI calculator** | **2–3 days** | **Bridges research to practice; shows real-world impact pathway** | **All prior** |

---

## Status Tracking

### Completed (✅)
- Phase 22: ✅ Statistical bootstrap analysis (stats_bootstrap.py, CI plots)
- Phase 23: ✅ OOD sensitivity analysis (real model on real shifted sims; ood_heatmap, failure modes)
- Phase 24: ✅ Extended scheduler comparison (real implementations only — SLURM/K8s/Yarn baselines dropped, see COMPLETION_SUMMARY.md; v3.2 adds a real EASY-backfill baseline in the 7-scheduler benchmark)
- Phase 25: ✅ Real trace integration (v3.2: two real PWA traces evaluated — zero-shot transfer R² ≈ 0 on both; retrained R²(log) 0.494 on SDSC SP2, 0.101 on LANL CM-5; signal is machine-dependent)
- Phase 26: ✅ Scaling validation (real simulation + real inference timing)
- Phase 27: ✅ Fairness & SLA analysis (real per-run/per-job data)
- Phase 28: ✅ Manuscript draft (numbers synced to regenerated artifacts)
- Phase 29: ✅ Reproducibility (seeded, cwd-independent, UTF-8-safe pipelines)
- Phase 30: ✅ Deployment guide (../DEPLOYMENT.md)

### Open (⚠️)
- None — all phases 22–30 are complete as of v3.2 (July 2026).

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

1. **Phase 24**: Scheduler comparison must show proactive advantage is **not artifact** of narrow baseline set.
2. **Phase 25**: Cross-dataset R² must be documented; sim-to-real gap quantified.
3. **Phase 26**: Scaling must prove linear or sub-linear inference cost.
4. **Phase 27**: Fairness proof must address starvation risk formally.
5. **Phase 28**: Manuscript must include failure-mode discussion (Phase 23 output).
6. **Phase 29**: Reproducibility script must run end-to-end on clean environment.
7. **Phase 30**: Operations guide must include fallback procedures.

---

## Artifact Outputs Summary

### Code & Scripts
- `phases_22_30/phase_24_extended_schedulers/scheduler_comparison.py`
- `phases_22_30/phase_25_real_traces/trace_preprocessing.py`
- `phases_22_30/phase_26_scaling/scaling_benchmark.py`
- `phases_22_30/phase_27_fairness/fairness_sla_analysis.py`
- `phases_22_30/run_all_experiments_v2.sh`

### Documents
- `phases_22_30/phase_28_manuscript/manuscript.tex`
- `phases_22_30/phase_27_fairness/fairness_formal_analysis.md`
- `DEPLOYMENT.md` (project root)
- `README_REPRODUCIBILITY.md` (project root)

### Data Artifacts
- CSV reports (comparison, traces, scaling, fairness metrics)
- Heatmaps, forest plots, violin plots
- Model checkpoints (trace-specific)

---

## Timeline

- **Week 1**: Phase 24–26 (scheduler comparison, trace integration, scaling)
- **Week 2**: Phase 27–28 (fairness, manuscript draft)
- **Week 3**: Phase 29–30 (release, deployment guide)

**Total effort**: ~3 weeks (intensive research sprints).

---

## Success Criteria

- [x] All phases produce publication-quality tables/plots
- [x] Phase 23 failure modes explicitly discussed in Phase 28 manuscript
- [x] Phase 29 reproducibility passes fresh environment test (verified on this machine, July 2026)
- [ ] Phase 30 operations guide reviewed by HPC domain expert
- [ ] DOI assigned to versioned release


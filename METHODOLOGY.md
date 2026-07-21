# Methodology (Phases 01–21 + v3.3/v3.4 baselines)

## Core system
- Discrete-time cluster simulation with heterogeneous queue states (synthetic study).
- **Event-driven, second-exact trace replay** for the real-workload study (v3.4):
  state changes only at arrivals and completions, so there is no tick-quantisation
  bias. Capacity-only allocator — SWF records no per-node placement, and inventing
  node-level fragmentation would fabricate data.
- Wait-time regression model trained on engineered cluster-state features.
- Proactive scheduler reorders queue using predicted wait-time.

## v3.4: ranking-degeneracy analysis
The central v3.4 experiment tests whether the wait-time model can influence queue
*ordering* at all. At a fixed dispatch instant all queued jobs share the cluster
state, so only per-job features can differ — and in this feature set each of those
(`can_fit_now`, `gpu_fit_ratio`, `node_availability`, `queue_pressure`) is a
deterministic function of the requested size given the state. The predicted score is
therefore `g_S(size)`, and the ranking is a permutation of the size order.

`04_scheduler/ranking_degeneracy.py` instruments real dispatch decisions through a
`RANK_OBSERVER` hook in both benchmarks (no reimplementation of the simulators) and
measures, per instant: which features actually vary across the queue; whether two
equally-sized jobs ever receive different scores; Kendall τ against the size order;
the fraction of instants whose order is identical to smallest-first and to arrival
order; and the recovered size→priority table with its monotonicity. The matching
control policy is `04_scheduler/size_scheduler.py` (`SMALLEST` / `SMALLEST_FIRST`):
sort by requested size, no model.

## Enhancements
1. **Ablation**: remove each of 12 features and measure R² drop.
2. **Fairness**: evaluate max wait, Gini index, completion by size, starvation count.
3. **Scheduler baselines (14 as of v3.4)**: FCFS/first-fit (historical 'FIFO' key), strict head-blocking FIFO, SJF with true runtimes (oracle), SJF with f-model user estimates (est = runtime·f, f~U(1,C), C=5; Mu'alem & Feitelson 2001) and modal estimates (menu rounding, Tsafrir & Feitelson 2005), Priority+aging, HRRN, **SMALLEST (sort by requested size — the ML-free control implied by the degeneracy analysis)**, canonical two-condition EASY backfill (oracle and estimated runtimes), conservative backfill (per-job reservations on a capacity profile), preemptive SRPT (1-tick checkpoint penalty per preemption), Proactive (XGBoost), NN (MLP), predicted-wait EASY hybrid. Unified wait definition: wait = turnaround − true runtime (identical to start − arrival for non-preemptive policies; charges preemptive requeue time and checkpoint overhead as waiting). A runtime-estimate-quality sweep (C ∈ {1,2,3,5,10} + modal) isolates how much of classical schedulers' advantage survives realistic estimate error.
4. **SHAP**: summary, dependence, and force plots.
5. **Real traces (validated as of v3.2; used for scheduling as of v3.4)**: two cleaned Parallel Workloads Archive traces are committed — LANL CM-5 1994 (1024 procs, 122,060 kept jobs) and SDSC SP2 1998 (128 procs, 43,117 kept jobs). `02_data/build_real_trace_datasets.py` reconstructs each job's submit-instant cluster state by replaying the recorded schedule; `02_data/real_trace_validation.py` evaluates prediction quality. **v3.4** adds `04_scheduler/trace_driven_benchmark.py`, which replays the traces through all 12 policies. Key point: SWF field 9 records the user's *requested time*, so the study uses the **real runtime estimates the traces contain** rather than the simulated f-model — real error is both larger and differently shaped (SDSC median 6.9× over-estimate; LANL 36.3% under-estimates, which the over-estimate-only f-model cannot generate). Jobs with missing estimates fall back to the trace median, deliberately *not* the true runtime, so estimate-driven policies get no free oracle. Protocol: chronological 60% train split, 20 evenly spaced windows (3-day warm-up not measured + 7 measured days), windows spaced evenly rather than selected by load.
6. **Scaling**: 4/8/16/32 nodes (8/32/128/256 GPUs), overhead and inference analysis.
7. **Online learning**: incremental updates on streaming data.
8. **Concept drift**: rolling MAE trigger for adaptive retraining.
9. **ROI**: GPU-hour savings, energy savings, and annual cost-benefit metrics.
10. **Reproducibility and dashboard**: one-command pipeline plus interactive explorers.

## Statistical treatment
- Seeded paired runs for scheduler comparisons (40-run FIFO-vs-proactive benchmark; 20-run 13-scheduler benchmark with out-of-training seeds).
- Paired t-test, Wilcoxon signed-rank, and bootstrap 95% confidence intervals (Phase 22); Benjamini–Hochberg correction across metrics; zero-variance comparisons reported as "n/a" rather than as significance.
- v3.3: every scheduler is compared pairwise against both PROACTIVE and FCFS with Holm step-down correction applied separately to the t-test and Wilcoxon families (`05_results/schedulers/multi_scheduler_significance.csv`), plus Cohen's dz effect sizes; per-C paired CIs in the estimate sweep.
- **v3.4 equivalence testing**: claims that two policies perform *the same* use paired TOST (two one-sided tests, `simstats.tost_equivalence`) with an equivalence margin of 10% of the reference mean, reporting both one-sided p-values, p_TOST, and the 90% CI of the paired difference. A large p from a difference test is **not** evidence of sameness; without TOST the size-sort-vs-XGBoost comparison (p = 0.65) would read as "no significant difference" and be discarded rather than recognised as the result. Trace comparisons are paired by window; synthetic ones by run.
- **v3.4 metrics**: mean bounded slowdown `max(turnaround/max(runtime, τ), 1)` (τ = 60 s on traces, 1 tick synthetic) is reported alongside mean wait as a first-class metric — it is the standard batch-scheduling measure and mean wait alone hides the effect on short jobs.
- Mean and max wait reporting.
- Fairness measured via per-job Gini coefficient, run-level Jain index, starvation counts, and SLA compliance (Phase 27).
- Cross-dataset (proxy) and OOD diagnostics via MAE and R² on freshly simulated shifted workloads (Phase 23).

## Reproducibility
- The entire pipeline is seeded: `bash run_all_experiments.sh` regenerates the dataset, model, and every result deterministically on a fresh checkout.
- `PYTHONUTF8=1` is exported by the pipeline scripts so Unicode console output works on Windows (cp1252) as well as Linux/macOS.

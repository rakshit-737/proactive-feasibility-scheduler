# Methodology (Phases 01–21)

## Core system
- Discrete-time cluster simulation with heterogeneous queue states.
- Wait-time regression model trained on engineered cluster-state features.
- Proactive scheduler reorders queue using predicted wait-time.

## Enhancements
1. **Ablation**: remove each of 12 features and measure R² drop.
2. **Fairness**: evaluate max wait, Gini index, completion by size, starvation count.
3. **Scheduler baselines**: FIFO, SJF, Priority, Proactive, NN.
4. **SHAP**: summary, dependence, and force plots.
5. **Real traces (scaffolded, not yet validated)**: an SWF parser exists (`02_data/load_real_traces.py`), but no real trace is committed — the evaluated trace is a **synthetic LANL-schema proxy** derived from the training distribution. Obtaining a real SWF trace (e.g. from the Parallel Workloads Archive, https://www.cs.huji.ac.il/labs/parallel/workload/) and re-evaluating is documented future work.
6. **Scaling**: 4/8/16/32 nodes (8/32/128/256 GPUs), overhead and inference analysis.
7. **Online learning**: incremental updates on streaming data.
8. **Concept drift**: rolling MAE trigger for adaptive retraining.
9. **ROI**: GPU-hour savings, energy savings, and annual cost-benefit metrics.
10. **Reproducibility and dashboard**: one-command pipeline plus interactive explorers.

## Statistical treatment
- Seeded paired runs for scheduler comparisons (40-run benchmark).
- Paired t-test, Wilcoxon signed-rank, and bootstrap 95% confidence intervals (Phase 22); Benjamini–Hochberg correction across metrics; zero-variance comparisons reported as "n/a" rather than as significance.
- Mean and max wait reporting.
- Fairness measured via per-job Gini coefficient, run-level Jain index, starvation counts, and SLA compliance (Phase 27).
- Cross-dataset (proxy) and OOD diagnostics via MAE and R² on freshly simulated shifted workloads (Phase 23).

## Reproducibility
- The entire pipeline is seeded: `bash run_all_experiments.sh` regenerates the dataset, model, and every result deterministically on a fresh checkout.
- `PYTHONUTF8=1` is exported by the pipeline scripts so Unicode console output works on Windows (cp1252) as well as Linux/macOS.

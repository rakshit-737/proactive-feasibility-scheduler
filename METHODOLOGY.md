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
5. **Real traces**: parse LANL-format workloads and evaluate cross-dataset performance.
6. **Scaling**: 4/8/16/32 nodes (8/32/128/256 GPUs), overhead and inference analysis.
7. **Online learning**: incremental updates on streaming data.
8. **Concept drift**: rolling MAE trigger for adaptive retraining.
9. **ROI**: GPU-hour savings, energy savings, and annual cost-benefit metrics.
10. **Reproducibility and dashboard**: one-command pipeline plus interactive explorers.

## Statistical treatment
- Paired runs for scheduler comparisons.
- Mean and max wait reporting.
- Fairness measured via Gini coefficient.
- Cross-dataset and OOD diagnostics via MAE and R².

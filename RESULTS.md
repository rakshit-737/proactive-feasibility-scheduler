# Results Summary (Phases 01–21)

## High-level findings
- Best model quality: **R² ≈ 0.937** on synthetic holdout.
- Proactive scheduling: **~7.5% mean wait-time reduction** vs FIFO (40-run benchmark context).
- Phase 12 ablation ranks the most critical wait-time features by R² drop.
- Phase 13 fairness analysis confirms reduced starvation with anti-starvation bumping.
- Phase 14 benchmark compares 5 schedulers using MAE, wait, fairness, and throughput.
- Phase 15 SHAP plots provide local and global explainability artifacts.
- Phase 16 synthetic-to-real validation reports cross-dataset generalization gaps.
- Phase 17 scaling confirms proactive advantages persist across larger cluster sizes.
- Phase 18 online adaptation recovers performance under workload drift.
- Phase 19 ROI estimates annual GPU-hour and cloud-cost savings.

## Artifact index
- Models: `05_results/models/*`
- Schedulers: `05_results/schedulers/*`
- Fairness: `05_results/fairness/*`
- Scaling: `05_results/scaling/*`
- SHAP: `05_results/shap/*`
- Traces: `05_results/traces/*`
- ROI: `05_results/roi/*`

## Publication-ready package
The repository now includes reproducibility scripts, container setup, structured result outputs, dashboard support, and GitHub Pages-ready documentation for end-to-end thesis regeneration.

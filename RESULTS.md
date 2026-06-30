# Results Summary (Phases 01–21)

## High-level findings
- Model quality: **R² ≈ 0.83, MAE ≈ 5.35** on a proper 20% holdout (5-fold CV MAE 5.68 ± 0.71). Note: an in-sample (full-data) evaluation reads R² ≈ 0.96 / MAE ≈ 2.56, but that overstates generalisation and should not be quoted as model quality.
- Proactive scheduling: **7.5% mean wait-time reduction** vs FIFO (40-run paired benchmark, p = 1.10e-06), at **identical GPU utilisation**.
- Trade-off: proactive reordering **worsens tail latency** (mean max-wait ~58 → 128 ts) and **fairness** (Gini 0.51 → 0.79). The mean-wait gain is not free.
- Real-trace transfer: the synthetic-trained model does **not** generalise to the LANL trace (cross-dataset R² ≈ −0.02); retraining/adaptation is required before any production claim.
- Phase 12 ablation ranks features by R² drop; job_gpu and queue_length dominate.
- Phase 13 fairness analysis quantifies the tail-latency/Gini trade-off and tests an anti-starvation bumping variant.
- Phase 14 benchmark compares 5 schedulers (FIFO, SJF, Priority, Proactive, NN); SJF wins on mean wait, FIFO wins on fairness.
- Phase 15 SHAP plots provide local and global explainability artifacts.
- Phases 16–19 cover scaling, online learning, concept drift, and an assumption-heavy ROI estimate.

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

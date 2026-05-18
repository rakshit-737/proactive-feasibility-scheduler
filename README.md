# Proactive Feasibility Scheduler

Simulation-driven proactive GPU job scheduling with ML-based wait-time prediction.

## Quick start
```bash
pip install -r requirements.txt
bash run_all_experiments.sh
```

## Key features
- 21-phase research pipeline (simulation, ML, benchmarking, robustness, explainability, ROI)
- Ablation analysis over 12 features
- Fairness and starvation-prevention analysis
- 5-scheduler benchmark (FIFO, SJF, Priority, Proactive, NN)
- Real-trace validation, scaling analysis, concept drift adaptation
- Reproducibility kit (shell script + Docker + requirements)
- Interactive dashboard (`dashboard.py`) and GitHub Pages docs (`docs/`)

## Results folders
- `05_results/models`
- `05_results/schedulers`
- `05_results/scaling`
- `05_results/fairness`
- `05_results/shap`
- `05_results/traces`
- `05_results/roi`

## Documentation
- Research progress: `research_progress.html`
- Project report: `project_report.html`
- Methodology: `METHODOLOGY.md`
- Results summary: `RESULTS.md`
- Reproducibility: `README_REPRODUCIBILITY.md`
- GitHub Pages entry: `docs/index.html`

## Citation
If you use this work, cite the repository and include the phase/version context (v3.0, phases 01–21).

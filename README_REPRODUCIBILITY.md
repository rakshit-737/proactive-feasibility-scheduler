# Reproducibility Guide

## One-click pipeline
```bash
bash run_all_experiments.sh
```
Step 0 of the script regenerates the dataset (`02_data/improved_wait_dataset.csv`) and retrains the model (`03_models/wait_model_v2.pkl`) before running any analysis, so the pipeline reproduces from a clean checkout without relying on pre-committed artifacts. Scripts resolve paths relative to their own location, so they can be run from any working directory, and the generators are seeded (global seed 42, per-run 42+i) — two runs produce identical datasets and results. The pipeline exports `PYTHONUTF8=1` so Unicode console output also works on Windows (cp1252) shells.

## Phases 22–27
```bash
bash phases_22_30/run_all_experiments_v2.sh
```
Runs the statistical-rigor, OOD, scheduler-landscape, trace, scaling, and fairness/SLA phases against the regenerated artifacts.

## Local setup
```bash
pip install -r requirements.txt
```

## Docker setup
```bash
docker build -t proactive-scheduler .
docker run --rm -it proactive-scheduler
```

## Dashboard
```bash
streamlit run dashboard.py
```

Generated artifacts are written into `05_results/models`, `05_results/schedulers`, `05_results/scaling`, `05_results/fairness`, `05_results/shap`, `05_results/traces`, and `05_results/roi`.

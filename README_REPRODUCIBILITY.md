# Reproducibility Guide

## One-click pipeline
```bash
bash run_all_experiments.sh
```
Step 0 of the script regenerates the dataset (`02_data/improved_wait_dataset.csv`) and retrains the model (`03_models/wait_model_v2.pkl`) before running any analysis, so the pipeline reproduces from a clean checkout without relying on pre-committed artifacts. Scripts resolve paths relative to the project root, so they can be run from any working directory.

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

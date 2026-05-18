# Reproducibility Guide

## One-click pipeline
```bash
bash run_all_experiments.sh
```

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

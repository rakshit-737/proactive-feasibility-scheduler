# ARCHIVED (v1 pipeline): kept for provenance; superseded by the v2 data pipeline
# (02_data/generate_improved_dataset.py and downstream checks).
import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(os.path.join(PROJECT_ROOT, "02_data", "dataset.csv"))
print(df["label"].value_counts())
print(f"\nClass balance: {df['label'].mean():.2%} feasible")
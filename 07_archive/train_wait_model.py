# ARCHIVED (v1 pipeline): kept for provenance; superseded by 03_models/train_wait_model.py
# and 03_models/train_improved_model.py (wait_model_v2).
import os
import sys

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(PROJECT_ROOT, "02_data", "wait_dataset.csv"))

# This archived script was written for the obsolete 8-feature v1 schema
# (incl. 'smaller_jobs_in_queue' and 'demand_ratio'), which no longer exists
# in the current wait_dataset.csv. Fail fast instead of training on the
# wrong schema and crashing later with a cryptic IndexError.
EXPECTED_V1_FEATURES = ["job_gpu", "total_free", "queue_length", "running_jobs",
                        "max_free_node", "variance_free", "smaller_jobs_in_queue", "demand_ratio"]
actual_features = [c for c in df.columns if c != "wait_time"]
if actual_features != EXPECTED_V1_FEATURES:
    sys.exit(
        "ARCHIVED SCRIPT: 07_archive/train_wait_model.py expects the obsolete 8-feature v1 "
        "schema, but 02_data/wait_dataset.csv now has a different schema "
        f"({len(actual_features)} features: {actual_features}).\n"
        "This script is kept for provenance only; use 03_models/train_wait_model.py "
        "or 03_models/train_improved_model.py instead."
    )

X = df.drop("wait_time", axis=1)
y = df["wait_time"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

# save model
with open(os.path.join(SCRIPT_DIR, "wait_model.pkl"), "wb") as f:
    pickle.dump(model, f)

print("Model saved as wait_model.pkl")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

feature_names = list(X.columns)

importance = model.feature_importances_
sorted_idx = np.argsort(importance)[::-1]

print("\nFeature Importance:")
for i in sorted_idx:
    print(f"  {feature_names[i]}: {importance[i]:.4f}")

plt.figure(figsize=(8, 5))
plt.bar([feature_names[i] for i in sorted_idx], importance[sorted_idx])
plt.xticks(rotation=45, ha='right')
plt.title("Feature Importance for Wait Time Prediction")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "feature_importance.png"))
print("\nPlot saved as feature_importance.png")
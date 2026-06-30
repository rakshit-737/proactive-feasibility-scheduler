import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

# Resolve paths relative to the project root so the script works no matter what
# the current working directory is (previously it assumed the dataset was in
# the CWD, which broke when run from run_all_experiments.sh at project root).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load data ──────────────────────────────────────────────────
df = pd.read_csv(os.path.join(PROJECT_ROOT, "02_data", "improved_wait_dataset.csv"))
print(f"Dataset shape: {df.shape}")
print(f"Wait time stats:\n{df['wait_time'].describe()}\n")

FEATURES = [
    "job_gpu", "total_free", "queue_length", "running_jobs",
    "max_free_node", "variance_free",
    "can_fit_now", "gpu_fit_ratio", "fragmentation",
    "queue_pressure", "node_availability", "avg_free_per_node"
]

X = df[FEATURES]
y = df["wait_time"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train ────────────────────────────────────────────────────────────────────
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("=" * 45)
print(f"  MAE      : {mae:.4f}")
print(f"  R²       : {r2:.4f}")
print("=" * 45)

# 5-fold CV MAE
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"  CV MAE   : {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("=" * 45)

# ── Feature importance ───────────────────────────────────────────────────────
importance = model.feature_importances_
sorted_idx = np.argsort(importance)[::-1]

print("\nFeature Importance:")
for i in sorted_idx:
    bar = "█" * int(importance[i] * 40)
    print(f"  {FEATURES[i]:<22} {importance[i]:.4f}  {bar}")

# ── Plot feature importance ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#38bdf8' if importance[i] < 0.25 else '#fb923c' for i in sorted_idx]
ax.bar([FEATURES[i] for i in sorted_idx], importance[sorted_idx], color=colors)
ax.set_title("Feature Importance — Improved Wait-Time Model", fontsize=13)
ax.set_ylabel("Importance")
plt.xticks(rotation=40, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "feature_importance_v2.png"), dpi=150)
print("\nPlot saved: feature_importance_v2.png")

# ── Save model ────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "wait_model_v2.pkl"), "wb") as f:
    pickle.dump({"model": model, "features": FEATURES}, f)

print("Model saved : wait_model_v2.pkl")
print(f"\nFeature list for scheduler: {FEATURES}")

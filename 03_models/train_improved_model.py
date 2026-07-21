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

# Shared figure style (light/dark pair, one palette for all 46 repo figures).
# PROJECT_ROOT is derived from __file__, so this import resolves whether the
# script is run from 03_models/ or from the repo root.
import sys; sys.path.insert(0, PROJECT_ROOT)
from vizstyle import figure, finish, save_both, PALETTE, color_of, label_of, bar_ends, legend_roles

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
# Axes are labelled by feature NAME, spelled out: the raw column names are
# cluster-state shorthand, so each tick says what the signal actually measures.
FEATURE_LABEL = {
    "job_gpu":           "GPUs requested by the job",
    "total_free":        "Free GPUs, cluster-wide",
    "queue_length":      "Jobs waiting in the queue",
    "running_jobs":      "Jobs currently running",
    "max_free_node":     "Largest free block on one node",
    "variance_free":     "Variance of free GPUs per node",
    "can_fit_now":       "Can start immediately (0/1)",
    "gpu_fit_ratio":     "Free GPUs vs GPUs requested",
    "fragmentation":     "Fragmentation (SD of free GPUs/node)",
    "queue_pressure":    "Queued GPU demand vs free GPUs",
    "node_availability": "Share of nodes that fit the job",
    "avg_free_per_node": "Mean free GPUs per node",
}

# This chart is not about schedulers, so it carries exactly ONE series colour;
# importance is already encoded by bar length and by rank order, and giving the
# large bars a second hue would make colour restate the rank.
plot_order = sorted_idx[::-1]            # largest ends up at the top in barh
top_feature = sorted_idx[0]

for mode in ('light', 'dark'):
    p = PALETTE[mode]
    fig, ax = figure(mode, figsize=(9, 5.4))

    ax.barh([FEATURE_LABEL[FEATURES[i]] for i in plot_order],
            importance[plot_order], color=p['series_1'], height=0.66)
    bar_ends(ax, 'h')
    ax.set_xlabel("Share of total importance")
    ax.set_xlim(0, float(importance.max()) * 1.16)

    # Direct-label the single dominant feature only, not every bar.
    ax.text(float(importance[top_feature]) * 1.02, len(plot_order) - 1,
            f"{importance[top_feature]:.2f}", va='center', ha='left',
            fontsize=9, color=p['ink'], fontweight='semibold')

    finish(fig, mode,
           title="What the improved wait-time model actually keys on",
           subtitle=(f"XGBoost, {len(FEATURES)} cluster-state features · "
                     f"hold-out MAE {mae:.2f}, R² {r2:.3f}"),
           source="02_data/improved_wait_dataset.csv")
    fig.subplots_adjust(top=0.82)
    save_both(fig, os.path.join(MODEL_DIR, "feature_importance_v2"), mode)

print("\nPlot saved: feature_importance_v2.png")

# ── Save model ────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "wait_model_v2.pkl"), "wb") as f:
    pickle.dump({"model": model, "features": FEATURES}, f)

print("Model saved : wait_model_v2.pkl")
print(f"\nFeature list for scheduler: {FEATURES}")

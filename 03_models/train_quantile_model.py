import pandas as pd
import numpy as np
import pickle
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Resolve paths relative to the project root (same pattern as train_improved_model.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(PROJECT_ROOT, "02_data", "improved_wait_dataset.csv"))
print(f"Dataset shape: {df.shape}")

# Same 12 features and split as train_improved_model.py (wait_model_v2)
FEATURES = [
    "job_gpu", "total_free", "queue_length", "running_jobs",
    "max_free_node", "variance_free",
    "can_fit_now", "gpu_fit_ratio", "fragmentation",
    "queue_pressure", "node_availability", "avg_free_per_node"
]
QUANTILES = np.array([0.1, 0.5, 0.9])

X = df[FEATURES]
y = df["wait_time"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train quantile model ─────────────────────────────────────────────────────
# xgboost 3.2 supports a vector quantile_alpha: one model, predict -> (n, 3)
# columns ordered as [q10, q50, q90]. If the API ever changes shape, fall back
# to one model per quantile (vector_alpha=False in the saved bundle).
model = XGBRegressor(
    objective='reg:quantileerror',
    quantile_alpha=QUANTILES,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
vector_alpha = (pred.ndim == 2 and pred.shape[1] == len(QUANTILES))

if vector_alpha:
    print(f"Vector quantile_alpha OK: predict shape = {pred.shape}")
    models = None
    q_pred = pred
else:
    # Fallback: one model per quantile (kept for API robustness — not expected
    # to trigger on xgboost 3.2.0, where predict returns (n, 3)).
    print(f"Vector alpha unsupported (predict shape {pred.shape}) — training one model per quantile")
    models = []
    cols = []
    for q in QUANTILES:
        m = XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=float(q),
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        m.fit(X_train, y_train)
        models.append(m)
        cols.append(m.predict(X_test))
    q_pred = np.column_stack(cols)
    model = None

# ── Evaluate on holdout ──────────────────────────────────────────────────────
# Guard against quantile crossing (rare but possible): sort each row so that
# q10 <= q50 <= q90 before computing interval metrics.
n_crossed = int(np.sum(np.diff(q_pred, axis=1) < 0))
if n_crossed > 0:
    print(f"Note: {n_crossed} crossed quantile pairs on holdout — sorted per row to enforce monotonicity")
q_pred = np.sort(q_pred, axis=1)

q10, q50, q90 = q_pred[:, 0], q_pred[:, 1], q_pred[:, 2]
y_true = y_test.values

q50_mae = mean_absolute_error(y_true, q50)
coverage = float(np.mean((y_true >= q10) & (y_true <= q90)))
mean_width = float(np.mean(q90 - q10))

print("=" * 55)
print(f"  q50 MAE            : {q50_mae:.4f}   (point model v2 MAE ~ 4.69)")
print(f"  [q10,q90] coverage : {coverage * 100:.1f}%  (target ~80%)")
print(f"  mean interval width: {mean_width:.4f}")
print("=" * 55)

# ── Spread-fallback threshold ────────────────────────────────────────────────
# Relative spread s = (q90 - q10) / max(q50, 1). spread_tau is the 95th
# percentile of s on the in-distribution holdout: at scheduling time, ticks
# whose mean queue spread exceeds spread_tau are treated as out-of-distribution
# and fall back to FIFO ordering.
rel_spread = (q90 - q10) / np.maximum(q50, 1.0)
spread_tau = float(np.percentile(rel_spread, 95))
print(f"  relative spread    : median={np.median(rel_spread):.3f}  p95(spread_tau)={spread_tau:.3f}")

# ── Save bundle ──────────────────────────────────────────────────────────────
bundle = {
    'model': model,                 # single vector-alpha model (None if fallback used)
    'models': models,               # per-quantile model list (None if vector alpha OK)
    'vector_alpha': vector_alpha,
    'features': FEATURES,
    'quantiles': QUANTILES,
    'holdout_coverage': coverage,
    'holdout_q50_mae': float(q50_mae),
    'holdout_mean_width': mean_width,
    'spread_tau': spread_tau,
}
out_path = os.path.join(MODEL_DIR, "wait_model_quantile.pkl")
with open(out_path, "wb") as f:
    pickle.dump(bundle, f)
print(f"Model saved : {out_path}")

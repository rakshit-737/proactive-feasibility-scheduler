import os
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "02_data", "improved_wait_dataset.csv")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "05_results", "xgboost_tuning_results.csv")

FEATURES = [
    "job_gpu", "total_free", "queue_length", "running_jobs",
    "max_free_node", "variance_free", "can_fit_now", "gpu_fit_ratio",
    "fragmentation", "queue_pressure", "node_availability", "avg_free_per_node"
]

param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}


def evaluate(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return mean_absolute_error(y_test, pred), r2_score(y_test, pred)


def main():
    df = pd.read_csv(DATA_PATH)
    x = df[FEATURES]
    y = df["wait_time"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    baseline = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    baseline_mae, baseline_r2 = evaluate(baseline, x_train, x_test, y_train, y_test)

    search = GridSearchCV(
        estimator=XGBRegressor(random_state=42),
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(x_train, y_train)

    best_model = search.best_estimator_
    tuned_pred = best_model.predict(x_test)
    tuned_mae = mean_absolute_error(y_test, tuned_pred)
    tuned_r2 = r2_score(y_test, tuned_pred)

    improvement_mae = baseline_mae - tuned_mae
    improvement_r2 = tuned_r2 - baseline_r2

    rows = [
        {"model": "baseline", "mae": baseline_mae, "r2": baseline_r2},
        {"model": "tuned", "mae": tuned_mae, "r2": tuned_r2},
    ]
    result_df = pd.DataFrame(rows)
    result_df.to_csv(RESULTS_PATH, index=False)

    print("=== XGBoost Hyperparameter Tuning ===")
    print(f"Baseline MAE: {baseline_mae:.4f} | Baseline R²: {baseline_r2:.4f}")
    print(f"Tuned MAE:    {tuned_mae:.4f} | Tuned R²:    {tuned_r2:.4f}")
    print(f"ΔMAE (baseline - tuned): {improvement_mae:.4f}")
    print(f"ΔR²  (tuned - baseline): {improvement_r2:.4f}")
    print(f"Best params: {search.best_params_}")
    print(f"Saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

import os
import time
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "02_data", "improved_wait_dataset.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "05_results", "model_comparison_table1.csv")

FEATURES = [
    "job_gpu", "total_free", "queue_length", "running_jobs",
    "max_free_node", "variance_free", "can_fit_now", "gpu_fit_ratio",
    "fragmentation", "queue_pressure", "node_availability", "avg_free_per_node"
]


def evaluate_model(name, model, x_train, x_test, y_train, y_test):
    start = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - start

    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": name,
        "mae": mae,
        "r2": r2,
        "training_time_sec": train_time,
    }


def main():
    df = pd.read_csv(DATA_PATH)
    x = df[FEATURES]
    y = df["wait_time"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    models = [
        ("Linear Regression", LinearRegression()),
        ("Random Forest", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
        ("LightGBM", LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)),
        ("CatBoost", CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, random_seed=42, verbose=False)),
        ("Gradient Boosting (sklearn)", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)),
        (
            "Neural Network (MLP)",
            Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", learning_rate_init=0.001, max_iter=500, random_state=42)),
            ]),
        ),
        ("XGBoost", XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)),
    ]

    results = []
    for name, model in models:
        row = evaluate_model(name, model, x_train, x_test, y_train, y_test)
        results.append(row)
        print(f"{name:<30} MAE={row['mae']:.4f}  R²={row['r2']:.4f}  time={row['training_time_sec']:.3f}s")

    result_df = pd.DataFrame(results).sort_values("mae")
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Table 1: Multi-Model Regression Comparison ===")
    print(result_df.to_string(index=False, formatters={
        "mae": "{:.4f}".format,
        "r2": "{:.4f}".format,
        "training_time_sec": "{:.3f}".format,
    }))
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

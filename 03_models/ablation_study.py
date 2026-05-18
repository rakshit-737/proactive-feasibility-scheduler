import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, '02_data', 'improved_wait_dataset.csv')
MODEL_BUNDLE_PATH = os.path.join(PROJECT_ROOT, '03_models', 'wait_model_v2.pkl')
OUT_DIR = os.path.join(PROJECT_ROOT, '05_results', 'models')
LEGACY_OUT_DIR = os.path.join(PROJECT_ROOT, '05_results')
os.makedirs(OUT_DIR, exist_ok=True)

with open(MODEL_BUNDLE_PATH, 'rb') as f:
    bundle = pickle.load(f)
FEATURES = bundle['features']

def train_and_eval(df, features):
    x = df[features]
    y = df['wait_time']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return r2_score(y_test, pred), mean_absolute_error(y_test, pred)

def main():
    df = pd.read_csv(DATA_PATH)
    base_r2, base_mae = train_and_eval(df, FEATURES)

    rows = []
    for feat in FEATURES:
        reduced = [f for f in FEATURES if f != feat]
        r2, mae = train_and_eval(df, reduced)
        rows.append({
            'removed_feature': feat,
            'remaining_features': len(reduced),
            'baseline_r2': base_r2,
            'ablation_r2': r2,
            'r2_drop': base_r2 - r2,
            'baseline_mae': base_mae,
            'ablation_mae': mae,
            'mae_increase': mae - base_mae,
        })

    out = pd.DataFrame(rows).sort_values('r2_drop', ascending=False).reset_index(drop=True)
    out['importance_rank'] = np.arange(1, len(out) + 1)

    csv_model = os.path.join(OUT_DIR, 'ablation_study_results.csv')
    csv_legacy = os.path.join(LEGACY_OUT_DIR, 'ablation_study_results.csv')
    out.to_csv(csv_model, index=False)
    out.to_csv(csv_legacy, index=False)

    plt.figure(figsize=(10, 5.5))
    colors = ['#fb923c' if x > out['r2_drop'].median() else '#38bdf8' for x in out['r2_drop']]
    plt.bar(out['removed_feature'], out['r2_drop'], color=colors)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.ylabel('R² drop after feature removal')
    plt.title('Feature importance ranking by ablation (higher drop = more important)')
    plt.tight_layout()
    png_model = os.path.join(OUT_DIR, 'ablation_importance.png')
    png_legacy = os.path.join(LEGACY_OUT_DIR, 'ablation_importance.png')
    plt.savefig(png_model, dpi=160)
    plt.savefig(png_legacy, dpi=160)

    print('Baseline R²:', round(base_r2, 4), '| Baseline MAE:', round(base_mae, 4))
    print('Top 5 important features by ablation:')
    print(out[['importance_rank', 'removed_feature', 'r2_drop']].head(5).to_string(index=False))
    print('Saved:', csv_model)
    print('Saved:', png_model)

if __name__ == '__main__':
    main()

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, '03_models', 'wait_model_v2.pkl')
DATA_PATH = os.path.join(PROJECT_ROOT, '02_data', 'improved_wait_dataset.csv')
OUT_DIR = os.path.join(PROJECT_ROOT, '05_results', 'shap')
os.makedirs(OUT_DIR, exist_ok=True)

with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)
model = bundle['model']
FEATURES = bundle['features']

def main():
    df = pd.read_csv(DATA_PATH)
    x = df[FEATURES]
    sample = x.sample(n=min(400, len(x)), random_state=42)

    explainer = shap.Explainer(model, x)
    shap_values = explainer(sample)

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    summary_path = os.path.join(OUT_DIR, 'shap_summary.png')
    plt.tight_layout()
    plt.savefig(summary_path, dpi=160, bbox_inches='tight')
    plt.close()

    for feat in FEATURES:
        plt.figure()
        shap.dependence_plot(feat, shap_values.values, sample, feature_names=FEATURES, show=False)
        dep_path = os.path.join(OUT_DIR, f'shap_dependence_{feat}.png')
        plt.tight_layout()
        plt.savefig(dep_path, dpi=140, bbox_inches='tight')
        plt.close()

    force_indexes = [0, min(1, len(sample) - 1), min(2, len(sample) - 1)]
    base_value = float(np.array(shap_values.base_values).reshape(-1)[0])
    for idx in force_indexes:
        fig = shap.force_plot(
            base_value=base_value,
            shap_values=shap_values.values[idx],
            features=sample.iloc[idx],
            matplotlib=True,
            show=False,
            feature_names=FEATURES,
        )
        force_path = os.path.join(OUT_DIR, f'shap_force_{idx}.png')
        plt.savefig(force_path, dpi=150, bbox_inches='tight')
        plt.close()

    print('Saved SHAP outputs in', OUT_DIR)

if __name__ == '__main__':
    main()

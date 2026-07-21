import os
import pickle
import shutil
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from vizstyle import (figure, finish, save_both, PALETTE, color_of, label_of,
                      bar_ends, legend_roles)
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

    png_model = os.path.join(OUT_DIR, 'ablation_importance.png')
    png_legacy = os.path.join(LEGACY_OUT_DIR, 'ablation_importance.png')
    stem_model = png_model[:-len('.png')]
    stem_legacy = png_legacy[:-len('.png')]

    # ── figure ───────────────────────────────────────────────────────────────
    # This chart is not about schedulers, so it carries exactly ONE series
    # colour for every bar. The previous version split the bars into two hues
    # at the median r2_drop, i.e. colour encoded RANK rather than an entity --
    # the reader saw two categories that do not exist in the data.
    plot = out.iloc[::-1]                      # rank 1 ends up at the top
    ypos = np.arange(len(plot))
    top = out.iloc[0]
    rest_max = out['r2_drop'].iloc[1:].max()
    subtitle = (f"Removing {top['removed_feature']} costs {top['r2_drop']:.3f} R²; "
                f"every other feature costs at most {rest_max:.3f}.\n"
                'A value at or below zero means the feature adds nothing the '
                'remaining eleven do not already carry.')

    for mode in ('light', 'dark'):
        p = PALETTE[mode]
        fig, ax = figure(mode, figsize=(10, 6))
        ax.barh(ypos, plot['r2_drop'], height=0.68, color=p['series_1'])
        ax.axvline(0, color=p['axis'], linewidth=0.8, zorder=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels(plot['removed_feature'])   # by NAME, never by index
        ax.set_ylim(-0.7, len(plot) - 0.3)
        lo, hi = out['r2_drop'].min(), out['r2_drop'].max()
        ax.set_xlim(min(0.0, lo) - 0.02 * hi, hi * 1.09)
        ax.set_xlabel('R² lost when the feature is removed (higher = more important)')
        ax.set_ylabel('Feature removed from the wait-time model')
        bar_ends(ax, 'h')

        # Direct-label only the three ranks that carry the story, not every bar.
        pad = 0.006 * max(out['r2_drop'].max(), 1e-9)
        for rank in range(3):
            row = out.iloc[rank]
            ax.text(row['r2_drop'] + pad, len(plot) - 1 - rank,
                    f"{row['r2_drop']:.3f}", va='center', ha='left',
                    fontsize=9, color=p['ink_2'])

        fig.tight_layout(rect=(0, 0.02, 1, 0.82))
        finish(fig, mode,
               title='One feature carries the wait-time model',
               subtitle=subtitle,
               source='05_results/models/ablation_study_results.csv')
        written = save_both(fig, stem_model, mode)
        shutil.copyfile(
            written,
            f'{stem_legacy}.png' if mode == 'light' else f'{stem_legacy}-dark.png')

    print('Baseline R²:', round(base_r2, 4), '| Baseline MAE:', round(base_mae, 4))
    print('Top 5 important features by ablation:')
    print(out[['importance_rank', 'removed_feature', 'r2_drop']].head(5).to_string(index=False))
    print('Saved:', csv_model)
    print('Saved:', png_model)

if __name__ == '__main__':
    main()

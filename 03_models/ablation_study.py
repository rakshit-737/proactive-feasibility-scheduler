"""Leave-one-feature-out ablation of the wait-time model, with error bars.

Two defects in the original version of this script made its table over-readable:

(a) COLLINEARITY. Several of the twelve features are near-duplicates of one
    another, so dropping either member of a pair costs almost nothing and the
    resulting small (or negative) ablation drop is evidence about REDUNDANCY,
    not about importance. `collinearity_table` quantifies that directly --
    the pairwise Pearson matrix plus, for every feature, the R-squared obtained
    when it is regressed on the other eleven and the variance inflation factor
    1 / (1 - R-squared) implied by it.

(b) NO ERROR BARS. The original fit each ablation exactly once, on one fixed
    random 80/20 row split, so a drop of 0.002 and a drop of 0.02 were printed
    in the same typeface with no way to tell either from zero.
    `crossval_ablation` re-fits every ablation over 20 leave-one-run-out folds
    built from the `run_id` bookkeeping column -- whole simulations are held
    out, never random rows, because rows inside one run share cluster state and
    a random row split leaks that state across the boundary. The per-fold drops
    are PAIRED (same fold, full model minus ablated model), and the interval is
    a Student-t 95% interval on those paired differences. The fold scheme is
    deterministic: leave-one-run-out needs no shuffling and therefore no seed.

The single fixed-split columns are kept verbatim so the previously published
numbers stay checkable; the cross-validated columns are added beside them.
"""

import os
import pickle
import shutil
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
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

GROUP_COL = 'run_id'
REDUNDANT_R = 0.95          # |r| above this is called out as a redundant pair
CI_LEVEL = 0.95

with open(MODEL_BUNDLE_PATH, 'rb') as f:
    bundle = pickle.load(f)
FEATURES = bundle['features']


def _model():
    """The one estimator configuration used everywhere in this script."""
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )


def train_and_eval(df, features):
    """Legacy single fixed 80/20 ROW split. Kept so the published numbers stay
    reproducible -- but see `crossval_ablation`: a random row split lets two
    jobs from the same simulation land on opposite sides of the boundary."""
    x = df[features]
    y = df['wait_time']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = _model()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return r2_score(y_test, pred), mean_absolute_error(y_test, pred)


# ── (a) collinearity ─────────────────────────────────────────────────────────
def collinearity_table(df, features):
    """Pairwise Pearson matrix + per-feature R-squared-on-the-other-eleven/VIF.

    The R-squared is from an ordinary least-squares fit of one feature on the
    other eleven plus an intercept, solved with `lstsq` so an exactly singular
    design (two features that are affine transforms of each other) still
    returns a value instead of raising. VIF = 1 / (1 - R-squared), reported as
    infinity when the R-squared is numerically one.
    """
    x = df[features].to_numpy(dtype=float)
    corr = df[features].corr(method='pearson')

    rows = []
    for i, feat in enumerate(features):
        y = x[:, i]
        others = np.delete(x, i, axis=1)
        design = np.column_stack([np.ones(len(others)), others])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = float('nan') if ss_tot == 0.0 else 1.0 - float((resid ** 2).sum()) / ss_tot
        r2 = min(r2, 1.0)
        vif = float('inf') if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2)

        off = corr.loc[feat].drop(labels=[feat]).abs()
        partner = off.idxmax()
        row = {
            'feature': feat,
            'r2_on_other_11': r2,
            'vif': vif,
            'max_abs_corr_other': float(off.max()),
            'most_correlated_with': partner,
            'redundant_above_0p95': bool(off.max() > REDUNDANT_R),
        }
        for other in features:
            row[f'corr_{other}'] = float(corr.loc[feat, other])
        rows.append(row)

    out = pd.DataFrame(rows)
    pairs = []
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            r = float(corr.loc[a, b])
            if abs(r) > REDUNDANT_R:
                pairs.append((a, b, r))
    return out, pairs


# ── (b) error bars ───────────────────────────────────────────────────────────
def _t_interval(values, level=CI_LEVEL):
    """Student-t interval on the mean of `values`. Returns (mean, std, lo, hi)."""
    v = np.asarray(values, dtype=float)
    n = v.size
    mean = float(v.mean())
    if n < 2:
        return mean, 0.0, mean, mean
    std = float(v.std(ddof=1))
    half = float(stats.t.ppf(0.5 + level / 2.0, n - 1)) * std / np.sqrt(n)
    return mean, std, mean - half, mean + half


def _fold_r2(train, test, features):
    model = _model()
    model.fit(train[features], train['wait_time'])
    return float(r2_score(test['wait_time'], model.predict(test[features])))


def crossval_ablation(df, features, group_col=GROUP_COL):
    """Leave-one-run-out ablation. Returns (per_feature_dict, baseline_stats).

    Folds are whole simulations, so no cluster state is shared between the
    training and test side of a fold. Drops are paired within a fold.
    """
    groups = sorted(df[group_col].unique())
    n_folds = len(groups)
    base_r2 = np.empty(n_folds, dtype=float)
    red_r2 = {f: np.empty(n_folds, dtype=float) for f in features}

    for k, g in enumerate(groups):
        test = df[df[group_col] == g]
        train = df[df[group_col] != g]
        base_r2[k] = _fold_r2(train, test, features)
        for feat in features:
            reduced = [f for f in features if f != feat]
            red_r2[feat][k] = _fold_r2(train, test, reduced)

    stats_by_feature = {}
    for feat in features:
        r2_mean, r2_std, r2_lo, r2_hi = _t_interval(red_r2[feat])
        drops = base_r2 - red_r2[feat]          # paired, same fold on both sides
        d_mean, d_std, d_lo, d_hi = _t_interval(drops)
        stats_by_feature[feat] = {
            'cv_scheme': f'leave-one-{group_col}-out',
            'n_folds': n_folds,
            'r2_mean': r2_mean,
            'r2_std': r2_std,
            'r2_ci_low': r2_lo,
            'r2_ci_high': r2_hi,
            'drop_mean': d_mean,
            'drop_std': d_std,
            'drop_ci_low': d_lo,
            'drop_ci_high': d_hi,
            'drop_ci_spans_zero': bool(d_lo <= 0.0 <= d_hi),
        }

    b_mean, b_std, b_lo, b_hi = _t_interval(base_r2)
    baseline = {
        'baseline_r2_mean': b_mean,
        'baseline_r2_std': b_std,
        'baseline_r2_ci_low': b_lo,
        'baseline_r2_ci_high': b_hi,
        'n_folds': n_folds,
    }
    return stats_by_feature, baseline


def main():
    df = pd.read_csv(DATA_PATH)

    # ── (a) collinearity artefact ────────────────────────────────────────────
    coll, redundant_pairs = collinearity_table(df, FEATURES)
    coll_csv = os.path.join(OUT_DIR, 'feature_collinearity.csv')
    coll.to_csv(coll_csv, index=False)

    # ── legacy single-split numbers, unchanged ───────────────────────────────
    base_r2, base_mae = train_and_eval(df, FEATURES)

    # ── (b) cross-validated numbers with intervals ───────────────────────────
    cv, cv_base = crossval_ablation(df, FEATURES)

    rows = []
    for feat in FEATURES:
        reduced = [f for f in FEATURES if f != feat]
        r2, mae = train_and_eval(df, reduced)
        row = {
            'removed_feature': feat,
            'remaining_features': len(reduced),
            'baseline_r2': base_r2,
            'ablation_r2': r2,
            'r2_drop': base_r2 - r2,
            'baseline_mae': base_mae,
            'ablation_mae': mae,
            'mae_increase': mae - base_mae,
        }
        row.update(cv_base)
        row.update(cv[feat])
        rows.append(row)

    # Ranked by the CROSS-VALIDATED drop: ranking twelve features by a single
    # random split ranks them partly by that split's noise.
    out = pd.DataFrame(rows).sort_values('drop_mean', ascending=False).reset_index(drop=True)
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
    # The bars now show the mean drop over the 20 leave-one-run-out folds and
    # carry the 95% interval on that mean, so a bar whose whisker crosses the
    # zero rule is visibly one this fold count cannot separate from no effect.
    # That is a statement about RESOLUTION, not a verdict of no effect: an
    # interval that contains zero contains every other value it spans too, so
    # the subtitle reports the bound the data support and never reads the
    # crossing as proof that the feature is redundant. Failing to reject is not
    # accepting -- this repository uses TOST wherever equivalence is the claim.
    plot = out.iloc[::-1]                      # rank 1 ends up at the top
    ypos = np.arange(len(plot))
    err = np.vstack([
        (plot['drop_mean'] - plot['drop_ci_low']).to_numpy(),
        (plot['drop_ci_high'] - plot['drop_mean']).to_numpy(),
    ])
    top = out.iloc[0]
    rest_max = out['drop_mean'].iloc[1:].max()
    spans_zero = out[out['drop_ci_spans_zero'].astype(bool)]
    n_zero = len(spans_zero)
    # The largest loss still inside a zero-crossing interval. That upper bound
    # is what such an interval licenses; the null it happens to contain is not.
    zero_hi = float(spans_zero['drop_ci_high'].max()) if n_zero else 0.0
    subtitle = (f"Removing {top['removed_feature']} costs "
                f"{top['drop_mean']:.3f} R² [{top['drop_ci_low']:.3f}, "
                f"{top['drop_ci_high']:.3f}]; every other feature costs at most "
                f"{rest_max:.3f}.\n"
                f'Whiskers are 95% t intervals over {int(top["n_folds"])} '
                'leave-one-run-out folds. '
                f'{n_zero} of {len(out)} intervals cross zero, so\nthose drops are '
                'not distinguishable from zero: each is still consistent with a '
                f'loss of up to {zero_hi:.3f} R².')

    for mode in ('light', 'dark'):
        p = PALETTE[mode]
        fig, ax = figure(mode, figsize=(10, 6))
        ax.barh(ypos, plot['drop_mean'], height=0.68, color=p['series_1'],
                xerr=err, error_kw={'ecolor': p['ink_2'], 'elinewidth': 1.3,
                                    'capsize': 3, 'capthick': 1.3, 'zorder': 4})
        ax.axvline(0, color=p['axis'], linewidth=0.8, zorder=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels(plot['removed_feature'])   # by NAME, never by index
        ax.set_ylim(-0.7, len(plot) - 0.3)
        lo = float(out['drop_ci_low'].min())
        hi = float(out['drop_ci_high'].max())
        ax.set_xlim(min(0.0, lo) - 0.02 * hi, hi * 1.12)
        ax.set_xlabel('R² lost when the feature is removed '
                      '(mean of 20 held-out runs, 95% CI)')
        ax.set_ylabel('Feature removed from the wait-time model')
        bar_ends(ax, 'h')

        # Direct-label only the three ranks that carry the story, not every bar.
        pad = 0.006 * max(hi, 1e-9)
        for rank in range(3):
            row = out.iloc[rank]
            ax.text(row['drop_ci_high'] + pad, len(plot) - 1 - rank,
                    f"{row['drop_mean']:.3f}", va='center', ha='left',
                    fontsize=9, color=p['ink_2'])

        fig.tight_layout(rect=(0, 0.02, 1, 0.80))
        finish(fig, mode,
               title='One feature carries the wait-time model',
               subtitle=subtitle,
               source='05_results/models/ablation_study_results.csv')
        written = save_both(fig, stem_model, mode)
        shutil.copyfile(
            written,
            f'{stem_legacy}.png' if mode == 'light' else f'{stem_legacy}-dark.png')

    print('Single fixed split  — baseline R²:', round(base_r2, 4),
          '| MAE:', round(base_mae, 4))
    print(f'Leave-one-{GROUP_COL}-out ({cv_base["n_folds"]} folds) — baseline R²:',
          round(cv_base['baseline_r2_mean'], 4),
          f'[{cv_base["baseline_r2_ci_low"]:.4f}, {cv_base["baseline_r2_ci_high"]:.4f}]')
    print('Ablation, ranked by cross-validated drop:')
    cols = ['importance_rank', 'removed_feature', 'drop_mean', 'drop_ci_low',
            'drop_ci_high', 'drop_ci_spans_zero']
    print(out[cols].to_string(index=False))
    print(f'{int(out["drop_ci_spans_zero"].sum())} of {len(out)} feature drops are '
          'NOT distinguishable from zero.')
    print('Redundant feature pairs (|r| > %.2f):' % REDUNDANT_R)
    for a, b, r in redundant_pairs:
        print(f'  {a} ~ {b}: r = {r:.6f}')
    print('Saved:', coll_csv)
    print('Saved:', csv_model)
    print('Saved:', png_model)


if __name__ == '__main__':
    main()

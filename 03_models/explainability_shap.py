"""SHAP explanation of the wait-time model, drawn in the repository figure style.

WHAT THIS PRODUCES (unchanged set of artefacts, restyled)
---------------------------------------------------------
  05_results/shap/shap_summary.png             global beeswarm over 400 sampled rows
  05_results/shap/shap_dependence_<feature>.png  one per model feature (12)
  05_results/shap/shap_force_<i>.png           per-row contribution breakdown (3)

Each of the above is now written as a light/dark pair via vizstyle.save_both, so
the light-mode filename is exactly what it always was and `<stem>-dark.png` is new.

WHY THE PLOTS ARE HAND-DRAWN RATHER THAN shap.*_plot
----------------------------------------------------
shap's own plotting helpers hard-code a red/blue ramp, a "#333333" axis colour and
a white canvas, and shap.dependence_plot auto-picks an *interaction* feature and
spends a third hue on it. That breaks three rules of the repo's figure standard at
once (extra hues, colour that tracks the panel rather than the entity, no dark
mode). The SHAP *values* below are computed exactly as before -- same explainer,
same background, same sample, same random_state -- and only the rendering changed.
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
import shap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from vizstyle import figure, finish, save_both, bar_ends, PALETTE  # noqa: E402

MODEL_PATH = os.path.join(PROJECT_ROOT, '03_models', 'wait_model_v2.pkl')
DATA_PATH = os.path.join(PROJECT_ROOT, '02_data', 'improved_wait_dataset.csv')
OUT_DIR = os.path.join(PROJECT_ROOT, '05_results', 'shap')
os.makedirs(OUT_DIR, exist_ok=True)

SOURCE = '02_data/improved_wait_dataset.csv — 03_models/wait_model_v2.pkl'
UNIT = 'simulation time steps'

with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)
model = bundle['model']
FEATURES = bundle['features']


# ─────────────────────────────────────────────────────────────────────────────
# Presentation helpers (no statistics are produced here that reach any output)
# ─────────────────────────────────────────────────────────────────────────────

def _value_cmap(mode):
    """Sequential ramp for a CONTINUOUS quantity (a feature's own value).

    One hue plus the neutral ink: muted grey at the low end, the single series
    blue at the high end. No categorical colour ramp anywhere in this file.
    """
    p = PALETTE[mode]
    return LinearSegmentedColormap.from_list('vs_value', [p['muted'], p['series_1']])


def _norm_values(v):
    """Map a feature column onto 0..1 for the ramp, clipped at the 5th/95th
    percentile so a single outlier cannot flatten the whole row."""
    v = np.asarray(v, dtype=float)
    lo, hi = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    if hi <= lo:
        return np.full(v.shape, 0.5)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def _swarm_offsets(values, row_height=0.36, nbins=100):
    """Deterministic beeswarm offsets: points sharing an x-bin fan out symmetrically
    about the row. Purely a layout device -- no RNG, so the figure is reproducible."""
    v = np.asarray(values, dtype=float)
    n = v.size
    ys = np.zeros(n)
    if n == 0:
        return ys
    vmin, vmax = float(v.min()), float(v.max())
    if vmax <= vmin:
        return ys
    quant = np.round(nbins * (v - vmin) / (vmax - vmin)).astype(int)
    layer, last_bin = 0, None
    for i in np.argsort(quant, kind='stable'):
        if quant[i] != last_bin:
            layer, last_bin = 0, quant[i]
        ys[i] = np.ceil(layer / 2.0) * ((layer % 2) * 2 - 1)
        layer += 1
    reach = float(np.max(np.abs(ys)))
    if reach > 0:
        ys *= row_height / (reach + 1.0)
    return ys


def _fmt(v):
    v = float(v)
    return f'{int(v)}' if v == int(v) and abs(v) < 1e6 else f'{v:.3g}'


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def make_summary(shap_matrix, sample_values, features, stem):
    """Global beeswarm. Rows are ordered by mean |SHAP| (what shap.summary_plot
    ordered by too) and labelled by feature NAME, never by index."""
    order = np.argsort(np.abs(shap_matrix).mean(axis=0))       # ascending -> top row
    n_rows, n_pts = len(order), shap_matrix.shape[0]
    top = features[order[-1]]

    for mode in ('light', 'dark'):
        p = PALETTE[mode]
        fig, ax = figure(mode, figsize=(11.4, 7.2))
        cmap = _value_cmap(mode)

        ax.axvline(0, color=p['axis'], linewidth=0.9, zorder=1)
        for row, j in enumerate(order):
            s = shap_matrix[:, j]
            ax.scatter(s, row + _swarm_offsets(s), c=_norm_values(sample_values[:, j]),
                       cmap=cmap, vmin=0, vmax=1, s=13, linewidths=0, alpha=0.85,
                       zorder=3)

        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels([features[j] for j in order])
        ax.set_ylim(-0.72, n_rows - 0.28)
        ax.set_xlabel(f'SHAP value — signed change in predicted wait ({UNIT})')
        bar_ends(ax, 'h')

        sm = ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.015, aspect=26)
        cb.outline.set_visible(False)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(['low', 'high'])
        cb.ax.tick_params(length=0, labelsize=8.5, labelcolor=p['ink_2'])
        cb.set_label("the feature's own value", color=p['ink_2'], fontsize=9)

        finish(fig, mode,
               title='What moves the wait-time prediction, and in which direction',
               subtitle=f'One dot per feature per sampled cluster state '
                        f'({n_pts} rows), features ordered by mean |SHAP|. Dots right '
                        f'of the rule push the predicted wait up, dots left push it '
                        f'down;\ncolour is the feature value itself, so a colour split '
                        f'across the rule means the feature changes sign. {top} '
                        f'dominates.',
               source=SOURCE)
        fig.subplots_adjust(top=0.80, bottom=0.09, left=0.175, right=0.985)
        save_both(fig, stem, mode)


def make_dependence(shap_matrix, sample_values, features, j, stem):
    """One feature against its own SHAP value. A single series, one hue: this chart
    is not about schedulers, so it gets series_1 and nothing else."""
    feat = features[j]
    xv, yv = sample_values[:, j], shap_matrix[:, j]
    levels = np.unique(xv)
    for mode in ('light', 'dark'):
        p = PALETTE[mode]
        fig, ax = figure(mode, figsize=(7.8, 4.9))
        ax.axhline(0, color=p['axis'], linewidth=0.9, zorder=1)
        ax.scatter(xv, yv, s=22, alpha=0.6, color=p['series_1'], linewidths=0,
                   zorder=3)
        if levels.size <= 8:
            # Few distinct values: tick the values the feature actually takes
            # rather than an arbitrary 0.0/0.2/... ruler.
            ax.set_xticks(levels)
            ax.set_xticklabels([_fmt(v) for v in levels])
        ax.set_xlabel(feat)
        ax.set_ylabel(f'SHAP value for {feat} ({UNIT})')
        ax.set_axisbelow(True)
        finish(fig, mode,
               title=f'How {feat} moves the predicted wait',
               subtitle='Each dot is one sampled cluster state. Vertical spread at a '
                        'given x is what the other features do to this one.',
               source=SOURCE)
        fig.subplots_adjust(top=0.78, bottom=0.13, left=0.105, right=0.985)
        save_both(fig, stem, mode)


def make_contributions(shap_matrix, sample_values, features, base_value, idx, stem):
    """Per-row breakdown, replacing shap's force plot.

    Two hues and no more: series_1 where the feature pushed the prediction up,
    series_2 where it pushed it down. A legend is present because two series are
    drawn, and the sign is also readable from which side of the rule a bar sits.
    """
    contrib = shap_matrix[idx]
    order = np.argsort(np.abs(contrib))                # ascending -> largest on top
    prediction = base_value + float(contrib.sum())
    span = float(np.max(np.abs(contrib))) or 1.0

    for mode in ('light', 'dark'):
        p = PALETTE[mode]
        fig, ax = figure(mode, figsize=(9.8, 6.2))
        y = np.arange(len(order))
        vals = contrib[order]
        ax.barh(y, vals, height=0.66, zorder=3,
                color=[p['series_1'] if v >= 0 else p['series_2'] for v in vals])
        ax.axvline(0, color=p['axis'], linewidth=0.9, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([f'{features[j]} = {_fmt(sample_values[idx, j])}'
                            for j in order])
        ax.set_xlabel(f'contribution to this row’s predicted wait ({UNIT})')
        ax.set_xlim(min(vals.min() * 1.30, -0.08 * span),
                    max(vals.max() * 1.30, 0.08 * span))
        bar_ends(ax, 'h')

        # Direct-label the three largest contributions only.
        for row in range(len(order))[-3:]:
            v = vals[row]
            ax.text(v + (0.02 * span if v >= 0 else -0.02 * span), row,
                    f'{v:+.2f}', va='center', fontsize=8.5, color=p['ink_2'],
                    ha='left' if v >= 0 else 'right')

        ax.legend(handles=[Patch(facecolor=p['series_1'], label='pushes the wait up'),
                           Patch(facecolor=p['series_2'], label='pushes the wait down')],
                  loc='lower left' if vals[-1] < 0 else 'lower right')

        finish(fig, mode,
               title=f'Why the model predicted this wait — sampled row {idx}',
               subtitle=f'Baseline (the model’s average output over the data) '
                        f'{base_value:.2f} {UNIT}; this row’s features move it to '
                        f'{prediction:.2f}.\nBars are ordered by size of effect and '
                        f'labelled with the feature value that produced them.',
               source=SOURCE)
        fig.subplots_adjust(top=0.80, bottom=0.10, left=0.235, right=0.985)
        save_both(fig, stem, mode)


def main():
    df = pd.read_csv(DATA_PATH)
    x = df[FEATURES]
    sample = x.sample(n=min(400, len(x)), random_state=42)

    explainer = shap.Explainer(model, x)
    shap_values = explainer(sample)

    shap_matrix = np.asarray(shap_values.values, dtype=float)
    sample_values = sample.to_numpy(dtype=float)

    make_summary(shap_matrix, sample_values, FEATURES,
                 os.path.join(OUT_DIR, 'shap_summary'))

    for j, feat in enumerate(FEATURES):
        make_dependence(shap_matrix, sample_values, FEATURES, j,
                        os.path.join(OUT_DIR, f'shap_dependence_{feat}'))

    force_indexes = [0, min(1, len(sample) - 1), min(2, len(sample) - 1)]
    base_value = float(np.array(shap_values.base_values).reshape(-1)[0])
    for idx in force_indexes:
        make_contributions(shap_matrix, sample_values, FEATURES, base_value, idx,
                           os.path.join(OUT_DIR, f'shap_force_{idx}'))

    print('Saved SHAP outputs in', OUT_DIR)


if __name__ == '__main__':
    main()

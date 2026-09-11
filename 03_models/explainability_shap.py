"""SHAP explanation of the wait-time model, computed on HELD-OUT rows only.

WHAT THIS PRODUCES
------------------
  05_results/shap/shap_summary.png              global beeswarm over the explained rows
  05_results/shap/shap_dependence_<feature>.png one per model feature (12)
  05_results/shap/shap_force_<i>.png            per-row contribution breakdown (3)
  05_results/shap/shap_provenance.csv           what the explanations were computed on
  05_results/shap/shap_explained_rows.csv       the dataset row labels actually explained

Each figure is written as a light/dark pair via vizstyle.save_both, so the
light-mode filename is exactly what it always was and `<stem>-dark.png` is new.

WHY THE SPLIT IS RECONSTRUCTED HERE
-----------------------------------
`03_models/train_improved_model.py` fits on `train_test_split(X, y,
test_size=0.2, random_state=42)` over the twelve features in that order. This
file reproduces that call exactly -- same frame, same feature order, same
random_state -- and then ASSERTS the reconstruction by scoring the loaded model
on the reconstructed test split: it must reproduce the hold-out MAE the bundle
itself recorded at fit time, or the script aborts. A silently wrong reconstruction would
be worse than explaining training rows, because the artefact would then claim a
hold-out provenance it does not have.

WHAT WAS WRONG BEFORE, PRECISELY
--------------------------------
The previous version sampled 400 rows from the FULL 2200-row frame and passed
the FULL frame as the masker.

  * The SAMPLE was in fact already clean, but only by an undocumented accident:
    `X.sample(n=400, random_state=42)` is `RandomState(42).permutation(2200)[:400]`,
    and `train_test_split(..., random_state=42)` takes `permutation(2200)[:440]`
    as its test set -- the same permutation, so the 400 explained rows were the
    first 400 of the 440 test rows. Change either seed, or the row count, and
    ~80% of the explained rows become training rows (with random_state=7,
    312 of 400 do). Nothing recorded that, so nothing defended it.
  * The BACKGROUND was genuinely contaminated: shap subsampled 100 rows from all
    2200, so roughly 80 of the 100 reference rows were rows the model had fitted.
    That moved the base value the explanations are measured against.

Both are now explicit: rows are drawn from the test split by name, the masker is
the test split, and 05_results/shap/shap_provenance.csv records it.

WHY THE PROVENANCE IS MEASURED RATHER THAN DECLARED
---------------------------------------------------
A provenance row assembled from this script's own intentions ("split=test",
"n_background=len(background)") is a self-report: reintroduce the defect and the
row keeps saying what the script meant rather than what it did. So every claim in
shap_provenance.csv is now derived from the objects actually handed to shap:

  * `split`, `rows_from_training`, `rows_from_test` are computed by testing the
    explained frame's own index against the reconstructed train and test indices,
    and the run aborts unless the answer is "all test, none train".
  * `explained_index_sha256` fingerprints the row labels of that same frame, and
    the labels themselves are written to shap_explained_rows.csv. A reader can
    recompute the fingerprint from that file and check the row set against an
    independently reconstructed split. Overlap counts cannot separate a held-out
    sampler from a whole-dataset one on this dataset (see `explanation_frames`);
    the row set can, because the two draws share only 364 of their 400 rows.
  * `n_background_rows_used` is read off the masker, not off the frame handed to
    it, and `background_rows_not_in_test_split` compares the masker's own value
    matrix against the held-out split.

Two fields remain declarations and are named so: `sample_source_rows_declared`
(the size of the frame this script pointed the sampler at) and the constants at
the bottom of the row. They are kept because they are legible, not because they
are evidence.

WHY THE PLOTS ARE HAND-DRAWN RATHER THAN shap.*_plot
----------------------------------------------------
shap's own plotting helpers hard-code a red/blue ramp, a "#333333" axis colour and
a white canvas, and shap.dependence_plot auto-picks an *interaction* feature and
spends a third hue on it. That breaks three rules of the repo's figure standard at
once (extra hues, colour that tracks the panel rather than the entity, no dark
mode). Only the rendering is bespoke; the SHAP values are shap's own.
"""

import hashlib
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
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from vizstyle import figure, finish, save_both, bar_ends, PALETTE  # noqa: E402

MODEL_PATH = os.path.join(PROJECT_ROOT, '03_models', 'wait_model_v2.pkl')
DATA_PATH = os.path.join(PROJECT_ROOT, '02_data', 'improved_wait_dataset.csv')
OUT_DIR = os.path.join(PROJECT_ROOT, '05_results', 'shap')
PROVENANCE_PATH = os.path.join(OUT_DIR, 'shap_provenance.csv')
EXPLAINED_INDEX_PATH = os.path.join(OUT_DIR, 'shap_explained_rows.csv')
os.makedirs(OUT_DIR, exist_ok=True)

SOURCE = ('02_data/improved_wait_dataset.csv — 03_models/wait_model_v2.pkl '
          '— held-out test split only')
UNIT = 'simulation time steps'

with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)
model = bundle['model']
FEATURES = bundle['features']

# --- the split the model was trained under -----------------------------------
# Read from the model bundle, not copied into this source. The bundle records the
# split it was fitted under and the hold-out score it earned there, so the
# reconstruction below is checked against THIS model rather than against a
# constant that was true of some earlier model on some particular machine.
#
# Why that distinction is load-bearing: XGBoost's histogram build reduces in
# parallel, so refitting on a machine with a different core count (or a different
# library build) yields a slightly different model -- measured here at MAE 4.6927
# / 4.6699 / 4.6373 / 4.6935 on 1 / 2 / 4 / 16 threads, same data, same seed --
# and a 2-vCPU Linux runner produced 4.7314. Checked against a constant, every
# one of those reads as "the split is wrong", which is a false alarm that hides
# the true one. Checked against the bundle, a genuine split mismatch still fails
# -- the model would be scoring rows it was fitted on, and the MAE would collapse
# far outside this tolerance -- while an honestly refitted model passes.
TARGET = 'wait_time'

# A bundle without this metadata predates the change and cannot be checked; that
# is a broken checkout, not a reason to skip the check.
if 'split' not in bundle or 'holdout' not in bundle:
    raise SystemExit(
        'wait_model_v2.pkl carries no split/holdout metadata. Re-run '
        '03_models/train_improved_model.py to regenerate it.')
TEST_SIZE = float(bundle['split']['test_size'])
SPLIT_RANDOM_STATE = int(bundle['split']['random_state'])
EXPECTED_TEST_MAE = float(bundle['holdout']['mae'])
MAE_TOLERANCE = 5e-4          # half a unit in the last published decimal

# The test split holds 440 rows. We explain 400 of them -- the same count the
# figures have always carried, so the beeswarm density and every dependence
# panel stay visually comparable with the previously published versions -- and
# use ALL 440 as the reference distribution, since a background is cheap to
# widen and a wider one is a better estimate of E[f(X)] on held-out data.
N_EXPLAIN = 400
EXPLAIN_RANDOM_STATE = 42



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
               subtitle=f'One dot per feature per held-out cluster state '
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
               subtitle='Each dot is one held-out cluster state. Vertical spread at a '
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
               title=f'Why the model predicted this wait — held-out row {idx}',
               subtitle=f'Baseline (the model’s average output over the held-out '
                        f'background) '
                        f'{base_value:.2f} {UNIT}; this row’s features move it to '
                        f'{prediction:.2f}.\nBars are ordered by size of effect and '
                        f'labelled with the feature value that produced them.',
               source=SOURCE)
        fig.subplots_adjust(top=0.80, bottom=0.10, left=0.235, right=0.985)
        save_both(fig, stem, mode)


# ─────────────────────────────────────────────────────────────────────────────
# Measurements. Everything the provenance file claims is produced here, from the
# frames actually handed to shap -- never from what this script meant to do.
# ─────────────────────────────────────────────────────────────────────────────

def index_fingerprint(index):
    """sha256 over the sorted row labels present in `index`.

    Order-independent, so it identifies the SET of dataset rows that was
    explained and nothing else. This is the field an auditor can recompute:
    from shap_explained_rows.csv, and from their own reconstruction of the
    split. See `explanation_frames` for why a count of training rows cannot
    stand in for it on this dataset.
    """
    labels = sorted(int(i) for i in index)
    payload = ','.join(str(i) for i in labels).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def classify_rows(index, train_index, test_index):
    """Where a frame's rows actually sit. Returns (n_train, n_test, label).

    The label is a measurement, not a declaration: 'test' is returned only when
    every row is in the reconstructed test index and none is in the train index.
    'unknown' means rows were found in neither, i.e. they are not rows of the
    frame the split was reconstructed from at all.
    """
    idx = pd.Index(index)
    n = len(idx)
    n_train = int(idx.isin(train_index).sum())
    n_test = int(idx.isin(test_index).sum())
    if n == 0:
        label = 'empty'
    elif n_test == n and n_train == 0:
        label = 'test'
    elif n_train == n and n_test == 0:
        label = 'train'
    elif n_train and n_test:
        label = 'mixed'
    else:
        label = 'unknown'
    return n_train, n_test, label


def rows_absent_from(matrix, reference):
    """How many rows of `matrix` carry a feature vector found nowhere in `reference`.

    Value-level rather than label-level, because `shap.maskers.Independent`
    stores a bare ndarray: the background rows shap uses have no row labels left
    to check. Honest limit: the dataset holds 33 duplicated feature vectors, so a
    training row that happens to duplicate a held-out one would pass this check.
    It detects a background drawn from a wider frame; it does not prove
    label-level provenance, which is what `background_rows_from_training` (an
    index test on the frame supplied to the masker) is for.
    """
    ref = {tuple(row) for row in np.asarray(reference, dtype=float)}
    return int(sum(tuple(row) not in ref for row in np.asarray(matrix, dtype=float)))


def reconstruct_split():
    """Rebuild the exact split `train_improved_model.py` fitted under, and prove it.

    Returns (x_train, x_test, y_test, test_mae, test_r2).

    The proof is the model's own MAE on the reconstructed test split: if the
    frame, the feature order, the test fraction or the seed differed by anything
    at all, the loaded model would score differently and we abort. "Held out"
    has to be checkable, not asserted.
    """
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f'dataset is missing required columns: {missing}')

    # run_id and arrival_time are bookkeeping, NOT features: selecting by
    # FEATURES (the model bundle's own list, in its own order) keeps them out.
    x = df[FEATURES]
    y = df[TARGET]
    x_train, x_test, _, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=SPLIT_RANDOM_STATE)

    pred = model.predict(x_test)
    test_mae = float(mean_absolute_error(y_test, pred))
    test_r2 = float(r2_score(y_test, pred))

    if abs(test_mae - EXPECTED_TEST_MAE) > MAE_TOLERANCE:
        raise SystemExit(
            'SPLIT RECONSTRUCTION FAILED.\n'
            f'  model MAE on the reconstructed test split : {test_mae:.6f}\n'
            f'  hold-out MAE recorded in the bundle       : {EXPECTED_TEST_MAE:.6f}\n'
            f'  tolerance                                 : {MAE_TOLERANCE:g}\n'
            'The rows this script would call "held out" are therefore not the rows '
            'the model was held out from, so every SHAP value below would carry a '
            'false provenance. Refusing to write it. The split parameters and the '
            'expected score both come from the bundle, so this is not a '
            'thread-count or platform difference: either the dataset changed under '
            'the model, or train_improved_model.py no longer fits the split it '
            'records. Regenerate the model and the dataset together.')

    return x_train, x_test, y_test, test_mae, test_r2


def explanation_frames(x_test):
    """The rows to explain and the reference distribution the SHAP values are
    measured against. Both come from ONE frame, named once, here.

    Returns (source, sample, background).

    `source` is returned, and its length recorded as
    `sample_source_rows_declared`, because "no explained row is a training row"
    is a weaker guarantee than it looks on this dataset.
    `X.sample(n=400, random_state=42)` over the full 2200 rows is
    `RandomState(42).permutation(2200)[:400]`, and `train_test_split(...,
    random_state=42)` takes `permutation(2200)[:440]` as its test set -- so
    sampling the FULL frame with this seed happens to return 400 held-out rows
    and an overlap count of zero. The overlap count therefore cannot, by itself,
    tell a held-out sampler from a whole-dataset one. The size of the frame the
    sampler was pointed at can, and so can the identity of the rows it returned:
    the two draws share only 364 of their 400 rows, so `index_fingerprint` of the
    explained frame separates them even though the overlap count does not. The
    frame size is written down as a declaration; the fingerprint is the evidence.
    """
    source = x_test
    n = min(N_EXPLAIN, len(source))
    sample = source.sample(n=n, random_state=EXPLAIN_RANDOM_STATE)
    background = source
    return source, sample, background


def write_explained_index(sample):
    """Write the row labels of the rows actually explained, and fingerprint them.

    Line k+1 of the file is the dataset row the figures call "held-out row k", so
    shap_force_0.png can be traced back to a row of
    02_data/improved_wait_dataset.csv. Draw order is preserved here; the
    fingerprint is taken over the sorted labels, so it identifies the row set
    independently of the order.
    """
    labels = [int(i) for i in sample.index]
    pd.DataFrame({'row_index': labels}).to_csv(EXPLAINED_INDEX_PATH, index=False)
    return index_fingerprint(sample.index)


def write_provenance(*, n_explained, explained_index_sha256, split,
                     rows_from_training, rows_from_test,
                     sample_source_rows_declared, n_background_rows_supplied,
                     n_background_rows_used, background_rows_from_training,
                     background_rows_not_in_test_split, test_split_size,
                     test_mae, test_r2, base_value):
    """Record what the explanations were computed on.

    An explanation that cannot say what it was computed on is not evidence, so
    this file is part of the result, not a log. Keyword-only because a mis-wired
    call site would otherwise quietly write a number under the wrong name, which
    is exactly the failure this file exists to prevent.

    Two background counts, deliberately:
      * `n_background_rows_supplied` -- rows in the frame handed to the masker.
      * `n_background_rows_used`     -- rows the masker actually holds, read back
        off it. `shap.maskers.Independent` subsamples to `max_samples` (100 by
        default) and only warns, so the supplied count alone can stay at 440
        while the reference distribution is a hundredth-size sample of it. A gap
        between the two names that happening instead of hiding it.
    """
    row = {
        'n_explained': int(n_explained),
        'explained_index_sha256': str(explained_index_sha256),
        'split': str(split),
        'rows_from_training': int(rows_from_training),
        'rows_from_test': int(rows_from_test),
        'sample_source_rows_declared': int(sample_source_rows_declared),
        'n_background_rows_supplied': int(n_background_rows_supplied),
        'n_background_rows_used': int(n_background_rows_used),
        'background_rows_from_training': int(background_rows_from_training),
        'background_rows_not_in_test_split': int(background_rows_not_in_test_split),
        'test_split_size': int(test_split_size),
        'model_test_mae': round(float(test_mae), 6),
        'model_test_r2': round(float(test_r2), 6),
        'expected_test_mae': float(EXPECTED_TEST_MAE),
        'shap_base_value': round(float(base_value), 6),
        'split_random_state': int(SPLIT_RANDOM_STATE),
        'explain_random_state': int(EXPLAIN_RANDOM_STATE),
        'test_size': float(TEST_SIZE),
    }
    pd.DataFrame([row]).to_csv(PROVENANCE_PATH, index=False)
    return row


def main():
    x_train, x_test, _, test_mae, test_r2 = reconstruct_split()
    source, sample, background = explanation_frames(x_test)
    n_explain = len(sample)

    # Belt and braces on top of the MAE check, and the source of the provenance
    # row: neither an explained row nor a background row may carry an index that
    # landed in the training half. Measured on the frames themselves.
    rows_from_training, rows_from_test, split_label = classify_rows(
        sample.index, x_train.index, x_test.index)
    bg_from_training, _, bg_label = classify_rows(
        background.index, x_train.index, x_test.index)
    if rows_from_training or bg_from_training or split_label != 'test' or bg_label != 'test':
        raise SystemExit(
            f'{rows_from_training} of {n_explain} explained rows (classified '
            f'"{split_label}") and {bg_from_training} of {len(background)} background '
            f'rows (classified "{bg_label}") are training rows. Explanations of '
            'memorised rows, or against a memorised reference distribution, are not '
            'explanations of behaviour. Refusing to write it.')

    # max_samples is pinned to the whole background frame because shap otherwise
    # subsamples it to 100 rows and only warns; n_background_rows_used below is
    # read back off the masker so that a subsample shows up instead of hiding.
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.Explainer(model, masker)
    shap_values = explainer(sample)

    shap_matrix = np.asarray(shap_values.values, dtype=float)
    sample_values = sample.to_numpy(dtype=float)
    base_values = np.asarray(shap_values.base_values, dtype=float).reshape(-1)

    # The recorded row labels only describe these figures if the explanation
    # matrix is row-for-row the frame whose labels we are about to write down.
    if shap_matrix.shape != (n_explain, len(FEATURES)):
        raise SystemExit(
            f'shap returned a {shap_matrix.shape} matrix for a frame of {n_explain} '
            f'rows x {len(FEATURES)} features. The explained rows can no longer be '
            'identified with the rows this script selected, so the provenance record '
            'would be unfounded. Refusing to write it.')

    # What shap actually used as its reference distribution, read off the masker.
    masker_data = np.asarray(masker.data, dtype=float)
    n_background_used = int(masker_data.shape[0])
    bg_not_in_test = rows_absent_from(masker_data, x_test)

    make_summary(shap_matrix, sample_values, FEATURES,
                 os.path.join(OUT_DIR, 'shap_summary'))

    for j, feat in enumerate(FEATURES):
        make_dependence(shap_matrix, sample_values, FEATURES, j,
                        os.path.join(OUT_DIR, f'shap_dependence_{feat}'))

    # Each explanation carries its OWN base value. The previous version read
    # base_values[0] once and reused it for all three force plots, so two of the
    # three captions stated a baseline that was not theirs -- harmless while a
    # fixed background makes every base value identical, wrong the moment it does
    # not (a per-row or clustered masker), and unverifiable either way.
    force_indexes = [i for i in (0, 1, 2) if i < len(sample)]
    for idx in force_indexes:
        make_contributions(shap_matrix, sample_values, FEATURES,
                           float(base_values[idx]), idx,
                           os.path.join(OUT_DIR, f'shap_force_{idx}'))

    fingerprint = write_explained_index(sample)
    row = write_provenance(
        n_explained=n_explain,
        explained_index_sha256=fingerprint,
        split=split_label,
        rows_from_training=rows_from_training,
        rows_from_test=rows_from_test,
        sample_source_rows_declared=len(source),
        n_background_rows_supplied=len(background),
        n_background_rows_used=n_background_used,
        background_rows_from_training=bg_from_training,
        background_rows_not_in_test_split=bg_not_in_test,
        test_split_size=len(x_test),
        test_mae=test_mae,
        test_r2=test_r2,
        base_value=base_values[0])

    order = np.argsort(np.abs(shap_matrix).mean(axis=0))[::-1]
    print('Held-out SHAP, ordered by mean |SHAP|:')
    for rank, j in enumerate(order, start=1):
        print(f'  {rank:2d}. {FEATURES[j]:<20s} {np.abs(shap_matrix[:, j]).mean():.4f}')
    print(f'\nProvenance: {row}')
    print(f'Explained row labels: {EXPLAINED_INDEX_PATH}')
    print('Saved SHAP outputs in', OUT_DIR)


if __name__ == '__main__':
    main()

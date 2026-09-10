"""The SHAP explanations must be computed on rows the model never fitted.

WHAT WAS WRONG
--------------
`03_models/explainability_shap.py` used to sample the rows it explained from the
FULL 2200-row dataset and pass that same full frame as the masker. The model was
fitted on a random 80% of it, so the reference distribution the explanations were
measured against was ~80% memorised rows.

WHY THE OBVIOUS TEST IS NOT ENOUGH
----------------------------------
"No explained row is a training row" is a weaker statement on this dataset than
it sounds, because two independent seeds collide:

    X.sample(n=400, random_state=42)              == RandomState(42).permutation(2200)[:400]
    train_test_split(X, y, 0.2, random_state=42)  -> test = permutation(2200)[:440]

Same permutation. So sampling the WHOLE dataset with this seed returns 400 rows
that all happen to sit inside the 440-row test split, and an overlap count of
zero. Point the sampler back at the full dataset and the overlap count does not
move -- with `random_state=7` it would be 312 of 400, but nobody is going to
change that seed while reintroducing the defect.

WHY THE FIRST VERSION OF THIS FILE STILL DID NOT CATCH IT
---------------------------------------------------------
It asserted `rows_from_training == 0` and `split == 'test'` out of the provenance
CSV. Both fields were written from the script's own variables, so a reverted
sampler that still returned `source = x_test` while drawing from the full frame
kept every assertion green: the CSV reported what the script MEANT. A provenance
row that reports intent is a self-report, not evidence.

So the provenance is now measured at the source -- `split` is classified by
testing the explained frame's index against the reconstructed split, and
`explained_index_sha256` fingerprints the row labels actually handed to shap --
and this file checks the MEASUREMENTS, independently:

  * the row labels themselves, in 05_results/shap/shap_explained_rows.csv, are
    re-fingerprinted here (sha256 recomputed in this file, never imported from
    the module under test) and checked row by row against a split reconstructed
    from the dataset;
  * `test_explained_rows_are_the_draw_the_provenance_declares` redraws the
    sample from the reconstructed TEST SPLIT with the recorded parameters and
    demands the same row set. This is the assertion the overlap count cannot
    make: the two draws share only 364 of their 400 rows, so pointing the
    sampler back at the full dataset moves 36 labels and fails here while
    `rows_from_training` stays at 0;
  * `n_background_rows_used` is read off the masker by the script rather than off
    the frame handed to it, so a silent `max_samples` subsample (shap's default
    is 100) shows up as a gap against `n_background_rows_supplied`;
  * the LIVE selection code is exercised directly. The artefact tests read a
    committed CSV, which a revert leaves stale and green; `test_live_*` imports
    the module, calls its selection functions and fingerprints what comes back,
    so it fails on a reverted sampler before the pipeline is re-run.

TOLERANCES
----------
MAE is compared at 1e-4 (the published figure has four decimals). The base value
is compared at 5e-2 against the model's own mean prediction over the test split:
shap's interventional tree path estimate and a plain mean differ in the third
decimal, while the contaminated background differs in the FIRST -- 0.53 away.
"""

import hashlib
import os
import pickle

import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from conftest import PROJECT_ROOT, require

PROVENANCE = '05_results/shap/shap_provenance.csv'
EXPLAINED_ROWS = '05_results/shap/shap_explained_rows.csv'
DATASET = '02_data/improved_wait_dataset.csv'
MODEL = '03_models/wait_model_v2.pkl'

# Mirrors 03_models/train_improved_model.py. Not knobs: if the trainer changes,
# these must change with it and the reconstruction below is what proves it.
TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42
PUBLISHED_TEST_MAE = 4.6935
MAE_TOL = 1e-4
BASE_VALUE_TOL = 5e-2


def fingerprint(labels):
    """sha256 over sorted row labels.

    Deliberately re-implemented here rather than imported from
    `explainability_shap`: a fingerprint checked with the fingerprinting code of
    the module under test proves only that the module agrees with itself.
    """
    ordered = sorted(int(i) for i in labels)
    return hashlib.sha256(','.join(str(i) for i in ordered).encode('utf-8')).hexdigest()


@pytest.fixture(scope='module')
def provenance():
    """The single row of 05_results/shap/shap_provenance.csv."""
    path = require(PROVENANCE, 'SHAP provenance record')
    df = pd.read_csv(path, float_precision='round_trip')
    assert len(df) == 1, (
        f'{PROVENANCE} must describe exactly one SHAP computation, found {len(df)} '
        f'rows. A provenance file with several rows cannot say which figures it '
        f'refers to.')
    return df.iloc[0]


@pytest.fixture(scope='module')
def explained_rows():
    """The dataset row labels the explanations were actually computed on."""
    path = require(EXPLAINED_ROWS, 'SHAP explained-row labels')
    df = pd.read_csv(path)
    assert list(df.columns) == ['row_index'], (
        f'{EXPLAINED_ROWS} must hold exactly one column, row_index; found '
        f'{list(df.columns)}.')
    return [int(v) for v in df['row_index']]


@pytest.fixture(scope='module')
def bundle():
    # The pickle is this repository's own committed model artefact, written by
    # 03_models/train_improved_model.py and read the same way by conftest's
    # `wait_model` fixture. It is never fetched from anywhere.
    with open(require(MODEL, 'trained wait model'), 'rb') as f:
        return pickle.load(f)


@pytest.fixture(scope='module')
def split(bundle):
    """The split reconstructed independently of the script under test."""
    df = pd.read_csv(os.path.join(PROJECT_ROOT, DATASET))
    features = bundle['features']
    return train_test_split(df[features], df['wait_time'],
                            test_size=TEST_SIZE,
                            random_state=SPLIT_RANDOM_STATE)


# ── What the artefact claims ─────────────────────────────────────────────────

def test_provenance_declares_the_held_out_split(provenance):
    """The headline claim: explained on test rows, none of them fitted.

    These fields are classifications the script made by testing the explained
    frame's index against the reconstructed split, so they are measurements --
    but they are its OWN measurements, which is why the three tests after this
    one redo them here from the row labels.
    """
    assert str(provenance['split']) == 'test', (
        f"SHAP provenance says split={provenance['split']!r}. Explanations are only "
        f"evidence about behaviour if they are computed on rows the model was not "
        f"fitted on.")
    assert int(provenance['rows_from_training']) == 0, (
        f"{int(provenance['rows_from_training'])} explained rows came from the "
        f"training split. Explanations of memorised rows are not explanations of "
        f"behaviour.")
    assert int(provenance['rows_from_test']) == int(provenance['n_explained']), (
        f"{int(provenance['rows_from_test'])} of {int(provenance['n_explained'])} "
        f'explained rows were found in the test split. Rows in neither half are not '
        f'rows of the dataset the split was reconstructed from at all.')


def test_explained_row_labels_match_the_recorded_fingerprint(provenance, explained_rows):
    """The two artefacts must describe the same computation.

    shap_explained_rows.csv is the evidence; shap_provenance.csv summarises it.
    Re-fingerprinting the labels here is what stops one of the two going stale
    against the other -- a regenerated CSV beside an old provenance row, or the
    reverse, would otherwise pass every count-based assertion in this file.
    """
    assert len(explained_rows) == len(set(explained_rows)), (
        f'{EXPLAINED_ROWS} repeats row labels; {len(explained_rows)} lines hold '
        f'{len(set(explained_rows))} distinct rows. A row explained twice is counted '
        f'twice in every mean |SHAP| in the summary figure.')
    assert len(explained_rows) == int(provenance['n_explained']), (
        f"provenance says n_explained={int(provenance['n_explained'])}, but "
        f'{EXPLAINED_ROWS} lists {len(explained_rows)} rows.')
    assert fingerprint(explained_rows) == str(provenance['explained_index_sha256']), (
        f'the sha256 of the row labels in {EXPLAINED_ROWS} is '
        f'{fingerprint(explained_rows)}, but the provenance row records '
        f"{provenance['explained_index_sha256']}. The two artefacts describe "
        f'different SHAP computations, so neither can be trusted to describe the '
        f'published figures.')


def test_no_explained_row_is_a_training_row(explained_rows, split):
    """The membership check, done here rather than taken on trust.

    The provenance row's own count is checked above; this one reconstructs the
    split from the dataset and tests the published labels against it directly.
    """
    x_train, x_test, _, _ = split
    train_labels = set(int(i) for i in x_train.index)
    test_labels = set(int(i) for i in x_test.index)

    fitted = sorted(set(explained_rows) & train_labels)
    assert not fitted, (
        f'{len(fitted)} of the {len(explained_rows)} explained rows are training '
        f'rows (first few: {fitted[:5]}). The model memorised them, so their SHAP '
        f'values describe recall rather than behaviour.')
    stray = sorted(set(explained_rows) - test_labels)
    assert not stray, (
        f'{len(stray)} explained rows (first few: {stray[:5]}) are in neither half '
        f'of the reconstructed split. They are not rows of '
        f'{DATASET} as this file reconstructs it.')


def test_explained_rows_are_the_draw_the_provenance_declares(provenance, explained_rows,
                                                             split):
    """The assertion the overlap count cannot make.

    `sample.index.isin(train.index).sum()` is 0 whether the sampler reads the
    440-row test split or all 2200 rows, because sample(random_state=42) and
    train_test_split(random_state=42) share a permutation. The row SET is not:
    drawing 400 of 2200 and drawing 400 of the 440 held-out rows agree on only
    364 labels. So redraw it here, from the split reconstructed in this file,
    with the parameters the provenance itself declares -- and demand the same
    rows back. A sampler pointed at the full dataset moves 36 labels and fails.

    Reading n and the seed from the provenance rather than hard-coding them
    keeps this test honest about what it checks: not "the sample is this fixed
    list", but "the sample is a draw of the declared size, with the declared
    seed, from the HELD-OUT SPLIT rather than from something wider".
    """
    _, x_test, _, _ = split
    seed = int(provenance['explain_random_state'])
    n = len(explained_rows)
    expected = set(int(i) for i in x_test.sample(n=n, random_state=seed).index)
    got = set(explained_rows)

    assert got == expected, (
        f'{len(got - expected)} of the {n} explained rows are not in the draw this '
        f'provenance declares. Redrawing x_test.sample(n={n}, random_state={seed}) '
        f'from the {len(x_test)}-row held-out split reproduces '
        f'{len(got & expected)} of them. The sampler was pointed at a different '
        f'frame -- on this dataset, sampling the full 2200 rows with this seed '
        f'reproduces exactly 364 of 400 and leaves rows_from_training at 0, so this '
        f'is the assertion that separates the two. (A pandas or scikit-learn upgrade '
        f'that changed either draw would also land here; regenerate the artefacts '
        f'and confirm the count.)')


def test_n_explained_fits_inside_the_test_split(provenance):
    """You cannot explain more held-out rows than there are held-out rows."""
    n_explained = int(provenance['n_explained'])
    n_test = int(provenance['test_split_size'])
    assert 0 < n_explained <= n_test, (
        f'n_explained={n_explained} against a test split of {n_test} rows. A count '
        f'above the split size means rows were drawn from somewhere else.')


def test_background_is_the_whole_test_split(provenance):
    """The masker is the reference distribution; it must be held out too, and shap
    must have actually used all of it.

    `n_background_rows_supplied` is the frame handed to the masker;
    `n_background_rows_used` is read back off the masker after construction.
    They differ whenever shap's `max_samples` subsampling fires (default 100,
    warned but not raised), which is exactly the case the supplied count alone
    cannot see.
    """
    supplied = int(provenance['n_background_rows_supplied'])
    used = int(provenance['n_background_rows_used'])
    n_test = int(provenance['test_split_size'])

    assert supplied == n_test, (
        f'{supplied} rows were handed to the masker but the test split holds '
        f'{n_test}. Equal means the background is the held-out split and nothing '
        f'else; larger means training rows are in the reference distribution.')
    assert used == supplied, (
        f'the masker holds {used} rows but {supplied} were supplied. shap subsampled '
        f'the background (its default is max_samples=100 with a warning, not an '
        f'error), so the reference distribution is not the frame this provenance '
        f'names.')
    assert int(provenance['background_rows_from_training']) == 0, (
        f"{int(provenance['background_rows_from_training'])} background rows are "
        f'training rows. SHAP values are differences against E[f(X)] over the '
        f'background, so a memorised background moves every number in every figure.')
    assert int(provenance['background_rows_not_in_test_split']) == 0, (
        f"{int(provenance['background_rows_not_in_test_split'])} of the rows the "
        f'masker actually holds carry a feature vector that appears nowhere in the '
        f'held-out split, so the background shap used is not the frame that was '
        f'supplied to it.')


def test_sampler_source_frame_is_declared_as_the_test_split(provenance):
    """The weaker, legible check: the script says it pointed the sampler at 440 rows.

    Named `_declared` in the artefact because that is all it is -- the script's
    own account of its input frame, which a reverted sampler can keep truthful
    while drawing from somewhere else.
    `test_explained_rows_are_the_draw_the_provenance_declares` is the measured
    version of this claim; this one only catches the naive revert.
    """
    source = int(provenance['sample_source_rows_declared'])
    n_test = int(provenance['test_split_size'])
    assert source == n_test, (
        f'the rows to explain were sampled from a {source}-row frame, but the test '
        f'split holds {n_test} rows. The sampler is reading something wider than the '
        f'held-out split -- almost certainly the full dataset, ~80% of which the '
        f'model was fitted on.')


# ── Whether the claim is true ────────────────────────────────────────────────

def test_declared_split_matches_an_independent_reconstruction(provenance, split):
    """Reconstruct the trainer's split here and compare sizes."""
    _, x_test, _, _ = split
    assert int(provenance['test_split_size']) == len(x_test), (
        f"provenance says the test split holds {int(provenance['test_split_size'])} "
        f'rows; reconstructing train_test_split(X, y, test_size={TEST_SIZE}, '
        f'random_state={SPLIT_RANDOM_STATE}) here gives {len(x_test)}.')


def test_declared_mae_is_the_model_on_that_split(provenance, bundle, split):
    """The reconstruction proof itself: a wrong split scores differently.

    Without this the provenance file could truthfully report "0 training rows"
    about a split that is not the split this model was held out from.
    """
    _, x_test, _, y_test = split
    measured = float(mean_absolute_error(y_test, bundle['model'].predict(x_test)))

    assert measured == pytest.approx(PUBLISHED_TEST_MAE, abs=MAE_TOL), (
        f'the model scores MAE {measured:.6f} on the reconstructed test split, but '
        f'the published hold-out MAE is {PUBLISHED_TEST_MAE}. Either the split '
        f'reconstruction no longer matches the trainer, or the model and the '
        f'dataset were not regenerated together -- in both cases "held out" in the '
        f'SHAP artefacts is unverified.')
    assert float(provenance['model_test_mae']) == pytest.approx(measured, abs=MAE_TOL), (
        f"provenance records model_test_mae={float(provenance['model_test_mae']):.6f}; "
        f'measured here on the same split: {measured:.6f}.')


def test_base_value_fingerprints_a_held_out_background(provenance, bundle, split):
    """E[f(X)] over the background IS the base value, so it identifies the background.

    Test split -> ~17.39. The contaminated version this replaced recorded 16.86.
    """
    _, x_test, _, _ = split
    mean_pred = float(bundle['model'].predict(x_test).mean())
    recorded = float(provenance['shap_base_value'])
    assert recorded == pytest.approx(mean_pred, abs=BASE_VALUE_TOL), (
        f'the recorded SHAP base value is {recorded:.6f}, but the model averages '
        f'{mean_pred:.6f} over the held-out split. The base value is the mean output '
        f'over the masker background, so a gap this size means the background is not '
        f'the held-out split.')


# ── Whether the live code still does it ──────────────────────────────────────

def test_live_selection_draws_only_held_out_rows(bundle, split):
    """Exercise the script's own selection, not the artefact it left behind.

    A revert leaves the committed CSVs in place and every test above green until
    the pipeline is re-run. This one imports the module and asks it directly, and
    it checks the rows that come back rather than the frame the function says it
    read them from -- `source` is a return value the defect can keep truthful.
    """
    require(MODEL, 'trained wait model')
    import explainability_shap as mod

    x_train, x_test, _, test_mae, _ = mod.reconstruct_split()
    source, sample, background = mod.explanation_frames(x_test)

    assert test_mae == pytest.approx(PUBLISHED_TEST_MAE, abs=MAE_TOL)
    assert len(source) == len(x_test), (
        f'explanation_frames sampled from a {len(source)}-row frame while the test '
        f'split holds {len(x_test)}. The sampler has been pointed back at a frame '
        f'that includes training rows.')
    assert 0 < len(sample) <= len(x_test)
    assert int(sample.index.isin(x_train.index).sum()) == 0
    assert set(background.index) == set(x_test.index), (
        'the SHAP background is no longer the held-out split; explanations would be '
        'measured against a different reference distribution than the provenance '
        'claims.')

    # The measured part: the rows returned must be the draw from the held-out
    # split, not merely rows that happen to have landed in it.
    seed = int(mod.EXPLAIN_RANDOM_STATE)
    expected = set(int(i) for i in x_test.sample(n=len(sample), random_state=seed).index)
    assert set(int(i) for i in sample.index) == expected, (
        f'explanation_frames returned rows that are not x_test.sample(n={len(sample)}, '
        f'random_state={seed}). Sampling the full dataset with this seed reproduces '
        f'364 of these 400 labels and keeps every overlap count at 0, so this is the '
        f'check that catches it.')


def test_live_selection_reproduces_the_published_fingerprint(provenance, bundle):
    """Tie the live code to the artefact the paper cites.

    If the sampler is reverted, the module still runs and still returns held-out
    rows -- but a different set of them, so the fingerprint it would write no
    longer matches the one the published figures were computed under.
    """
    require(MODEL, 'trained wait model')
    import explainability_shap as mod

    _, x_test, _, _, _ = mod.reconstruct_split()
    _, sample, _ = mod.explanation_frames(x_test)

    live = fingerprint(sample.index)
    assert live == str(provenance['explained_index_sha256']), (
        f'the live selection code fingerprints as {live}, but the published '
        f"provenance records {provenance['explained_index_sha256']}. The committed "
        f'SHAP figures were computed on a different set of rows than the code now '
        f'selects, so either the code was reverted or the artefacts are stale; '
        f're-run 03_models/explainability_shap.py and inspect the diff.')

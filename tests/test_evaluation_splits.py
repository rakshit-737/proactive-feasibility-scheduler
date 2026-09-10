"""Guards on the HONEST re-evaluation of the wait-time model (C1).

WHAT IS BEING DEFENDED
----------------------
03_models/evaluate_splits.py scores one fixed model configuration under three
train/test protocols and writes 05_results/models/evaluation_splits.csv:

  random         the published `train_test_split(test_size=0.2, random_state=42)`
                 over rows, which leaks near-duplicate rows of the same
                 simulation run across the partition;
  run_wise       GroupKFold over run_id -- no run on both sides. This is the
                 number that answers "does the model work on a cluster it has
                 not seen?", and it is the one the paper should quote;
  chronological  train on early arrivals within each run, test on late ones.

Two failure modes would silently undo that work, and each has a test here:

  1. THE SPLIT STOPS BEING HONEST. If the run-wise splitter ever stops grouping
     -- someone "simplifies" GroupKFold to KFold, drops the groups= argument,
     shuffles rows into folds -- the run-wise row would quietly become a second
     random split and the honest number would vanish while the CSV still
     carried its name. `test_run_wise_folds_never_put_a_run_on_both_sides`
     checks the SPLITTER ITSELF, not the CSV, across three group layouts
     (blocked, interleaved, permuted), because a groups-ignoring splitter looks
     leak-free on the blocked layout the real dataset happens to have and only
     gives itself away when the runs are interleaved.

  2. THE MODEL CONFIGURATION MOVES. The comparison is only about the split if
     the model is held fixed. The `random` row is therefore anchored to the
     published 0.8368 / 4.6935: any change to the estimator, the feature list
     or the dataset moves it and turns this file red.

The chronological guard is likewise a direct check of the splitter: within every
run, EVERY test arrival must be strictly later than EVERY training arrival. The
tie test is the one that matters -- 110 jobs share about 75 distinct arrival
times per run, so a `<=` where the code has a `<` would put the same arrival
instant on both sides and re-introduce exactly the leakage the split exists to
remove.

INPUTS
------
02_data/improved_wait_dataset.csv is COMMITTED, so its absence is a broken
checkout and these tests assert rather than skip. evaluation_splits.csv is
GENERATED, so `require()` skips when the pipeline has not been run.
"""

import os

import numpy as np
import pandas as pd
import pytest

import evaluate_splits as ev
from conftest import PROJECT_ROOT, require

DATASET = "02_data/improved_wait_dataset.csv"
ARTEFACT = "05_results/models/evaluation_splits.csv"

REQUIRED_COLUMNS = ["split", "n_folds", "n_train", "n_test",
                    "r2_mean", "r2_std", "mae_mean", "mae_std", "note"]
NAMED_SPLITS = ["random", "run_wise", "chronological"]

# The published anchor and the tolerance the task fixes.
PUBLISHED_R2 = 0.8368
PUBLISHED_MAE = 4.6935
ANCHOR_TOL = 1e-3


def dataset():
    """The committed feature table. Absence is a broken checkout, so assert."""
    full = os.path.join(PROJECT_ROOT, DATASET)
    assert os.path.exists(full), (
        f"committed dataset missing: {DATASET}. Restore it with "
        f'"git checkout -- {DATASET}" rather than regenerating.')
    return pd.read_csv(full, float_precision="round_trip")


def artefact():
    """The generated split-comparison table, or SKIP if the pipeline has not run."""
    return pd.read_csv(require(ARTEFACT, "honest split comparison"),
                       float_precision="round_trip")


# ---------------------------------------------------------------------------
# (a) the artefact exists and says what it claims to say
# ---------------------------------------------------------------------------

@pytest.mark.artefact
def test_artefact_carries_the_three_named_splits_with_the_agreed_schema():
    df = artefact()
    assert list(df.columns)[:len(REQUIRED_COLUMNS)] == REQUIRED_COLUMNS, (
        f"evaluation_splits.csv columns changed: {list(df.columns)}")
    missing = [s for s in NAMED_SPLITS if s not in set(df["split"])]
    assert not missing, (
        f"evaluation_splits.csv no longer reports {missing}. The three protocols are the "
        "artefact's whole purpose: dropping one hides the comparison it exists to make.")
    assert df["split"].is_unique, (
        "one row per split/predictor; duplicates make the table unreadable")
    assert (df["n_folds"] >= 1).all()
    assert (df["n_train"] > 0).all() and (df["n_test"] > 0).all()
    # A grouped protocol reported as a single fold would be the noisy single
    # hold-out the run-wise row exists to avoid.
    run_wise = df.loc[df["split"] == "run_wise"].iloc[0]
    assert run_wise["n_folds"] >= 5, (
        f"run_wise collapsed to {run_wise['n_folds']} fold(s); with only 20 runs a single "
        "grouped hold-out is too noisy to quote")


@pytest.mark.artefact
def test_random_row_reproduces_the_published_anchor():
    """0.8368 / 4.6935 -- if this moves, the model config or the data moved, not the split."""
    row = artefact().set_index("split").loc["random"]
    assert abs(row["r2_mean"] - PUBLISHED_R2) < ANCHOR_TOL, (
        f"random-split R2 is {row['r2_mean']:.6f}, published is {PUBLISHED_R2}. The split "
        "comparison is only about the split if the model configuration is held fixed.")
    assert abs(row["mae_mean"] - PUBLISHED_MAE) < ANCHOR_TOL, (
        f"random-split MAE is {row['mae_mean']:.6f}, published is {PUBLISHED_MAE}")
    assert row["n_train"] == 1760 and row["n_test"] == 440


@pytest.mark.artefact
def test_honest_splits_score_below_the_optimistic_one_and_above_a_constant():
    """The finding itself: leakage flatters, and the honest model still beats a constant.

    This is what goes red if the honest protocols are quietly replaced by the
    random one while keeping their names.
    """
    df = artefact().set_index("split")
    assert df.loc["run_wise", "r2_mean"] < df.loc["random", "r2_mean"], (
        "the grouped split scores at least as high as the leaky one -- either the grouping "
        "stopped working or the two protocols are now the same partition")
    assert df.loc["chronological", "r2_mean"] < df.loc["random", "r2_mean"]
    for split in NAMED_SPLITS:
        for kind in ("mean", "median"):
            base = f"{split}_baseline_{kind}"
            assert base in df.index, f"{base} missing: the constant baseline gives the reader scale"
            assert df.loc[split, "r2_mean"] > df.loc[base, "r2_mean"], (
                f"under the {split} protocol the model does not beat a constant predictor")
            assert df.loc[base, "r2_mean"] <= 0.0, (
                f"{base} has positive out-of-sample R2, which a fold-constant cannot have")


# ---------------------------------------------------------------------------
# (b) the run-wise splitter itself -- checked directly, never through the CSV
# ---------------------------------------------------------------------------

def _group_layouts(n_groups=20, per_group=110):
    """Three row orderings of the same 20 runs.

    `blocked` is the real dataset's layout (all of run 0, then all of run 1...).
    A splitter that ignores groups entirely still looks leak-free on it, because
    contiguous chunks happen to be whole runs -- so `interleaved` and `permuted`
    are what actually catch a groups-ignoring splitter.
    """
    blocked = np.repeat(np.arange(n_groups), per_group)
    interleaved = np.tile(np.arange(n_groups), per_group)
    rng = np.random.default_rng(0)          # test-local shuffle; not a pipeline seed
    permuted = rng.permutation(blocked)
    return {"blocked": blocked, "interleaved": interleaved, "permuted": permuted}


@pytest.mark.parametrize("layout", sorted(_group_layouts()))
def test_run_wise_folds_never_put_a_run_on_both_sides(layout):
    groups = _group_layouts()[layout]
    folds = ev.run_wise_folds(groups, n_splits=5)
    assert len(folds) == 5

    seen_test_groups = set()
    for i, (train_idx, test_idx) in enumerate(folds):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups & test_groups
        assert not overlap, (
            f"[{layout} layout, fold {i}] run(s) {sorted(overlap)} appear in BOTH the training "
            "and the test side. Rows of one run share a cluster trajectory, so this is the exact "
            "leakage the grouped protocol exists to remove -- the run-wise score is meaningless "
            "if the splitter does not group.")
        assert not (seen_test_groups & test_groups), "a run is held out by two folds"
        seen_test_groups |= test_groups
        assert not (set(train_idx) & set(test_idx)), "a row is in both sides"

    assert seen_test_groups == set(groups), "every run must be held out exactly once"


def test_run_wise_folds_never_leak_on_the_real_run_id_column():
    df = dataset()
    groups = df["run_id"].to_numpy()
    for i, (train_idx, test_idx) in enumerate(ev.run_wise_folds(groups, n_splits=5)):
        overlap = set(groups[train_idx]) & set(groups[test_idx])
        assert not overlap, f"fold {i} shares run(s) {sorted(overlap)} across the split"
        assert len(train_idx) == 1760 and len(test_idx) == 440, (
            "the 5-fold grouped split must keep the published 1760/440 sizes, so the "
            "random-vs-run_wise gap cannot be blamed on training-set size")


# ---------------------------------------------------------------------------
# (c) the chronological splitter -- strictly forward in time, within each run
# ---------------------------------------------------------------------------

def test_chronological_test_arrivals_are_strictly_later_within_each_run():
    df = dataset()
    arrival = df["arrival_time"].to_numpy()
    groups = df["run_id"].to_numpy()
    (train_idx, test_idx), = ev.chronological_folds(arrival, groups)

    assert len(train_idx) + len(test_idx) == len(df), "the split must partition every row"
    assert not (set(train_idx) & set(test_idx))

    for g in np.unique(groups):
        tr = arrival[train_idx][groups[train_idx] == g]
        te = arrival[test_idx][groups[test_idx] == g]
        assert len(tr) and len(te), f"run {g} has an empty side; the split is degenerate there"
        assert tr.max() < te.min(), (
            f"run {g}: latest training arrival {tr.max()} is not strictly before earliest test "
            f"arrival {te.min()}. Deployment means fitting on the past and predicting the "
            "future; an arrival instant on both sides is leakage, not chronology.")


def test_chronological_splitter_keeps_tied_arrival_times_on_one_side():
    """Arrival times tie heavily, so `<` vs `<=` at the cut is the whole game."""
    groups = np.repeat([0, 1], 10)
    arrival = np.tile(np.array([0, 0, 0, 0, 5, 5, 5, 5, 9, 9]), 2)
    (train_idx, test_idx), = ev.chronological_folds(arrival, groups, quantile=0.8)
    for g in (0, 1):
        tr = arrival[train_idx][groups[train_idx] == g]
        te = arrival[test_idx][groups[test_idx] == g]
        assert len(tr) and len(te)
        assert tr.max() < te.min(), (
            f"run {g}: a tied arrival time landed on both sides of the chronological cut")


def test_chronological_cut_is_per_run_not_global():
    """A global time threshold would be an arbitrary mixture of runs, not 'early vs late'."""
    groups = np.repeat([0, 1], 10)
    # run 0 lives in 0..9, run 1 in 100..109: a single global cut would put all
    # of run 0 in train and all of run 1 in test.
    arrival = np.concatenate([np.arange(10), 100 + np.arange(10)])
    (train_idx, test_idx), = ev.chronological_folds(arrival, groups, quantile=0.8)
    for g in (0, 1):
        assert (groups[train_idx] == g).any(), f"run {g} contributes no training rows"
        assert (groups[test_idx] == g).any(), f"run {g} contributes no test rows"


# ---------------------------------------------------------------------------
# (d) the thing being evaluated is the published model, unchanged
# ---------------------------------------------------------------------------

def test_model_config_is_the_published_one():
    params = ev.make_model().get_params()
    expected = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6,
                "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42}
    for key, value in expected.items():
        assert params[key] == value, (
            f"evaluate_splits trains {key}={params[key]}, but 03_models/train_improved_model.py "
            f"publishes {key}={value}. The three rows only compare SPLITS if the model is fixed.")


def test_bookkeeping_columns_are_not_model_features():
    assert "run_id" not in ev.FEATURES and "arrival_time" not in ev.FEATURES, (
        "run_id and arrival_time describe how a row was produced. Feeding either to the model "
        "would be train/serve skew: neither is knowable for a job on a live cluster.")
    assert len(ev.FEATURES) == 12
    assert set(ev.FEATURES).issubset(dataset().columns)

"""Re-evaluate the wait-time model under THREE train/test protocols, honestly.

WHY THIS SCRIPT EXISTS
----------------------
The published model score -- R2 0.8368, MAE 4.6935 -- comes from
`sklearn.train_test_split(test_size=0.2, random_state=42)` over the 2200 rows of
02_data/improved_wait_dataset.csv. Those rows are NOT independent draws. They are
20 simulation runs of 110 jobs each: rows from one run share a single cluster
trajectory, and rows adjacent in time within a run observe almost the same
cluster state (the same jobs running, the same queue, the same fragmentation).
A uniformly random row split therefore puts near-duplicate rows on BOTH sides of
the partition, and the score it produces answers "can the model interpolate
inside a trajectory it has already seen?" -- not the question an operator asks,
which is "does this model work on a cluster it has not seen?".

This script holds the MODEL CONFIGURATION fixed -- the exact XGBRegressor from
03_models/train_improved_model.py -- and varies ONLY the split, so every
difference between the numbers is attributable to the evaluation protocol.

THE THREE PROTOCOLS
-------------------
  random         the status quo. Reproduced here so the table is anchored: if
                 this row stops printing 0.8368 / 4.6935 then something other
                 than the split has moved, and the script says so loudly.
  run_wise       GroupKFold(n_splits=5) with groups=run_id. No run contributes
                 rows to both sides. THIS IS THE NUMBER THAT ANSWERS THE
                 GENERALISATION QUESTION and it is the one that should be
                 quoted as the model's accuracy.
  chronological  within each run, train on early arrivals and test on late ones.
                 This is the split that matches deployment: fit on the past,
                 predict the future.

WHY FIVE FOLDS FOR THE RUN-WISE SPLIT
-------------------------------------
With only 20 groups any single grouped hold-out is itself noisy, so the run-wise
row is a mean +/- SD over folds rather than one split. n_splits=5 puts 4 whole
runs (440 rows) in each test fold and 16 runs (1760 rows) in each training fold
-- EXACTLY the 1760/440 sizes of the published random split. That is the point:
training-set size is held constant, so the random-vs-run_wise gap cannot be
explained away as "the grouped model just saw less data". `run_wise_loro`
(leave-one-run-out, 20 folds) is reported alongside it as a sensitivity check on
that choice, and its much larger fold-to-fold SD is the direct evidence that a
single grouped hold-out would have been unquotable.

WHY THE CHRONOLOGICAL SPLIT IS PER-RUN
--------------------------------------
`arrival_time` restarts at 0 in every run (each run is an independent 0..150
simulation), so a single global time threshold would not be "early vs late" at
all -- it would be an arbitrary mixture of whole runs and partial runs. Each run
is therefore cut at ITS OWN arrival-time quantile (0.8), with train = arrivals
strictly BEFORE the cut and test = arrivals at or after it. Ties at the cut all
land in test, which keeps the guarantee that matters: within every run, every
test arrival is strictly later than every training arrival. Note this split is
still generous -- the model sees the early part of every trajectory it is tested
on -- so it isolates forward-in-time extrapolation, while run_wise isolates
unseen-trajectory generalisation. Neither dominates the other; the two together
bracket the honest range.

CONSTANT BASELINES
------------------
Each protocol also scores two constant predictors fitted on ITS OWN training
fold: the training mean and the training median. R2 of the training mean is 0 by
construction in-sample but can go NEGATIVE out of sample, and under the
chronological split it does, heavily -- late arrivals wait longer than the early
mean. The reader needs that context to tell "the model is barely better than a
constant" apart from "the model is much better than a constant on a target whose
own distribution shifts".

Output: 05_results/models/evaluation_splits.csv (one row per split/predictor).
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from xgboost import XGBRegressor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "02_data", "improved_wait_dataset.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "05_results", "models")
OUT_PATH = os.path.join(OUT_DIR, "evaluation_splits.csv")

# The 12 model features. run_id and arrival_time are BOOKKEEPING columns that
# describe how a row was produced; they are never fed to the model.
FEATURES = [
    "job_gpu", "total_free", "queue_length", "running_jobs",
    "max_free_node", "variance_free", "can_fit_now", "gpu_fit_ratio",
    "fragmentation", "queue_pressure", "node_availability", "avg_free_per_node",
]
TARGET = "wait_time"
GROUP_COL = "run_id"
TIME_COL = "arrival_time"

# The published random-split anchor, from 03_models/train_improved_model.py.
PUBLISHED_R2 = 0.8368
PUBLISHED_MAE = 4.6935
ANCHOR_TOL = 1e-3

RUN_WISE_SPLITS = 5
CHRONO_QUANTILE = 0.8
RANDOM_TEST_SIZE = 0.2
RANDOM_STATE = 42


def make_model():
    """The model configuration under test -- copied from train_improved_model.py.

    Every protocol trains THIS configuration and nothing else, so the table
    compares evaluation protocols rather than models.
    """
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )


def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET, GROUP_COL, TIME_COL] if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{path} is missing required column(s): {missing}. run_id and arrival_time are "
            "bookkeeping columns written by 02_data/generate_improved_dataset.py; without them "
            "no honest split can be built.")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Splitters. Each returns a list of (train_index, test_index) arrays over the
# ROW POSITIONS of the dataset, so they can be checked directly by a test
# without re-reading the artefact.
# ---------------------------------------------------------------------------

def random_folds(n_rows, test_size=RANDOM_TEST_SIZE, random_state=RANDOM_STATE):
    """The status quo: one uniformly random 80/20 row split.

    Splitting an index array reproduces the published partition exactly --
    train_test_split permutes positions, so it does not matter whether the
    frame or its index is handed in.

    The returned indices are deliberately NOT sorted. train_test_split emits
    them in shuffled order, and XGBoost's row subsampling (subsample=0.8) is
    order-dependent: re-sorting the same 1760 training rows moves the anchor to
    R2 0.8382 / MAE 4.6773. Preserving the order is what makes this row a
    byte-for-byte reproduction of the published fit rather than a near miss.
    """
    train_idx, test_idx = train_test_split(
        np.arange(n_rows), test_size=test_size, random_state=random_state)
    return [(train_idx, test_idx)]


def run_wise_folds(groups, n_splits=RUN_WISE_SPLITS):
    """Grouped folds: an ENTIRE run is either training or test, never both.

    This is the honest generalisation protocol. GroupKFold is deterministic and
    takes no seed, so nothing here touches the repository's seed families.
    """
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    if n_splits > n_groups:
        raise ValueError(f"n_splits={n_splits} exceeds the {n_groups} available groups")
    splitter = GroupKFold(n_splits=n_splits)
    return [(tr, te) for tr, te in splitter.split(np.zeros((len(groups), 1)), groups=groups)]


def chronological_folds(arrival, groups, quantile=CHRONO_QUANTILE):
    """One forward-in-time split, cut inside each run at its own time quantile.

    train = {arrival < cut_run}, test = {arrival >= cut_run}. Because the cut is
    a strict inequality on one side and non-strict on the other, every test
    arrival in a run is STRICTLY later than every training arrival in that run,
    even though arrival times tie (110 jobs share ~75 distinct arrival times).
    """
    arrival = np.asarray(arrival)
    groups = np.asarray(groups)
    train_mask = np.zeros(len(arrival), dtype=bool)
    for g in np.unique(groups):
        in_group = groups == g
        cut = np.quantile(arrival[in_group], quantile)
        train_mask |= in_group & (arrival < cut)
    return [(np.flatnonzero(train_mask), np.flatnonzero(~train_mask))]


# ---------------------------------------------------------------------------
# Predictors: the model under test, and the two constant baselines.
# ---------------------------------------------------------------------------

def predict_model(x_train, y_train, x_test):
    return make_model().fit(x_train, y_train).predict(x_test)


def predict_train_mean(x_train, y_train, x_test):
    return np.full(len(x_test), float(np.mean(y_train)))


def predict_train_median(x_train, y_train, x_test):
    return np.full(len(x_test), float(np.median(y_train)))


PREDICTORS = {
    "": predict_model,
    "_baseline_mean": predict_train_mean,
    "_baseline_median": predict_train_median,
}


def score_folds(x, y, folds, predictor):
    """Fit/score `predictor` on every fold; return the aggregated row fields.

    The SD is the population SD across folds (numpy default, ddof=0), matching
    the CV spread already printed by train_improved_model.py. For a single-fold
    protocol it is 0.0 by construction, not an estimate of anything.
    """
    r2s, maes, n_trains, n_tests = [], [], [], []
    for train_idx, test_idx in folds:
        pred = predictor(x[train_idx], y[train_idx], x[test_idx])
        r2s.append(r2_score(y[test_idx], pred))
        maes.append(mean_absolute_error(y[test_idx], pred))
        n_trains.append(len(train_idx))
        n_tests.append(len(test_idx))
    return {
        "n_folds": len(folds),
        "n_train": int(round(float(np.mean(n_trains)))),
        "n_test": int(round(float(np.mean(n_tests)))),
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "_per_fold_r2": [float(v) for v in r2s],
    }


NOTES = {
    "random": (
        "PUBLISHED PROTOCOL, OPTIMISTIC: train_test_split(test_size=0.2, random_state=42) over "
        "rows; the same run, and adjacent instants of it, appear on both sides"),
    "run_wise": (
        "HEADLINE: GroupKFold(n_splits=5) on run_id; 16 training runs / 4 held-out runs per fold, "
        "1760/440 rows as in the random split; no run on both sides"),
    "run_wise_loro": (
        "sensitivity check on the fold count: leave-one-run-out (20 folds, 19 training runs); "
        "larger training set, much larger fold-to-fold spread"),
    "chronological": (
        "DEPLOYMENT ORDER: within each run, train on arrivals before that run's own 0.8 "
        "arrival_time quantile, test on arrivals at or after it"),
}
BASELINE_NOTE = {
    "_baseline_mean": "constant predictor = mean(y_train) of the same fold",
    "_baseline_median": "constant predictor = median(y_train) of the same fold",
}


def build_rows(df):
    x = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()
    groups = df[GROUP_COL].to_numpy()
    arrival = df[TIME_COL].to_numpy()

    protocols = [
        ("random", random_folds(len(df))),
        ("run_wise", run_wise_folds(groups, n_splits=RUN_WISE_SPLITS)),
        ("chronological", chronological_folds(arrival, groups)),
        ("run_wise_loro", run_wise_folds(groups, n_splits=int(len(np.unique(groups))))),
    ]

    rows, per_fold = [], {}
    for name, folds in protocols:
        for suffix, predictor in PREDICTORS.items():
            stats = score_folds(x, y, folds, predictor)
            per_fold[name + suffix] = stats.pop("_per_fold_r2")
            note = NOTES[name] if not suffix else f"{BASELINE_NOTE[suffix]}, {name} folds"
            rows.append({"split": name + suffix, **stats, "note": note})
    return rows, per_fold


def check_anchor(rows):
    """Shout if the random split no longer reproduces the published numbers."""
    anchor = next(r for r in rows if r["split"] == "random")
    d_r2 = abs(anchor["r2_mean"] - PUBLISHED_R2)
    d_mae = abs(anchor["mae_mean"] - PUBLISHED_MAE)
    if d_r2 > ANCHOR_TOL or d_mae > ANCHOR_TOL:
        print("!" * 78)
        print("!! ANCHOR BROKEN: the random split no longer reproduces the published score.")
        print(f"!!   published R2 {PUBLISHED_R2:.4f} / MAE {PUBLISHED_MAE:.4f}")
        print(f"!!   measured  R2 {anchor['r2_mean']:.4f} / MAE {anchor['mae_mean']:.4f}")
        print("!! The split is not what moved -- the data, the features or the config did.")
        print("!" * 78)
        return False
    print(f"Anchor OK: random split reproduces published R2 {anchor['r2_mean']:.4f} / "
          f"MAE {anchor['mae_mean']:.4f}")
    return True


def main():
    df = load_dataset()
    print(f"Dataset: {df.shape[0]} rows, {df[GROUP_COL].nunique()} runs, "
          f"arrival_time {df[TIME_COL].min()}..{df[TIME_COL].max()} per run")

    rows, per_fold = build_rows(df)
    check_anchor(rows)

    print()
    header = f"{'split':<34}{'folds':>6}{'n_train':>9}{'n_test':>8}{'R2':>18}{'MAE':>18}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['split']:<34}{r['n_folds']:>6}{r['n_train']:>9}{r['n_test']:>8}"
              f"{r['r2_mean']:>11.4f} +-{r['r2_std']:<5.4f}"
              f"{r['mae_mean']:>11.4f} +-{r['mae_std']:<5.4f}")

    print("\nPer-fold R2, run-wise (5 folds)   : "
          + ", ".join(f"{v:.4f}" for v in per_fold["run_wise"]))
    print("Per-fold R2, leave-one-run-out    : "
          + ", ".join(f"{v:.4f}" for v in per_fold["run_wise_loro"]))

    rnd = next(r for r in rows if r["split"] == "random")
    run_wise = next(r for r in rows if r["split"] == "run_wise")
    chrono = next(r for r in rows if r["split"] == "chronological")
    print("\nREAD THIS ROW AS THE MODEL'S ACCURACY: run_wise, "
          f"R2 {run_wise['r2_mean']:.4f} +- {run_wise['r2_std']:.4f}, "
          f"MAE {run_wise['mae_mean']:.4f} +- {run_wise['mae_std']:.4f}.")
    print(f"The random split flatters it by {rnd['r2_mean'] - run_wise['r2_mean']:+.4f} R2; "
          f"predicting the future within a run costs {chrono['r2_mean'] - rnd['r2_mean']:+.4f} R2 "
          f"and {chrono['mae_mean'] - rnd['mae_mean']:+.4f} MAE.")
    print("None of this touches the ranking-degeneracy result, which is a statement about the "
          "score's FUNCTIONAL FORM (a function of requested size alone at a fixed instant) and "
          "holds whatever the score's accuracy is.")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = pd.DataFrame(rows)[
        ["split", "n_folds", "n_train", "n_test",
         "r2_mean", "r2_std", "mae_mean", "mae_std", "note"]]
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {os.path.relpath(OUT_PATH, PROJECT_ROOT)} ({len(out)} rows)")


if __name__ == "__main__":
    main()

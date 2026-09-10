"""C6 -- the ablation table must not be readable as if it had no uncertainty.

Two defects used to make 03_models/ablation_study.py over-readable:

(a) COLLINEARITY. Several of the twelve features are near-duplicates. In the
    synthetic generator (02_data/generate_improved_dataset.py, extract_features)
    `avg_free_per_node = total_free / cluster.num_nodes` with a constant node
    count, so the two are an exact affine pair (r = 1.000000); and
    `fragmentation = np.std(cluster.nodes)` while
    `variance_free = np.var(cluster.nodes)`, so one is the square root of the
    other (r = 0.950720). Dropping either member of such a pair costs almost
    nothing BECAUSE its twin still carries the information, which is a fact
    about redundancy, not about importance. 05_results/models/
    feature_collinearity.csv has to keep saying so.

(b) NO ERROR BARS. The original script fit each ablation once on one fixed
    random ROW split, so a drop of 0.002 and a drop of 0.02 were printed
    identically. The script now re-fits every ablation over 20 leave-one-run-out
    folds (whole simulations held out, never random rows) and reports a paired
    95% t interval on the drop. The FINDING is that most of the intervals cross
    zero: nine of twelve feature drops are not distinguishable from no effect at
    this fold count. That is a bound on what the measurement can resolve, NOT a
    demonstration that those features contribute nothing -- an interval that
    contains zero also contains every other value it spans.

(c) RUN-WISE FOLDS. The interval in (b) is only honest while the fold boundary
    is a whole simulation. A random ROW split would train on the run it then
    scores, and the intervals would come out too narrow. So the fold scheme is
    itself an assertion: `cv_scheme` / `n_folds` in the artefact, and the real
    splitter checked directly against a run appearing on both sides of a fold.

These tests fail if any of those repairs is silently reverted -- in particular
if the folds are collapsed back to a single fit, which destroys the interval, or
regrouped onto rows, which understates it.
"""

import importlib
import math

import pandas as pd
import pytest

ABLATION_ARTEFACT = '05_results/models/ablation_study_results.csv'
COLLINEARITY_ARTEFACT = '05_results/models/feature_collinearity.csv'
WAIT_DATASET = '02_data/improved_wait_dataset.csv'

REDUNDANT_R = 0.95

# The folds are whole simulations, keyed by this bookkeeping column, and the
# artefact has to say so in its own `cv_scheme` field.
GROUP_COL = 'run_id'
CV_SCHEME = f'leave-one-{GROUP_COL}-out'

# The twelve model features. Named here rather than read from the artefact so
# that an artefact that silently loses a feature fails instead of passing.
FEATURES = [
    'job_gpu', 'total_free', 'queue_length', 'running_jobs', 'max_free_node',
    'variance_free', 'can_fit_now', 'gpu_fit_ratio', 'fragmentation',
    'queue_pressure', 'node_availability', 'avg_free_per_node',
]


def _load(path, what):
    from conftest import require
    return pd.read_csv(require(path, what))


@pytest.fixture(scope='module')
def ablation():
    return _load(ABLATION_ARTEFACT, 'ablation study with cross-validated intervals')


@pytest.fixture(scope='module')
def collinearity():
    return _load(COLLINEARITY_ARTEFACT, 'feature collinearity / VIF table')


@pytest.fixture(scope='module')
def wait_dataset():
    return _load(WAIT_DATASET, 'wait-time dataset with run_id bookkeeping')


@pytest.fixture(scope='module')
def ablation_module():
    """The ablation script itself; `03_models` is on sys.path via conftest."""
    from conftest import require
    pytest.importorskip('xgboost', reason='ablation_study imports XGBRegressor')
    require('03_models/wait_model_v2.pkl', 'trained wait model bundle')
    return importlib.import_module('ablation_study')


# ── (a) the collinearity artefact ────────────────────────────────────────────
def test_collinearity_artefact_covers_every_model_feature(collinearity):
    assert set(collinearity['feature']) == set(FEATURES), (
        'feature_collinearity.csv must describe exactly the twelve model features')
    for col in ('r2_on_other_11', 'vif', 'max_abs_corr_other', 'most_correlated_with'):
        assert col in collinearity.columns, f'missing diagnostic column: {col}'


def test_collinearity_artefact_names_a_pair_above_r_0p95(collinearity):
    """At least one feature PAIR must be flagged above |r| = 0.95.

    Both the pairwise matrix and the summary columns have to agree, so that
    neither can be quietly dropped without turning this red.
    """
    indexed = collinearity.set_index('feature')

    pairs = []
    for i, a in enumerate(FEATURES):
        for b in FEATURES[i + 1:]:
            r = float(indexed.loc[a, f'corr_{b}'])
            assert math.isclose(r, float(indexed.loc[b, f'corr_{a}']), abs_tol=1e-12), (
                f'correlation matrix is not symmetric for {a}/{b}')
            if abs(r) > REDUNDANT_R:
                pairs.append((a, b, r))

    assert pairs, (
        'feature_collinearity.csv reports no feature pair above |r| = 0.95. The '
        'ablation drops for near-duplicate features are only interpretable if '
        'this redundancy is measured and stated.')

    flagged = set(indexed.index[indexed['redundant_above_0p95'].astype(bool)])
    assert flagged, 'redundant_above_0p95 flags nothing while the matrix does'
    for a, b, _ in pairs:
        assert a in flagged and b in flagged, (
            f'{a}/{b} exceed |r| = {REDUNDANT_R} in the matrix but are not flagged')


def test_collinearity_artefact_reports_the_exactly_affine_pair(collinearity):
    """total_free and avg_free_per_node differ by a constant node count only."""
    indexed = collinearity.set_index('feature')
    r = float(indexed.loc['total_free', 'corr_avg_free_per_node'])
    assert abs(r) > 0.999999, (
        f'total_free ~ avg_free_per_node should be an exact affine pair, got r={r}')
    for feat in ('total_free', 'avg_free_per_node'):
        vif = float(indexed.loc[feat, 'vif'])
        assert math.isinf(vif) or vif > 100.0, (
            f'{feat} is perfectly explained by the other eleven features; its VIF '
            f'should be unbounded, got {vif}')


# ── (b) the ablation intervals ───────────────────────────────────────────────
def test_ablation_artefact_keeps_the_original_single_split_columns(ablation):
    """The pre-existing columns must survive so nothing downstream breaks."""
    for col in ('removed_feature', 'remaining_features', 'baseline_r2', 'ablation_r2',
                'r2_drop', 'baseline_mae', 'ablation_mae', 'mae_increase',
                'importance_rank'):
        assert col in ablation.columns, f'ablation artefact lost column: {col}'
    assert set(ablation['removed_feature']) == set(FEATURES)


def test_ablation_artefact_has_confidence_interval_columns(ablation):
    for col in ('r2_mean', 'r2_std', 'r2_ci_low', 'r2_ci_high',
                'drop_mean', 'drop_ci_low', 'drop_ci_high', 'n_folds'):
        assert col in ablation.columns, (
            f'ablation artefact has no {col} column: the table is being published '
            'without error bars again')


def test_ablation_intervals_are_resampled_not_a_single_fit(ablation):
    """A single fit cannot produce an interval. Guard the fold count directly."""
    folds = ablation['n_folds'].astype(int)
    assert (folds >= 2).all(), (
        'every ablation row must be summarised over at least two held-out folds; '
        f'got n_folds={sorted(set(folds))}. One fit yields a point, not an interval.')
    widths = ablation['drop_ci_high'] - ablation['drop_ci_low']
    assert (widths > 0).all(), (
        'a zero-width confidence interval means the resampling collapsed to a '
        'single fit')


def test_every_interval_brackets_its_point_estimate(ablation):
    for _, row in ablation.iterrows():
        lo, mid, hi = row['drop_ci_low'], row['drop_mean'], row['drop_ci_high']
        assert lo <= mid <= hi, (
            f"{row['removed_feature']}: drop CI [{lo}, {hi}] does not bracket "
            f'drop_mean={mid}')
        r_lo, r_mid, r_hi = row['r2_ci_low'], row['r2_mean'], row['r2_ci_high']
        assert r_lo <= r_mid <= r_hi, (
            f"{row['removed_feature']}: r2 CI [{r_lo}, {r_hi}] does not bracket "
            f'r2_mean={r_mid}')


def test_at_least_one_feature_drop_is_indistinguishable_from_zero(ablation):
    """THE FINDING. Most feature drops are noise, and the table must show it.

    If every interval excluded zero, the published table would be claiming that
    all twelve features carry measurable signal -- which the run-wise resampling
    says is false.
    """
    spans = (ablation['drop_ci_low'] <= 0.0) & (ablation['drop_ci_high'] >= 0.0)
    assert spans.any(), (
        'no ablation interval spans zero. Either the resampling collapsed to a '
        'single fit (a point estimate never spans zero) or the intervals are no '
        'longer being computed from held-out runs.')
    # job_gpu is the one feature the degeneracy result depends on: its drop is
    # large and its interval must stay clear of zero.
    job_gpu = ablation[ablation['removed_feature'] == 'job_gpu'].iloc[0]
    assert job_gpu['drop_ci_low'] > 0.0, (
        'removing job_gpu must remain a distinguishable loss; the ranking-'
        'degeneracy claim rests on requested size being the only live feature')


# ── (c) the folds must stay RUN-WISE ─────────────────────────────────────────
# WHICH rows travel together is the whole content of the (b) repair. Rows inside
# one simulation share cluster state, so a random ROW split trains on the very
# run it then scores, and every interval it produces is too narrow -- the
# intervals above only mean anything while whole runs are held out. The two
# tests below fail if the study drifts back to a non-grouped scheme: the first
# reads the scheme the artefact recorded, the second drives the real splitter in
# 03_models/ablation_study.py and checks that no run_id ever lands on both sides
# of a fold.
def test_ablation_artefact_records_a_run_wise_fold_scheme(ablation, wait_dataset):
    """The artefact must name the grouped scheme and one fold per simulation."""
    assert 'cv_scheme' in ablation.columns, (
        'ablation artefact has no cv_scheme column: nothing in the published '
        'table then says whether whole runs or random rows were held out')

    schemes = set(ablation['cv_scheme'].astype(str))
    assert schemes == {CV_SCHEME}, (
        f'ablation artefact reports cv_scheme={sorted(schemes)}; it must be '
        f'{CV_SCHEME!r} on every row. A row-wise or ungrouped scheme leaks '
        'cluster state across the fold boundary and shrinks every interval.')

    assert GROUP_COL in wait_dataset.columns, (
        f'{WAIT_DATASET} has no {GROUP_COL} column, so the ablation cannot be '
        'holding out whole runs')
    n_runs = int(wait_dataset[GROUP_COL].nunique())
    folds = set(ablation['n_folds'].astype(int))
    assert folds == {n_runs}, (
        f'ablation artefact reports n_folds={sorted(folds)} but the dataset '
        f'contains {n_runs} distinct {GROUP_COL} values. Leave-one-run-out has '
        'exactly one fold per run; any other count means the folds are no '
        'longer the runs (e.g. a k-fold over shuffled rows).')


def _grouped_frame(n_runs=4, per_run=5):
    """Tiny stand-in dataset: `n_runs` simulations of `per_run` rows each."""
    rows = []
    for run in range(n_runs):
        for j in range(per_run):
            rows.append({
                'a': float(run + j),
                'b': float(j * j),
                'wait_time': float(run * per_run + j),
                GROUP_COL: run,
            })
    return pd.DataFrame(rows)


def test_crossval_ablation_never_puts_a_run_on_both_sides_of_a_fold(
        ablation_module, monkeypatch):
    """Drive the splitter directly; the estimator is replaced by a recorder.

    Nothing is fitted here -- `_fold_r2` is swapped for a probe that records the
    two sides of every fold -- so this test is about the SPLIT and nothing else.
    """
    frame = _grouped_frame()
    all_runs = set(frame[GROUP_COL])
    calls = []

    def recorder(train, test, features):
        calls.append({
            'train_runs': set(train[GROUP_COL]),
            'test_runs': set(test[GROUP_COL]),
            'train_rows': set(train.index),
            'test_rows': set(test.index),
        })
        return 0.5 + 0.01 * len(features) + 0.001 * float(min(test[GROUP_COL]))

    monkeypatch.setattr(ablation_module, '_fold_r2', recorder)
    per_feature, baseline = ablation_module.crossval_ablation(
        frame, ['a', 'b'], group_col=GROUP_COL)

    assert calls, 'crossval_ablation evaluated no folds at all'
    assert baseline['n_folds'] == len(all_runs), (
        f"crossval_ablation reports {baseline['n_folds']} folds for "
        f'{len(all_runs)} runs; leave-one-run-out means one fold per run')

    for call in calls:
        assert len(call['test_runs']) == 1, (
            f"a fold held out {len(call['test_runs'])} runs; leave-one-run-out "
            'holds out exactly one')
        shared_runs = call['train_runs'] & call['test_runs']
        assert not shared_runs, (
            f'{GROUP_COL} {sorted(shared_runs)} appears on BOTH sides of a '
            'fold. The ablation has stopped holding out whole runs, so rows '
            'sharing cluster state are trained on and scored on together and '
            'every published interval is too narrow.')
        assert not (call['train_rows'] & call['test_rows']), (
            'the same rows appear in train and test within one fold')
        assert call['train_runs'] | call['test_runs'] == all_runs, (
            'a fold dropped runs entirely instead of splitting them')
        held = next(iter(call['test_runs']))
        whole_run = set(frame.index[frame[GROUP_COL] == held])
        assert call['test_rows'] == whole_run, (
            f'the held-out side of the fold for {GROUP_COL}={held} is not the '
            'whole run: part of it stayed in training')

    held_out = {next(iter(call['test_runs'])) for call in calls}
    assert held_out == all_runs, (
        f'runs {sorted(all_runs - held_out)} were never held out')

    for feat in ('a', 'b'):
        assert per_feature[feat]['cv_scheme'] == CV_SCHEME, (
            f"crossval_ablation labelled its folds "
            f"{per_feature[feat]['cv_scheme']!r}, not {CV_SCHEME!r}")
        assert per_feature[feat]['n_folds'] == len(all_runs)

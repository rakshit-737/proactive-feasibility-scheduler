"""The Phase 23 OOD taxonomy must actually DISCRIMINATE between scenarios.

THE DEFECT THESE TESTS EXIST TO PREVENT
---------------------------------------
`phases_22_30/phase_23_sensitivity/sensitivity_ood_analysis.py` used to label
each of its 72 domain-shift scenarios via a chain of hard-coded cut-points. The
third branch fired on `mape >= 35.0`. The MINIMUM MAPE anywhere in the sweep is
54.01, so that branch fired for every scenario that reached it, and the two
branches ahead of it matched nothing at all. The committed artefact held

    failure_mode == "DISTRIBUTION_MISMATCH"   for 72 of 72 rows

— one value, zero entropy, seven of eight categories unreachable. The labels
were worth nothing, and nothing in the repository noticed, because nothing ever
asserted that the taxonomy produced more than one answer.

The replacement is a continuous severity score plus bands and labels derived
from the OBSERVED distribution (quantiles of the sweep itself, and comparisons
against the grid mean or against zero) rather than from constants that can
silently fall out of date the way 35.0 did.

WHAT EACH SECTION GUARDS
------------------------
  (a) the committed CSV carries more than one category, band and risk level,
      and ALL FOUR derived columns — severity_score, severity_band,
      failure_mode and risk_level — are reproducible from the raw r2 / MAPE /
      improvement / failure-rate columns sitting in that same file. Recomputing
      only severity, as this file once did, would have left a regression in how
      the two label columns are assigned completely invisible;
  (b) every band, mode and risk level is REACHABLE — proved by feeding the
      classifier synthetic rows engineered to land in each one;
  (c) the cut-points move with the data: a frame whose MAPE is uniformly above
      any plausible absolute threshold is still discriminated. This is the
      mutation guard. Re-introducing an absolute `mape >= 35.0` branch collapses
      that frame to a single label and turns section (c) red;
  (d) severity is oriented so larger always means worse, so the score can never
      quietly become flattering, and the mildest RISK label carries an absolute
      floor (the policy helps AND the model beats predicting the mean wait) on
      top of its within-grid quartile — a quartile is populated by construction,
      so the band alone would keep handing out LOW_RISK however bad the grid;
  (e) the twelve pre-existing columns survive, so downstream readers keep
      working.

Fast and pure: no simulation runs, no tracked artefact is written. The module
under test is loaded by path (`phases_22_30/` holds no importable package) and
importing it has no side effects — everything that writes lives behind main().
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import PROJECT_ROOT, require

CSV_PATH = 'phases_22_30/phase_23_sensitivity/ood_failure_modes.csv'
MODULE_PATH = os.path.join(PROJECT_ROOT, 'phases_22_30', 'phase_23_sensitivity',
                           'sensitivity_ood_analysis.py')

# The twelve columns the artefact carried before the taxonomy was replaced, in
# their original order. New columns are appended; none of these may be dropped
# or reordered, or a downstream reader that positions by index breaks silently.
LEGACY_COLUMNS = [
    'scenario', 'arrival_rate_multiplier', 'cluster_size', 'job_dist_type',
    'r2_score', 'mape', 'improvement_pct', 'failure_mode', 'risk_level',
    'improvement_degradation_pct', 'failure_rate_pct', 'n_samples',
]

# The stale constant that made the old taxonomy degenerate. Kept here as a
# number to assert AGAINST, never to classify with.
OLD_MAPE_CUTPOINT = 35.0


@pytest.fixture(scope='module')
def ood():
    """Phase 23's analysis module, loaded from its path."""
    pytest.importorskip('sklearn', reason='sensitivity_ood_analysis imports sklearn.metrics')
    spec = importlib.util.spec_from_file_location(
        'phase23_sensitivity_ood_under_test', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def committed():
    """The committed OOD artefact, or skip on a clone that has not run the sweep."""
    return pd.read_csv(require(CSV_PATH, 'Phase 23 OOD sweep'))


def _frame(rows):
    """Build the minimal frame `classify_scenarios` consumes.

    Each row is (mape, r2_score, improvement_pct, failure_rate_pct) — the four
    severity dimensions. Nothing else is read by the classifier.
    """
    return pd.DataFrame(
        rows, columns=['mape', 'r2_score', 'improvement_pct', 'failure_rate_pct'])


def _relabel_from_raw_columns(ood, frame):
    """Rebuild every derived column from the raw columns of `frame`.

    Composed from the primitives instead of calling `classify_scenarios`, so a
    mis-wiring inside that wrapper — feeding the R2 column where the improvement
    column belongs, say — surfaces as a mismatch against the committed artefact
    rather than being faithfully reproduced by the test that is meant to catch
    it. The tests below check the wrapper separately.
    """
    zmatrix = ood.severity_zmatrix(frame)
    severity = ood.compute_severity_score(zmatrix)
    bands = ood.assign_severity_bands(severity, ood.derive_severity_cutpoints(severity))
    modes = ood.assign_failure_modes(zmatrix)
    risks = ood.assign_risk_levels(
        bands,
        frame['improvement_pct'].to_numpy(dtype=float),
        frame['r2_score'].to_numpy(dtype=float),
    )
    return severity, bands, modes, risks


def _first_mismatch(stored, recomputed, frame):
    """Human-readable location of the first disagreement, for the failure text."""
    for i, (was, now) in enumerate(zip(stored, recomputed)):
        if was != now:
            scenario = frame['scenario'].iloc[i] if 'scenario' in frame else i
            return f'row {i} ({scenario}): stored {was!r}, recomputed {now!r}'
    return f'lengths differ: {len(stored)} stored vs {len(recomputed)} recomputed'


# ---------------------------------------------------------------------------
# (a) The committed artefact is not degenerate
# ---------------------------------------------------------------------------

def test_committed_failure_mode_is_not_constant(committed):
    """THE EXACT DEFECT: every row used to carry the same label."""
    distinct = committed['failure_mode'].nunique()
    assert distinct > 1, (
        f'failure_mode has {distinct} distinct value(s) across '
        f'{len(committed)} scenarios: '
        f'{sorted(committed["failure_mode"].unique())}. A taxonomy that returns '
        'one value carries no information — this is the Phase 23 defect '
        '(72/72 rows == "DISTRIBUTION_MISMATCH") returning.'
    )
    assert 'DISTRIBUTION_MISMATCH' not in set(committed['failure_mode']), (
        'the retired hard-coded taxonomy is back in the artefact'
    )


def test_committed_severity_band_spans_every_quartile(committed, ood):
    """Quantile bands are populated BY CONSTRUCTION, so all four must appear."""
    bands = set(committed['severity_band'])
    assert bands == set(ood.SEVERITY_BAND_LABELS), (
        f'bands present {sorted(bands)} != defined {sorted(ood.SEVERITY_BAND_LABELS)}; '
        'quartile bands over 72 scenarios cannot legitimately leave one empty'
    )
    counts = committed['severity_band'].value_counts()
    assert counts.min() >= len(committed) // len(ood.SEVERITY_BAND_LABELS) - 1, (
        f'quartile bands are lopsided: {counts.to_dict()} — the cut-points are '
        'not the quantiles of the data they claim to describe'
    )


def test_committed_risk_level_uses_all_three_levels(committed, ood):
    """The old artefact never produced LOW_RISK; the new one must reach all three."""
    levels = set(committed['risk_level'])
    assert levels == set(ood.RISK_LEVELS), (
        f'risk_level values {sorted(levels)} != {sorted(ood.RISK_LEVELS)}'
    )


def test_committed_severity_is_reproducible_from_its_own_raw_columns(committed, ood):
    """Ties the artefact to the code: a stale CSV fails here.

    Recomputing severity from the CSV's own r2/mape/improvement/failure-rate
    columns must return the stored severity_score, so nobody can hand-edit a
    band or ship a CSV written by a different classifier.
    """
    recomputed = ood.compute_severity_score(ood.severity_zmatrix(committed))
    np.testing.assert_allclose(
        recomputed, committed['severity_score'].to_numpy(dtype=float), atol=1e-9,
        err_msg='stored severity_score does not match a recomputation from the '
                'raw columns in the same file')

    cutpoints = ood.derive_severity_cutpoints(recomputed)
    expected = ood.assign_severity_bands(recomputed, cutpoints)
    assert list(committed['severity_band']) == expected, (
        'severity_band does not match the quantile bands of the severity in '
        'this same file')


def test_committed_failure_mode_is_reproducible_from_its_own_raw_columns(committed, ood):
    """The same tie-back as severity, for the label column that replaced the
    degenerate one.

    `failure_mode` is the column that used to be the constant string
    "DISTRIBUTION_MISMATCH" for 72 of 72 rows. Asserting only that it is now
    non-constant says nothing about whether the values are the RIGHT ones, so
    recompute the dominant-axis label from the four raw dimensions in this file
    and require an exact match, row for row.
    """
    _, _, modes, _ = _relabel_from_raw_columns(ood, committed)
    stored = list(committed['failure_mode'])
    assert modes == stored, (
        'stored failure_mode does not match a recomputation from the raw '
        'r2/MAPE/improvement/failure-rate columns in the same file — '
        + _first_mismatch(stored, modes, committed))

    # ...and the production entry point agrees, so a mis-wiring in
    # classify_scenarios cannot hide behind a hand-composed recomputation.
    reclassified, _ = ood.classify_scenarios(committed)
    assert list(reclassified['failure_mode']) == stored, (
        'classify_scenarios disagrees with the committed failure_mode: '
        + _first_mismatch(stored, list(reclassified['failure_mode']), committed))


def test_committed_risk_level_is_reproducible_from_its_own_raw_columns(committed, ood):
    """Ditto for risk_level, the other label column nothing used to recompute.

    risk_level is the column a reader is most likely to quote, so a silent drift
    between the code and the shipped artefact matters most here. It is derived
    from the severity band plus two sign tests on raw columns, and all of that
    is rebuilt from this file's own numbers.
    """
    _, _, _, risks = _relabel_from_raw_columns(ood, committed)
    stored = list(committed['risk_level'])
    assert risks == stored, (
        'stored risk_level does not match a recomputation from the raw columns '
        'in the same file — ' + _first_mismatch(stored, risks, committed))

    reclassified, _ = ood.classify_scenarios(committed)
    assert list(reclassified['risk_level']) == stored, (
        'classify_scenarios disagrees with the committed risk_level: '
        + _first_mismatch(stored, list(reclassified['risk_level']), committed))


def test_no_committed_low_risk_row_is_flattering(committed, ood):
    """LOW_RISK in the shipped artefact means more than "bottom quartile".

    The band half of the rule is purely relative — a quartile is populated by
    construction, so on its own it would keep labelling scenarios LOW_RISK in a
    grid where the model was anti-predictive everywhere. Every LOW_RISK row must
    therefore also clear both absolute floors: the ordering helps, and the wait
    model beats predicting the mean wait.
    """
    low = committed[committed['risk_level'] == 'LOW_RISK']
    assert not low.empty, 'no LOW_RISK rows to check — the fixture is vacuous'

    offenders = low[~((low['improvement_pct'] > 0) & (low['r2_score'] > 0))]
    assert offenders.empty, (
        'LOW_RISK is being handed to scenarios that do not clear the floor:\n'
        + offenders[['scenario', 'r2_score', 'improvement_pct',
                     'severity_band']].to_string(index=False)
        + '\nan R2 <= 0 means the wait model is worse than predicting the mean '
          'wait; calling that regime low risk is exactly the flattering label '
          'this rule exists to prevent'
    )
    assert set(low['severity_band']) == {ood.SEVERITY_BAND_LABELS[0]}, (
        f'LOW_RISK escaped the mildest band: {sorted(set(low["severity_band"]))}')

    # And the label stays honest in the absolute: it is a rank within a grid
    # whose BEST calibration is still a ~54% mean absolute percentage error.
    assert low['mape'].min() > OLD_MAPE_CUTPOINT, (
        'a LOW_RISK scenario now has MAPE below the retired 35.0 gate — the '
        'note calling every band in this grid badly calibrated needs revisiting')


def test_the_old_cutpoint_would_still_be_degenerate_on_this_data(committed):
    """Documents WHY 35.0 was worthless: nothing in the sweep is below it.

    This is the fact that made seven of eight old categories unreachable, and it
    is asserted rather than remembered so the justification stays checkable.
    """
    worst_case = committed['mape'].min()
    assert worst_case > OLD_MAPE_CUTPOINT, (
        f'minimum MAPE {worst_case:.2f} is no longer above the retired {OLD_MAPE_CUTPOINT} '
        'cut-point; the note explaining the taxonomy replacement needs updating'
    )


# ---------------------------------------------------------------------------
# (b) Every category is reachable — proved with synthetic rows
# ---------------------------------------------------------------------------

def test_every_severity_band_is_reachable(ood):
    """Synthetic rows spanning the severity range must hit all four bands."""
    rows = [(50.0 + 10.0 * i, 0.9 - 0.2 * i, 40.0 - 5.0 * i, 2.0 * i) for i in range(8)]
    classified, cutpoints = ood.classify_scenarios(_frame(rows))
    assert set(classified['severity_band']) == set(ood.SEVERITY_BAND_LABELS), (
        f'unreachable band(s): {set(ood.SEVERITY_BAND_LABELS) - set(classified["severity_band"])}'
    )
    assert len(cutpoints) == len(ood.SEVERITY_BAND_LABELS) - 1
    assert np.all(np.diff(cutpoints) >= 0), 'band cut-points are not monotonic'


def test_every_failure_mode_is_reachable(ood):
    """One row engineered to be dominated by each axis, plus an all-better row.

    Row 0 is the neutral bulk. Rows 1-4 each push exactly one dimension far into
    its WORSE direction, so that dimension must win the argmax. Row 5 is better
    than the mean on all four axes, so no axis dominates.
    """
    rows = [
        (70.0, 0.0, 25.0, 30.0),    # neutral
        (200.0, 0.0, 25.0, 30.0),   # MAPE blows up          -> CALIBRATION_DOMINATED
        (70.0, -9.0, 25.0, 30.0),   # R2 collapses           -> FIT_DOMINATED
        (70.0, 0.0, -60.0, 30.0),   # policy hurts           -> POLICY_DOMINATED
        (70.0, 0.0, 25.0, 95.0),    # jobs never finish      -> COMPLETION_DOMINATED
        (55.0, 0.9, 60.0, 0.0),     # better on every axis   -> NO_DOMINANT_AXIS
    ]
    classified, _ = ood.classify_scenarios(_frame(rows))
    modes = list(classified['failure_mode'])
    assert modes[1] == 'CALIBRATION_DOMINATED', modes
    assert modes[2] == 'FIT_DOMINATED', modes
    assert modes[3] == 'POLICY_DOMINATED', modes
    assert modes[4] == 'COMPLETION_DOMINATED', modes
    assert modes[5] == ood.NO_DOMINANT_AXIS, modes
    assert set(modes) >= set(ood.FAILURE_MODE_LABELS), (
        f'unreachable mode(s): {set(ood.FAILURE_MODE_LABELS) - set(modes)}'
    )


def test_every_risk_level_is_reachable(ood):
    """HIGH needs the worst band or a negative gain; LOW needs the mildest band."""
    rows = [
        (52.0, 0.95, 45.0, 0.0),    # mildest band, gain > 0 -> LOW_RISK
        (70.0, 0.10, 25.0, 30.0),   # middling               -> MEDIUM_RISK
        (75.0, 0.05, 22.0, 35.0),   # middling               -> MEDIUM_RISK
        (200.0, -9.0, -60.0, 95.0),  # worst band            -> HIGH_RISK
    ]
    classified, _ = ood.classify_scenarios(_frame(rows))
    levels = list(classified['risk_level'])
    assert levels[0] == 'LOW_RISK', levels
    assert levels[-1] == 'HIGH_RISK', levels
    assert set(levels) == set(ood.RISK_LEVELS), (
        f'unreachable risk level(s): {set(ood.RISK_LEVELS) - set(levels)}'
    )


def test_a_negative_gain_is_high_risk_regardless_of_band(ood):
    """The policy making waits WORSE than FIFO is never below HIGH_RISK.

    Zero is a definition (does the ordering help or hurt?), not a fitted
    threshold, so this rule cannot go stale.
    """
    rows = [(52.0, 0.95, -0.5, 0.0)] + [(70.0, 0.0, 25.0, 30.0)] * 5
    classified, _ = ood.classify_scenarios(_frame(rows))
    assert classified['severity_band'].iloc[0] == ood.SEVERITY_BAND_LABELS[0], (
        'the fixture no longer puts the regressing row in the mildest band, so '
        'it no longer tests that a negative gain OVERRIDES a mild band')
    assert classified['risk_level'].iloc[0] == 'HIGH_RISK'


def test_low_risk_needs_a_model_that_beats_the_mean_predictor(ood):
    """The mildest band alone must NOT be enough to earn LOW_RISK.

    All three rows below sit in the mildest quartile of their grid and all three
    have a policy that helps. They differ only in R2: positive, negative, and
    undefined. Only the first may be called LOW_RISK — an R2 <= 0 means the wait
    model does worse than predicting the mean wait, and an R2 that could not be
    computed is an unmeasured axis. Neither deserves the mildest label on an
    out-of-distribution scenario.

    Drop the R2 clause from `assign_risk_levels` and rows 1 and 2 go LOW_RISK.
    """
    good = [
        (52.0, 0.95, 45.0, 0.0),      # helps, model beats the mean -> LOW_RISK
        (52.5, -0.20, 44.0, 0.0),     # helps, model worse than mean -> not LOW
        (52.2, np.nan, 44.5, 0.0),    # helps, R2 undefined          -> not LOW
    ]
    bad = [(150.0 + 5.0 * i, -2.0, 5.0, 70.0) for i in range(9)]
    classified, _ = ood.classify_scenarios(_frame(good + bad))

    mildest = ood.SEVERITY_BAND_LABELS[0]
    assert list(classified['severity_band'].iloc[:3]) == [mildest] * 3, (
        'the fixture no longer puts all three rows in the mildest band, so it '
        'no longer isolates the R2 floor from the band')
    assert (classified['improvement_pct'].iloc[:3] > 0).all(), (
        'the fixture no longer holds the policy sign fixed across the three rows')

    levels = list(classified['risk_level'].iloc[:3])
    assert levels[0] == 'LOW_RISK', levels
    assert levels[1] == 'MEDIUM_RISK', (
        f'a scenario whose wait model is worse than a constant was labelled '
        f'{levels[1]}; the mildest band must not be sufficient on its own')
    assert levels[2] == 'MEDIUM_RISK', (
        f'a scenario with an undefined R2 was labelled {levels[2]}; an axis that '
        'could not be measured must not earn the mildest label')


# ---------------------------------------------------------------------------
# (c) MUTATION GUARD: the cut-points must be data-derived, not absolute
# ---------------------------------------------------------------------------

def test_uniformly_high_mape_is_still_discriminated(ood):
    """The exact shape of the sweep, where an absolute MAPE gate collapses.

    Every row here has MAPE far above any plausible absolute cut-point — as in
    the real artefact, whose minimum is 54.01. A classifier that tests
    `mape >= 35.0` before anything else returns ONE label for all of these. A
    classifier that standardises within the grid still separates them.
    """
    rng = np.random.default_rng(0)  # test-local fixture RNG; not a pipeline seed
    n = 40
    rows = list(zip(
        rng.uniform(54.0, 177.0, n),   # observed MAPE range: all >> 35.0
        rng.uniform(-2.3, 0.86, n),
        rng.uniform(-9.7, 52.6, n),
        rng.uniform(0.0, 79.0, n),
    ))
    classified, _ = ood.classify_scenarios(_frame(rows))

    assert (classified['mape'] > OLD_MAPE_CUTPOINT).all(), 'fixture no longer exercises the defect'
    assert classified['failure_mode'].nunique() > 1, (
        'the taxonomy collapsed to a single label on uniformly-high-MAPE data — '
        'this is the hard-coded absolute cut-point defect returning'
    )
    assert classified['severity_band'].nunique() == len(ood.SEVERITY_BAND_LABELS)
    assert classified['risk_level'].nunique() > 1


def test_bands_are_invariant_to_the_units_the_dimensions_arrive_in(ood):
    """Shift and rescale every dimension; the RANKING must be unchanged.

    Standardising within the grid makes the bands invariant to units and offset.
    An absolute threshold is not: once the data moves past it, every row lands
    on the same side — which is precisely how `mape >= 35.0` went degenerate.
    """
    rows = [(50.0 + 10.0 * i, 0.9 - 0.2 * i, 40.0 - 5.0 * i, 2.0 * i) for i in range(8)]
    base, _ = ood.classify_scenarios(_frame(rows))
    moved = _frame([(m * 3.0 + 500.0, r * 3.0 + 500.0, p * 3.0 + 500.0, f * 3.0 + 500.0)
                    for m, r, p, f in rows])
    shifted, _ = ood.classify_scenarios(moved)

    assert list(base['severity_band']) == list(shifted['severity_band']), (
        'band assignment changed when the data was rescaled — the cut-points '
        'are anchored to absolute values, not to the distribution')
    assert list(base['failure_mode']) == list(shifted['failure_mode'])


def test_cutpoints_are_read_off_the_distribution_not_stored(ood):
    """Two differently-SHAPED severity distributions must yield different cuts.

    Affine invariance above is necessary but not sufficient: a hard-coded
    constant would also be 'stable'. What separates a data-derived cut-point
    from a stored one is that changing the shape of the data moves it.
    """
    even = ood.derive_severity_cutpoints(np.linspace(0.0, 1.0, 100))
    skewed = ood.derive_severity_cutpoints(
        np.concatenate([np.zeros(90), np.full(10, 10.0)]))

    assert not np.allclose(even, skewed), (
        f'cut-points {even} and {skewed} are identical across two different '
        'distributions, which is what a hard-coded constant looks like')
    np.testing.assert_allclose(even, np.quantile(np.linspace(0.0, 1.0, 100),
                                                 ood.SEVERITY_BAND_QUANTILES))


# ---------------------------------------------------------------------------
# (d) Severity must stay oriented so that larger means worse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('column,worse_value,better_value', [
    ('mape', 200.0, 40.0),
    ('r2_score', -5.0, 0.95),
    ('improvement_pct', -40.0, 60.0),
    ('failure_rate_pct', 95.0, 0.0),
])
def test_severity_increases_with_every_bad_dimension(ood, column, worse_value, better_value):
    """A silent sign flip would make the score flattering instead of honest."""
    neutral = {'mape': 70.0, 'r2_score': 0.0, 'improvement_pct': 25.0,
               'failure_rate_pct': 30.0}
    worse = dict(neutral, **{column: worse_value})
    better = dict(neutral, **{column: better_value})
    frame = pd.DataFrame([neutral, worse, better])[list(neutral)]

    classified, _ = ood.classify_scenarios(frame)
    severity = classified['severity_score'].to_numpy(dtype=float)
    assert severity[1] > severity[2], (
        f'a worse {column} ({worse_value}) did not score as more severe than a '
        f'better one ({better_value})')


def test_a_dimension_that_does_not_vary_contributes_nothing(ood):
    """A constant column cannot discriminate, and must not divide by zero."""
    z = ood.oriented_zscore(np.full(5, 77.0), +1.0)
    assert np.all(z == 0.0)
    assert np.all(np.isfinite(z))


# ---------------------------------------------------------------------------
# (e) The pre-existing columns survive
# ---------------------------------------------------------------------------

def test_legacy_columns_are_preserved_in_order(committed, ood):
    """New columns are APPENDED; the original twelve keep name and position."""
    assert list(committed.columns[:len(LEGACY_COLUMNS)]) == LEGACY_COLUMNS
    for added in ('severity_score', 'severity_band', *ood.Z_COLUMNS):
        assert added in committed.columns, f'missing added column {added}'


def test_committed_labels_come_from_the_declared_vocabularies(committed, ood):
    assert set(committed['failure_mode']) <= set(ood.FAILURE_MODE_LABELS)
    assert set(committed['severity_band']) <= set(ood.SEVERITY_BAND_LABELS)
    assert set(committed['risk_level']) <= set(ood.RISK_LEVELS)
    assert len(committed) == 72, 'the sweep is 6 arrival rates x 4 sizes x 3 profiles'

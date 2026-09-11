"""Tests for 04_scheduler/non_degeneracy_sweep.py.

THREE DEFECTS ARE PINNED HERE, each by a test that goes RED when the defect is
reintroduced (each was verified by mutating a scratch COPY of the module):

  1. LEAKAGE. `causal_user_history` computed as a full-trace groupby mean would
     manufacture a fake non-degeneracy out of information no scheduler could
     have, and the whole experiment would prove nothing. This is the test that
     matters most.
  2. REGRESSION AGAINST THE PUBLISHED RESULT. The 8-feature baseline must still
     yield ZERO equal-size/different-prediction violations, so this parallel
     experiment agrees with `ranking_degeneracy.py`.
  3. SILENT REVERT TO SIZE-ONLY FEATURES. At least one augmented feature set
     must yield violations > 0; otherwise a patch that quietly dropped the
     extra columns would leave a green suite and a meaningless artefact.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, '04_scheduler'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, '02_data'))
sys.path.insert(0, PROJECT_ROOT)

import non_degeneracy_sweep as nds          # noqa: E402
import trace_driven_benchmark as tdb        # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# 1. THE LEAKAGE TEST
# ─────────────────────────────────────────────────────────────────────────────

def _leak_trace():
    """A five-job trace built so that a full-trace groupby is catastrophic.

    User 1 submits an EARLY job (job 1) and, much later, a LATE job (job 3)
    whose wait is 1,000,000 seconds. Job 5 is a SECOND early job of user 1 that
    completes before job 3 is submitted, so job 3 legitimately has history.

    The early job (job 1) has NO completed prior job of its own user and must
    therefore carry NaN history. The late job's enormous wait must never reach
    it -- forwards or sideways.
    """
    return pd.DataFrame([
        # job_id, submit, recorded_wait, runtime, user_id
        {'job_id': 1, 'submit': 0,       'recorded_wait': 10,      'runtime': 100, 'user_id': 1},
        {'job_id': 2, 'submit': 50,      'recorded_wait': 5,       'runtime': 10,  'user_id': 2},
        {'job_id': 5, 'submit': 200,     'recorded_wait': 20,      'runtime': 30,  'user_id': 1},
        {'job_id': 3, 'submit': 1000,    'recorded_wait': 1000000, 'runtime': 100, 'user_id': 1},
        {'job_id': 4, 'submit': 2000000, 'recorded_wait': 1,       'runtime': 10,  'user_id': 1},
    ]).sort_values('submit').reset_index(drop=True)


def test_user_history_does_not_leak_a_later_extreme_wait():
    """THE test. A user's LATE extreme wait must not touch their EARLIER job."""
    df = _leak_trace()
    hw, hr = nds.causal_user_history(df)
    by_id = {int(j): (hw[i], hr[i]) for i, j in enumerate(df['job_id'])}

    extreme = 1000000.0
    full_trace_mean = df[df['user_id'] == 1]['recorded_wait'].mean()
    assert full_trace_mean > extreme / 5, 'the fixture must make leakage visible'

    # Job 1 is user 1's first job: no completed prior history at all.
    assert np.isnan(by_id[1][0]), 'first job of a user must be cold-start NaN'
    assert np.isnan(by_id[1][1])

    # Job 5 (submit 200) sees ONLY job 1, which completed at 0+10+100 = 110.
    assert by_id[5][0] == pytest.approx(10.0)
    assert by_id[5][1] == pytest.approx(100.0)
    # And nothing anywhere near the late extreme wait, nor the full-trace mean.
    assert by_id[5][0] < extreme / 1000
    assert by_id[5][0] != pytest.approx(full_trace_mean)

    # Job 3 (submit 1000) sees jobs 1 and 5, both completed (110 and 250), and
    # NOT its own 1,000,000-second wait.
    assert by_id[3][0] == pytest.approx((10.0 + 20.0) / 2.0)

    # Job 4, submitted long after everything, finally sees the extreme wait.
    assert by_id[4][0] == pytest.approx((10.0 + 20.0 + 1000000.0) / 3.0)


def test_user_history_excludes_jobs_submitted_earlier_but_not_yet_completed():
    """Submitted-earlier is NOT enough: a wait is unobservable until start, and
    a runtime until end. A running predecessor must contribute nothing."""
    df = pd.DataFrame([
        {'job_id': 1, 'submit': 0,  'recorded_wait': 0, 'runtime': 10000, 'user_id': 7},
        {'job_id': 2, 'submit': 10, 'recorded_wait': 0, 'runtime': 5,     'user_id': 7},
    ])
    hw, hr = nds.causal_user_history(df)
    # Job 1 is still RUNNING at t=10 (completes at 10000), so job 2 has no
    # observable history even though job 1 was submitted first.
    assert np.isnan(hw[1]), 'a still-running predecessor must not be counted'
    assert np.isnan(hr[1])


def test_history_is_prefix_stable_under_appended_future():
    """Truncating the future must not change any earlier job's feature value --
    the property a full-trace groupby violates by construction."""
    df = _leak_trace()
    hw_full, hr_full = nds.causal_user_history(df)
    prefix = df[df['submit'] <= 1000].reset_index(drop=True)
    hw_pre, hr_pre = nds.causal_user_history(prefix)
    np.testing.assert_allclose(hw_full[:len(prefix)], hw_pre, equal_nan=True)
    np.testing.assert_allclose(hr_full[:len(prefix)], hr_pre, equal_nan=True)


# ─────────────────────────────────────────────────────────────────────────────
# Wiring guards: the sweep must keep using the published training recipe
# ─────────────────────────────────────────────────────────────────────────────

def test_xgb_hyperparameters_match_the_published_trace_model():
    """The sweep duplicates train_trace_model's column list and nothing else.
    Read the published hyper-parameters out of its SOURCE so a change there
    cannot drift silently away from the copy here."""
    import inspect
    src = inspect.getsource(tdb.train_trace_model)
    for key, val in nds.XGB_PARAMS.items():
        if key == 'random_state':
            assert 'random_state=SEED' in src
            assert val == 42, 'the trace seed is SEED=42 and must not change'
            continue
        assert f'{key}={val}' in src, f'{key}={val} not in published train_trace_model'


def test_partition_is_constant_on_both_traces_and_is_not_a_feature():
    """SWF field 16 is constant on both committed traces, so it cannot separate
    co-queued jobs. It must stay out of the candidate list."""
    assert 'partition' not in nds.EXTRA_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# 2. + 3. The sweep itself, on a small real slice
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def sdsc_slice():
    """One evaluation window of the SDSC trace, with the sweep's attributes.

    SDSC rather than LANL: it is the smaller trace (43k jobs) and one window
    replays in ~1.5 s, which keeps this file runnable on its own.
    """
    swf = os.path.join(tdb.DATA_DIR, tdb.TRACES['sdsc']['swf'])
    capacity, df, _ = tdb.parse_swf_jobs(swf)
    attr = nds.build_job_attributes(df, swf)
    t_min, t_max = int(attr['submit'].min()), int(attr['submit'].max())
    split = int(t_min + tdb.TRAIN_FRACTION * (t_max - t_min))
    windows = tdb.carve_windows(attr, split, 1, tdb.WARMUP_DAYS, tdb.MEASURE_DAYS)
    assert windows, 'the SDSC trace must yield at least one evaluation window'
    return capacity, attr, split, windows[:1]


def _declared(set_name):
    """The extras the module ITSELF declares for this feature set.

    Read from nds.FEATURE_SETS rather than hard-coded here on purpose: a change
    that quietly gave the baseline an extra column, or took one away from an
    augmented set, must reach these tests instead of being bypassed by a literal
    list written in the test file.
    """
    for name, extras in nds.FEATURE_SETS:
        if name == set_name:
            return list(extras)
    raise AssertionError(f'{set_name!r} is not a declared feature set')


def _violations(sdsc_slice, set_name):
    capacity, attr, split, windows = sdsc_slice
    extras = _declared(set_name)
    summary, rows = nds.run_variant('sdsc', capacity, attr, split, windows,
                                    set_name, extras)
    assert rows, 'the window must have produced simulated jobs'
    assert summary['ranking_instants'] > 0, 'no ranking decision was observed'
    return summary


def test_baseline_feature_set_still_has_zero_violations(sdsc_slice):
    """AGREEMENT WITH THE PUBLISHED RESULT: with the 8 base features every
    equal-size pair of co-queued jobs receives the same score."""
    s = _violations(sdsc_slice, 'baseline')
    assert s['equal_size_diff_pred_violations'] == 0
    assert s['n_features'] == len(tdb.BASE_FEATURES)
    assert _declared('baseline') == [], 'the baseline must declare NO extras'


@pytest.mark.parametrize('name', ['+est_runtime', '+user_hist_wait'])
def test_augmented_feature_set_breaks_the_degeneracy(sdsc_slice, name):
    """A genuine per-job attribute must produce equal-size/different-prediction
    violations. Zero here would mean the extra columns never reached the model
    -- a silent revert to the size-only feature set."""
    extras = _declared(name)
    assert extras, f'{name} must declare at least one extra feature'
    base = _violations(sdsc_slice, 'baseline')
    s = _violations(sdsc_slice, name)
    assert s['equal_size_diff_pred_violations'] > 0, (
        f'{name} produced no violations: the extra column did not reach the model')
    assert s['n_features'] == len(tdb.BASE_FEATURES) + len(extras)
    # And the queue is genuinely being separated more finely than by size alone.
    assert s['mean_distinct_predictions'] > base['mean_distinct_predictions']


def test_feature_matrix_patch_is_removed_after_the_block():
    """The published build_feature_matrix must be restored even on an error --
    leaving it patched would silently corrupt any later benchmark run in the
    same process."""
    original = tdb.build_feature_matrix
    with pytest.raises(RuntimeError):
        with nds.extended_features(['est_runtime'], {}):
            assert tdb.build_feature_matrix is not original
            raise RuntimeError('boom')
    assert tdb.build_feature_matrix is original

"""Every headline number, read back from the COMMITTED artefact and pinned.

WHY THIS FILE EXISTS
--------------------
Before it, not one published number in this repository was asserted anywhere. The
suite pinned invariants and known defects -- valuable, but it would have stayed
green through a regeneration that moved the central result. The 45,432 dispatch
instants, the 7.9% improvement, the TOST equivalence and the twelve-policy trace
table were each defended only by a human noticing.

These tests are the artefact-to-prose contract. They do not exercise the
scheduler; they assert that what is committed still says what the paper says.
When a legitimate change moves one of them, the failure is the point: update the
literal here, in the changelog, and in the prose, in one commit, deliberately.

WHY THEY FAIL RATHER THAN SKIP
------------------------------
`conftest.require()` skips when a GENERATED artefact is missing, so a fresh clone
can run the suite before the pipeline has. That policy is wrong for these inputs:
every file read here is COMMITTED to git. Its absence is a broken checkout or an
accidental deletion, not an un-run pipeline -- and a skip would silently remove
the only guard on the numbers the project is judged by. So `golden()` asserts.

TOLERANCE
---------
`REL = 1e-6` for floats, exact for counts, strings and booleans. That is far
looser than CSV round-trip error and far tighter than any drift a real
behavioural change could produce: scheduling is discrete, so a changed queue
order moves a mean by percent, not by parts per million. A failure here is never
noise to be absorbed by widening the tolerance.
"""

import os

import pandas as pd
import pytest

from conftest import PROJECT_ROOT

pytestmark = pytest.mark.golden

REL = 1e-6


def golden(rel_path):
    """Absolute path of a COMMITTED artefact; fail loudly if it is missing."""
    full = os.path.join(PROJECT_ROOT, rel_path)
    assert os.path.exists(full), (
        f'committed golden artefact missing: {rel_path}\n'
        f'Golden inputs are tracked in git, so absence means a broken checkout or an '
        f'accidental deletion -- not an un-run pipeline. Restore it with '
        f'"git checkout -- {rel_path}" rather than regenerating, so that what is '
        f'compared is what is committed.')
    return full


def read(rel_path):
    return pd.read_csv(golden(rel_path), float_precision='round_trip')


# ---------------------------------------------------------------------------
# 1. The ranking-degeneracy result: 45,432 instants, zero counterexamples
# ---------------------------------------------------------------------------

SETTINGS = ('synthetic (12 features)',
            'SDSC SP2 (1998) (8 features)',
            'LANL CM-5 (1994) (8 features)')

INSTANTS = {
    'synthetic (12 features)': 3646,
    'SDSC SP2 (1998) (8 features)': 11843,
    'LANL CM-5 (1994) (8 features)': 29943,
}
TOTAL_INSTANTS = 45_432


def test_the_instant_count_and_the_zero_violation_claim():
    """THE headline: no two equally-sized co-queued jobs ever got different scores.

    3,646 of these instants are synthetic and 41,786 are replayed from the two real
    SWF traces. Prose that calls all 45,432 "real" is wrong, which is why the split
    is asserted here and not just the total.
    """
    df = read('05_results/degeneracy/ranking_degeneracy.csv')
    assert list(df['setting']) == list(SETTINGS), 'setting order is part of the artefact'
    assert dict(zip(df['setting'], df['ranking_instants'].astype(int))) == INSTANTS
    assert int(df['ranking_instants'].sum()) == TOTAL_INSTANTS
    assert int(df.loc[df['setting'] != SETTINGS[0], 'ranking_instants'].sum()) == 41_786
    assert (df['equal_size_diff_pred_violations'].astype(int) == 0).all()
    assert dict(zip(df['setting'], df['features_total'].astype(int))) == {
        SETTINGS[0]: 12, SETTINGS[1]: 8, SETTINGS[2]: 8}
    # A perfectly size-ordered queue somewhere in every setting: the score is a
    # size lookup table, so the minimum rank correlation against size is exactly -1.
    assert df['kendall_tau_vs_size_min'].tolist() == pytest.approx([-1.0] * 3, rel=REL)


def test_the_run_that_produced_the_committed_tables_was_complete():
    """The totals row is the only record that the committed CSVs came from a full run.

    Without it, a --quick or --allow-partial run writes a smaller total that is
    indistinguishable from the published one by inspection.
    """
    t = read('05_results/degeneracy/ranking_degeneracy_totals.csv')
    assert len(t) == 1
    row = t.iloc[0]
    assert int(row['total_instants']) == TOTAL_INSTANTS
    assert int(row['total_violations']) == 0
    assert int(row['settings_present']) == int(row['settings_expected']) == 3
    assert str(row['missing_traces']) == 'none'
    assert not bool(row['partial'])
    assert int(row['runs']) == int(row['published_runs']) == 20
    assert int(row['windows']) == int(row['published_windows']) == 20


def test_the_totals_row_agrees_with_the_per_setting_table():
    """One number, one source. These two files are written by the same run and must
    never drift apart; if they do, one of them was hand-edited."""
    df = read('05_results/degeneracy/ranking_degeneracy.csv')
    t = read('05_results/degeneracy/ranking_degeneracy_totals.csv').iloc[0]
    assert int(df['ranking_instants'].sum()) == int(t['total_instants'])
    assert int(df['equal_size_diff_pred_violations'].sum()) == int(t['total_violations'])
    assert len(df) == int(t['settings_present'])


# ---------------------------------------------------------------------------
# 2. Two DIFFERENT quantities that prose once conflated
# ---------------------------------------------------------------------------

ORDER_IS_ARRIVAL_ORDER = {
    'synthetic (12 features)': 19.720241360394954,
    'SDSC SP2 (1998) (8 features)': 18.111964873765093,
    'LANL CM-5 (1994) (8 features)': 27.465517817185987,
}
ALL_SCORES_TIED = {
    'synthetic (12 features)': 17.11464618760285,
    'SDSC SP2 (1998) (8 features)': 14.413577640799628,
    'LANL CM-5 (1994) (8 features)': 20.729385830831915,
}


def test_the_all_ties_fraction_is_measured_and_is_not_the_arrival_order_fraction():
    """RETRACTION GUARD (v3.6).

    README, RESULTS, the manuscript and the changelog all said "in 18-27% of
    instants ALL SCORES TIE". That 18-27% is pct_order_identical_to_arrival --
    the fraction where the induced ORDER equals arrival order. All-tied implies
    that; the converse does not hold, so the column only upper-bounds the claim.
    Measured directly the all-ties fraction is 14-21%.

    Both numbers support the degeneracy argument. Only the conflation was wrong,
    so both are pinned here, together with the inequality that makes them
    different quantities.
    """
    df = read('05_results/degeneracy/ranking_degeneracy.csv')
    arrival = dict(zip(df['setting'], df['pct_order_identical_to_arrival']))
    tied = dict(zip(df['setting'], df['pct_all_scores_tied']))
    for setting in SETTINGS:
        assert arrival[setting] == pytest.approx(ORDER_IS_ARRIVAL_ORDER[setting], rel=REL)
        assert tied[setting] == pytest.approx(ALL_SCORES_TIED[setting], rel=REL)
        # The bound the collector guarantees by construction.
        assert tied[setting] <= arrival[setting] + 1e-12, setting
        # ... and they are genuinely different, so quoting one for the other is
        # a real error rather than a rounding choice.
        assert arrival[setting] - tied[setting] > 1.0, setting


def test_the_ml_order_is_usually_exactly_the_size_order():
    df = read('05_results/degeneracy/ranking_degeneracy.csv')
    got = dict(zip(df['setting'], df['pct_order_identical_to_size']))
    assert got[SETTINGS[0]] == pytest.approx(80.2523313219967, rel=REL)
    assert got[SETTINGS[1]] == pytest.approx(71.52748459005319, rel=REL)
    assert got[SETTINGS[2]] == pytest.approx(77.63083191396987, rel=REL)


def test_the_size_to_priority_table_is_monotone_only_about_half_the_time():
    """57-63%: the learned table is NOT simply smallest-first, which is why the
    equivalence result needs a statistical test rather than an appeal to the shape."""
    df = read('05_results/degeneracy/ranking_degeneracy.csv')
    got = dict(zip(df['setting'], df['pct_size_table_monotone']))
    assert got[SETTINGS[0]] == pytest.approx(63.137685134393855, rel=REL)
    assert got[SETTINGS[1]] == pytest.approx(57.11390694925272, rel=REL)
    assert got[SETTINGS[2]] == pytest.approx(56.90144608088702, rel=REL)


def test_a_nine_job_queue_receives_only_two_or_three_distinct_priority_levels():
    df = read('05_results/degeneracy/ranking_degeneracy.csv')
    distinct = dict(zip(df['setting'], df['mean_distinct_predictions']))
    queues = dict(zip(df['setting'], df['mean_queue_len']))
    assert distinct[SETTINGS[0]] == pytest.approx(2.8049917718047177, rel=REL)
    assert distinct[SETTINGS[1]] == pytest.approx(3.0883222156548173, rel=REL)
    assert distinct[SETTINGS[2]] == pytest.approx(2.2772601275757274, rel=REL)
    assert all(8.4 < q < 10.2 for q in queues.values())


# ---------------------------------------------------------------------------
# 3. Seven of twelve features never vary across a queue
# ---------------------------------------------------------------------------

SYNTHETIC_CONSTANT_FEATURES = {
    'total_free', 'queue_length', 'running_jobs', 'max_free_node',
    'variance_free', 'fragmentation', 'avg_free_per_node',
}
TRACE_CONSTANT_FEATURES = {'total_free', 'queue_length', 'running_jobs', 'free_frac'}


def test_seven_of_twelve_synthetic_features_are_constant_across_the_queue():
    """The mechanism behind the headline: at one dispatch instant every queued job
    sees the same cluster, so a cluster-only feature cannot reorder anything."""
    df = read('05_results/degeneracy/feature_variation.csv')
    syn = df[df['setting'] == SETTINGS[0]]
    assert len(syn) == 12
    zero = set(syn.loc[syn['pct_instants_varying_across_queue'] == 0.0, 'feature'])
    assert zero == SYNTHETIC_CONSTANT_FEATURES
    assert len(zero) == 7
    pct = dict(zip(syn['feature'], syn['pct_instants_varying_across_queue']))
    # queue_pressure varies exactly as often as job_gpu: it is a deterministic
    # function of the requested size given the state, which is the v3.5 finding.
    assert pct['job_gpu'] == pytest.approx(82.88535381239714, rel=REL)
    assert pct['queue_pressure'] == pytest.approx(pct['job_gpu'], rel=REL)


@pytest.mark.parametrize('setting', [SETTINGS[1], SETTINGS[2]])
def test_four_of_eight_trace_features_are_constant_across_the_queue(setting):
    df = read('05_results/degeneracy/feature_variation.csv')
    sub = df[df['setting'] == setting]
    assert len(sub) == 8
    zero = set(sub.loc[sub['pct_instants_varying_across_queue'] == 0.0, 'feature'])
    assert zero == TRACE_CONSTANT_FEATURES
    pct = dict(zip(sub['feature'], sub['pct_instants_varying_across_queue']))
    assert pct['queue_pressure'] == pytest.approx(pct['job_procs'], rel=REL)


# ---------------------------------------------------------------------------
# 4. Sorting by size is statistically equivalent to the whole ML pipeline
# ---------------------------------------------------------------------------

def _equiv_row(df, a, b, metric):
    r = df[(df['scheduler_a'] == a) & (df['scheduler_b'] == b) & (df['metric'] == metric)]
    assert len(r) == 1, f'expected exactly one ({a}, {b}, {metric}) row, got {len(r)}'
    return r.iloc[0]


def test_tost_says_size_sort_is_equivalent_to_the_xgboost_pipeline():
    """The paper's second headline. A large p from a DIFFERENCE test would mean only
    "we failed to detect a difference"; TOST rejects both one-sided nulls, so this is
    positive evidence of equivalence within a +/-1.59 ts margin."""
    df = read('05_results/schedulers/multi_scheduler_equivalence.csv')
    assert len(df) == 6
    r = _equiv_row(df, 'SMALLEST', 'PROACTIVE', 'mean_wait')
    assert int(r['n']) == 20
    assert r['mean_a'] == pytest.approx(16.074545454545454, rel=REL)
    assert r['mean_b'] == pytest.approx(15.947727272727272, rel=REL)
    assert r['pct_diff'] == pytest.approx(0.7952116289012406, rel=REL)
    assert r['margin_frac'] == pytest.approx(0.1, rel=REL)
    assert r['margin'] == pytest.approx(1.5947727272727272, rel=REL)
    assert r['mean_diff'] == pytest.approx(0.12681818181818177, rel=REL)
    assert r['ci_low'] == pytest.approx(0.02537041679179944, rel=REL)
    assert r['ci_high'] == pytest.approx(0.2282659468445641, rel=REL)
    assert r['p_tost'] == pytest.approx(2.6231406607136957e-16, rel=REL)
    assert bool(r['equivalent']) is True
    # The whole interval sits inside the margin: that IS the equivalence claim.
    assert abs(r['ci_low']) < r['margin'] and abs(r['ci_high']) < r['margin']


def test_the_mlp_reproduces_the_size_sort_exactly_not_merely_equivalently():
    """NN vs SMALLEST is not "close": it is the same ordering. A zero difference with
    a zero-width interval is the strongest form the degeneracy claim takes."""
    df = read('05_results/schedulers/multi_scheduler_equivalence.csv')
    for metric in ('mean_wait', 'mean_bounded_slowdown'):
        z = _equiv_row(df, 'SMALLEST', 'NN', metric)
        assert z['mean_diff'] == 0.0
        assert z['pct_diff'] == 0.0
        assert z['ci_low'] == 0.0 and z['ci_high'] == 0.0
        assert bool(z['equivalent']) is True
        # and the NN row mirrors the SMALLEST row against PROACTIVE, cell for cell
        smallest = _equiv_row(df, 'SMALLEST', 'PROACTIVE', metric)
        nn = _equiv_row(df, 'NN', 'PROACTIVE', metric)
        for col in ('mean_a', 'mean_b', 'pct_diff', 'mean_diff', 'ci_low', 'ci_high', 'p_tost'):
            assert nn[col] == pytest.approx(smallest[col], rel=REL), (metric, col)


RUN_METRICS = ['mean_wait', 'max_wait', 'p95_wait', 'mean_turnaround',
               'mean_bounded_slowdown', 'throughput', 'fairness_gini', 'gpu_util',
               'preemptions']


def test_the_mlp_and_the_size_sort_are_bit_identical_on_every_run():
    """Not approximately equal on the aggregate -- identical on all nine metrics of
    all twenty runs. Asserted with check_exact, because approximate equality here
    would let a real divergence hide inside a tolerance."""
    df = read('05_results/schedulers/multi_scheduler_runs.csv')
    assert len(df) == 14 * 20
    assert df['scheduler'].nunique() == 14
    nn = df[df['scheduler'] == 'NN'].set_index('run').sort_index()[RUN_METRICS]
    sm = df[df['scheduler'] == 'SMALLEST'].set_index('run').sort_index()[RUN_METRICS]
    assert list(nn.index) == list(sm.index) == list(range(1, 21))
    pd.testing.assert_frame_equal(nn, sm, check_exact=True)


SYNTHETIC_MEAN_WAIT = {
    'SJF': 12.34,
    'SJF_EST': 13.317727272727272,
    'SRPT': 14.023181818181818,
    'HRRN': 15.553636363636365,
    'PROACTIVE': 15.947727272727272,
    'NN': 16.074545454545454,
    'SMALLEST': 16.074545454545454,
    'STATIC_PRIORITY': 16.53181818181818,
    'FIFO': 17.215909090909093,
    'PROACTIVE_BF': 19.201363636363638,
    'BACKFILL_EST': 19.227727272727272,
    'BACKFILL': 19.247727272727275,
    'CONS_BF': 21.843636363636364,
    'FIFO_STRICT': 26.174545454545452,
}


def test_the_fourteen_scheduler_synthetic_table():
    """Includes the renamed STATIC_PRIORITY key. The old name asserted an aging
    behaviour the policy does not have; the number is unchanged by the rename."""
    df = read('05_results/schedulers/multi_scheduler_benchmark.csv')
    assert len(df) == 14
    got = dict(zip(df['scheduler'], df['mean_wait']))
    assert set(got) == set(SYNTHETIC_MEAN_WAIT)
    assert 'PRIORITY' not in got, 'the retired key is back; see v3.6 in CHANGELOG.md'
    for key, value in SYNTHETIC_MEAN_WAIT.items():
        assert got[key] == pytest.approx(value, rel=REL), key
    assert got['NN'] == got['SMALLEST']


# ---------------------------------------------------------------------------
# 5. The 40-run paired study
# ---------------------------------------------------------------------------

STAT_SUMMARY = {
    'runs': 40.0,
    'baseline_wait_mean': 18.268863636363637,
    'proactive_wait_mean': 16.640454545454546,
    'mean_wait_difference': 1.6284090909090907,
    'std_wait_difference': 1.8466342383297363,
    'mean_improvement_pct': 7.897862356147857,
    'std_improvement_pct': 9.428638754929786,
    'ttest_pvalue': 2.001470727467938e-06,
    'wilcoxon_pvalue': 2.6737125153886154e-06,
    'wait_diff_ci95_low': 1.0378268100508574,
    'wait_diff_ci95_high': 2.218991371767324,
    'improvement_ci95_low': 4.882437392343807,
    'improvement_ci95_high': 10.913287319951907,
}


def test_the_forty_run_improvement_and_its_interval():
    """The 7.9% headline. NOTE the interval is a STUDENT-T interval
    (benchmark_statistical.py computes stats.t.ppf), not a bootstrap: prose called
    it a bootstrap CI for several releases. The one real percentile bootstrap in the
    repository is phase 22's, pinned separately below."""
    df = read('05_results/benchmark_statistical_summary.csv')
    got = dict(zip(df['metric'], df['value']))
    assert set(got) == set(STAT_SUMMARY)
    assert int(got['runs']) == 40
    for key, value in STAT_SUMMARY.items():
        assert got[key] == pytest.approx(value, rel=REL), key


def test_the_phase_22_bootstrap_interval_is_a_different_interval():
    """Both are 95% intervals over the SAME 40 runs, and they are not the same
    numbers. Pinning both is what stops one being quoted under the other's name."""
    df = read('phases_22_30/phase_22_stats/stats_summary.csv')
    row = df[df['metric'] == 'wait_improvement_pct']
    assert len(row) == 1
    row = row.iloc[0]
    assert int(row['n']) == 40
    assert row['mean'] == pytest.approx(STAT_SUMMARY['mean_improvement_pct'], rel=REL)
    assert row['ci_lower'] == pytest.approx(4.910216097222412, rel=1e-5)
    assert row['ci_upper'] == pytest.approx(10.671388946113114, rel=1e-5)
    assert row['ci_upper'] != pytest.approx(STAT_SUMMARY['improvement_ci95_high'], rel=1e-3)
    # Utilisation is not merely similar between the two policies: it is IDENTICAL in
    # all 40 runs, which is why phase 22 refuses to test it at all.
    util = df[df['metric'] == 'paired_ttest_gpu_utilization'].iloc[0]
    assert str(util['note']) == 'n/a (zero variance)'


# ---------------------------------------------------------------------------
# 6. The trace-driven table: twelve policies, two real traces
# ---------------------------------------------------------------------------

TRACE_MEAN_WAIT = {
    ('lanl', 'SRPT_ORACLE'): 878.3746746811528,
    ('lanl', 'SJF_ORACLE'): 1380.4325198298222,
    ('lanl', 'HRRN_USEREST'): 1771.8124708686748,
    ('lanl', 'SJF_USEREST'): 1889.184549944727,
    ('lanl', 'PROACTIVE_EST'): 2219.442479504783,
    ('lanl', 'PROACTIVE'): 2229.324032143856,
    ('lanl', 'FCFS'): 2329.148276133066,
    ('lanl', 'SMALLEST_FIRST'): 2549.341513987956,
    ('lanl', 'EASY_ORACLE'): 2758.84706337263,
    ('lanl', 'EASY_USEREST'): 4810.486058017742,
    ('lanl', 'CONS_BF_USEREST'): 6815.033337057475,
    ('lanl', 'FIFO_STRICT'): 12343.719541058064,
    ('sdsc', 'SRPT_ORACLE'): 3607.022747328494,
    ('sdsc', 'SJF_ORACLE'): 6650.464388749965,
    ('sdsc', 'SJF_USEREST'): 6946.289699633102,
    ('sdsc', 'PROACTIVE_EST'): 7497.492671520018,
    ('sdsc', 'HRRN_USEREST'): 7578.276322693348,
    ('sdsc', 'SMALLEST_FIRST'): 8697.251757691663,
    ('sdsc', 'PROACTIVE'): 8701.696149082263,
    ('sdsc', 'FCFS'): 10477.45336896466,
    ('sdsc', 'EASY_ORACLE'): 11024.666788161996,
    ('sdsc', 'EASY_USEREST'): 11705.170880242451,
    ('sdsc', 'CONS_BF_USEREST'): 12449.33183966191,
    ('sdsc', 'FIFO_STRICT'): 45308.47843492056,
}
TRACE_JOBS_MEASURED = {'lanl': 925.25, 'sdsc': 308.5}
TRACE_OFFERED_LOAD = {'lanl': 0.6750734049479167, 'sdsc': 0.7232552508318866}


def test_the_twelve_policy_trace_table_in_full():
    """Values are SECONDS here; the paper's table is minutes. Pinning all 24 cells
    means a change to any policy on either machine has to be acknowledged."""
    df = read('05_results/trace_schedulers/trace_scheduler_summary.csv')
    assert len(df) == 24
    got = {(t, s): w for t, s, w in zip(df['trace'], df['scheduler'], df['mean_wait'])}
    assert set(got) == set(TRACE_MEAN_WAIT)
    for key, value in TRACE_MEAN_WAIT.items():
        assert got[key] == pytest.approx(value, rel=REL), key


def test_the_trace_protocol_is_the_published_one():
    df = read('05_results/trace_schedulers/trace_scheduler_summary.csv')
    for trace, grp in df.groupby('trace'):
        assert len(grp) == 12
        assert grp['jobs_measured'].tolist() == pytest.approx(
            [TRACE_JOBS_MEASURED[trace]] * 12, rel=REL)
        assert grp['offered_load'].tolist() == pytest.approx(
            [TRACE_OFFERED_LOAD[trace]] * 12, rel=REL)
    windows = read('05_results/trace_schedulers/trace_scheduler_windows.csv')
    assert len(windows) == 12 * 2 * 20, '12 policies x 2 traces x 20 windows'
    for trace, grp in windows.groupby('trace'):
        assert grp['window'].nunique() == 20


def test_a_runtime_signal_beats_the_learned_score_on_both_real_machines():
    """The finding that removes the model's claimed niche: every oracle or
    user-estimate runtime policy beats PROACTIVE, on both machines."""
    df = read('05_results/trace_schedulers/trace_scheduler_summary.csv')
    got = {(t, s): w for t, s, w in zip(df['trace'], df['scheduler'], df['mean_wait'])}
    for trace in ('lanl', 'sdsc'):
        for policy in ('SRPT_ORACLE', 'SJF_ORACLE', 'SJF_USEREST'):
            assert got[(trace, policy)] < got[(trace, 'PROACTIVE')], (trace, policy)


# ---------------------------------------------------------------------------
# 7. Numbers the operator-facing documentation quotes
#
# The ROI study that used to be pinned here was DELETED in v3.6. It converted a
# wait-time percentage into GPU-hours saved, but this repository's own 40-run
# benchmark records utilisation identical to six decimal places and the same 110
# jobs completing under both policies -- the cluster does the same compute either
# way. The saving being monetised was measured at zero, so the model was not an
# uncertain estimate but a category error, and no pin can rescue it.
# ---------------------------------------------------------------------------

def test_the_fairness_cost_of_reordering():
    """Proactive buys mean wait with tail latency and equity. Pinned because the
    repository once carried two disagreeing copies of this table, and prose quoted
    the stale one (max wait 125 and 87 instead of 122.65 and 88.15)."""
    df = read('05_results/fairness/fairness_metrics.csv').set_index('scheduler')
    assert df.loc['fifo', 'max_wait'] == pytest.approx(57.85, rel=REL)
    assert df.loc['proactive', 'max_wait'] == pytest.approx(122.65, rel=REL)
    assert df.loc['proactive_starvation', 'max_wait'] == pytest.approx(88.15, rel=REL)
    assert df.loc['fifo', 'fairness_gini'] == pytest.approx(0.5261904919201371, rel=REL)
    assert df.loc['proactive', 'fairness_gini'] == pytest.approx(0.7936274161073568, rel=REL)
    assert df.loc['proactive_starvation', 'fairness_gini'] == pytest.approx(
        0.6921033216783936, rel=REL)


def test_phase_27_agrees_with_the_root_fairness_table():
    """These two files disagreed for several releases (124.8 vs 122.65, 87.15 vs
    88.15) purely because the phase-27 copy was stale and unreachable. One quantity,
    one value: if they diverge again, one of them was not regenerated."""
    root = read('05_results/fairness/fairness_metrics.csv').set_index('scheduler')
    ph27 = read('phases_22_30/phase_27_fairness/fairness_metrics.csv').set_index('scheduler')
    for name in ('fifo', 'proactive', 'proactive_starvation'):
        assert ph27.loc[name, 'max_job_wait'] == pytest.approx(
            root.loc[name, 'max_wait'], rel=REL), name
        assert ph27.loc[name, 'gini_per_job'] == pytest.approx(
            root.loc[name, 'fairness_gini'], rel=REL), name
        assert ph27.loc[name, 'starvation_count'] == pytest.approx(
            root.loc[name, 'starvation_count'], rel=REL), name


def test_the_contended_cluster_advantage():
    """14.5%, not the 14.4% several documents carried."""
    df = read('05_results/scaling/scaling_analysis.csv')
    row = df[(df['nodes'] == 4)].iloc[0]
    assert row['wait_advantage_pct'] == pytest.approx(14.521943154401422, rel=REL)
    # and it vanishes once the cluster is big enough to run everything at once
    assert df[df['nodes'] >= 16]['wait_advantage_pct'].tolist() == pytest.approx([0.0, 0.0])


def test_the_wait_model_quality():
    df = read('05_results/model_comparison_table1.csv')
    xgb = df[df['model'].str.contains('XGB', case=False)].iloc[0]
    assert xgb['r2'] == pytest.approx(0.836840033531189, rel=REL)
    assert xgb['mae'] == pytest.approx(4.693508148193359, rel=REL)


def test_real_user_estimates_are_not_the_f_model():
    """The project's second, independent contribution: real runtime estimates do not
    look like runtime x U(1, C). LANL under-estimates 36.3% of the time, which the
    f-model cannot produce at all."""
    df = read('05_results/trace_schedulers/trace_estimate_quality.csv').set_index('trace')
    assert df.loc['lanl', 'under_estimate_frac'] == pytest.approx(0.36273686681291695, rel=REL)
    assert df.loc['sdsc', 'under_estimate_frac'] == pytest.approx(0.001230069394480934, rel=REL)
    # The f-model cannot under-estimate at all -- runtime x U(1, C) is always >= runtime --
    # so 36.3% of LANL's estimates are outside anything it can generate.
    assert df.loc['lanl', 'fmodel_under_estimate_frac'] == 0.0
    assert df.loc['sdsc', 'fmodel_under_estimate_frac'] == 0.0
    assert df.loc['sdsc', 'ratio_median'] == pytest.approx(6.907137375287798, rel=REL)
    assert df.loc['lanl', 'ratio_median'] == pytest.approx(1.5143017386427369, rel=REL)
    assert df.loc['sdsc', 'fmodel_C5_ratio_median'] == 3.0


def test_the_simulator_fidelity_gap_is_recorded():
    """A limitation, pinned so it cannot quietly disappear: on SDSC the capacity-only
    simulator produces a mean wait far below what the machine actually recorded."""
    df = read('05_results/trace_schedulers/trace_fidelity.csv')
    means = df.groupby('trace')[['sim_fcfs_mean_wait_min', 'recorded_mean_wait_min']].mean()
    assert means.loc['sdsc', 'sim_fcfs_mean_wait_min'] == pytest.approx(174.62, rel=1e-4)
    assert means.loc['sdsc', 'recorded_mean_wait_min'] == pytest.approx(630.92, rel=1e-4)
    assert means.loc['lanl', 'sim_fcfs_mean_wait_min'] == pytest.approx(38.82, rel=1e-3)
    assert means.loc['lanl', 'recorded_mean_wait_min'] == pytest.approx(33.25, rel=1e-3)


# ---------------------------------------------------------------------------
# 8. Phase D: the non-degeneracy condition, demonstrated
#
# Phase D turned a STATED condition into a measured one. These pins guard the
# shape of that result, not merely its values: the baseline arm must keep
# reproducing the published diagnostic (otherwise the parallel experiment is not
# measuring the same thing), every augmented arm must keep breaking the
# degeneracy, and no augmented arm may quietly start beating the heuristic it
# loses to -- because "necessary but not sufficient" is the finding.
# ---------------------------------------------------------------------------

DEGEN_BASELINE = {
    'sdsc': dict(instants=11843, tau=0.7518832993306538, tied=14.413577640800472),
    'lanl': dict(instants=29943, tau=0.6206617494336178, tied=20.72938583308286),
}


def test_the_sweep_baseline_reproduces_the_published_diagnostic():
    """The sweep is only readable if its control agrees with the published run.

    If these drift apart, the sweep is instrumenting something other than the
    published policy and every augmented row becomes uninterpretable.
    """
    sweep = read('05_results/degeneracy/non_degeneracy_sweep.csv')
    published = read('05_results/degeneracy/ranking_degeneracy.csv')
    pub = {'sdsc': 'SDSC SP2 (1998) (8 features)', 'lanl': 'LANL CM-5 (1994) (8 features)'}
    for trace, expected in DEGEN_BASELINE.items():
        row = sweep[(sweep['trace'] == trace) & (sweep['feature_set'] == 'baseline')]
        assert len(row) == 1, trace
        row = row.iloc[0]
        assert int(row['equal_size_diff_pred_violations']) == 0, trace
        assert int(row['ranking_instants']) == expected['instants'], trace
        assert row['kendall_tau_vs_size_mean'] == pytest.approx(expected['tau'], rel=REL)
        assert row['pct_all_scores_tied'] == pytest.approx(expected['tied'], rel=REL)
        # ... and identical to the published diagnostic, not merely close to a literal
        pub_row = published[published['setting'] == pub[trace]].iloc[0]
        assert int(row['ranking_instants']) == int(pub_row['ranking_instants']), trace
        assert row['kendall_tau_vs_size_mean'] == pytest.approx(
            pub_row['kendall_tau_vs_size_mean'], rel=REL), trace
        assert row['pct_all_scores_tied'] == pytest.approx(
            pub_row['pct_all_scores_tied'], rel=REL), trace


def test_every_genuine_per_job_feature_breaks_the_degeneracy():
    """Proposition 2, measured. Five attributes that do not factor through size
    given the state; all twelve augmented arms must produce counterexamples."""
    sweep = read('05_results/degeneracy/non_degeneracy_sweep.csv')
    assert len(sweep) == 14, '7 feature sets x 2 traces'
    assert set(sweep['n_windows']) == {20}, 'the published protocol, not a smoke run'
    aug = sweep[sweep['feature_set'] != 'baseline']
    assert len(aug) == 12
    assert (aug['equal_size_diff_pred_violations'] > 0).all(), (
        'an augmented feature set stopped producing counterexamples: either the extra '
        'column is no longer reaching the model, or it is no longer a genuine per-job '
        'attribute')
    for trace in ('sdsc', 'lanl'):
        base = sweep[(sweep['trace'] == trace) & (sweep['feature_set'] == 'baseline')].iloc[0]
        arms = aug[aug['trace'] == trace]
        # every predicted direction, as a direction rather than a literal
        assert (arms['kendall_tau_vs_size_mean'] < base['kendall_tau_vs_size_mean']).all()
        assert (arms['pct_all_scores_tied'] < base['pct_all_scores_tied']).all()
        assert (arms['mean_distinct_predictions'] > base['mean_distinct_predictions']).all()
        assert (arms['pct_order_identical_to_size'] < base['pct_order_identical_to_size']).all()


def test_breaking_the_degeneracy_does_not_beat_the_heuristic():
    """THE finding: necessary, not sufficient.

    If a future change makes one of these variants genuinely win, this test must
    fail so the claim is restated deliberately rather than drifting.
    """
    util = read('05_results/degeneracy/non_degeneracy_utility.csv')
    assert len(util) == 14
    assert not util['beats_sjf_userest'].any(), (
        'an augmented variant now beats SJF on user estimates -- the "necessary but '
        'not sufficient" claim needs restating')
    assert (util['pct_vs_sjf_userest'] > 0).all(), 'every variant is slower in the paired mean'
    best = util.loc[util['pct_vs_sjf_userest'].idxmin()]
    assert best['trace'] == 'lanl' and best['feature_set'] == '+all'
    assert best['pct_vs_sjf_userest'] == pytest.approx(1.4779942341373555, rel=REL)
    assert bool(best['tost_equivalent_to_sjf']) is True, (
        'the best case is a statistical TIE with the heuristic, which is the point')


# ---------------------------------------------------------------------------
# 9. Phase D: robustness of the claim under attack
# ---------------------------------------------------------------------------

def test_three_attacks_fail_and_only_enqueue_caching_succeeds():
    """The scope of the claim, pinned.

    A1 (learning-to-rank), A2 (monotone transform) and A3 (history window) must
    keep failing: the degeneracy is a property of the INPUTS, not the loss, and a
    lagged cluster state is still shared by every co-queued job. A4 must keep
    succeeding: it is the documented scope limitation, and a paper that stopped
    reporting it would be overclaiming.
    """
    att = read('05_results/degeneracy/robustness_attacks.csv')
    assert len(att) == 14
    assert not att['partial'].any(), 'full protocol, not a smoke run'
    assert set(att['windows']) == {20}
    for trace in ('sdsc', 'lanl'):
        rows = att[att['trace'] == trace]
        survive = rows[rows['attack_id'].isin(['A0', 'A1', 'A2', 'A3'])]
        assert (survive['violations'] == 0).all(), trace
        assert (survive['verdict'] == 'NOT BROKEN').all(), trace
        broke = rows[rows['attack_id'] == 'A4']
        assert len(broke) == 1 and int(broke.iloc[0]['violations']) > 0, trace
        assert broke.iloc[0]['verdict'] == 'BROKEN', trace
    a4 = att[att['attack_id'] == 'A4'].set_index('trace')
    assert int(a4.loc['sdsc', 'violations']) == 20482
    assert int(a4.loc['lanl', 'violations']) == 37741


def test_the_monotone_attack_is_numerically_identical_to_the_baseline():
    """A2 must fail BY CONSTRUCTION -- a strictly monotone map cannot reorder.
    If these rows ever diverge, the instrument is broken, not the theory."""
    att = read('05_results/degeneracy/robustness_attacks.csv')
    for trace in ('sdsc', 'lanl'):
        base = att[(att['trace'] == trace) & (att['attack_id'] == 'A0')].iloc[0]
        for _, row in att[(att['trace'] == trace) & (att['attack_id'] == 'A2')].iterrows():
            for col in ('ranking_instants', 'violations'):
                assert int(row[col]) == int(base[col]), (trace, col)
            for col in ('mean_kendall_tau_vs_size', 'pct_all_scores_tied',
                        'mean_distinct_levels', 'pct_order_identical_to_size'):
                assert row[col] == pytest.approx(base[col], rel=REL), (trace, col)


def test_the_one_successful_attack_does_not_schedule_better():
    """Breaking the degeneracy is not the same as scheduling well. On LANL the
    enqueue-cached policy is worse on mean wait AND bounded slowdown; on SDSC its
    small mean-wait edge is contradicted by median wait and bounded slowdown. Both
    traces put it behind an ML-free heuristic."""
    util = read('05_results/degeneracy/robustness_attack_utility.csv')
    by = util.set_index(['trace', 'scheduler'])
    for trace in ('sdsc', 'lanl'):
        cached = by.loc[(trace, 'PROACTIVE_ENQUEUE_CACHED')]
        proactive = by.loc[(trace, 'PROACTIVE')]
        sjf = by.loc[(trace, 'SJF_USEREST')]
        assert sjf['mean_wait'] < cached['mean_wait'], (
            'the heuristic no longer beats the non-degenerate variant on ' + trace)
        assert cached['mean_bounded_slowdown'] > proactive['mean_bounded_slowdown'], (
            'enqueue-caching is no longer worse on bounded slowdown on ' + trace)
    assert by.loc[('lanl', 'PROACTIVE_ENQUEUE_CACHED'), 'mean_wait'] > \
        by.loc[('lanl', 'PROACTIVE'), 'mean_wait'], 'LANL: worse on mean wait too'

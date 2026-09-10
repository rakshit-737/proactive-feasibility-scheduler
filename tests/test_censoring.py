"""Does the censoring audit actually remove the selection effect it exists to remove?

BACKGROUND
----------
`uncertainty_scheduler_benchmark.py` reports each policy's mean wait over the
jobs that STARTED inside the 300-tick horizon. Different policies strand
different jobs, so the published `improvement_vs_fifo_pct` can compare two means
taken over two different job populations. `04_scheduler/censoring_analysis.py`
adds the paired statistic that fixes this: a mean over the jobs that started
under BOTH policies.

WHAT THESE TESTS PIN
--------------------
Sections (a)-(c) are the point of the file. They feed `compare_pair` a
hand-built pair of policies with a KNOWN, extreme selection effect and assert
that the published-style number and the common-set number disagree in the way
selection bias predicts -- in BOTH directions:

  (a) a cherry-picking policy that starts only the short jobs must look far
      better than the baseline as published, and no better (in fact worse) on
      the common set;
  (b) a rescuing policy that starts the jobs the baseline strands must look far
      worse than the baseline as published, and roughly as good on the common
      set -- this is the direction the real `arr2.0_nodes4` scenario takes;
  (c) the started FRACTION must be reported alongside, since a common-set mean
      read without it hides how much of the workload the comparison discarded.

If the common-set path ever falls back to each policy's own started set --
the one-line regression that would silently un-do this whole analysis -- (a) and
(b) both go red, because the common-set number becomes the published number.

Sections (d)-(e) guard the recovery of the started set from the simulator's
return value and the arithmetic guards; section (f) checks the committed
artefact against the benchmark it audits, and skips on a fresh clone.

Section (g) pins the SIGN CONVENTION of `selection_gap_pp`, which is the part
the audit previously got backwards. The gap is
`improvement_pct_published - improvement_pct_common`, so a POSITIVE gap means
the published figure OVERSTATED the improvement (that much of it disappears
once both policies are scored on the same jobs) and a NEGATIVE gap means it
UNDERSTATED it (the paired comparison is MORE favourable). The committed
artefact takes the second direction: 24 of its 30 rows have a gap of exactly
0.0 -- four of the five scenarios start every job under every policy and carry
no censoring at all -- and every one of the 6 non-zero gaps is NEGATIVE. A note
column that emits "N pp of it disappears on the common set" for those rows
states the result the wrong way round, and (g) fails if it ever does again.
"""

import math

import numpy as np
import pandas as pd
import pytest

_POINT_MODEL = '03_models/wait_model_v2.pkl'
_QUANT_MODEL = '03_models/wait_model_quantile.pkl'
_AUDIT_CSV = '05_results/uncertainty/censoring_analysis.csv'
_BENCH_CSV = '05_results/uncertainty/uncertainty_ood_benchmark.csv'

HORIZON = 100  # stand-in for SIM_TIME in the hand-made fixtures


def _censoring():
    """Import censoring_analysis, skipping cleanly when its import-time inputs are absent.

    `censoring_analysis` imports `uncertainty_scheduler_benchmark`, which
    unpickles both trained models at module scope, so a fresh clone that has not
    run the pipeline must skip rather than error.
    """
    from conftest import require
    pytest.importorskip('xgboost',
                        reason='uncertainty_scheduler_benchmark unpickles an XGBoost model')
    require(_POINT_MODEL, 'trained wait model')
    require(_QUANT_MODEL, 'trained quantile model')
    import censoring_analysis
    return censoring_analysis


def _waits(started, censored_arrivals, horizon=HORIZON):
    """Build a `run_once`-shaped job_waits dict.

    `started` maps job id -> observed wait. `censored_arrivals` maps job id ->
    arrival time for jobs that never started; those enter the dict at their
    censoring bound `horizon - arrival`, exactly as the simulator records them.
    """
    waits = dict(started)
    for jid, arrival in censored_arrivals.items():
        waits[jid] = horizon - arrival
    return waits


# -- (a) a cherry-picking policy: published looks great, common set does not ----
def test_common_set_kills_the_cherry_picking_advantage():
    """A policy that starts ONLY the short jobs must not keep its published win.

    BASELINE starts all ten jobs: five short ones wait 10, five long ones wait
    50, so its published mean is 30. CHERRY starts only the five short jobs, and
    is even slightly SLOWER on them (12 each); the five long jobs never start.
    Its published mean is 12 -- a 60% "improvement" bought entirely by refusing
    to run the hard half of the workload.

    On the common set (the five short jobs, the only ones both policies ran)
    CHERRY is 20% WORSE. Published and common-set therefore disagree in SIGN,
    which is the whole reason the common-set column exists.
    """
    ca = _censoring()
    short, long_ = [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]

    baseline_waits = _waits({j: 10.0 for j in short} | {j: 50.0 for j in long_}, {})
    baseline_started = set(short) | set(long_)

    cherry_waits = _waits({j: 12.0 for j in short}, {j: 0 for j in long_})
    cherry_started = set(short)

    stats = ca.compare_pair(baseline_waits, baseline_started, cherry_waits, cherry_started)

    # The selection is real and large: CHERRY ran half the workload.
    assert stats['n_started_a'] == 10
    assert stats['n_started_b'] == 5
    assert stats['n_common'] == 5

    assert stats['mean_wait_a_published'] == pytest.approx(30.0)
    assert stats['mean_wait_b_published'] == pytest.approx(12.0)
    # The common-set means must be taken over the SHARED five jobs, so the
    # baseline's number drops from 30 to 10 -- it is no longer being charged for
    # the long jobs its rival simply refused to run.
    assert stats['mean_wait_a_common'] == pytest.approx(10.0)
    assert stats['mean_wait_b_common'] == pytest.approx(12.0)

    published = ca.improvement_pct(stats['mean_wait_a_published'],
                                   stats['mean_wait_b_published'])
    common = ca.improvement_pct(stats['mean_wait_a_common'],
                                stats['mean_wait_b_common'])

    assert published == pytest.approx(60.0), 'published-style improvement must be large'
    assert common == pytest.approx(-20.0), 'common-set improvement must be negative'

    # The discriminating assertion. If the common-set path ever falls back to
    # each policy's own started set, `common` becomes `published` (+60) and both
    # of these fail.
    assert published > 0.0 > common, (
        f'common-set statistic did not remove the selection effect: '
        f'published {published:+.2f}% vs common-set {common:+.2f}% -- these must '
        f'disagree in sign on a workload where one policy ran only the easy jobs')
    assert published - common > 50.0, (
        f'selection gap collapsed to {published - common:.2f} pp; a cherry-picking '
        f'policy must lose most of its published advantage on the common set')


# -- (b) the opposite direction: the policy that rescues stranded jobs ---------
def test_common_set_rescues_the_policy_that_starts_more_jobs():
    """Starting MORE jobs must not be punished by the published statistic.

    This is the direction the real `arr2.0_nodes4` scenario takes. STRANDER
    starts only the five short jobs (wait 10 each, published mean 10). RESCUER
    starts all ten: the same five short jobs at 12, plus the five the baseline
    stranded, which necessarily wait a long time (60). Its published mean is 36,
    so as published RESCUER looks catastrophically worse -- purely because it is
    the only one being charged for the hard jobs at all.

    On the common set the two policies are within 20% of each other. The
    common-set number must therefore be very much HIGHER than the published one,
    i.e. the published column understates the better policy.
    """
    ca = _censoring()
    short, long_ = [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]

    strander_waits = _waits({j: 10.0 for j in short}, {j: 0 for j in long_})
    strander_started = set(short)

    rescuer_waits = _waits({j: 12.0 for j in short} | {j: 60.0 for j in long_}, {})
    rescuer_started = set(short) | set(long_)

    stats = ca.compare_pair(strander_waits, strander_started, rescuer_waits, rescuer_started)
    published = ca.improvement_pct(stats['mean_wait_a_published'],
                                   stats['mean_wait_b_published'])
    common = ca.improvement_pct(stats['mean_wait_a_common'],
                                stats['mean_wait_b_common'])

    assert stats['n_common'] == 5
    assert published == pytest.approx(-260.0)
    assert common == pytest.approx(-20.0)
    assert common - published > 100.0, (
        f'common-set statistic did not correct the penalty on the policy that '
        f'started more jobs: published {published:+.2f}% vs common-set '
        f'{common:+.2f}%')


# -- (c) the censoring-aware view over ALL jobs --------------------------------
def test_started_fraction_and_lower_bound_cover_every_job():
    """The censoring-aware columns must describe the WHOLE workload, not a subset.

    The started fraction says how much of the workload each policy actually ran;
    the lower-bound mean averages over every job, a stranded one entering at its
    censoring bound (horizon - arrival). No wait time is invented for a job that
    never ran -- the bound is what the simulator already records.
    """
    ca = _censoring()
    short, long_ = [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]

    baseline_waits = _waits({j: 10.0 for j in short} | {j: 50.0 for j in long_}, {})
    cherry_waits = _waits({j: 12.0 for j in short}, {j: 0 for j in long_})

    stats = ca.compare_pair(baseline_waits, set(short) | set(long_),
                            cherry_waits, set(short))

    assert stats['n_jobs'] == 10
    assert stats['started_frac_a_pct'] == pytest.approx(100.0)
    assert stats['started_frac_b_pct'] == pytest.approx(50.0)

    # Lower-bound means run over all ten jobs on both sides: no selection at all.
    assert stats['mean_wait_a_lowerbound'] == pytest.approx(30.0)
    assert stats['mean_wait_b_lowerbound'] == pytest.approx((5 * 12.0 + 5 * 100.0) / 10)
    lower = ca.improvement_pct(stats['mean_wait_a_lowerbound'],
                               stats['mean_wait_b_lowerbound'])
    assert lower < 0.0, 'a policy that strands half the jobs must not lead on the all-jobs view'

    # And the note must name the artefact in words, so the CSV cannot report a
    # large selection gap while describing it as clean.
    row = {'policy_a': 'BASELINE', 'policy_b': 'CHERRY',
           'started_frac_a_pct': stats['started_frac_a_pct'],
           'started_frac_b_pct': stats['started_frac_b_pct'],
           'selection_gap_pp': 80.0}
    note = ca.note_for(row)
    assert 'selection artefact' in note
    assert '50.0%' in note

    clean = ca.note_for({'policy_a': 'A', 'policy_b': 'B',
                         'started_frac_a_pct': 100.0, 'started_frac_b_pct': 100.0,
                         'selection_gap_pp': 0.0})
    assert 'no censoring' in clean


# -- (d) recovering the started set from what the simulator returns ------------
def test_started_job_ids_separates_censored_from_started():
    """A censored wait is EXACTLY horizon - arrival; a real wait is strictly below it.

    `run_once` reports censored jobs at their bound rather than dropping them, so
    the two populations are separable without changing the benchmark. The
    boundary case matters: a job that starts on the last possible tick
    (horizon - 1) must still count as started.
    """
    ca = _censoring()
    arrivals = {0: 0, 1: 30, 2: 30, 3: 99, 4: 0}
    waits = {
        0: 99.0,    # started at the very last tick -> started
        1: 70.0,    # censored: 100 - 30
        2: 69.0,    # started one tick before the horizon
        3: 1.0,     # censored: 100 - 99
        4: 0.0,     # started immediately
    }
    assert ca.started_job_ids(waits, arrivals, sim_time=HORIZON) == {0, 2, 4}


def test_started_job_ids_defaults_to_the_benchmark_horizon():
    ca = _censoring()
    horizon = ca.usb.SIM_TIME
    arrivals = {0: 10, 1: 10}
    waits = {0: float(horizon - 10), 1: float(horizon - 11)}
    assert ca.started_job_ids(waits, arrivals) == {1}


# -- (e) arithmetic guards -----------------------------------------------------
def test_improvement_pct_refuses_a_degenerate_or_missing_baseline():
    ca = _censoring()
    assert ca.improvement_pct(20.0, 15.0) == pytest.approx(25.0)
    assert math.isnan(ca.improvement_pct(0.0, 0.0))
    assert math.isnan(ca.improvement_pct(-1.0, 1.0))
    assert math.isnan(ca.improvement_pct(float('nan'), 1.0))
    assert math.isnan(ca.improvement_pct(10.0, float('nan')))


def test_compare_pair_reports_nan_rather_than_inventing_an_empty_mean():
    """Two policies with no started job in common must yield NaN, not zero."""
    ca = _censoring()
    a_waits = _waits({0: 5.0}, {1: 0})
    b_waits = _waits({1: 7.0}, {0: 0})
    stats = ca.compare_pair(a_waits, {0}, b_waits, {1})
    assert stats['n_common'] == 0
    assert math.isnan(stats['mean_wait_a_common'])
    assert math.isnan(stats['mean_wait_b_common'])


# -- (f) the committed artefact must audit the benchmark it claims to audit ----
def test_committed_audit_reproduces_the_published_benchmark():
    """`*_published` must equal the committed benchmark cell for cell.

    The whole argument rests on the published and common-set columns differing
    only by the SELECTION. If the published column were computed by some other
    estimator, the gap would be uninterpretable.
    """
    from conftest import require
    audit = pd.read_csv(require(_AUDIT_CSV, 'censoring audit'))
    bench = pd.read_csv(require(_BENCH_CSV, 'uncertainty OOD benchmark'))
    ref = bench.set_index(['scenario', 'policy'])

    required = ['scenario', 'policy_a', 'policy_b', 'n_jobs', 'n_started_a', 'n_started_b',
                'n_common', 'mean_wait_a_published', 'mean_wait_b_published',
                'mean_wait_a_common', 'mean_wait_b_common', 'improvement_pct_published',
                'improvement_pct_common', 'note']
    missing = [c for c in required if c not in audit.columns]
    assert not missing, f'censoring_analysis.csv is missing columns: {missing}'
    assert len(audit) > 0

    for _, row in audit.iterrows():
        for side in ('a', 'b'):
            cell = ref.loc[(row['scenario'], row[f'policy_{side}'])]
            assert row[f'mean_wait_{side}_published'] == pytest.approx(
                cell['mean_wait'], rel=1e-9, abs=1e-9), (
                f"{row['scenario']}/{row[f'policy_{side}']}: published mean wait "
                f"does not match uncertainty_ood_benchmark.csv")
            assert row[f'n_started_{side}'] == pytest.approx(
                cell['n_started'], rel=1e-9, abs=1e-9)
        if row['policy_a'] == 'FIFO':
            cell = ref.loc[(row['scenario'], row['policy_b'])]
            assert row['improvement_pct_published'] == pytest.approx(
                cell['improvement_vs_fifo_pct'], rel=1e-8, abs=1e-8)


def test_committed_audit_is_internally_consistent():
    """Common set cannot exceed either policy's started set, and the gap must add up."""
    from conftest import require
    audit = pd.read_csv(require(_AUDIT_CSV, 'censoring audit'))

    for _, row in audit.iterrows():
        tag = f"{row['scenario']}/{row['policy_a']}-vs-{row['policy_b']}"
        assert row['n_common'] <= row['n_started_a'] + 1e-9, tag
        assert row['n_common'] <= row['n_started_b'] + 1e-9, tag
        assert row['n_started_a'] <= row['n_jobs'] + 1e-9, tag
        assert row['n_started_b'] <= row['n_jobs'] + 1e-9, tag

        pub, com, gap = (row['improvement_pct_published'],
                         row['improvement_pct_common'], row['selection_gap_pp'])
        if np.isfinite(pub) and np.isfinite(com):
            assert gap == pytest.approx(pub - com, rel=1e-8, abs=1e-8), tag

        # A row with no censoring on either side is a row where the published
        # comparison was already paired, so the two columns must agree exactly.
        if (row['n_started_a'] == pytest.approx(row['n_jobs'])
                and row['n_started_b'] == pytest.approx(row['n_jobs'])):
            assert row['n_common'] == pytest.approx(row['n_jobs']), tag
            if np.isfinite(pub):
                assert gap == pytest.approx(0.0, abs=1e-9), (
                    f'{tag}: no job is censored, so the published and common-set '
                    f'improvements cannot differ')


# -- (g) the SIGN of selection_gap_pp, stated the right way round --------------
def _censored_row(gap, policy_a='FIFO', policy_b='PROACTIVE'):
    """One aggregated row with censoring on both sides and the given gap."""
    return {'policy_a': policy_a, 'policy_b': policy_b,
            'started_frac_a_pct': 51.7, 'started_frac_b_pct': 59.7,
            'selection_gap_pp': gap}


def test_note_says_understated_when_the_common_set_improvement_is_larger():
    """gap < 0 means published < common, i.e. the published figure UNDERSTATED the pair.

    This is the direction EVERY non-zero row of the committed artefact takes
    (6 rows, all negative, all in `arr2.0_nodes4`). Nothing "disappears on the
    common set" there: correcting the selection makes `policy_b` look BETTER,
    not worse, because the unpaired statistic was charging the policy that
    started more jobs for the long waits of the jobs its rival never ran.
    """
    ca = _censoring()
    gap = -3.3
    note = ca.note_for(_censored_row(gap))

    assert 'UNDERSTATED' in note, (
        f'a negative gap means the published figure understated the pair; note said: {note}')
    assert 'OVERSTATED' not in note
    # The exact defect being pinned: the one-directional wording that was
    # emitted for negative gaps too, and which reversed the finding.
    assert 'disappears' not in note, (
        f'nothing disappears on the common set when the gap is negative -- the '
        f'common-set improvement is LARGER than the published one; note said: {note}')
    assert 'selection artefact' not in note, (
        f'a negative gap is the opposite of a flattering selection artefact; note said: {note}')
    assert f'{abs(gap):.1f} pp LARGER' in note, (
        f'the note must quantify how much larger the common-set improvement is; '
        f'note said: {note}')
    assert 'PROACTIVE' in note


def test_note_says_overstated_when_the_published_improvement_is_larger():
    """gap > 0 is the other direction: part of the published improvement was selection.

    No row of the committed artefact currently takes this direction, which is
    exactly why it needs a test: the wording must already be correct if the
    benchmark ever produces one.
    """
    ca = _censoring()
    gap = 3.3
    note = ca.note_for(_censored_row(gap))

    assert 'OVERSTATED' in note, (
        f'a positive gap means part of the published improvement was a selection '
        f'artefact; note said: {note}')
    assert 'UNDERSTATED' not in note
    assert 'selection artefact' in note
    assert f'{gap:.1f} pp of it' in note and 'disappears on the common set' in note, (
        f'a positive gap is the case where the improvement really does shrink on '
        f'the common set; note said: {note}')


def test_note_wording_flips_with_the_sign_and_stays_neutral_inside_the_band():
    """The two directions must not share wording, and a small gap must claim neither."""
    ca = _censoring()
    negative = ca.note_for(_censored_row(-4.0))
    positive = ca.note_for(_censored_row(4.0))
    assert negative != positive
    assert ('UNDERSTATED' in negative) and ('UNDERSTATED' not in positive)
    assert ('OVERSTATED' in positive) and ('OVERSTATED' not in negative)

    inside = ca.note_for(_censored_row(-0.4))
    assert 'UNDERSTATED' not in inside and 'OVERSTATED' not in inside, (
        f'a gap inside the {ca.GAP_FLAG_PP:.1f} pp band must not claim a direction; '
        f'note said: {inside}')
    assert f'within {ca.GAP_FLAG_PP:.1f} pp of published' in inside


def test_note_carries_the_lower_bound_caveat_wherever_censoring_exists():
    """`improvement_pct_lowerbound` is a ratio of two bounds, not a bound on the ratio.

    The column is kept rather than dropped, so the artefact itself has to carry
    the limitation: a lower bound on each of two means bounds neither the sign
    nor the size of their percent difference. Every censored row must say so,
    in the row, not only in the module docstring.
    """
    ca = _censoring()
    for gap in (-4.0, -0.4, 0.0, 4.0, float('nan')):
        note = ca.note_for(_censored_row(gap))
        assert 'improvement_pct_lowerbound' in note, (
            f'gap={gap}: censored row does not mention the derived column at all')
        assert 'NOT itself a bound' in note, (
            f'gap={gap}: the ratio of two lower bounds is not a bound; note said: {note}')
        assert 'descriptive only' in note

    # A row with no censoring needs no caveat: with nothing censored the bounds
    # are exact, so the derived column equals the published improvement.
    clean = ca.note_for({'policy_a': 'A', 'policy_b': 'B',
                         'started_frac_a_pct': 100.0, 'started_frac_b_pct': 100.0,
                         'selection_gap_pp': 0.0})
    assert 'NOT itself a bound' not in clean
    assert 'improvement_pct_lowerbound equals the published improvement' in clean


def test_compare_pair_decomposes_the_started_sets_in_both_directions():
    """A net started-job count hides whether the two sets are nested or exchanging.

    A starts {0..4}; B starts {3..9}. On the net count B "starts two more jobs",
    which reads like a one-way rescue. The decomposition shows it is not: three
    jobs start under A ONLY and five under B ONLY, so neither started set
    contains the other and each policy strands jobs its rival ran. This is the
    shape the real `arr2.0_nodes4` FIFO pairs have (common 91.9, FIFO-only 21.8,
    PROACTIVE-only 39.4), and it is why the audit may not be summarised as one
    policy rescuing a superset of the other's jobs.
    """
    ca = _censoring()
    a_started, b_started = {0, 1, 2, 3, 4}, {3, 4, 5, 6, 7, 8, 9}
    a_waits = _waits({j: 10.0 for j in a_started},
                     {j: 0 for j in range(10) if j not in a_started})
    b_waits = _waits({j: 10.0 for j in b_started},
                     {j: 0 for j in range(10) if j not in b_started})

    stats = ca.compare_pair(a_waits, a_started, b_waits, b_started)

    assert stats['n_common'] == 2
    assert stats['n_only_a'] == 3, (
        'jobs that started under A only must be reported; a zero here would let a '
        'two-way exchange be described as a one-way rescue')
    assert stats['n_only_b'] == 5
    # The decomposition must partition each started set exactly.
    assert stats['n_common'] + stats['n_only_a'] == stats['n_started_a']
    assert stats['n_common'] + stats['n_only_b'] == stats['n_started_b']
    # And it must contradict the "superset" reading that the net count invites.
    assert stats['n_started_b'] - stats['n_started_a'] == 2
    assert stats['n_only_a'] > 0 and stats['n_only_b'] > 0


# -- (h) the committed artefact must state its own direction correctly ---------
def test_committed_notes_agree_with_the_sign_of_their_own_gap():
    """Every delivered note must match the sign of the number in its own row.

    This is the artefact-level form of (g): the CSV is what a reader sees, and
    it once described six negative gaps with wording that meant the opposite.
    """
    from conftest import require
    audit = pd.read_csv(require(_AUDIT_CSV, 'censoring audit'))
    flag_pp = 1.0  # GAP_FLAG_PP, restated so this test does not import the module

    for _, row in audit.iterrows():
        gap, note = row['selection_gap_pp'], str(row['note'])
        tag = f"{row['scenario']}/{row['policy_a']}-vs-{row['policy_b']} (gap {gap:+.2f} pp)"
        uncensored = (row['started_frac_a_pct'] == pytest.approx(100.0)
                      and row['started_frac_b_pct'] == pytest.approx(100.0))
        if uncensored:
            assert 'no censoring' in note, tag
            continue
        assert 'improvement_pct_lowerbound' in note and 'NOT itself a bound' in note, (
            f'{tag}: censored row ships the derived lower-bound ratio with no caveat')
        if not np.isfinite(gap) or abs(gap) <= flag_pp:
            assert 'UNDERSTATED' not in note and 'OVERSTATED' not in note, tag
        elif gap < 0:
            assert 'UNDERSTATED' in note and 'disappears' not in note, (
                f'{tag}: the common-set improvement is LARGER than the published one, '
                f'so nothing disappears on the common set. Note reads: {note}')
        else:
            assert 'OVERSTATED' in note and 'disappears on the common set' in note, tag


def test_committed_gaps_are_mostly_zero_and_never_positive():
    """24 of 30 rows have no censoring at all; every NON-ZERO gap is negative.

    Both halves are load-bearing. "Four of five scenarios carry no censoring"
    is the headline -- the concern touches one overload scenario, not the
    benchmark -- and "every non-zero gap is negative" is the direction: where
    the selection bites it penalised the learned policies rather than
    flattering them. If a regeneration ever moves either count, the claim in
    this module's docstring and in `censoring_analysis`'s must be restated.
    """
    from conftest import require
    audit = pd.read_csv(require(_AUDIT_CSV, 'censoring audit'))
    gaps = audit['selection_gap_pp']

    assert len(audit) == 30
    assert int((gaps == 0.0).sum()) == 24, (
        f'expected 24 of 30 rows to have no censoring and hence a zero gap, got '
        f'{int((gaps == 0.0).sum())}')
    assert int((gaps < 0.0).sum()) == 6
    assert int((gaps > 0.0).sum()) == 0, (
        f'a positive gap would mean the published improvement was partly a selection '
        f'artefact; the committed rows are all the other way round. Positive rows: '
        f'{audit.loc[gaps > 0.0, ["scenario", "policy_a", "policy_b"]].to_dict("records")}')

    censored_scenarios = set(audit.loc[gaps != 0.0, 'scenario'])
    assert censored_scenarios == {'arr2.0_nodes4'}, (
        f'the selection effect is supposed to be confined to one overload scenario; '
        f'found it in {sorted(censored_scenarios)}')

    # A zero gap must be a genuinely uncensored row, not a coincidence.
    for _, row in audit[gaps == 0.0].iterrows():
        assert row['n_common'] == pytest.approx(row['n_jobs']), (
            f"{row['scenario']}/{row['policy_a']}-vs-{row['policy_b']}: zero gap but "
            f"the common set is smaller than the job set")


def test_committed_started_sets_decompose_and_show_a_two_way_exchange():
    """The -only columns must partition each started set, and expose the exchange.

    For the three FIFO pairs in `arr2.0_nodes4` BOTH -only counts are non-zero:
    each policy starts jobs the other strands, so the learned policies do not
    simply rescue a superset of FIFO's jobs. (`UCB` vs `GUARDED` is nested
    instead, which is why the shape is asserted per row rather than globally.)
    """
    from conftest import require
    audit = pd.read_csv(require(_AUDIT_CSV, 'censoring audit'))
    for col in ('n_only_a', 'n_only_b'):
        assert col in audit.columns, f'censoring_analysis.csv is missing {col}'

    for _, row in audit.iterrows():
        tag = f"{row['scenario']}/{row['policy_a']}-vs-{row['policy_b']}"
        assert row['n_common'] + row['n_only_a'] == pytest.approx(row['n_started_a']), tag
        assert row['n_common'] + row['n_only_b'] == pytest.approx(row['n_started_b']), tag

    fifo_pairs = audit[(audit['scenario'] == 'arr2.0_nodes4')
                       & (audit['policy_a'] == 'FIFO')]
    assert len(fifo_pairs) == 3
    for _, row in fifo_pairs.iterrows():
        tag = f"{row['scenario']}/FIFO-vs-{row['policy_b']}"
        assert row['n_only_a'] > 0.0, (
            f'{tag}: FIFO starts {row["n_only_a"]} jobs that {row["policy_b"]} never '
            f'starts, so this is a two-way exchange and must not be reported as '
            f'{row["policy_b"]} rescuing a superset of FIFO jobs')
        assert row['n_only_b'] > 0.0, tag
        assert row['n_only_b'] > row['n_only_a'], tag

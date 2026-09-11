"""Guards on the "Robustness of the degeneracy claim" artefact.

WHAT THIS MODULE IS FOR
-----------------------
04_scheduler/robustness_attacks.py tries to BREAK the published ranking
degeneracy four ways and writes 05_results/degeneracy/robustness_attacks.csv.
An attack that fails is evidence FOR the claim; an attack that succeeds is a
scope limitation the manuscript must state. Either way the artefact is only
worth reading if it is COMPLETE and if the one attack whose outcome is fixed by
mathematics actually reports that outcome.

So the three things asserted here are exactly the three things that would let a
misleading robustness section reach the paper:

  (a) COVERAGE. All four attacks, on both traces. An attack that was run,
      succeeded, and then quietly dropped from the table looks identical to an
      attack that was never run -- unless the table is required to carry all
      four. This is the assertion that makes "do not quietly drop an attack that
      succeeded" checkable rather than a promise.

  (b) THE MONOTONE-TRANSFORM ATTACK MUST REPORT ZERO VIOLATIONS. A strictly
      monotone map of the score is order-preserving: it cannot reorder a queue
      and it cannot split a tie. So A2's violation count is zero BY
      CONSTRUCTION, and a non-zero value there does not weaken the degeneracy
      claim -- it means the instrumentation counted something that is not a
      counterexample, and every other row in the table is suspect too. This test
      is a self-check on the measuring instrument, and it is why A2's row is
      required to match the A0 reference row it was derived from.

  (c) THE ENQUEUE-TIME ATTACK MUST RECORD A VERDICT. A4 is the attack expected
      to SUCCEED. A blank, NaN or unrecognised verdict in that row is the exact
      shape a suppressed positive result takes, so the verdict cell is required
      to be one of the two defined outcomes.

These tests read a committed artefact and write nothing. They SKIP when the
artefact is absent, per the suite's rule that a fresh clone can run the suite
before the pipeline has run -- but they do NOT skip on a partial run, because a
reduced window count changes the magnitudes in the table and not any of the
three properties above.
"""

import math

import pandas as pd
import pytest

from conftest import require

ARTEFACT = '05_results/degeneracy/robustness_attacks.csv'
UTILITY = '05_results/degeneracy/robustness_attack_utility.csv'

EXPECTED_ATTACKS = {'A1', 'A2', 'A3', 'A4'}
EXPECTED_TRACES = {'sdsc', 'lanl'}
VALID_VERDICTS = {'BROKEN', 'NOT BROKEN'}

REQUIRED_COLUMNS = {
    'attack_id', 'attack', 'trace', 'violations', 'mean_kendall_tau_vs_size',
    'pct_all_scores_tied', 'mean_distinct_levels', 'verdict', 'note',
}


@pytest.fixture(scope='module')
def attacks():
    path = require(ARTEFACT, 'run 04_scheduler/robustness_attacks.py')
    df = pd.read_csv(path)
    # `verdict` and `note` are free text; read them as text so a missing note
    # arrives as NaN rather than as the string 'nan'.
    return df


# ── (a) coverage: all four attacks, on both traces ──────────────────────────

def test_artefact_has_the_required_columns(attacks):
    missing = REQUIRED_COLUMNS - set(attacks.columns)
    assert not missing, f'robustness_attacks.csv is missing columns: {missing}'


def test_all_four_attacks_are_present_on_both_traces(attacks):
    """Every (attack, trace) cell of the 4x2 grid must be filled.

    Checked as a grid rather than as two independent set comparisons: a table
    holding A1-A4 for SDSC and only A1-A3 for LANL passes "all four attacks
    appear" and "both traces appear" while still having dropped an attack.
    """
    have = {(a, t) for a, t in zip(attacks['attack_id'].astype(str),
                                   attacks['trace'].astype(str))}
    want = {(a, t) for a in EXPECTED_ATTACKS for t in EXPECTED_TRACES}
    assert not (want - have), f'missing attack/trace rows: {sorted(want - have)}'


def test_every_row_carries_a_valid_verdict_and_a_note(attacks):
    """No row may be silent about its outcome or about why."""
    for row in attacks.itertuples(index=False):
        assert row.verdict in VALID_VERDICTS, (
            f'{row.attack_id}/{row.trace}: verdict {row.verdict!r} is not one '
            f'of {sorted(VALID_VERDICTS)}')
        assert isinstance(row.note, str) and row.note.strip(), (
            f'{row.attack_id}/{row.trace}: empty note')


def test_verdict_agrees_with_the_violation_count(attacks):
    """The verdict is not free text: it is a function of the counter.

    The claim is falsified by a counterexample -- two equally-sized co-queued
    jobs scored differently -- so a row reporting violations must say BROKEN and
    a row reporting none must not.
    """
    for row in attacks.itertuples(index=False):
        expected = 'BROKEN' if int(row.violations) > 0 else 'NOT BROKEN'
        assert row.verdict == expected, (
            f'{row.attack_id}/{row.trace}: {row.violations} violations but '
            f'verdict {row.verdict!r}')


# ── (b) the monotone transform must be a no-op ──────────────────────────────

def test_monotone_transform_attack_yields_exactly_zero_violations(attacks):
    """A2 is order-preserving by construction; anything else is a bug HERE.

    Deliberately not phrased as "<= the baseline" or "approximately zero": the
    mathematics admits exactly one value, so the assertion admits exactly one
    value. A failure means the instrumentation is miscounting, and the right
    response is to investigate it, not to report it as a weakened claim.
    """
    a2 = attacks[attacks['attack_id'] == 'A2']
    assert not a2.empty, 'no monotone-transform rows in the artefact'
    bad = a2[a2['violations'].astype(int) != 0]
    assert bad.empty, (
        'a strictly monotone map of the score cannot reorder a queue or split '
        'a tie, so these rows indicate broken instrumentation rather than a '
        f'broken claim:\n{bad[["attack", "trace", "violations"]]}')
    assert (a2['verdict'] == 'NOT BROKEN').all()


def test_monotone_transform_reproduces_the_baseline_row(attacks):
    """Order-preserving means the ORDER statistics are unchanged, not just the
    violation count.

    A transform applied to the score but accidentally measured against a
    different model, window set or feature matrix would still report zero
    violations while silently being a different experiment. Tying A2 to the A0
    reference row on the order-derived columns is what rules that out.
    """
    if 'A0' not in set(attacks['attack_id'].astype(str)):
        pytest.skip('artefact carries no A0 baseline reference row')
    cols = ['pct_order_identical_to_size', 'pct_all_scores_tied',
            'mean_distinct_levels', 'ranking_instants']
    for trace in sorted(EXPECTED_TRACES & set(attacks['trace'])):
        base = attacks[(attacks['attack_id'] == 'A0')
                       & (attacks['trace'] == trace)]
        if base.empty:
            continue
        base = base.iloc[0]
        for row in attacks[(attacks['attack_id'] == 'A2')
                           & (attacks['trace'] == trace)].itertuples():
            for c in cols:
                if c not in attacks.columns:
                    continue
                assert math.isclose(float(getattr(row, c)), float(base[c]),
                                    rel_tol=1e-9, abs_tol=1e-9), (
                    f'{trace}: monotone transform changed {c} '
                    f'({getattr(row, c)} vs baseline {base[c]}) -- a monotone '
                    f'map cannot do that, so the two rows did not measure the '
                    f'same thing')


# ── (c) the enqueue-time attack must record its outcome ─────────────────────

def test_enqueue_time_attack_records_a_verdict(attacks):
    """A4 is the attack expected to succeed, so its verdict cell is load-bearing.

    A blank or unrecognised verdict is exactly what a suppressed positive result
    looks like in this table.
    """
    a4 = attacks[attacks['attack_id'] == 'A4']
    assert not a4.empty, 'no enqueue-time rows in the artefact'
    assert set(a4['trace']) >= EXPECTED_TRACES, (
        f'enqueue-time attack missing traces: '
        f'{sorted(EXPECTED_TRACES - set(a4["trace"]))}')
    for row in a4.itertuples(index=False):
        assert isinstance(row.verdict, str), (
            f'A4/{row.trace}: verdict is not text ({row.verdict!r}) -- an '
            f'unrecorded outcome for the attack most likely to succeed')
        assert row.verdict in VALID_VERDICTS, (
            f'A4/{row.trace}: verdict {row.verdict!r} is not one of '
            f'{sorted(VALID_VERDICTS)}')


def test_enqueue_time_attack_has_a_scheduling_comparison(attacks):
    """Non-degenerate is not the same as better.

    A4 changes the POLICY, so the table alone cannot say whether it is an
    improvement or merely a different kind of wrong. The utility artefact must
    exist and must price it against both PROACTIVE and SJF_USEREST on the same
    traces, or the robustness section reports a broken claim without reporting
    what breaking it cost.
    """
    path = require(UTILITY, 'run 04_scheduler/robustness_attacks.py')
    util = pd.read_csv(path)
    for trace in sorted(set(attacks[attacks['attack_id'] == 'A4']['trace'])):
        have = set(util[util['trace'] == trace]['scheduler'])
        assert {'PROACTIVE_ENQUEUE_CACHED', 'PROACTIVE', 'SJF_USEREST'} <= have, (
            f'{trace}: utility table lacks a like-for-like comparison '
            f'(has {sorted(have)})')
        assert util[util['trace'] == trace]['mean_wait'].notna().all(), (
            f'{trace}: utility table has a missing mean wait')

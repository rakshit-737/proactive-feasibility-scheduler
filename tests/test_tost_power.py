"""Tests for 04_scheduler/tost_power.py — the power analysis behind the
"LANL is INCONCLUSIVE, not different" reading of the trace-driven results.

WHAT IS ACTUALLY AT RISK HERE
-----------------------------
`tost_power.py` exists to say a specific uncomfortable thing: the published
20-window TOST on LANL had ~0.5% chance of certifying equivalence, so its
failure to certify is not evidence of a difference. Every number in that
argument is produced by ONE decision rule — `tost_power.tost_reject` — and a
power estimate is only as trustworthy as that rule. Invert it, loosen it, or let
it drift away from `simstats.tost_equivalence`, and the artefact fills with
confident numbers that mean the opposite of what they say, with nothing on the
surface looking wrong.

The rule can break in two directions, and they need DIFFERENT tests.

INVERSION -- certifying equivalence where the effect is large -- is caught by
the two CALIBRATION checks:

  * a paired-difference distribution centred FAR OUTSIDE the margin must yield
    power to conclude equivalence near 0;
  * one centred at 0 with negligible variance must yield power near 1.

plus a third at the margin boundary, where TOST's power is pinned to alpha by
construction. A power function that fails any of these cannot be believed
anywhere else, and an inverted decision rule fails all three.

LOOSENING -- rejecting a little too readily, which inflates every power number
in the artefact while nothing on the surface looks wrong -- is the realistic
regression, and the calibration checks above are far too coarse to see it: with
the decision rule run at an effective alpha 1.75x too large, every one of them
still passed. It is caught here by two DETERMINISTIC tests that put the rule at
the alpha boundary instead of sampling near it:

  * `test_tost_reject_holds_the_alpha_knife_edge` evaluates the rule on two
    constructed samples whose p_TOST straddles alpha by 4e-7 and requires the
    verdict to flip exactly between them;
  * `test_tost_reject_refuses_the_zone_just_above_alpha` sweeps the paired mean
    through both tails and requires agreement with `simstats` at every point,
    including the "just too weak to certify" band where p_TOST sits at
    1-10x alpha.

`test_monte_carlo_size_at_and_beyond_the_margin_stays_at_alpha` then repeats the
size check through the actual power estimator, so a loosening that lives in the
sampling path rather than the rule is caught too.

The remaining tests keep the module honest about its inputs: the decision rule
must agree exactly with `simstats.tost_equivalence` (the rule the published
p-values came from), and the pair list must still be the pair list
`trace_driven_benchmark.py` actually tests.

Artefact-dependent tests SKIP when 05_results/trace_schedulers/tost_power.csv
has not been generated, per the suite's rule; they never regenerate it.
"""

import ast
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from conftest import require

import simstats
import tost_power
from tost_power import (
    power_grid,
    tost_power_bootstrap,
    tost_reject,
    ttest_reject,
    window_supply,
)

ARTEFACT = '05_results/trace_schedulers/tost_power.csv'

# Small enough to keep the suite fast, large enough for the assertions below.
# The two extreme calibration checks are nowhere near their thresholds: they
# come out at 0.000 and 1.000 against bounds of 0.01 and 0.99. The SIZE checks
# are the tight ones and are worth stating honestly: the Monte-Carlo standard
# error at p = alpha is 0.0049, the worst size observed across the sweep is
# 0.051, and the bound is 0.070 -- about four standard errors of headroom, which
# is what a loosening of the decision rule has to clear before this suite sees
# it. Every draw is seeded, so none of these numbers is flaky; the exact guard
# against a smaller loosening is the deterministic knife-edge test, not these.
REPS = 2000
CAL_SEED = 31337        # tost_power.POWER_SEED; outside every protected family


def _power(mean_d, sd_d, n, margin, reps=REPS, alpha=0.05, seed=CAL_SEED):
    return tost_power.tost_power(mean_d, sd_d, n, margin,
                                 np.random.default_rng(seed), reps, alpha)


# ---------------------------------------------------------------------------
# (a) CALIBRATION — the real test
# ---------------------------------------------------------------------------
# TOST asks "is the true paired difference INSIDE +/- margin?". Its power is
# therefore a function of where the truth sits relative to the margin, and it is
# pinned at three places by mathematics rather than by convention:
#   far outside  -> ~0        (you cannot certify equivalence that is false)
#   exactly on the boundary -> alpha  (the size of the test)
#   dead centre with no noise -> ~1   (equivalence is obvious)
# Any of these coming out wrong means the reported achieved_power and the
# reported n-for-80% are fiction.


def test_power_is_near_zero_far_outside_the_margin():
    """A truth ten margins away must almost never be certified as equivalent.

    If this passes at a high value, the module is reporting that a large,
    real difference can be certified as "the same" — which would let the
    paper claim equivalence exactly where it does not hold.
    """
    p = _power(mean_d=10.0, sd_d=1.0, n=20, margin=1.0)
    assert 0.0 <= p <= 0.01, (
        f'TOST power 10 margins outside the equivalence bound should be ~0, '
        f'got {p}')


def test_power_is_near_one_at_zero_difference_with_tiny_variance():
    """A truth dead-centre in the margin with negligible noise must be
    certified essentially always. A low value here means the module would
    report that even a perfectly equivalent pair needs an unreachable number
    of windows, inflating every n it publishes."""
    p = _power(mean_d=0.0, sd_d=1e-3, n=20, margin=1.0)
    assert p >= 0.99, (
        f'TOST power at a zero difference with negligible variance should be '
        f'~1, got {p}')


def test_power_at_the_margin_boundary_is_about_alpha():
    """On the boundary the two one-sided tests are exactly size-alpha, so power
    must sit at alpha — not above it (anti-conservative: equivalence claimed
    too readily) and not at zero (the rule is not testing what it claims)."""
    p = _power(mean_d=1.0, sd_d=1.0, n=20, margin=1.0, alpha=0.05)
    # +/- ~4 Monte-Carlo standard errors of alpha. The band used to run to 0.09,
    # which tolerated a rule rejecting at nearly twice the nominal alpha.
    assert 0.030 <= p <= 0.070, (
        f'TOST power on the equivalence boundary should be ~alpha=0.05, '
        f'got {p}')


def test_bootstrap_power_is_calibrated_the_same_way():
    """The non-parametric variant is a check on the normal assumption, so it
    has to answer the same two extremes the same way; otherwise the
    `achieved_power_bootstrap` column is not comparable to `achieved_power`."""
    rng = np.random.default_rng(CAL_SEED)
    far = np.full(20, 10.0) + rng.normal(0, 1.0, 20)
    p_far = tost_power_bootstrap(far, 20, 1.0, np.random.default_rng(CAL_SEED),
                                 REPS)
    assert p_far <= 0.01, f'bootstrap power far outside the margin: {p_far}'

    tight = np.full(20, 0.0) + rng.normal(0, 1e-3, 20)
    p_tight = tost_power_bootstrap(tight, 20, 1.0,
                                   np.random.default_rng(CAL_SEED), REPS)
    assert p_tight >= 0.99, f'bootstrap power at zero difference: {p_tight}'


def test_power_increases_with_n_inside_the_margin():
    """More windows must make an equivalence that is TRUE easier to certify.
    A rule that ignored n, or used it backwards, could still pass the two
    extremes above while producing meaningless n-for-80% numbers."""
    ps = [_power(mean_d=0.0, sd_d=1.0, n=n, margin=0.6) for n in (5, 20, 80)]
    assert ps[0] < ps[1] < ps[2], f'power should rise with n, got {ps}'


def test_power_decreases_with_n_outside_the_margin():
    """The claim that drives the LANL verdict: once the true effect is OUTSIDE
    the margin, extra windows push equivalence power toward 0, so no window
    budget can rescue the equivalence claim. tost_power.py stops its scan early
    on that basis, so the basis has to hold."""
    ps = [_power(mean_d=1.5, sd_d=1.0, n=n, margin=1.0) for n in (5, 30, 120)]
    assert ps[0] > ps[2], f'power should fall with n outside the margin: {ps}'
    assert ps[2] <= 0.01


def test_monte_carlo_size_at_and_beyond_the_margin_stays_at_alpha():
    """TOST's SIZE, measured through the estimator the artefact actually uses.

    Wherever the true paired difference sits AT or BEYOND the margin the
    equivalence hypothesis is false, so the probability of certifying it is at
    most alpha -- exactly alpha on the boundary, less further out. This is the
    property a LOOSENED decision rule breaks: it lifts the whole power surface,
    which is what would inflate `achieved_power` for every row of the artefact
    at once. The single-point boundary check above can only see that in one
    configuration; this sweeps n, sd and both tails so a loosening cannot hide
    in a corner of the parameter space.
    """
    bound = 0.070          # alpha + ~4 Monte-Carlo standard errors at REPS
    cases = [(1.0, 1.0, 20), (1.0, 1.0, 8), (1.0, 1.0, 50),
             (1.05, 1.0, 20), (1.0, 0.4, 20), (1.0, 3.0, 20)]
    worst = 0.0
    for mean, sd, n in cases:
        for sign in (1.0, -1.0):
            p = _power(mean_d=sign * mean, sd_d=sd, n=n, margin=1.0)
            worst = max(worst, p)
            assert p <= bound, (
                f'size {p} at mean={sign * mean}, sd={sd}, n={n} exceeds '
                f'{bound}: the rule certifies equivalence more often than alpha '
                f'where equivalence is false, so every achieved_power in the '
                f'artefact is inflated')
    # ...and it must not sit near zero on the boundary either, which would mean
    # the rule had been tightened into uselessness rather than loosened.
    assert worst >= 0.030, (
        f'largest size on the boundary is only {worst}; a test that never '
        f'reaches its own alpha is not the test simstats runs')


# ---------------------------------------------------------------------------
# (b) The decision rule must BE the project's decision rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4])
def test_tost_reject_matches_simstats_tost_equivalence(seed):
    """`tost_reject` is a vectorised restatement of `simstats.tost_equivalence`.
    If the two ever disagree, the power analysis is describing a test the
    project does not run and every n it reports is for the wrong test."""
    rng = np.random.default_rng(seed)
    a = rng.normal(100, 20, 20)
    b = rng.normal(100, 20, 20)
    res = simstats.tost_equivalence(a, b, margin_frac=0.10, alpha=0.05)
    mine = bool(tost_reject(a - b, res['margin'], 0.05)[0])
    assert mine == res['equivalent'], (
        f'tost_reject={mine} but simstats says equivalent={res["equivalent"]} '
        f'(p_tost={res["p_tost"]:.4g}, margin={res["margin"]:.4g})')


def test_tost_reject_needs_at_least_two_observations():
    """One window has no within-sample variance and no df; silently returning
    a verdict there would put a fabricated power number in the artefact."""
    with pytest.raises(ValueError):
        tost_reject(np.array([[1.0]]), margin=1.0)


def test_ttest_reject_matches_scipy():
    """The difference-test arm answers 'how many windows to call them
    DIFFERENT', which is the only question left once equivalence is out of
    reach. It must be the same t-test the significance table uses."""
    rng = np.random.default_rng(7)
    d = rng.normal(0.4, 1.0, 25)
    expected = stats.ttest_1samp(d, 0.0).pvalue < 0.05
    assert bool(ttest_reject(d, 0.05)[0]) == expected


def test_pairs_match_the_pairs_the_benchmark_tests():
    """tost_power.EQUIV_PAIRS duplicates a list defined inside
    trace_driven_benchmark.main(), which cannot be imported. A power analysis
    of pairs the project does not test would be answering a question nobody
    asked, so the duplication is pinned here rather than trusted."""
    src = Path(__file__).resolve().parents[1] / '04_scheduler'
    text = (src / 'trace_driven_benchmark.py').read_text(encoding='utf-8')
    m = re.search(r'EQUIV_PAIRS\s*=\s*\[(.*?)\]', text, re.S)
    assert m, 'EQUIV_PAIRS not found in trace_driven_benchmark.py'
    body = re.sub(r'#[^\n]*', '', m.group(1))
    benchmark_pairs = [tuple(p) for p in ast.literal_eval('[' + body + ']')]
    assert [tuple(p) for p in tost_power.EQUIV_PAIRS] == benchmark_pairs


def _diffs_with(mean, sd=1.0, n=20):
    """A length-n paired-difference sample with EXACTLY this mean and this sd.

    Deterministic: the verdict taken on it is a property of the decision rule
    and not of a draw, which is what lets the tests below sit micrometres from
    the alpha boundary without being flaky. The pattern is symmetric about its
    own mean, so `mean` is reproduced to floating-point exactness.
    """
    base = np.arange(n, dtype=float) - (n - 1) / 2.0
    base /= base.std(ddof=1)
    return mean + sd * base


def _simstats_p_tost(diffs, margin, alpha=0.05):
    """p_TOST for `diffs` from `simstats.tost_equivalence` -- the published rule.

    simstats derives its margin as margin_frac * mean(b), so the reference
    sample here is a constant vector chosen to make that product exactly
    `margin`. Going through simstats rather than recomputing the t statistics
    inline is the point: the expectation these tests hold `tost_reject` to comes
    from the function the paper's p-values came from, so mutating `tost_reject`
    cannot move the expectation along with it.
    """
    d = np.asarray(diffs, dtype=float)
    b = np.full(d.shape[0], margin / 0.10)
    return simstats.tost_equivalence(b + d, b, margin_frac=0.10,
                                     alpha=alpha)['p_tost']


def _alpha_knife_edge(margin, sd, n, alpha, sign=1):
    """The paired mean at which p_TOST is EXACTLY alpha.

    The upper one-sided null binds when (mean - margin) / se = t_alpha(n-1), so
    mean = margin + se * t.ppf(alpha, n-1); `sign=-1` mirrors it onto the lower
    null. Just inside this value the project certifies equivalence; just outside
    it, it must refuse to.
    """
    se = sd / math.sqrt(n)
    return sign * (margin + se * stats.t.ppf(alpha, n - 1))


@pytest.mark.parametrize('sign', [1, -1])
def test_tost_reject_holds_the_alpha_knife_edge(sign):
    """The rule must certify at p_TOST just BELOW alpha and refuse just above.

    This is the test that makes the module docstring's claim about loosening
    true. Sampling-based calibration cannot see a rule that rejects a little too
    readily -- a rule run at an effective alpha 1.75x too large passes every
    calibration check in this file -- because shifting the rejection threshold
    by 25% moves a Monte-Carlo power estimate by less than those tests' own
    tolerance. Here there is no tolerance to hide in: two constructed samples
    sit either side of the exact alpha boundary, their p_TOST values differ by
    4e-7, and the verdict has to flip between them. Every loosening -- a larger
    effective alpha, min() where max() belongs, one tail instead of two, a
    widened margin, the wrong df or ddof -- moves that boundary by far more than
    4e-7 and turns the second assertion red.
    """
    margin, sd, n, alpha = 1.0, 1.0, 20, 0.05
    edge = _alpha_knife_edge(margin, sd, n, alpha, sign)
    eps = 1e-6
    inside = _diffs_with(edge - sign * eps, sd, n)
    outside = _diffs_with(edge + sign * eps, sd, n)

    p_in = _simstats_p_tost(inside, margin, alpha)
    p_out = _simstats_p_tost(outside, margin, alpha)
    # The two samples really are knife-edge cases for the PROJECT's rule: both
    # p-values land within 1e-5 of alpha, one on each side of it.
    assert alpha - 1e-5 < p_in < alpha < p_out < alpha + 1e-5, (
        f'the constructed samples do not straddle alpha: {p_in}, {p_out}')

    assert bool(tost_reject(inside, margin, alpha)[0]), (
        f'p_TOST={p_in:.9f} is below alpha={alpha} yet tost_reject refused to '
        f'certify equivalence: the rule has drifted TIGHTER than simstats')
    assert not bool(tost_reject(outside, margin, alpha)[0]), (
        f'p_TOST={p_out:.9f} is above alpha={alpha} yet tost_reject certified '
        f'equivalence: the rule is LOOSER than simstats, which inflates every '
        f'power number this module writes into the artefact')


def test_tost_reject_refuses_the_zone_just_above_alpha():
    """Agreement with simstats across the decision boundary, not at random
    points far away from it.

    `test_tost_reject_matches_simstats_tost_equivalence` draws random samples,
    which land near p_TOST = alpha essentially never, so it cannot see a rule
    that rejects too readily. This sweeps the paired mean through both tails at
    a resolution fine enough to place several points in each band that matters,
    and demands the same verdict as simstats at every one -- in particular
    across the "too weak to certify" band at 1-10x alpha, where a loosened rule
    would claim an equivalence the project does not claim.
    """
    margin, sd, n, alpha = 1.0, 1.0, 20, 0.05
    just_above = 0
    for step in range(0, 101):
        for sign in (1.0, -1.0):
            mean = sign * step * 0.01
            d = _diffs_with(mean, sd, n)
            p = _simstats_p_tost(d, margin, alpha)
            got = bool(tost_reject(d, margin, alpha)[0])
            assert got == (p < alpha), (
                f'mean={mean:.2f}: simstats p_TOST={p:.6f} '
                f'(equivalent={p < alpha}) but tost_reject={got}')
            if alpha < p <= 10 * alpha:
                just_above += 1
    assert just_above >= 10, (
        f'only {just_above} sweep points landed just above alpha; the sweep no '
        f'longer exercises the band a loosened rule gets wrong')


def test_power_grid_is_ascending_and_bounded():
    grid = power_grid()
    assert grid == sorted(grid)
    assert len(set(grid)) == len(grid)
    assert grid[0] >= 3 and grid[-1] == tost_power.N_GRID_MAX


# ---------------------------------------------------------------------------
# (c) The artefact
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def table():
    return pd.read_csv(require(ARTEFACT, 'TOST power analysis'))


def test_artefact_has_the_documented_schema(table):
    assert list(table.columns) == tost_power.COLUMNS
    # 3 pairs x 2 metrics x 2 traces.
    assert len(table) == len(tost_power.EQUIV_PAIRS) * 2 * 2
    assert set(table['trace']) == {'sdsc', 'lanl'}
    assert (table['n_observed'] == 20).all()


def test_achieved_power_is_a_probability(table):
    """The headline column. A value outside [0, 1] is not a power."""
    for col in ('achieved_power', 'achieved_power_bootstrap'):
        vals = table[col].to_numpy(dtype=float)
        assert np.isfinite(vals).all(), f'{col} has non-finite entries'
        assert (vals >= 0.0).all() and (vals <= 1.0).all(), (
            f'{col} outside [0, 1]: {vals}')


def test_the_bootstrap_column_is_not_uniformly_the_higher_one(table):
    """Pins the docstring's row-by-row account of the two power columns.

    It is easy, and wrong, to summarise this artefact as "the bootstrap
    estimate sits consistently a little above the parametric one, so the
    parametric headline number is the conservative one". On the committed
    artefact the bootstrap is higher on 10 of 12 rows but NOT on the other two,
    and both exceptions fall on the SMALLEST_FIRST vs PROACTIVE pair the paper
    leans on: the two are exactly equal on SDSC mean_wait, and the bootstrap is
    LOWER on SDSC mean_bounded_slowdown. The module docstring now says so; this
    test is what stops that correction being quietly reverted, and what forces
    a re-read of it if the relationship ever changes.
    """
    par = table['achieved_power'].to_numpy(dtype=float)
    boot = table['achieved_power_bootstrap'].to_numpy(dtype=float)
    assert not (boot >= par).all(), (
        'the bootstrap column is now >= the parametric column on every row, so '
        'the module docstring, which records a row where it is strictly lower, '
        'is stale -- re-derive the comparison before quoting either column as '
        'the conservative one')

    row = table[(table['trace'] == 'sdsc')
                & (table['scheduler_a'] == 'SMALLEST_FIRST')
                & (table['scheduler_b'] == 'PROACTIVE')
                & (table['metric'] == 'mean_bounded_slowdown')]
    assert len(row) == 1
    row = row.iloc[0]
    assert row['achieved_power_bootstrap'] < row['achieved_power'], (
        f'the documented exception has moved: SDSC SMALLEST_FIRST vs PROACTIVE '
        f'/ mean_bounded_slowdown now has bootstrap='
        f'{row["achieved_power_bootstrap"]} >= parametric='
        f'{row["achieved_power"]}')


def test_power_outside_the_margin_cannot_exceed_alpha(table):
    """Where the observed effect sits outside the equivalence margin, TOST is
    at or below its own size, so achieved power must be <= alpha. A row that
    breaks this is reporting an equivalence test that certifies effects it was
    built to reject."""
    outside = table[table['effect_outside_margin']]
    assert len(outside) > 0, 'expected some pairs with effects outside the margin'
    for r in outside.itertuples():
        assert r.achieved_power <= r.alpha + 0.01, (
            f'{r.trace} {r.pair} {r.metric}: effect outside the margin but '
            f'achieved_power={r.achieved_power}')


def test_effect_outside_margin_flag_agrees_with_the_numbers(table):
    for r in table.itertuples():
        assert bool(r.effect_outside_margin) == (abs(r.mean_diff) >= r.margin)


def test_equivalence_is_unreachable_when_the_effect_is_outside_the_margin(table):
    """No window budget certifies an equivalence that is false, so those rows
    must report no n-for-80%. If one ever carried a number, the artefact would
    be telling the paper to go collect windows that cannot help."""
    for r in table[table['effect_outside_margin']].itertuples():
        assert math.isnan(r.n_for_80pct_power), (
            f'{r.trace} {r.pair} {r.metric} claims equivalence is reachable at '
            f'n={r.n_for_80pct_power} despite an effect outside the margin')


def test_settleable_flag_follows_from_the_n_columns(table):
    """`settleable_with_disjoint_windows` is the deliverable: can this trace,
    with the windows it can actually supply, reach a verdict in EITHER
    direction? It must be derivable from the columns beside it, not asserted."""
    for r in table.itertuples():
        eq_ok = (not math.isnan(r.n_for_80pct_power)
                 and r.n_for_80pct_power <= r.n_disjoint_windows_available)
        diff_ok = (not math.isnan(r.n_for_80pct_power_difference_holm)
                   and (r.n_for_80pct_power_difference_holm
                        <= r.n_disjoint_windows_available))
        assert bool(r.settleable_with_disjoint_windows) == (eq_ok or diff_ok), (
            f'{r.trace} {r.pair} {r.metric}: settleable flag disagrees with '
            f'n_for_80pct_power={r.n_for_80pct_power}, '
            f'holm={r.n_for_80pct_power_difference_holm}, '
            f'supply={r.n_disjoint_windows_available}')


def test_the_lanl_headline_pair_is_underpowered_and_unsettleable(table):
    """The finding this artefact exists to record.

    LANL's SMALLEST_FIRST vs PROACTIVE mean-wait TOST is the row the paper
    reads as "not equivalent". It must remain on record that the test had
    almost no chance of concluding otherwise, and that the trace cannot supply
    the windows a difference verdict would need. If either half of that is
    silently reverted — a power number quietly rising, or the supply number
    quietly growing past the requirement — this fails.
    """
    r = table[(table['trace'] == 'lanl')
              & (table['scheduler_a'] == 'SMALLEST_FIRST')
              & (table['scheduler_b'] == 'PROACTIVE')
              & (table['metric'] == 'mean_wait')]
    assert len(r) == 1
    r = r.iloc[0]
    assert r['achieved_power'] < 0.20, (
        f'LANL SMALLEST_FIRST vs PROACTIVE achieved_power={r["achieved_power"]}'
        f' — the underpowered finding no longer holds; re-derive it before '
        f'relaxing this bound')
    assert not r['settleable_with_disjoint_windows']
    assert 'INCONCLUSIVE' in r['note']


def test_window_supply_matches_the_artefact_and_is_internally_ordered():
    """The supply counts are read from the committed .swf.gz through the
    benchmark's own parser and split, so they cannot drift from the protocol.
    Recompute them and check the artefact carries the same numbers.

    Skips when the traces are absent; parses them once (no simulation).
    """
    require('02_data/LANL-CM5-1994-4.1-cln.swf.gz', 'LANL trace')
    require('02_data/SDSC-SP2-1998-4.2-cln.swf.gz', 'SDSC trace')
    supply = window_supply(('sdsc', 'lanl'))
    for key, s in supply.items():
        assert s['n_disjoint_protocol'] >= 1
        # A window that the harness would drop for having < 50 jobs is not
        # supply; on these traces none is dropped, and the ordering below must
        # hold whatever the trace.
        assert s['n_disjoint_protocol_ge50jobs'] <= s['n_disjoint_protocol']
        assert s['n_disjoint_protocol'] <= s['n_disjoint_measured_only']
        assert s['n_disjoint_protocol'] <= s['n_disjoint_whole_trace']
        assert s['eval_span_days'] < s['trace_span_days']

    table = pd.read_csv(require(ARTEFACT, 'TOST power analysis'))
    for key, s in supply.items():
        got = table.loc[table['trace'] == key,
                        'n_disjoint_windows_available'].unique()
        assert list(got) == [s['n_disjoint_protocol']], (
            f'{key}: artefact says {got}, protocol supplies '
            f'{s["n_disjoint_protocol"]}')


def test_the_grid_ceiling_claim_is_arithmetically_true():
    """The comment on N_GRID_MAX justifies the ceiling with arithmetic, so the
    arithmetic is read back out of the comment and recomputed here rather than
    trusted.

    An earlier version of that comment called N_GRID_MAX = 500 "an order of
    magnitude" past what the traces could deliver. It is not: the whole-trace
    ceiling is 72 windows and 500 / 72 = 6.9. The numbers a reader is asked to
    accept -- the per-trace supply, the whole-trace ceiling, and both multiples
    of N_GRID_MAX -- are parsed out of the comment itself, so rewriting it with
    a wrong multiple fails here instead of standing as a load-bearing sentence
    nobody checks.

    Skips when the traces are absent; parses them once (no simulation).
    """
    src = Path(tost_power.__file__).read_text(encoding='utf-8')
    block = src[src.index('# The grid stops at'):src.index('N_GRID_MAX = ')]
    prose = ' '.join(re.sub(r'^\s*#\s?', '', ln) for ln in block.splitlines())
    prose = re.sub(r'\s+', ' ', prose)

    m_supply = re.search(r'traces supply (\d+) and (\d+) disjoint', prose)
    m_ceiling = re.search(r'reaches more than (\d+)', prose)
    m_ratio = re.search(r'about (\d+)x the whole-trace ceiling and about '
                        r'(\d+)x the windows', prose)
    assert m_supply and m_ceiling and m_ratio, (
        f'the N_GRID_MAX comment no longer states its own arithmetic in a form '
        f'this test can check; re-read it and update the test: {prose!r}')
    said_sdsc, said_lanl = int(m_supply.group(1)), int(m_supply.group(2))
    said_ceiling = int(m_ceiling.group(1))
    said_x_ceiling, said_x_usable = int(m_ratio.group(1)), int(m_ratio.group(2))

    require('02_data/LANL-CM5-1994-4.1-cln.swf.gz', 'LANL trace')
    require('02_data/SDSC-SP2-1998-4.2-cln.swf.gz', 'SDSC trace')
    supply = window_supply(('sdsc', 'lanl'))
    got_sdsc = supply['sdsc']['n_disjoint_protocol']
    got_lanl = supply['lanl']['n_disjoint_protocol']
    ceiling = max(s['n_disjoint_whole_trace'] for s in supply.values())
    usable = max(got_sdsc, got_lanl)

    assert (said_sdsc, said_lanl) == (got_sdsc, got_lanl), (
        f'the comment claims the traces supply {said_sdsc} (SDSC) and '
        f'{said_lanl} (LANL) disjoint windows; the protocol supplies '
        f'{got_sdsc} and {got_lanl}')
    assert said_ceiling == ceiling, (
        f'the comment puts the whole-trace ceiling at {said_ceiling} windows; '
        f'it is {ceiling}')
    assert tost_power.N_GRID_MAX > ceiling, (
        f'N_GRID_MAX={tost_power.N_GRID_MAX} no longer runs past the {ceiling} '
        f'windows the traces could deliver, so a row reported as unattainable '
        f'may simply be off the end of the grid')

    x_ceiling = tost_power.N_GRID_MAX / ceiling
    x_usable = tost_power.N_GRID_MAX / usable
    assert round(x_ceiling) == said_x_ceiling, (
        f'the comment says N_GRID_MAX is about {said_x_ceiling}x the '
        f'whole-trace ceiling; {tost_power.N_GRID_MAX} / {ceiling} = '
        f'{x_ceiling:.2f}, which rounds to {round(x_ceiling)}x')
    assert round(x_usable) == said_x_usable, (
        f'the comment says N_GRID_MAX is about {said_x_usable}x the usable '
        f'supply; {tost_power.N_GRID_MAX} / {usable} = {x_usable:.2f}, which '
        f'rounds to {round(x_usable)}x')


def test_lanl_cannot_supply_the_windows_its_headline_pair_would_need(table):
    """The comparison that is the whole deliverable, stated as an assertion:
    for LANL's SMALLEST_FIRST vs PROACTIVE mean-wait row, the windows needed to
    reach a Holm-corrected difference verdict exceed the windows the trace can
    supply. That gap is why the honest verdict is INCONCLUSIVE."""
    r = table[(table['trace'] == 'lanl')
              & (table['scheduler_a'] == 'SMALLEST_FIRST')
              & (table['metric'] == 'mean_wait')].iloc[0]
    needed = r['n_for_80pct_power_difference_holm']
    assert not math.isnan(needed)
    assert needed > r['n_disjoint_windows_available'], (
        f'LANL now needs only {needed} windows and can supply '
        f'{r["n_disjoint_windows_available"]} — the "cannot be settled" '
        f'conclusion would no longer hold')

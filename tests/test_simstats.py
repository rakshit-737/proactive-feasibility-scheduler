"""Tests for 04_scheduler/simstats.py — the statistics behind every published claim.

`simstats` is the single source of truth for three things that appear in the
paper's tables and in the CHANGELOG's headline results:

  * gini          — the fairness number reported for every scheduler.
  * holm_correction — the family-wise error control that decides which pairwise
                    differences are allowed to be called "significant".
  * tost_equivalence / equivalence_table — the machinery behind every claim that
                    two policies perform *the same* (e.g. "the ML pipeline is
                    redundant against a size sort"). A large p from a difference
                    test is NOT sameness; TOST is what licenses the sameness
                    claim, so its behaviour must be pinned exactly.
  * pairwise_significance — the paired per-run comparison table.

If any of these silently changed, published numbers would move without anyone
noticing. These tests fix the contract. They import the helper via the duck-type
path set up in conftest.py (`import simstats`) and use tiny hand-checkable inputs
— no simulation, no artefacts, no writes.

conftest.py puts 04_scheduler on sys.path, so the bare import below works.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from simstats import (
    equivalence_table,
    gini,
    holm_correction,
    pairwise_significance,
    tost_equivalence,
)


# ---------------------------------------------------------------------------
# (a) gini
# ---------------------------------------------------------------------------
# The Gini coefficient is the headline fairness metric. The invariants below are
# the mathematical guarantees a reader assumes when they see "fairness_gini" in a
# results table: it is bounded in [0, 1], 0 means perfect equality, its maximum
# for a single beneficiary is (n-1)/n, it is scale-free (doubling every share
# leaves inequality unchanged), and — the part that actually bites in code — the
# degenerate empty / all-zero inputs must return 0.0, not NaN and not a crash,
# because an idle scheduler produces exactly those vectors.


def test_gini_all_equal_is_zero():
    """Perfect equality must read as zero inequality.

    If every user gets an identical share, the fairness metric must be exactly
    0.0; any drift here would make a perfectly fair scheduler look unfair.
    """
    assert gini([5, 5, 5, 5]) == 0.0
    assert gini([1.0]) == 0.0  # a single actor is trivially "equal" to itself


def test_gini_single_beneficiary_is_n_minus_1_over_n():
    """Maximal concentration among n actors gives exactly (n-1)/n.

    One actor holding everything is the most unequal a sample of size n can be,
    and the closed form is (n-1)/n. This pins both the numerator and the
    normalisation of the formula against an exact analytic value.
    """
    assert gini([0, 0, 0, 7]) == pytest.approx(3 / 4)
    assert gini([0, 0, 0, 0, 10]) == pytest.approx(4 / 5)
    # A hand-checkable non-degenerate value locks the whole formula, not just
    # its extremes: Gini([1,2,3]) = 2/9.
    assert gini([1, 2, 3]) == pytest.approx(2 / 9)


def test_gini_empty_and_all_zero_are_zero_not_nan():
    """Idle / empty inputs must be 0.0, never NaN and never a crash.

    An idle scheduler yields an empty allocation vector or an all-zero one.
    Returning NaN there would poison every downstream mean/CSV; a crash would
    take down the benchmark. The contract is a defined 0.0.
    """
    assert gini([]) == 0.0
    assert gini([0, 0, 0]) == 0.0
    assert gini(np.zeros(4)) == 0.0
    assert not np.isnan(gini([]))
    assert not np.isnan(gini([0, 0, 0]))


def test_gini_is_scale_invariant():
    """Multiplying every share by a constant must not change inequality.

    Inequality is about the *shape* of a distribution, not its units. Reporting
    GPU-seconds vs GPU-hours must give the same Gini, so the metric is
    comparable across workloads of different absolute size.
    """
    base = [1.0, 2.0, 3.0, 10.0, 0.5]
    assert gini(base) == pytest.approx(gini([100 * v for v in base]))
    assert gini(base) == pytest.approx(gini([0.001 * v for v in base]))


def test_gini_always_in_unit_interval():
    """Gini must stay within [0, 1] for every non-negative sample.

    Any value outside [0, 1] is meaningless as a fairness coefficient and would
    signal a formula error. This is a cheap property check over many random
    non-negative vectors (seeded for determinism).
    """
    rng = np.random.default_rng(20260721)
    for _ in range(300):
        n = int(rng.integers(1, 30))
        vals = rng.random(n) * rng.integers(1, 1000)
        g = gini(vals)
        assert 0.0 - 1e-12 <= g <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# (b) holm_correction
# ---------------------------------------------------------------------------
# Holm is the gatekeeper for the word "significant". With ~13 schedulers compared
# against each reference, uncorrected p-values would manufacture false positives.
# The step-down procedure must (1) reproduce a hand-computed example exactly,
# (2) never report an adjusted p below the raw p, (3) clip at 1.0, (4) be
# monotone non-decreasing along the sorted order, and — the subtle one —
# (5) treat NaN as "not a test": NaNs stay NaN and must NOT inflate the family
# size m, or every real p in the family would be over-corrected.


def test_holm_matches_hand_computed_example():
    """Exact reproduction of a worked step-down example.

    Holm sorts p ascending and multiplies the k-th smallest (0-indexed rank r)
    by (m - r), taking a running maximum so the sequence never decreases.
    For p = [0.01, 0.04, 0.03] with m = 3:
        rank 0 -> 3 * 0.01 = 0.03
        rank 1 -> 2 * 0.03 = 0.06   (running max = 0.06)
        rank 2 -> 1 * 0.04 = 0.04   (running max stays 0.06)
    Mapped back to input order -> [0.03, 0.06, 0.06]. Any change to the
    procedure breaks this exact expectation.
    """
    adj = holm_correction([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])


def test_holm_adjusted_never_below_raw():
    """A multiple-comparison correction may only make p-values larger.

    An adjusted p smaller than its raw p would be a correction that *weakens*
    error control — the opposite of Holm's purpose. Checked over random valid
    families (seeded).
    """
    rng = np.random.default_rng(7)
    for _ in range(200):
        p = rng.random(int(rng.integers(1, 12)))
        adj = holm_correction(p)
        assert np.all(adj >= p - 1e-12)


def test_holm_clipped_at_one():
    """Adjusted p-values are probabilities and must never exceed 1.0.

    The step-down multiplier can push a value above 1 (here 2 * 0.6 = 1.2);
    reporting p > 1 would be nonsensical. The result must be clamped to 1.0.
    """
    adj = holm_correction([0.6, 0.7])
    assert list(adj) == [1.0, 1.0]
    assert np.all(adj <= 1.0)


def test_holm_monotone_non_decreasing_in_sorted_order():
    """Along ascending raw p, the adjusted p-values must not decrease.

    Monotonicity in sorted order is the defining property of a step-down
    procedure (it is why a running maximum is used). If it failed, a *smaller*
    raw p could receive a *larger* adjusted p than a bigger one, corrupting the
    accept/reject boundary.
    """
    p = np.array([0.02, 0.001, 0.2, 0.05])
    adj = np.asarray(holm_correction(p))
    sorted_adj = adj[np.argsort(p)]
    assert np.all(np.diff(sorted_adj) >= -1e-12)


def test_holm_nan_does_not_count_toward_family_size():
    """NaN entries are 'no test': they stay NaN and must shrink the family m.

    This is the invariant the docstring promises and the one most likely to
    regress. If a NaN (a comparison that could not be run, e.g. Wilcoxon on an
    all-ties family) were counted in m, every real p would be multiplied by an
    m that is too large and over-corrected.

    Proof that m shrank: adding a NaN to [0.01, 0.04, 0.03] must leave the three
    real adjusted values IDENTICAL to the no-NaN case (m = 3). The smallest
    adjusted value is therefore 3 * 0.01 = 0.03. If the NaN had been counted
    (m = 4) it would instead be 4 * 0.01 = 0.04 — so asserting 0.03 (and not
    0.04) directly proves the family size did not include the NaN.
    """
    with_nan = holm_correction([0.01, 0.04, 0.03, np.nan])
    without_nan = holm_correction([0.01, 0.04, 0.03])

    # The NaN slot stays NaN and is positionally preserved.
    assert np.isnan(with_nan[3])
    assert not np.any(np.isnan(with_nan[:3]))

    # The three real values are unchanged by the presence of the NaN...
    assert with_nan[:3] == pytest.approx(without_nan)
    # ...and equal the m = 3 result, NOT the m = 4 result.
    assert with_nan[0] == pytest.approx(0.03)   # = 3 * 0.01, so m == 3
    assert with_nan[0] != pytest.approx(0.04)   # would be 4 * 0.01 if m == 4


def test_holm_empty_input_returns_empty():
    """An empty family must return an empty array, not raise.

    A reference with no other schedulers to compare against yields an empty
    p-vector; the helper must handle it gracefully.
    """
    adj = holm_correction([])
    assert len(adj) == 0


# ---------------------------------------------------------------------------
# (c) tost_equivalence
# ---------------------------------------------------------------------------
# TOST is the crux of the methodology: it is the ONLY tool in this repo that can
# license a claim of "these two schedulers perform the same". Its guarantees:
# equivalence is provable for identical samples (p_TOST ~ 0), refused for samples
# far apart, p_TOST is exactly max of the two one-sided p-values, the reported CI
# brackets the mean difference, the margin is margin_frac * mean(b) — and,
# critically, the underpowered case where the difference test is non-significant
# yet TOST also cannot certify equivalence must remain distinct from a positive
# equivalence result.


def test_tost_identical_samples_are_equivalent():
    """Two identical samples are trivially equivalent with p_TOST ~ 0.

    If a scheduler is compared against itself (or two runs coincide exactly),
    the difference is zero, well inside any positive margin, and TOST must
    certify equivalence with essentially zero p.
    """
    res = tost_equivalence([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    assert res['equivalent'] is True
    assert res['p_tost'] == pytest.approx(0.0)
    assert res['mean_diff'] == 0.0


def test_tost_far_apart_samples_are_not_equivalent():
    """A difference many multiples of the margin must NOT be certified equal.

    Two samples ~100 apart with a margin of ~10 are plainly different; TOST must
    refuse equivalence (p_TOST above alpha). This guards against a broken test
    that rubber-stamps everything as equivalent.
    """
    a = [200.0, 210.0, 190.0, 205.0, 195.0]
    b = [100.0, 101.0, 99.0, 100.0, 100.0]
    res = tost_equivalence(a, b, margin_frac=0.10, alpha=0.05)
    assert res['equivalent'] is False
    assert res['p_tost'] > 0.05


def test_tost_p_is_exactly_max_of_the_two_one_sided_tests():
    """p_TOST is by definition max(p_lower, p_upper), exactly.

    TOST rejects the composite null only if BOTH one-sided tests reject, so the
    combined p is the larger of the two. Pinning the exact identity (not an
    approximation) protects the definition against a refactor that averages or
    mis-combines them.
    """
    a = [200.0, 210.0, 190.0, 205.0, 195.0]
    b = [100.0, 101.0, 99.0, 100.0, 100.0]
    res = tost_equivalence(a, b)
    assert res['p_tost'] == max(res['p_lower'], res['p_upper'])


def test_tost_ci_brackets_the_mean_difference():
    """The reported CI must contain the paired mean difference.

    TOST works by inverting a confidence interval on the mean difference; a CI
    that did not bracket its own point estimate would be internally
    inconsistent and every reported interval suspect.
    """
    a = [80.0, 90.0, 100.0, 110.0, 130.0]
    b = [100.0, 100.0, 100.0, 100.0, 100.0]
    res = tost_equivalence(a, b)
    assert res['ci_low'] <= res['mean_diff'] <= res['ci_high']


def test_tost_margin_is_margin_frac_times_mean_of_b():
    """The equivalence margin Delta = margin_frac * mean(b), keyed on b.

    The margin defines "practically negligible" relative to the REFERENCE (b),
    not a. If it were keyed on a, or on a different fraction, every equivalence
    verdict in the paper would shift. mean(b) = 200 here, so a 10% margin is 20.
    """
    res = tost_equivalence([40.0, 50.0, 60.0], [190.0, 200.0, 210.0],
                           margin_frac=0.10)
    assert res['margin'] == pytest.approx(0.10 * 200.0)


def test_tost_underpowered_case_is_neither_different_nor_equivalent():
    """THE pillar of the methodology: 'not significant' != 'equivalent'.

    Here the paired difference is tiny in the mean (2.0, inside the margin of
    10) but the sample is noisy and small, so:
      * the ordinary paired difference test is NON-significant (p >> alpha) —
        i.e. we cannot show the two policies differ; AND
      * TOST ALSO fails (p_TOST > alpha) — we cannot show they are the same,
        because the CI spills outside +/- margin.
    This is the underpowered limbo the whole v3.4 methodology exists to expose:
    a large difference-test p read naively would wrongly be reported as
    "no difference / the same". This test pins that the two are distinct
    outcomes and that TOST refuses to certify equivalence here.
    """
    a = np.array([80.0, 90.0, 100.0, 110.0, 130.0])
    b = np.array([100.0, 100.0, 100.0, 100.0, 100.0])

    # The naive difference test cannot detect a difference...
    diff_p = stats.ttest_rel(a, b).pvalue
    assert diff_p > 0.05

    # ...yet TOST cannot certify equivalence either.
    res = tost_equivalence(a, b, margin_frac=0.10, alpha=0.05)
    assert res['equivalent'] is False
    assert res['p_tost'] > 0.05
    # And the reason is visible: the CI extends beyond +/- margin.
    assert res['ci_high'] > res['margin'] or res['ci_low'] < -res['margin']


# ---------------------------------------------------------------------------
# (d) equivalence_table / pairwise_significance
# ---------------------------------------------------------------------------
# These are the table-builders that turn a runs DataFrame into the published
# significance and equivalence CSVs. The load-bearing invariant is the PAIRING:
# both helpers pair observations by the `unit` column (run / window), and both
# sort_values(unit) before pairing. If someone dropped that sort, results would
# depend on incoming row order — a silent, data-dependent corruption. The tests
# below build a tiny synthetic runs frame, confirm the output shape/columns, and
# prove that shuffling the input rows leaves the output byte-for-byte identical.


def _synthetic_runs(seed=101, n_runs=8):
    """A tiny, deterministic runs frame: 3 schedulers x n_runs, ordered by unit.

    Values are constructed so that run index matters to the pairing: PROACTIVE
    rises with run, SMALLEST falls with run. If the pairing were by row position
    instead of by `run`, shuffling the rows would change the paired differences —
    which is exactly what the shuffle tests below rule out.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for sch, base, slope in (('PROACTIVE', 10.0, 1.0),
                             ('SMALLEST', 18.0, -1.0),
                             ('NN', 12.0, 0.3),
                             ('FIFO', 20.0, 0.6)):
        for r in range(n_runs):
            wait = base + slope * r + rng.normal(0, 0.5)
            slow = 1.0 + 0.05 * wait + rng.normal(0, 0.02)
            rows.append({'scheduler': sch, 'run': r,
                         'mean_wait': wait, 'mean_bounded_slowdown': slow})
    # Ordered by (scheduler, run) on purpose: the unshuffled frame is already
    # correctly aligned, so any regression that drops sort_values would only
    # surface once the rows are shuffled — which the tests then do.
    return pd.DataFrame(rows)


def test_equivalence_table_shape_and_columns():
    """One row per (scheduler_a, scheduler_b) pair, with the documented columns.

    Downstream code and the published CSV rely on these exact column names and
    on there being exactly one row per requested pair; a missing column or a
    dropped/duplicated pair would break the results table.
    """
    df = _synthetic_runs()
    pairs = [('SMALLEST', 'PROACTIVE'), ('NN', 'PROACTIVE')]
    out = equivalence_table(df, pairs, metric='mean_wait', unit='run')

    assert len(out) == len(pairs)
    expected_cols = {'scheduler_a', 'scheduler_b', 'metric', 'n', 'mean_a',
                     'mean_b', 'pct_diff', 'margin_frac', 'margin', 'mean_diff',
                     'ci_low', 'ci_high', 'p_lower', 'p_upper', 'p_tost',
                     'equivalent'}
    assert expected_cols.issubset(set(out.columns))
    assert list(zip(out['scheduler_a'], out['scheduler_b'])) == pairs


def test_equivalence_table_pairs_by_unit_not_row_order():
    """Equivalence results must depend on `unit`, never on input row order.

    This is the real-world risk the task calls out: if someone deletes the
    `.sort_values(unit)`, the pairing silently follows whatever order the rows
    happen to arrive in, and the entire equivalence table becomes a function of
    DataFrame ordering. Shuffling the rows (seeded) must leave the output
    identical to the ordered-input output.
    """
    df = _synthetic_runs()
    pairs = [('SMALLEST', 'PROACTIVE'), ('NN', 'PROACTIVE')]

    ordered = equivalence_table(df, pairs, metric='mean_wait', unit='run')
    shuffled_in = df.sample(frac=1.0, random_state=999).reset_index(drop=True)
    shuffled = equivalence_table(shuffled_in, pairs, metric='mean_wait',
                                 unit='run')

    pd.testing.assert_frame_equal(ordered, shuffled)


def test_pairwise_significance_shape_and_columns():
    """One row per (reference, scheduler) pair, carrying the reported statistics.

    The significance CSV promises a paired t-test p, a Wilcoxon p, Cohen's dz,
    and Holm-adjusted p per reference family. Missing any of these columns, or a
    wrong row count, would mean a broken or incomplete results table.
    """
    df = _synthetic_runs()
    out = pairwise_significance(df, references=('PROACTIVE',),
                               metric='mean_wait', unit='run')

    # Family = every scheduler except the reference itself.
    others = sorted(set(df['scheduler']) - {'PROACTIVE'})
    assert len(out) == len(others)
    assert set(out['scheduler']) == set(others)
    assert (out['reference'] == 'PROACTIVE').all()
    for col in ('ttest_p', 'wilcoxon_p', 'cohens_dz', 'mean_diff',
                'ttest_p_holm', 'wilcoxon_p_holm'):
        assert col in out.columns


def test_pairwise_significance_pairs_by_unit_not_row_order():
    """Paired significance must be invariant to input row order.

    A paired t-test / Wilcoxon is only valid if observation i of the scheduler
    is matched to observation i of the reference by `unit`. If the sort were
    dropped, shuffled input would pair mismatched runs and produce different
    p-values — a subtle way to fabricate or destroy significance. Shuffling the
    rows must not change the output.
    """
    df = _synthetic_runs()

    ordered = pairwise_significance(df, references=('PROACTIVE',),
                                   metric='mean_wait', unit='run')
    shuffled_in = df.sample(frac=1.0, random_state=4242).reset_index(drop=True)
    shuffled = pairwise_significance(shuffled_in, references=('PROACTIVE',),
                                    metric='mean_wait', unit='run')

    pd.testing.assert_frame_equal(
        ordered.sort_values('scheduler').reset_index(drop=True),
        shuffled.sort_values('scheduler').reset_index(drop=True),
    )


def test_pairwise_significance_holm_is_applied_within_each_reference_family():
    """Holm corrects WITHIN each reference family, not across the whole table.

    The docstring and the published significance CSV promise "Holm-adjusted p
    within each reference family". With two references and three comparators
    each, the correct family size is m = 3 per reference, not m = 6 for the
    combined table. Correcting globally would over-correct every row and could
    flip published significance verdicts, so the scope is pinned here by
    recomputing Holm independently over each reference's own p-values.
    """
    df = _synthetic_runs()
    out = pairwise_significance(df, references=('PROACTIVE', 'FIFO'),
                               metric='mean_wait', unit='run')

    assert len(out) == 2 * (df['scheduler'].nunique() - 1)
    for ref in ('PROACTIVE', 'FIFO'):
        fam = out[out['reference'] == ref]
        assert len(fam) == df['scheduler'].nunique() - 1
        expected = holm_correction(fam['ttest_p'].to_numpy())
        assert fam['ttest_p_holm'].to_numpy() == pytest.approx(expected)
        expected_w = holm_correction(fam['wilcoxon_p'].to_numpy())
        assert fam['wilcoxon_p_holm'].to_numpy() == pytest.approx(expected_w)


def test_equivalence_table_silently_skips_mismatched_length_pairs():
    """DOCUMENTS CURRENT BEHAVIOUR (not an endorsement).

    When the two schedulers of a pair have different numbers of observations,
    `equivalence_table` cannot pair them and drops the pair entirely — with no
    warning and no row in the output. A reader of the resulting CSV sees an
    absent row, not an error. This is pinned so the silence is at least
    deliberate and visible: if the behaviour ever becomes "raise" or "emit a
    NaN row", this test fails and forces the change to be conscious.
    """
    rows = []
    for r in range(6):
        rows.append({'scheduler': 'PROACTIVE', 'run': r, 'mean_wait': 10.0 + r})
    for r in range(4):  # deliberately fewer runs
        rows.append({'scheduler': 'SMALLEST', 'run': r, 'mean_wait': 12.0 + r})
    df = pd.DataFrame(rows)

    out = equivalence_table(df, [('SMALLEST', 'PROACTIVE')], metric='mean_wait',
                            unit='run')
    assert len(out) == 0                       # the pair vanished silently
    assert 'p_tost' in out.columns             # schema still intact


def test_pairwise_significance_raises_on_mismatched_or_missing_reference():
    """DOCUMENTS CURRENT BEHAVIOUR: it raises rather than skipping.

    Unlike `equivalence_table` (which guards with a length check and skips),
    `pairwise_significance` has no guard: an unequal number of observations, or
    a reference name absent from the frame, produces a numpy broadcast
    ValueError. Failing loudly is preferable to a silent wrong answer, so this
    is pinned as the contract — and it flags the inconsistency between the two
    helpers for anyone extending them.

    Note this makes the default `references=('PROACTIVE', 'FIFO')` a landmine
    for any runs frame that lacks a scheduler literally named 'FIFO'.
    """
    rows = []
    for r in range(6):
        rows.append({'scheduler': 'PROACTIVE', 'run': r, 'mean_wait': 10.0 + r})
    for r in range(4):
        rows.append({'scheduler': 'SMALLEST', 'run': r, 'mean_wait': 12.0 + r})
    df = pd.DataFrame(rows)

    with pytest.raises(ValueError):
        pairwise_significance(df, references=('PROACTIVE',), metric='mean_wait',
                             unit='run')

    # A reference name that does not appear at all: empty array, same failure.
    with pytest.raises(ValueError):
        pairwise_significance(df[df['scheduler'] == 'SMALLEST'],
                             references=('PROACTIVE',), metric='mean_wait',
                             unit='run')


def test_pairing_is_positional_and_unit_values_are_never_compared():
    """DOCUMENTS A GENUINE LATENT DEFECT — see the summary, do not "fix" here.

    Both helpers sort each scheduler's rows by `unit` and then pair them BY
    POSITION. They never check that the two schedulers actually share the same
    set of unit labels — only that the counts match (and `pairwise_significance`
    does not even check that).

    Consequence: if two schedulers have the same NUMBER of observations but
    different unit labels — e.g. the trace-driven benchmark drops a window for
    one policy and adds a different one, or two runs frames are concatenated
    with offset run ids — the "paired" t-test / Wilcoxon / TOST silently pairs
    unrelated observations and reports a confident p-value for a comparison
    that was never paired at all. There is no warning.

    Below, PROACTIVE has runs 0-3 and SMALLEST has runs 10-13: entirely
    disjoint units. The correct answer is "these cannot be paired". The current
    answer is a fully populated result row. This test asserts the *current*
    (wrong) behaviour so the defect is visible and any future guard shows up as
    a deliberate, reviewed change.
    """
    rows = []
    for i, r in enumerate([0, 1, 2, 3]):
        rows.append({'scheduler': 'PROACTIVE', 'run': r,
                     'mean_wait': 10.0 + 0.7 * i})
    for i, r in enumerate([10, 11, 12, 13]):   # DISJOINT unit labels
        rows.append({'scheduler': 'SMALLEST', 'run': r,
                     'mean_wait': 13.0 + 1.3 * i})
    df = pd.DataFrame(rows)

    sig = pairwise_significance(df, references=('PROACTIVE',),
                               metric='mean_wait', unit='run')
    # BUG: a paired test is reported even though no unit is shared.
    assert len(sig) == 1
    assert np.isfinite(sig['ttest_p'].iloc[0])

    eq = equivalence_table(df, [('SMALLEST', 'PROACTIVE')], metric='mean_wait',
                           unit='run')
    # BUG: TOST reports n = 4 "pairs" that do not correspond to common units.
    assert len(eq) == 1
    assert int(eq['n'].iloc[0]) == 4

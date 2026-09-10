"""How many paired windows would the trace-driven equivalence tests actually need?

WHY THIS EXISTS
---------------
`trace_driven_benchmark.py` runs paired TOST (via `simstats.equivalence_table`)
on 20 windows per trace. On SDSC that test CERTIFIES the headline claim:
SMALLEST_FIRST is equivalent to PROACTIVE on mean wait at a +/-10% margin
(p_TOST = 1.76e-12). On LANL the same test does NOT certify it
(p_TOST = 0.7655), and the paired difference test does not reject a difference
either (t p = 0.0249 raw, 0.2490 after Holm over the 11-scheduler family).

"Neither equivalent nor different" is INCONCLUSIVE, not "different". The paper
must not read the failed LANL equivalence test as evidence that the ML pipeline
adds something on LANL. This script quantifies the obvious suspect -- power --
and answers the only question that makes the LANL row interpretable:

    how many windows would LANL need, and can the trace supply them?

WHAT IS COMPUTED
----------------
For every pair the project actually tests (`EQUIV_PAIRS` in
trace_driven_benchmark.py) on both metrics and both traces:

  1. ACHIEVED POWER of the existing 20-window TOST at the project margin
     (margin_frac = 0.10 of the reference mean, exactly as
     `simstats.tost_equivalence` computes it), by Monte-Carlo simulation from
     the OBSERVED paired-difference distribution.
  2. The smallest n on a grid at which that power reaches 0.80.
  3. The smallest n at which the paired DIFFERENCE test reaches 0.80 power --
     both at nominal alpha and at the Holm worst case alpha/m that the
     project's own significance table applies. This is the second half of the
     answer: where the observed effect lies OUTSIDE the equivalence margin, no
     number of windows can ever certify equivalence, and the only thing extra
     windows can buy is a difference verdict.
  4. The number of DISJOINT windows the trace can actually supply under the
     project's own window protocol, and therefore whether the comparison is
     settleable at all.

TWO LIMITS, STATED UP FRONT
---------------------------
(a) POST-HOC. Power is computed at the OBSERVED effect size. The observed mean
    and sd of the paired difference are themselves estimates from n = 20, so
    every n reported here is an estimate with its own (unquantified) sampling
    error, not a guarantee. A sample-size number derived this way is a planning
    aid; it is not a promise that n windows will settle anything.

(b) FINITE TRACE. A trace has a bounded number of disjoint windows. Where the
    required n exceeds that supply, the honest conclusion is not "run more
    windows" -- it is that the comparison CANNOT be settled at this margin with
    disjoint windows from this trace, and the paper must say INCONCLUSIVE.

Two smaller assumptions, also worth naming:
  * The margin is held FIXED at 0.10 * (observed reference mean). In a real
    replication the margin would be recomputed from the new sample and would
    carry its own noise, so holding it fixed is mildly OPTIMISTIC.
  * The primary simulation draws Normal(mean_diff, sd_diff), which is the model
    the t-based TOST assumes. `achieved_power_bootstrap` re-does the n = 20
    number by resampling the 20 observed differences with replacement, as a
    check that the normal assumption is not doing the work.

WHICH OF THE TWO POWER ESTIMATES IS THE CONSERVATIVE ONE
--------------------------------------------------------
Do NOT say the bootstrap column sits consistently a little above the parametric
one, and therefore that the parametric headline number is always the
conservative one. On the committed artefact that is false, and it is false
precisely on the pair the paper leans on. Row by row across the 12 rows of
05_results/trace_schedulers/tost_power.csv, `achieved_power_bootstrap` is:

  * HIGHER on 10 of 12 rows, by anything from 0.00005 to 0.19 in absolute
    terms; the widest gap is LANL PROACTIVE vs FCFS / mean_wait (0.2488
    bootstrap against 0.0613 parametric), and it is NOT confined to rows whose
    effect lies outside the margin;
  * EQUAL on SDSC SMALLEST_FIRST vs PROACTIVE / mean_wait, where both read
    1.0000 and neither is the more conservative of the two;
  * LOWER on SDSC SMALLEST_FIRST vs PROACTIVE / mean_bounded_slowdown
    (0.9108 bootstrap against 0.9734 parametric).

So the relationship is NOT consistent, and it fails exactly where the paper
leans on it: both exceptions fall on the headline SMALLEST_FIRST vs PROACTIVE
pair, and on the SDSC bounded-slowdown row the PARAMETRIC estimate is the
OPTIMISTIC one. Treat the two columns as two estimates of the same quantity
that usually, but not always, sit on the same side of it, and where a row's two
numbers disagree quote both rather than whichever is smaller.

This does not soften the LANL result. On LANL SMALLEST_FIRST vs PROACTIVE /
mean_wait the two estimates differ by a factor of 12 -- 0.0047 parametric,
0.0580 bootstrap -- but 0.5% and 5.8% are both an enormous distance below the
0.80 the study asks for, so the 20-window test had almost no chance of
certifying equivalence under EITHER model, and the LANL verdict stays
INCONCLUSIVE on either number.

SEED
----
POWER_SEED = 31337. The protected seed families in this repository are 42+i,
1000+run, 20000+run, 800+run, 4000+id+nodes, 5000+i, 7000+run and SEED = 42;
31337 falls in none of them, so nothing here can collide with, or be mistaken
for, a seed that drives a published number. Each output row draws from its own
child generator `default_rng([POWER_SEED, row_index])`, so a row's numbers do
not depend on how many rows ran before it.

THIS SCRIPT CHANGES NO PUBLISHED NUMBER. It only reads
05_results/trace_schedulers/trace_scheduler_windows.csv and the committed
.swf.gz traces. It deliberately does NOT increase the benchmark's window count.

OUTPUT
------
  05_results/trace_schedulers/tost_power.csv

Usage:
  python tost_power.py                  # 20000 replicates (default)
  python tost_power.py --replicates 2000  # fast sanity run
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '02_data'))

from simstats import _aligned_pair, tost_equivalence  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, '05_results', 'trace_schedulers')
WINDOWS_CSV = os.path.join(OUT_DIR, 'trace_scheduler_windows.csv')
OUT_CSV = os.path.join(OUT_DIR, 'tost_power.csv')

# Seed chosen to sit outside every protected family in this repository; see the
# module docstring. Changing it changes only the Monte-Carlo noise of this
# script's own estimates, never a published number.
POWER_SEED = 31337
N_REPLICATES = 20000
ALPHA = 0.05
MARGIN_FRAC = 0.10          # identical to trace_driven_benchmark.py's call
TARGET_POWER = 0.80

# The grid stops at 500 windows on purpose. Under this protocol the SDSC and
# LANL traces supply 29 and 28 disjoint post-split windows respectively, and
# even ignoring the chronological train/eval split entirely neither trace
# reaches more than 72 (`window_supply` recomputes all three counts from the
# committed .swf.gz). So 500 is about 7x the whole-trace ceiling and about 17x
# the windows the protocol can actually deliver -- comfortably past anything the
# data could supply, though NOT the "order of magnitude" past the ceiling that
# an earlier version of this comment claimed: 500 / 72 = 6.9, not 10. Anything
# that has not reached 0.80 by n = 500 is reported as unattainable rather than
# extrapolated.
N_GRID_MAX = 500

# Once the true paired difference lies outside the equivalence margin, TOST
# power is non-increasing in n and tends to 0 -- more windows make the CI
# tighter around a mean that sits outside the margin, which is the opposite of
# equivalence. Scanning such a row to N_GRID_MAX only re-confirms an analytic
# fact, so the scan stops at 10x the observed n and reports the falling power.
HOPELESS_SCAN_N = 200


def power_grid(n_max=N_GRID_MAX):
    """Ascending candidate window counts, fine where the answers land.

    Resolution is 1 window up to 40, then 2, 5 and 10. A reported
    `n_for_80pct_power` is therefore the first GRID POINT at or above the true
    threshold, not the exact threshold -- at n > 300 it can overstate the true
    requirement by up to 9 windows. That coarseness is immaterial next to
    limit (a) in the module docstring.
    """
    grid = list(range(3, 41))
    grid += list(range(42, 101, 2))
    grid += list(range(105, 301, 5))
    grid += list(range(310, n_max + 1, 10))
    return [n for n in grid if n <= n_max]


# ─────────────────────────────────────────────────────────────────────────────
# The decision rules under test
# ─────────────────────────────────────────────────────────────────────────────

def tost_reject(diffs, margin, alpha=ALPHA):
    """Vectorised paired-TOST equivalence decision over rows of `diffs`.

    `diffs` is (replicates, n) of paired differences. Returns a boolean array:
    True where TOST REJECTS both one-sided nulls, i.e. concludes EQUIVALENCE
    within +/- `margin`.

    This is the same rule as `simstats.tost_equivalence` -- p_TOST =
    max(p_lower, p_upper) < alpha with df = n-1 -- restated for a whole batch of
    samples at once, with the margin supplied rather than recomputed from a
    reference sample. `test_tost_power.py` pins it against simstats on random
    data so the two cannot drift apart.

    This rule can fail in two directions, and they are guarded differently.
    INVERTING it -- certifying equivalence where the effect is large -- is
    caught by the Monte-Carlo calibration tests, which pin power to ~0 ten
    margins outside the bound and to ~1 at a zero difference. LOOSENING it --
    rejecting a little too readily, which silently inflates every power number
    in the artefact -- is the realistic regression, and those calibration tests
    are far too coarse to see it: a rule run at an effective alpha 1.75x too
    large passed every one of them. Loosening is caught instead by
    `test_tost_reject_holds_the_alpha_knife_edge`, which evaluates this rule on
    constructed samples whose p_TOST straddles alpha by 4e-7 and requires the
    verdict to flip exactly there, and by
    `test_tost_reject_refuses_the_zone_just_above_alpha`.
    """
    d = np.asarray(diffs, dtype=float)
    if d.ndim == 1:
        d = d[None, :]
    n = d.shape[1]
    if n < 2:
        raise ValueError(f'TOST needs at least 2 paired observations, got {n}')
    mean = d.mean(axis=1)
    se = d.std(axis=1, ddof=1) / np.sqrt(n)
    df = n - 1
    with np.errstate(divide='ignore', invalid='ignore'):
        p_lower = stats.t.sf((mean + margin) / se, df)   # H01: mu <= -margin
        p_upper = stats.t.cdf((mean - margin) / se, df)  # H02: mu >= +margin
    p_tost = np.maximum(p_lower, p_upper)
    # se == 0 (every difference identical) makes both t statistics infinite;
    # fall back on simstats' degenerate branch: equivalent iff |mean| < margin.
    degenerate = ~np.isfinite(p_tost)
    if degenerate.any():
        p_tost = np.where(degenerate, np.where(np.abs(mean) < margin, 0.0, 1.0),
                          p_tost)
    return p_tost < alpha


def ttest_reject(diffs, alpha=ALPHA):
    """Vectorised two-sided paired t-test decision (H0: mu_d = 0)."""
    d = np.asarray(diffs, dtype=float)
    if d.ndim == 1:
        d = d[None, :]
    n = d.shape[1]
    se = d.std(axis=1, ddof=1) / np.sqrt(n)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = 2.0 * stats.t.sf(np.abs(d.mean(axis=1) / se), n - 1)
    return np.where(np.isfinite(p), p, 1.0) < alpha


# ─────────────────────────────────────────────────────────────────────────────
# Monte-Carlo power
# ─────────────────────────────────────────────────────────────────────────────

def _draw(rng, mean_d, sd_d, n, replicates):
    return rng.normal(mean_d, sd_d, size=(replicates, n))


def tost_power(mean_d, sd_d, n, margin, rng, replicates=N_REPLICATES,
               alpha=ALPHA, chunk=4000):
    """P(TOST concludes equivalence) for n paired windows drawn N(mean_d, sd_d).

    Chunked so that a large n does not allocate a (replicates, n) array in one
    go; the chunking changes the draw ORDER but not the estimator, and the
    generator is consumed sequentially so the result is still deterministic
    given `rng`.
    """
    hits, done = 0, 0
    while done < replicates:
        r = min(chunk, replicates - done)
        hits += int(tost_reject(_draw(rng, mean_d, sd_d, n, r), margin,
                                alpha).sum())
        done += r
    return hits / replicates


def tost_power_bootstrap(observed, n, margin, rng, replicates=N_REPLICATES,
                         alpha=ALPHA, chunk=4000):
    """Same power, resampling the OBSERVED differences instead of assuming
    normality. A large gap between this and `tost_power` means the normal model
    is carrying the result. The gap has no guaranteed SIGN -- see "which of the
    two power estimates is the conservative one" in the module docstring."""
    obs = np.asarray(observed, dtype=float)
    hits, done = 0, 0
    while done < replicates:
        r = min(chunk, replicates - done)
        sample = rng.choice(obs, size=(r, n), replace=True)
        hits += int(tost_reject(sample, margin, alpha).sum())
        done += r
    return hits / replicates


def ttest_power(mean_d, sd_d, n, rng, replicates=N_REPLICATES, alpha=ALPHA,
                chunk=4000):
    """P(paired t-test rejects mu_d = 0) at n windows."""
    hits, done = 0, 0
    while done < replicates:
        r = min(chunk, replicates - done)
        hits += int(ttest_reject(_draw(rng, mean_d, sd_d, n, r), alpha).sum())
        done += r
    return hits / replicates


def smallest_n_for_power(power_fn, grid, target=TARGET_POWER, hopeless=False):
    """First grid point reaching `target` power.

    Returns (n_or_None, n_scanned_max, power_at_that_n). `hopeless=True` stops
    the scan at HOPELESS_SCAN_N: it is set only when the effect lies outside the
    equivalence margin, where TOST power provably falls toward 0 with n.
    """
    n_last, p_last = None, float('nan')
    for n in grid:
        p = power_fn(n)
        n_last, p_last = n, p
        if p >= target:
            return n, n_last, p_last
        if hopeless and n >= HOPELESS_SCAN_N:
            break
    return None, n_last, p_last


# ─────────────────────────────────────────────────────────────────────────────
# How many disjoint windows can each trace actually supply?
# ─────────────────────────────────────────────────────────────────────────────

def window_supply(trace_keys=('sdsc', 'lanl')):
    """Disjoint-window capacity of each trace under the PROJECT's protocol.

    Everything here is read from `trace_driven_benchmark` rather than restated:
    the job filter (`parse_swf_jobs`, which opens the committed .swf.gz through
    `swf_io.open_swf`), the 60% chronological train split, and the
    WARMUP_DAYS + MEASURE_DAYS window span. If the protocol changes, these
    numbers change with it instead of going quietly stale.

    Three counts, from strictest to loosest:

      n_disjoint_protocol
          Non-overlapping (warm-up + measured) spans inside the post-split
          evaluation region. This is what `carve_windows` produces and the only
          count directly comparable to the benchmark's n = 20.
      n_disjoint_measured_only
          Disjoint 7-day MEASURED periods, allowing each window's warm-up to
          overlap the previous window's measured period. Looser, and it would
          break the "non-overlapping windows" property the benchmark documents,
          but it bounds what a redesigned protocol could reach.
      n_disjoint_whole_trace
          Non-overlapping spans over the ENTIRE trace, ignoring the train/eval
          split. Unreachable without leaking training data into evaluation; it
          is the absolute ceiling of the trace, quoted so the reader can see
          that even abandoning the split does not rescue a large n.
    """
    import trace_driven_benchmark as tdb   # heavy (xgboost); imported on demand

    span_days = tdb.WARMUP_DAYS + tdb.MEASURE_DAYS
    span = span_days * tdb.DAY
    out = {}
    for key in trace_keys:
        meta = tdb.TRACES[key]
        capacity, df, _ = tdb.parse_swf_jobs(
            os.path.join(tdb.DATA_DIR, meta['swf']))
        t_min = int(df['submit'].min())
        t_max = int(df['submit'].max())
        split_time = int(t_min + tdb.TRAIN_FRACTION * (t_max - t_min))
        available = t_max - split_time
        n_proto = int(available // span)
        # Back-to-back tiling of the evaluation region, applying carve_windows'
        # own >= 50 jobs rule, so a window that the harness would silently drop
        # is not counted as supply.
        usable = 0
        for k in range(n_proto):
            w0 = split_time + k * span
            sel = df[(df['submit'] >= w0) & (df['submit'] < w0 + span)]
            if len(sel) >= 50:
                usable += 1
        measured = tdb.MEASURE_DAYS * tdb.DAY
        warm = tdb.WARMUP_DAYS * tdb.DAY
        out[key] = {
            'label': meta['label'],
            'capacity': capacity,
            'jobs_kept': int(len(df)),
            'trace_span_days': (t_max - t_min) / tdb.DAY,
            'eval_span_days': available / tdb.DAY,
            'window_span_days': span_days,
            'n_disjoint_protocol': n_proto,
            'n_disjoint_protocol_ge50jobs': usable,
            'n_disjoint_measured_only': int(max(available - warm, 0) // measured),
            'n_disjoint_whole_trace': int((t_max - t_min) // span),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

# The pairs the project actually tests. Kept in sync with EQUIV_PAIRS in
# trace_driven_benchmark.py (a local list because that one is defined inside
# main() and cannot be imported); test_tost_power.py parses the benchmark
# source and fails if the two lists diverge.
EQUIV_PAIRS = [
    ('SMALLEST_FIRST', 'PROACTIVE'),   # is the ML pipeline redundant?
    ('PROACTIVE_EST', 'SJF_USEREST'),  # does ML add anything over SJF?
    ('PROACTIVE', 'FCFS'),             # does ML beat doing nothing?
]
METRICS = ('mean_wait', 'mean_bounded_slowdown')
TRACES_ORDER = ('sdsc', 'lanl')

COLUMNS = ['trace', 'pair', 'scheduler_a', 'scheduler_b', 'metric',
           'n_observed', 'mean_diff', 'sd_diff', 'margin', 'margin_frac',
           'alpha', 'achieved_power', 'achieved_power_bootstrap',
           'n_for_80pct_power', 'n_for_80pct_power_difference',
           'n_for_80pct_power_difference_holm', 'holm_family_size',
           'n_disjoint_windows_available', 'settleable_with_disjoint_windows',
           'p_tost_observed', 'equivalent_observed', 'effect_outside_margin',
           'n_scan_max', 'power_at_n_scan_max', 'replicates', 'seed', 'note']


def _holm_family_size(runs_df, reference):
    """m for the Holm worst case: schedulers compared against `reference` in
    the project's own significance table (every scheduler but the reference)."""
    return int(runs_df['scheduler'].nunique() - 1)


def _note(outside, n_eq, n_diff_holm, supply, achieved):
    if not outside and n_eq is not None and n_eq <= supply:
        return (f'equivalence reachable: {n_eq} disjoint windows needed, '
                f'{supply} available')
    if not outside and n_eq is not None:
        return (f'equivalence needs {n_eq} disjoint windows but the trace '
                f'supplies only {supply}; INCONCLUSIVE at this margin')
    if not outside:
        return (f'observed effect is inside the margin but power stays below '
                f'{TARGET_POWER:.2f} out to n={N_GRID_MAX}; the trace supplies '
                f'{supply}; INCONCLUSIVE at this margin')
    base = ('observed effect lies OUTSIDE the margin, so TOST power falls '
            f'toward 0 with n (achieved {achieved:.4f} at n=20) and no number '
            'of windows can certify equivalence; ')
    if n_diff_holm is not None and n_diff_holm <= supply:
        return base + (f'a Holm-corrected difference verdict needs '
                       f'{n_diff_holm} windows and the trace supplies '
                       f'{supply}: settleable as DIFFERENT')
    if n_diff_holm is not None:
        return base + (f'a Holm-corrected difference verdict needs '
                       f'{n_diff_holm} windows but the trace supplies only '
                       f'{supply}: NOT settleable, report INCONCLUSIVE')
    return base + (f'a Holm-corrected difference verdict is out of reach by '
                   f'n={N_GRID_MAX} too, and the trace supplies {supply}: '
                   f'NOT settleable, report INCONCLUSIVE')


def build_table(runs, supply, replicates=N_REPLICATES, seed=POWER_SEED):
    grid = power_grid()
    rows = []
    row_index = 0
    for trace in TRACES_ORDER:
        sub = runs[runs['trace'] == trace]
        if sub.empty:
            continue
        avail = supply[trace]['n_disjoint_protocol']
        m_holm = _holm_family_size(sub, 'PROACTIVE')
        alpha_holm = ALPHA / m_holm
        for metric in METRICS:
            for a_name, b_name in EQUIV_PAIRS:
                a, b = _aligned_pair(sub, a_name, b_name, metric, 'window')
                d = a - b
                obs = tost_equivalence(a, b, margin_frac=MARGIN_FRAC,
                                       alpha=ALPHA)
                margin = obs['margin']
                mean_d = float(d.mean())
                sd_d = float(d.std(ddof=1))
                n_obs = int(len(d))
                outside = abs(mean_d) >= margin

                rng = np.random.default_rng([seed, row_index])
                row_index += 1

                achieved = tost_power(mean_d, sd_d, n_obs, margin, rng,
                                      replicates)
                achieved_bs = tost_power_bootstrap(d, n_obs, margin, rng,
                                                   replicates)
                n_eq, n_scan, p_scan = smallest_n_for_power(
                    lambda n: tost_power(mean_d, sd_d, n, margin, rng,
                                         replicates),
                    grid, hopeless=outside)
                n_diff, _, _ = smallest_n_for_power(
                    lambda n: ttest_power(mean_d, sd_d, n, rng, replicates,
                                          ALPHA), grid)
                n_diff_h, _, _ = smallest_n_for_power(
                    lambda n: ttest_power(mean_d, sd_d, n, rng, replicates,
                                          alpha_holm), grid)

                settleable = bool((n_eq is not None and n_eq <= avail)
                                  or (n_diff_h is not None
                                      and n_diff_h <= avail))
                rows.append({
                    'trace': trace,
                    'pair': f'{a_name} vs {b_name}',
                    'scheduler_a': a_name,
                    'scheduler_b': b_name,
                    'metric': metric,
                    'n_observed': n_obs,
                    'mean_diff': mean_d,
                    'sd_diff': sd_d,
                    'margin': float(margin),
                    'margin_frac': MARGIN_FRAC,
                    'alpha': ALPHA,
                    'achieved_power': achieved,
                    'achieved_power_bootstrap': achieved_bs,
                    'n_for_80pct_power': n_eq,
                    'n_for_80pct_power_difference': n_diff,
                    'n_for_80pct_power_difference_holm': n_diff_h,
                    'holm_family_size': m_holm,
                    'n_disjoint_windows_available': avail,
                    'settleable_with_disjoint_windows': settleable,
                    'p_tost_observed': float(obs['p_tost']),
                    'equivalent_observed': bool(obs['equivalent']),
                    'effect_outside_margin': bool(outside),
                    'n_scan_max': n_scan,
                    'power_at_n_scan_max': float(p_scan),
                    'replicates': int(replicates),
                    'seed': int(seed),
                    'note': _note(outside, n_eq, n_diff_h, avail, achieved),
                })
    return pd.DataFrame(rows, columns=COLUMNS)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--replicates', type=int, default=N_REPLICATES,
                    help='Monte-Carlo replicates per power estimate')
    ap.add_argument('--seed', type=int, default=POWER_SEED,
                    help='root seed for the power simulation only')
    args = ap.parse_args()

    if not os.path.exists(WINDOWS_CSV):
        raise SystemExit(
            f'{WINDOWS_CSV} is missing. Run trace_driven_benchmark.py first '
            f'(run_all_experiments.sh does this).')
    runs = pd.read_csv(WINDOWS_CSV)

    print('=' * 78)
    print('DISJOINT-WINDOW SUPPLY (from the committed .swf.gz, project protocol)')
    print('=' * 78)
    supply = window_supply(TRACES_ORDER)
    for key in TRACES_ORDER:
        s = supply[key]
        print(f"{s['label']:>18s} | trace {s['trace_span_days']:7.1f} d | "
              f"post-split {s['eval_span_days']:7.1f} d | "
              f"{s['window_span_days']}-day windows | "
              f"disjoint: {s['n_disjoint_protocol']:3d} "
              f"({s['n_disjoint_protocol_ge50jobs']} with >=50 jobs) | "
              f"measured-only {s['n_disjoint_measured_only']:3d} | "
              f"whole trace {s['n_disjoint_whole_trace']:3d}")

    print(f'\nSimulating power: {args.replicates} replicates per estimate, '
          f'root seed {args.seed} ...')
    table = build_table(runs, supply, args.replicates, args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)

    print('\n' + '=' * 78)
    print('ACHIEVED POWER OF THE PUBLISHED 20-WINDOW TOST, AND WHAT WOULD BE NEEDED')
    print('=' * 78)
    disp = table[['trace', 'pair', 'metric', 'mean_diff', 'margin',
                  'achieved_power', 'n_for_80pct_power',
                  'n_for_80pct_power_difference_holm',
                  'n_disjoint_windows_available',
                  'settleable_with_disjoint_windows']].copy()
    print(disp.to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    print('\nPower is computed AT THE OBSERVED EFFECT SIZE and is therefore '
          'post-hoc:\nit is an estimate from n=20, not a guarantee. Where the '
          'required n exceeds\nthe disjoint-window supply, the honest verdict '
          'is INCONCLUSIVE, not "different".')
    print(f'\nWrote {OUT_CSV}')


if __name__ == '__main__':
    main()

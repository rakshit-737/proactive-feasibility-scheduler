"""Verify the CLAIMS the paper makes, rather than the digits the artefacts hold.

    python tools/verify_claims.py                 # check the committed tree
    python tools/verify_claims.py --root DIR      # check a regenerated tree
    python tools/verify_claims.py --report P.md   # write the table to a file

WHY THIS EXISTS, AND WHY IT IS NOT A WEAKER verify_artifacts.py
---------------------------------------------------------------
`tools/verify_artifacts.py` re-runs the pipeline and diffs every artefact digit
for digit. That is the right instrument for "does this tree regenerate itself",
and it is how this repository catches a stale file or a silent edit.

It is the wrong instrument for "does this result hold somewhere else", because
XGBoost's histogram build reduces in parallel: refit the same data with the same
seed on a different core count, or a different library build, and the model
differs in the last few decimals. The model drives dispatch decisions, dispatch
decisions change which instants exist, and one perturbation lands in every
downstream count. Measured: a 2-vCPU Linux runner reproduced 45,268 dispatch
instants where this project's reference platform records 45,432.

A digit diff calls that a failure. It is not one. 45,432 is a property of the
reference platform; **zero violations** is the property of the world, and it held
on Linux unchanged. This script checks the second kind of statement -- the kind
the paper actually makes -- so that a platform which cannot reproduce the digits
can still confirm or refute the science.

The two tools are complementary and neither replaces the other:

    verify_artifacts.py   exact, platform-scoped, catches drift and staleness
    verify_claims.py      structural, platform-independent, catches a wrong claim

WHAT COUNTS AS A CLAIM HERE
---------------------------
Only statements `reports/honest_claims.md` says the repository may make. Each
check names the claim in the words the paper uses, so a reader can line them up.
Where a claim is directional ("every augmented arm breaks the degeneracy"), the
direction is asserted and the magnitude is reported but not pinned. Where a claim
is an exact identity ("the neural net and the size sort are the same policy"),
exactness is asserted, because there the identity IS the claim.

A check whose inputs are missing FAILS. This script is run against trees that are
supposed to be complete; a silent skip would let a truncated pipeline pass.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASS = 'PASS'
FAIL = 'FAIL'

# Directional checks allow the magnitude to move between platforms; these are the
# margins inside which a REPORTED number is still considered the same finding.
# They are deliberately loose, because they guard the claim, not the digit: the
# exact values are guarded by tests/test_golden_numbers.py on the reference
# platform. A value outside these bands is not platform noise, it is a different
# result.
INSTANT_TOL = 0.02          # 2% of the reference dispatch-instant count
REFERENCE_INSTANTS = 45432


class Checker:
    """Collects claim outcomes. Every check returns None and records a row."""

    def __init__(self, root):
        self.root = root
        self.rows = []
        self._cache = {}

    def read(self, rel):
        if rel not in self._cache:
            path = os.path.join(self.root, rel.replace('/', os.sep))
            if not os.path.exists(path):
                raise FileNotFoundError(rel)
            self._cache[rel] = pd.read_csv(path, float_precision='round_trip')
        return self._cache[rel]

    def check(self, claim, fn):
        """Run one claim check. `fn` returns a detail string or raises."""
        try:
            detail = fn()
            self.rows.append((PASS, claim, detail))
        except FileNotFoundError as exc:
            self.rows.append((FAIL, claim, f'artefact missing: {exc}'))
        except AssertionError as exc:
            self.rows.append((FAIL, claim, str(exc) or 'assertion failed'))
        except Exception as exc:                      # noqa: BLE001
            self.rows.append((FAIL, claim, f'{type(exc).__name__}: {exc}'))


# ─────────────────────────────────────────────────────────────────────────────
# 1. The degeneracy itself
# ─────────────────────────────────────────────────────────────────────────────

def claim_no_violations(c):
    df = c.read('05_results/degeneracy/ranking_degeneracy.csv')
    bad = df[df['equal_size_diff_pred_violations'] != 0]
    assert bad.empty, (
        'the central claim FAILED: equally-sized co-queued jobs received different '
        f'scores in {len(bad)} setting(s): {list(bad["setting"])}')
    return f'0 violations across {len(df)} settings, {int(df["ranking_instants"].sum()):,} instants'


def claim_totals_complete(c):
    t = c.read('05_results/degeneracy/ranking_degeneracy_totals.csv').iloc[0]
    assert not bool(t['partial']), 'the diagnostic ran PARTIAL; the totals are not the published protocol'
    assert int(t['settings_present']) == int(t['settings_expected']), (
        f'only {int(t["settings_present"])} of {int(t["settings_expected"])} settings present')
    assert int(t['total_violations']) == 0, f'total violations is {int(t["total_violations"])}, not 0'
    n = int(t['total_instants'])
    drift = abs(n - REFERENCE_INSTANTS) / REFERENCE_INSTANTS
    assert drift <= INSTANT_TOL, (
        f'{n:,} dispatch instants is {drift:.1%} from the reference {REFERENCE_INSTANTS:,} -- '
        'beyond platform noise; the protocol or the traces changed')
    return f'{n:,} instants ({drift:+.2%} vs reference), 0 violations, complete'


def claim_constant_features(c):
    df = c.read('05_results/degeneracy/feature_variation.csv')
    syn = df[df['setting'].str.contains('Synthetic', case=False, na=False)]
    assert not syn.empty, 'no synthetic setting in feature_variation.csv'
    zero = syn[syn['pct_instants_varying_across_queue'] == 0.0]
    assert len(zero) == 7, (
        f'{len(zero)} of 12 synthetic features are constant across the queue, expected 7 -- '
        'the feature set changed, so the "cannot separate two jobs" argument must be rechecked')
    # queue_pressure is a function of size given the state: it must vary exactly
    # as often as job_gpu does, not merely close to it. This is an identity.
    for setting, g in df.groupby('setting'):
        by = g.set_index('feature')['pct_instants_varying_across_queue']
        if 'queue_pressure' in by.index and 'job_gpu' in by.index:
            assert by['queue_pressure'] == by['job_gpu'], (
                f'{setting}: queue_pressure varies {by["queue_pressure"]}% but job_gpu '
                f'{by["job_gpu"]}% -- they are no longer the same function of size')
    return '7 of 12 synthetic features constant; queue_pressure identical to job_gpu everywhere'


# ─────────────────────────────────────────────────────────────────────────────
# 2. Equivalence to a size sort
# ─────────────────────────────────────────────────────────────────────────────

def claim_nn_is_the_size_sort(c):
    runs = c.read('05_results/schedulers/multi_scheduler_runs.csv')
    metrics = ['mean_wait', 'max_wait', 'p95_wait', 'mean_turnaround',
               'mean_bounded_slowdown', 'throughput', 'fairness_gini', 'gpu_util']
    nn = runs[runs['scheduler'] == 'NN'].sort_values('run').reset_index(drop=True)
    sm = runs[runs['scheduler'] == 'SMALLEST'].sort_values('run').reset_index(drop=True)
    assert len(nn) and len(sm), 'NN or SMALLEST missing from multi_scheduler_runs.csv'
    assert len(nn) == len(sm), f'{len(nn)} NN runs vs {len(sm)} SMALLEST runs'
    for m in metrics:
        if m not in runs.columns:
            continue
        assert np.array_equal(nn[m].to_numpy(), sm[m].to_numpy()), (
            f'NN and SMALLEST differ on {m} -- the "identical, not merely equivalent" '
            'claim no longer holds')
    return f'bit-identical on {len(metrics)} metrics across {len(nn)} runs'


def claim_synthetic_equivalence(c):
    df = c.read('05_results/schedulers/multi_scheduler_equivalence.csv')
    rows = df[(df['metric'] == 'mean_wait') &
              (((df['scheduler_a'] == 'SMALLEST') & (df['scheduler_b'] == 'PROACTIVE')) |
               ((df['scheduler_a'] == 'PROACTIVE') & (df['scheduler_b'] == 'SMALLEST')))]
    assert not rows.empty, 'no PROACTIVE/SMALLEST mean_wait equivalence row'
    assert rows['equivalent'].all(), (
        'PROACTIVE is no longer TOST-equivalent to SMALLEST on the synthetic benchmark')
    r = rows.iloc[0]
    return f'equivalent, pct_diff {r["pct_diff"]:+.3f}%, p_tost {r["p_tost"]:.2e}'


def claim_sdsc_equivalence(c):
    df = c.read('05_results/trace_schedulers/trace_scheduler_equivalence.csv')
    rows = df[(df['trace'].str.lower() == 'sdsc') & (df['metric'] == 'mean_wait') &
              (((df['scheduler_a'] == 'SMALLEST_FIRST') & (df['scheduler_b'] == 'PROACTIVE')) |
               ((df['scheduler_a'] == 'PROACTIVE') & (df['scheduler_b'] == 'SMALLEST_FIRST')))]
    assert not rows.empty, 'no SDSC PROACTIVE/SMALLEST_FIRST mean_wait row'
    assert rows['equivalent'].all(), 'SDSC equivalence to the size sort no longer holds'
    r = rows.iloc[0]
    return f'SDSC equivalent, pct_diff {r["pct_diff"]:+.3f}%, p_tost {r["p_tost"]:.2e}'


# ─────────────────────────────────────────────────────────────────────────────
# 3. Phase D: necessary, not sufficient
# ─────────────────────────────────────────────────────────────────────────────

def claim_augmented_break_degeneracy(c):
    df = c.read('05_results/degeneracy/non_degeneracy_sweep.csv')
    base = df[df['feature_set'] == 'baseline']
    aug = df[df['feature_set'] != 'baseline']
    assert not base.empty and not aug.empty, 'sweep has no baseline or no augmented arms'
    assert (base['equal_size_diff_pred_violations'] == 0).all(), (
        'the sweep baseline produced violations -- the control no longer reproduces the '
        'published policy, so the augmented arms are uninterpretable')
    assert (aug['equal_size_diff_pred_violations'] > 0).all(), (
        'an augmented feature set stopped breaking the degeneracy: '
        f'{list(aug[aug["equal_size_diff_pred_violations"] == 0]["feature_set"])}')
    for trace, g in df.groupby('trace'):
        b = g[g['feature_set'] == 'baseline']
        a = g[g['feature_set'] != 'baseline']
        if b.empty or a.empty:
            continue
        b0 = b.iloc[0]
        assert (a['pct_all_scores_tied'] < b0['pct_all_scores_tied']).all(), \
            f'{trace}: an augmented arm did not reduce the all-tied fraction'
        assert (a['mean_distinct_predictions'] > b0['mean_distinct_predictions']).all(), \
            f'{trace}: an augmented arm did not raise the number of distinct score levels'
    lo, hi = int(aug['equal_size_diff_pred_violations'].min()), int(aug['equal_size_diff_pred_violations'].max())
    return f'{len(aug)} augmented arms, all break it ({lo:,}-{hi:,} violations), all directions hold'


def claim_none_beat_the_heuristic(c):
    df = c.read('05_results/degeneracy/non_degeneracy_utility.csv')
    assert not df['beats_sjf_userest'].any(), (
        'an augmented variant now BEATS shortest-job-first on user estimates -- the '
        '"necessary but not sufficient" finding must be restated, not silently kept: '
        f'{list(df[df["beats_sjf_userest"]]["feature_set"])}')
    assert (df['pct_vs_sjf_userest'] > 0).all(), \
        'a variant is faster than the heuristic in the paired mean'
    best = df.loc[df['pct_vs_sjf_userest'].idxmin()]
    return (f'0 of {len(df)} beat SJF on user estimates; best is '
            f'{best["trace"]}/{best["feature_set"]} at {best["pct_vs_sjf_userest"]:+.2f}%')


def claim_attacks(c):
    df = c.read('05_results/degeneracy/robustness_attacks.csv')
    survive = df[df['attack_id'].isin(['A0', 'A1', 'A2', 'A3'])]
    broke = df[df['attack_id'] == 'A4']
    assert not survive.empty and not broke.empty, 'attack rows missing'
    assert (survive['violations'] == 0).all(), (
        'an attack that should fail now produces counterexamples: '
        f'{list(survive[survive["violations"] > 0]["attack_id"].unique())} -- the claim is '
        'narrower than the paper states')
    assert (broke['violations'] > 0).all(), (
        'A4 (enqueue-time caching) no longer breaks the degeneracy; the documented scope '
        'limitation may be wrong')
    # A2 is a strictly monotone reparameterisation: it CANNOT reorder anything.
    # If its order-derived columns ever differ from the control, the instrument is
    # broken rather than the theory.
    for trace, g in df.groupby('trace'):
        a0 = g[g['attack_id'] == 'A0']
        a2 = g[g['attack_id'] == 'A2']
        if a0.empty or a2.empty:
            continue
        for col in ('ranking_instants', 'violations'):
            assert (a2[col] == a0.iloc[0][col]).all(), (
                f'{trace}: the monotone attack changed {col}; a strictly monotone map '
                'cannot reorder a queue, so the harness is wrong')
    return f'A0-A3 zero violations, A4 breaks it ({int(broke["violations"].min()):,}-{int(broke["violations"].max()):,})'


def claim_breaking_it_costs_more(c):
    df = c.read('05_results/degeneracy/robustness_attack_utility.csv')
    by = df.set_index(['trace', 'scheduler'])
    seen = []
    for trace in df['trace'].unique():
        try:
            cached = by.loc[(trace, 'PROACTIVE_ENQUEUE_CACHED')]
            proactive = by.loc[(trace, 'PROACTIVE')]
            sjf = by.loc[(trace, 'SJF_USEREST')]
        except KeyError:
            continue
        assert sjf['mean_wait'] < cached['mean_wait'], (
            f'{trace}: the ML-free heuristic no longer beats the non-degenerate variant')
        assert cached['mean_bounded_slowdown'] > proactive['mean_bounded_slowdown'], (
            f'{trace}: enqueue-caching is no longer worse on bounded slowdown')
        seen.append(trace)
    assert seen, 'no trace had all three schedulers present'
    return f'on {", ".join(seen)}: SJF beats the non-degenerate variant, which is worse on slowdown'


# ─────────────────────────────────────────────────────────────────────────────
# 4. Supporting claims the prose depends on
# ─────────────────────────────────────────────────────────────────────────────

def claim_improvement_is_real_and_is_reordering(c):
    s = c.read('05_results/benchmark_statistical_summary.csv').set_index('metric')['value']
    pct = float(s['mean_improvement_pct'])
    lo, hi = float(s['improvement_ci95_low']), float(s['improvement_ci95_high'])
    assert pct > 0, f'the wait improvement is {pct:.3f}%, no longer positive'
    assert lo > 0, f'the 95% CI [{lo:.2f}, {hi:.2f}] now includes zero'
    assert float(s['ttest_pvalue']) < 0.05, 'the paired t-test is no longer significant'
    return f'{pct:.2f}% [{lo:.2f}, {hi:.2f}], p={float(s["ttest_pvalue"]):.2e}'


def claim_utilisation_unchanged(c):
    runs = c.read('05_results/schedulers/multi_scheduler_runs.csv')
    if 'gpu_util' not in runs.columns:
        raise AssertionError('gpu_util column absent')
    fifo = runs[runs['scheduler'] == 'FIFO'].sort_values('run')['gpu_util'].to_numpy()
    pro = runs[runs['scheduler'] == 'PROACTIVE'].sort_values('run')['gpu_util'].to_numpy()
    assert len(fifo) and len(fifo) == len(pro), 'FIFO/PROACTIVE run counts differ'
    assert np.allclose(fifo, pro, rtol=0, atol=1e-6), (
        'utilisation now differs between FIFO and PROACTIVE -- the "reordering, not a '
        'throughput gain" framing changes, and the deleted ROI study would need revisiting')
    return f'identical across {len(fifo)} paired runs (max diff {np.abs(fifo - pro).max():.2e})'


def claim_split_ordering(c):
    df = c.read('05_results/models/evaluation_splits.csv').set_index('split')
    need = ['random', 'run_wise', 'chronological']
    for k in need:
        assert k in df.index, f'split "{k}" missing'
    r = df['r2_mean']
    assert r['chronological'] < r['run_wise'] < r['random'], (
        'the split ordering reversed: random should be the most optimistic and '
        f'chronological the least, got random={r["random"]:.4f}, '
        f'run_wise={r["run_wise"]:.4f}, chronological={r["chronological"]:.4f}')
    return (f'random {r["random"]:.3f} > run-wise {r["run_wise"]:.3f} > '
            f'chronological {r["chronological"]:.3f}')


def claim_estimates_are_not_the_f_model(c):
    df = c.read('05_results/trace_schedulers/trace_estimate_quality.csv').set_index('trace')
    assert 'lanl' in df.index, 'lanl row missing from trace_estimate_quality.csv'
    under = float(df.loc['lanl', 'under_estimate_frac'])
    fmodel = float(df.loc['lanl', 'fmodel_under_estimate_frac'])
    assert under > 0.05, (
        f'LANL under-estimation is now {under:.2%}; the argument that real estimates '
        'cannot come from runtime x U(1,C) rests on it being substantial')
    assert fmodel == 0.0, (
        f'the f-model produced {fmodel:.2%} under-estimates; by construction it cannot '
        'produce any, so the comparison is broken')
    return f'LANL under-estimates {under:.2%} of the time; the f-model, by construction, 0%'


CHECKS = [
    ('no two equally-sized co-queued jobs ever scored differently', claim_no_violations),
    ('the diagnostic ran the full protocol and totals agree', claim_totals_complete),
    ('7 of 12 synthetic features cannot separate two queued jobs', claim_constant_features),
    ('the neural net IS the size sort, not merely equivalent to it', claim_nn_is_the_size_sort),
    ('PROACTIVE is TOST-equivalent to SMALLEST (synthetic)', claim_synthetic_equivalence),
    ('PROACTIVE is TOST-equivalent to the size sort (SDSC)', claim_sdsc_equivalence),
    ('every genuine per-job feature breaks the degeneracy', claim_augmented_break_degeneracy),
    ('breaking it beats nothing: 0 of 12 beat SJF on user estimates', claim_none_beat_the_heuristic),
    ('three attacks fail, only enqueue-time caching succeeds', claim_attacks),
    ('the one successful attack schedules worse', claim_breaking_it_costs_more),
    ('the wait improvement is positive and significant', claim_improvement_is_real_and_is_reordering),
    ('the improvement is reordering: utilisation is unchanged', claim_utilisation_unchanged),
    ('random-split accuracy is optimistic; chronological is worst', claim_split_ordering),
    ('real user estimates are not the f-model', claim_estimates_are_not_the_f_model),
]


def render(rows, root):
    out = ['# Claim verification', '',
           f'tree: {root}', '',
           'Structural checks of the statements `reports/honest_claims.md` permits.',
           'Digits are allowed to move between platforms; claims are not.', '',
           '| status | claim | detail |', '| --- | --- | --- |']
    for status, claim, detail in rows:
        out.append(f'| {status} | {claim} | {detail} |')
    failed = sum(1 for s, _, _ in rows if s == FAIL)
    out += ['', f'**{len(rows) - failed} of {len(rows)} claims hold.**']
    if failed:
        out.append('')
        out.append('A FAIL here is not drift and is not a platform difference. It means an '
                   'artefact in this tree does not support a statement the repository makes. '
                   'Either the result changed, or the claim was wrong.')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=ROOT, help='tree to check (default: this repository)')
    ap.add_argument('--report', help='write the table to this path as well')
    args = ap.parse_args()

    c = Checker(args.root)
    for claim, fn in CHECKS:
        c.check(claim, lambda fn=fn: fn(c))

    report = render(c.rows, args.root)
    print(report)
    if args.report:
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write(report + '\n')
        print(f'\nreport written to {args.report}')
    return 1 if any(s == FAIL for s, _, _ in c.rows) else 0


if __name__ == '__main__':
    sys.exit(main())

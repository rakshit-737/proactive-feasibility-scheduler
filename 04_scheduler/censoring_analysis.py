"""Selection-bias (censoring) audit of the uncertainty / OOD scheduling benchmark.

THE PROBLEM
-----------
`uncertainty_scheduler_benchmark.py` runs each policy for SIM_TIME=300 ticks and
reports `mean_wait` over the jobs that STARTED inside that horizon. Under load
(the `arr2.0_nodes4` scenario in particular) a large fraction of jobs never
start, and -- crucially -- DIFFERENT POLICIES START DIFFERENT JOBS. The
published `improvement_vs_fifo_pct` therefore compares two means taken over two
different job populations. A policy that starts only the easy jobs and strands
the hard ones scores a short mean wait for exactly the wrong reason; a policy
that manages to start MORE jobs is penalised, because the extra jobs it rescues
are by construction the ones that waited longest. Neither direction of that bias
is visible in the published column.

WHAT THIS SCRIPT MEASURES
-------------------------
It re-runs the identical simulations -- importing the simulator, the policies,
the scenario grid and the 7000+run seeds from `uncertainty_scheduler_benchmark`
rather than restating any of them -- and for every scenario and every PAIR of
policies reports the mean wait three ways:

  (a) AS PUBLISHED   each policy's mean over ITS OWN started jobs. Reproduces
                     the committed `uncertainty_ood_benchmark.csv` exactly.
  (b) COMMON SET     each policy's mean over the jobs that started under BOTH
                     policies, matched by job id. This is the only genuinely
                     paired comparison: identical jobs on both sides, so the
                     difference is the policies' doing and nothing else.
  (c) CENSORING-AWARE  the started FRACTION of each policy, reported next to the
                     common-set mean so the selection and the effect are read
                     together, plus a mean over ALL jobs in which a job that
                     never started contributes its censoring LOWER BOUND
                     (SIM_TIME - arrival_time, which is what the simulator
                     already records for such a job). No wait time is invented:
                     the true wait of a stranded job is >= that bound, so
                     `mean_wait_*_lowerbound` is a lower bound on the policy's
                     true all-jobs mean wait, and it is taken over the identical
                     full job set for both policies, so it carries no selection
                     at all. A Kaplan-Meier fit was rejected as over-machinery
                     here: censoring is administrative and shared (one horizon,
                     identical for every job and every policy), so the bound is
                     both tighter to state and exactly reproducible. What the
                     DERIVED `improvement_pct_lowerbound` does and does not mean
                     is a separate question, treated at the end of this
                     docstring.

THE DELIVERABLE AND ITS SIGN
----------------------------
The deliverable is the gap between (a) and (b):

    selection_gap_pp = improvement_pct_published - improvement_pct_common

The SIGN carries the meaning, and it must be read in BOTH directions:

  gap > 0   the published improvement OVERSTATES the pair. Part of it
            disappears once both policies are scored on the same jobs, i.e.
            that part was a selection artefact.
  gap < 0   the published improvement UNDERSTATES the pair. The paired
            comparison is MORE favourable to `policy_b` than the published one,
            i.e. the unpaired statistic was biased AGAINST that policy.
  gap = 0   no job is censored under either policy, so the published comparison
            was already paired and there is nothing to correct.

The `note` column states which of the three a row is, in words.

WHAT THIS AUDIT ACTUALLY FOUND
------------------------------
Read off the committed 30-row artefact:

* 24 of the 30 rows have `selection_gap_pp` exactly 0.0. Four of the five
  scenarios (`in_dist`, `arr0.5_nodes4`, `arr0.5_nodes16`, `arr2.0_nodes16`)
  start EVERY job under EVERY policy, so they carry no censoring at all and
  nothing to correct. That is the headline: the concern touches one scenario,
  not the benchmark.
* The 6 remaining rows are all the `arr2.0_nodes4` overload scenario, and every
  NON-ZERO gap is NEGATIVE (-3.25, -2.22, -1.96, -1.93, -1.84, -0.38 pp). The
  direction is the opposite of the worry: on the common set the learned
  policies come out BETTER than the published column says. The published
  statistic understated them by up to 3.3 pp; it did not flatter them.
* The selection effect is nonetheless REAL, and against FIFO it is a TWO-WAY
  EXCHANGE rather than a one-way rescue. For `arr2.0_nodes4`, FIFO vs PROACTIVE
  (means over the 10 paired runs): 91.9 jobs start under BOTH policies, 21.8
  start under FIFO ONLY, and 39.4 start under PROACTIVE ONLY. Each policy
  starts jobs the other strands. PROACTIVE starts more on net (131.3 vs 113.7),
  which is why the unpaired published mean penalises it -- but "FIFO strands
  more jobs" is a weaker claim than "FIFO's started set is a subset of
  PROACTIVE's", and the latter is false here. (The two remaining FIFO pairs
  behave the same way; among the learned policies the exchange is tiny, and
  `UCB` vs `GUARDED` is in fact nested rather than two-way. The `n_only_a` and
  `n_only_b` columns say which shape each row has, so this is read off the
  artefact rather than assumed.)

WHAT IT DOES NOT ESTABLISH
--------------------------
It does not show the censoring concern is unfounded, and this script makes no
such claim. It shows the concern is confined to one of five scenarios and that,
where it bites, it runs in the direction that penalised the learned policies.
The common-set mean removes the selection only for the 91.9 jobs both FIFO and
PROACTIVE ran; the 61.2 jobs in that exchange are DESCRIBED (their counts, the
started fractions, the lower-bound means) but not adjudicated, because no
policy-free wait exists for a job that one policy never started.

ON `improvement_pct_lowerbound`
-------------------------------
`mean_wait_a_lowerbound` and `mean_wait_b_lowerbound` are genuine lower bounds
on each policy's true all-jobs mean wait. Their RATIO is not a bound on
anything: a lower bound on each of two means bounds neither the sign nor the
magnitude of their percent difference, since the two one-sided errors point the
same way and do not cancel. The column is kept rather than dropped, because the
two means it divides are the only selection-free numbers in the table and a
reader who wants the comparison would otherwise form the same ratio unwarned;
instead every censored row's `note` carries the caveat, so the artefact cannot
be read without it. Treat the column as descriptive, not as a bound.

Writes 05_results/uncertainty/censoring_analysis.csv.
"""

import os
import random
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import uncertainty_scheduler_benchmark as usb  # noqa: E402  (needs HERE on sys.path)

OUT_DIR = os.path.join(PROJECT_ROOT, '05_results', 'uncertainty')

# A row is flagged as selection-driven when the published and common-set
# improvements differ by more than this many percentage points.
GAP_FLAG_PP = 1.0

COLUMNS = [
    'scenario', 'policy_a', 'policy_b',
    'n_jobs', 'n_started_a', 'n_started_b', 'n_common', 'n_only_a', 'n_only_b',
    'started_frac_a_pct', 'started_frac_b_pct',
    'mean_wait_a_published', 'mean_wait_b_published',
    'mean_wait_a_common', 'mean_wait_b_common',
    'mean_wait_a_lowerbound', 'mean_wait_b_lowerbound',
    'improvement_pct_published', 'improvement_pct_common',
    'improvement_pct_lowerbound', 'selection_gap_pp', 'note',
]


# -- Pure statistics (no simulator, no model: unit-testable on hand-made data) --
def started_job_ids(job_waits, arrivals, sim_time=None):
    """Ids of jobs that actually started, recovered from a `run_once` result.

    `run_once` returns `job_waits` for EVERY job: a started job contributes
    `start_time - arrival_time`, a stranded one contributes the censoring bound
    `sim_time - arrival_time`. Because a start can happen no later than tick
    `sim_time - 1`, a started job's wait is STRICTLY below its own bound, so the
    two populations separate with no ambiguity and without needing the benchmark
    to change what it returns.
    """
    if sim_time is None:
        sim_time = usb.SIM_TIME
    return {jid for jid, w in job_waits.items() if w < sim_time - arrivals[jid]}


def _mean(values):
    values = list(values)
    return float(np.mean(values)) if values else float('nan')


def improvement_pct(base_mean, treatment_mean):
    """Percent reduction in mean wait of `treatment` relative to `base`."""
    if not np.isfinite(base_mean) or not np.isfinite(treatment_mean):
        return float('nan')
    if base_mean <= 0:
        return float('nan')
    return 100.0 * (base_mean - treatment_mean) / base_mean


def compare_pair(waits_a, started_a, waits_b, started_b):
    """Three views of one (policy_a, policy_b) comparison on ONE simulation run.

    `waits_*` map job id -> wait, censored entries already at their lower bound;
    `started_*` are the id sets that actually started. Every returned mean is a
    plain average over an explicitly named id set -- the common-set means are
    taken over `started_a & started_b`, NOT over each policy's own set, which is
    the entire point of the statistic.

    The started sets are also reported DECOMPOSED three ways -- common, a-only,
    b-only -- because a net count ("b starts more jobs than a") hides whether
    the two sets are NESTED or merely OVERLAPPING, and only the decomposition
    distinguishes them. Both readings occur in the committed artefact: the three
    FIFO pairs in `arr2.0_nodes4` overlap in both directions (each policy starts
    jobs the other strands, so the exchange is two-way and neither started set
    contains the other), while `UCB` vs `GUARDED` is nested (a-only is 0.0 in
    every run). The caller must read the two -only columns, not the net.
    """
    own_a = sorted(started_a)
    own_b = sorted(started_b)
    common = sorted(set(started_a) & set(started_b))
    n_jobs = len(waits_a)
    return {
        'n_jobs': n_jobs,
        'n_started_a': len(own_a),
        'n_started_b': len(own_b),
        'n_common': len(common),
        'n_only_a': len(set(started_a) - set(started_b)),
        'n_only_b': len(set(started_b) - set(started_a)),
        'started_frac_a_pct': 100.0 * len(own_a) / n_jobs if n_jobs else float('nan'),
        'started_frac_b_pct': 100.0 * len(own_b) / n_jobs if n_jobs else float('nan'),
        'mean_wait_a_published': _mean(waits_a[j] for j in own_a),
        'mean_wait_b_published': _mean(waits_b[j] for j in own_b),
        'mean_wait_a_common': _mean(waits_a[j] for j in common),
        'mean_wait_b_common': _mean(waits_b[j] for j in common),
        'mean_wait_a_lowerbound': _mean(waits_a.values()),
        'mean_wait_b_lowerbound': _mean(waits_b.values()),
    }


# Appended to every row that has censoring on at least one side. The ratio of
# two lower bounds is not a bound on the ratio, and the artefact has to say so
# where the number is read, not only in this module's docstring.
LOWERBOUND_CAVEAT = ('; caveat: improvement_pct_lowerbound divides two lower-bound means, '
                     'so it is NOT itself a bound on the improvement (bounding each of two '
                     'means bounds neither the sign nor the size of their ratio) -- '
                     'descriptive only')


def note_for(row):
    """Plain-language verdict for one aggregated pair row.

    THE SIGN CONVENTION, stated once and obeyed in both directions:
    `selection_gap_pp` is `improvement_pct_published - improvement_pct_common`,
    so a POSITIVE gap means the published figure OVERSTATED the improvement
    (that much of it disappears once both policies are scored on the same jobs)
    and a NEGATIVE gap means the published figure UNDERSTATED it (the paired
    comparison is more favourable to `policy_b` than the published one, i.e.
    the unpaired statistic was biased against that policy). Emitting the
    "disappears on the common set" wording for a negative gap states the result
    backwards, which is the defect this function's tests pin.
    """
    full_a = np.isclose(row['started_frac_a_pct'], 100.0)
    full_b = np.isclose(row['started_frac_b_pct'], 100.0)
    if full_a and full_b:
        return ('no censoring: every job starts under both policies, '
                'common set = full set, published improvement is unbiased; '
                'with nothing censored the lower-bound means are exact, so '
                'improvement_pct_lowerbound equals the published improvement here')
    gap = row['selection_gap_pp']
    stub = (f"censoring differs: {row['started_frac_a_pct']:.1f}% of jobs start under "
            f"{row['policy_a']} vs {row['started_frac_b_pct']:.1f}% under {row['policy_b']}")
    if not np.isfinite(gap):
        return stub + '; improvement undefined (base mean wait is zero)' + LOWERBOUND_CAVEAT
    if gap > GAP_FLAG_PP:
        verdict = (f'; the published improvement OVERSTATED this pair: {gap:.1f} pp of it '
                   f'disappears on the common set, so that much was a selection artefact')
    elif gap < -GAP_FLAG_PP:
        verdict = (f'; the published improvement UNDERSTATED this pair: on the common set '
                   f'the improvement is {abs(gap):.1f} pp LARGER, so the selection ran '
                   f"against {row['policy_b']} rather than flattering it")
    else:
        verdict = f'; common-set improvement within {GAP_FLAG_PP:.1f} pp of published'
    return stub + verdict + LOWERBOUND_CAVEAT


# -- Simulation driver (reuses uncertainty_scheduler_benchmark end to end) ------
def run_all_scenarios(verbose=True):
    """Re-run the published grid and return one record per (scenario, run, pair)."""
    records = []
    for sc in usb.SCENARIOS:
        n_jobs = int(round(usb.BASE_JOBS * sc['arrival_mult']))
        if verbose:
            print(f"\n--- Scenario {sc['name']} | nodes={sc['nodes']} jobs={n_jobs} ---")
        for run in range(usb.NUM_RUNS):
            # Identical seeding to uncertainty_scheduler_benchmark.main(): the
            # 7000+run family, set once before job generation, so this script
            # replays the SAME jobs the published numbers were computed on.
            random.seed(7000 + run)
            np.random.seed(7000 + run)
            jobs = usb.generate_jobs(n_jobs, usb.ARRIVAL_WINDOW)
            arrivals = {j.job_id: j.arrival_time for j in jobs}

            state = {}
            for pol in usb.POLICIES:
                res = usb.run_once(jobs, pol, sc['nodes'])
                ids = started_job_ids(res['job_waits'], arrivals)
                if len(ids) != res['n_started']:
                    raise AssertionError(
                        'started-set recovery disagrees with the simulator for '
                        f"{sc['name']}/{pol}/run{run + 1}: "
                        f"{len(ids)} recovered vs {res['n_started']} reported")
                state[pol] = (res['job_waits'], ids)

            for i, pol_a in enumerate(usb.POLICIES):
                for pol_b in usb.POLICIES[i + 1:]:
                    waits_a, ids_a = state[pol_a]
                    waits_b, ids_b = state[pol_b]
                    rec = {'scenario': sc['name'], 'policy_a': pol_a,
                           'policy_b': pol_b, 'run': run + 1}
                    rec.update(compare_pair(waits_a, ids_a, waits_b, ids_b))
                    records.append(rec)

            if verbose:
                print('  run {:2d}: '.format(run + 1) + ' | '.join(
                    f'{p} started {len(state[p][1])}/{n_jobs}' for p in usb.POLICIES))
    return pd.DataFrame(records)


def aggregate(per_run):
    """Average each per-run quantity over runs, THEN form the improvements.

    Aggregate-then-divide is deliberate: it is exactly what
    `uncertainty_scheduler_benchmark.main()` does, so `improvement_pct_published`
    reproduces the committed column and the gap against
    `improvement_pct_common` is attributable to the SELECTION and to nothing
    else -- not to a change of estimator.
    """
    value_cols = [c for c in per_run.columns
                  if c not in ('scenario', 'policy_a', 'policy_b', 'run')]
    agg = (per_run.groupby(['scenario', 'policy_a', 'policy_b'], as_index=False)[value_cols]
           .mean())
    agg['improvement_pct_published'] = [
        improvement_pct(r['mean_wait_a_published'], r['mean_wait_b_published'])
        for _, r in agg.iterrows()]
    agg['improvement_pct_common'] = [
        improvement_pct(r['mean_wait_a_common'], r['mean_wait_b_common'])
        for _, r in agg.iterrows()]
    agg['improvement_pct_lowerbound'] = [
        improvement_pct(r['mean_wait_a_lowerbound'], r['mean_wait_b_lowerbound'])
        for _, r in agg.iterrows()]
    agg['selection_gap_pp'] = (agg['improvement_pct_published']
                               - agg['improvement_pct_common'])
    agg['note'] = [note_for(r) for _, r in agg.iterrows()]

    scen_order = [s['name'] for s in usb.SCENARIOS]
    agg['scenario'] = pd.Categorical(agg['scenario'], categories=scen_order, ordered=True)
    agg['policy_a'] = pd.Categorical(agg['policy_a'], categories=usb.POLICIES, ordered=True)
    agg['policy_b'] = pd.Categorical(agg['policy_b'], categories=usb.POLICIES, ordered=True)
    agg = agg.sort_values(['scenario', 'policy_a', 'policy_b']).reset_index(drop=True)
    return agg[COLUMNS]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    per_run = run_all_scenarios()
    agg = aggregate(per_run)

    csv_path = os.path.join(OUT_DIR, 'censoring_analysis.csv')
    agg.to_csv(csv_path, index=False, encoding='utf-8')

    show = agg[['scenario', 'policy_a', 'policy_b', 'n_started_a', 'n_started_b',
                'n_common', 'n_only_a', 'n_only_b',
                'improvement_pct_published', 'improvement_pct_common',
                'improvement_pct_lowerbound', 'selection_gap_pp']]
    print('\n=== Censoring audit: published vs common-set improvement ===')
    print(show.to_string(index=False, formatters={
        'n_started_a': '{:.1f}'.format,
        'n_started_b': '{:.1f}'.format,
        'n_common': '{:.1f}'.format,
        'n_only_a': '{:.1f}'.format,
        'n_only_b': '{:.1f}'.format,
        'improvement_pct_published': '{:.2f}'.format,
        'improvement_pct_common': '{:.2f}'.format,
        'improvement_pct_lowerbound': '{:.2f}'.format,
        'selection_gap_pp': '{:+.2f}'.format,
    }))

    gaps = agg['selection_gap_pp']
    n_zero = int((gaps == 0.0).sum())
    n_neg = int((gaps < 0.0).sum())
    n_pos = int((gaps > 0.0).sum())
    print(f'\nselection_gap_pp across {len(agg)} rows: {n_zero} are EXACTLY 0.0 '
          f'(no job censored on either side, so nothing to correct), '
          f'{n_neg} negative, {n_pos} positive.')
    print('  negative = the published improvement UNDERSTATED the pair '
          '(common-set improvement is larger)')
    print('  positive = the published improvement OVERSTATED it '
          '(that much disappears on the common set)')
    per_scen = agg.groupby('scenario', observed=True)['selection_gap_pp'].apply(
        lambda s: bool((s == 0.0).all()))
    clean_scen = [str(name) for name, is_clean in per_scen.items() if is_clean]
    print(f'  scenarios with NO censoring at all: {len(clean_scen)} of '
          f'{len(usb.SCENARIOS)} -- {", ".join(clean_scen)}')

    flagged = agg[agg['selection_gap_pp'].abs() > GAP_FLAG_PP]
    worst = agg['selection_gap_pp'].abs().max()
    print(f'\nRows whose published improvement moves by more than {GAP_FLAG_PP:.1f} pp '
          f'on the common set: {len(flagged)} of {len(agg)}')
    for _, r in flagged.iterrows():
        direction = ('understated' if r['selection_gap_pp'] < 0 else 'overstated')
        print(f"  {r['scenario']:<16s} {r['policy_a']} vs {r['policy_b']}: "
              f"published {r['improvement_pct_published']:+.2f}% -> "
              f"common-set {r['improvement_pct_common']:+.2f}% "
              f"({r['selection_gap_pp']:+.2f} pp; published {direction} the pair)")
    print(f'Largest absolute selection gap: {worst:.2f} pp')

    censored = agg[agg['n_common'] < agg['n_jobs'] - 1e-9]
    if len(censored):
        print('\nStarted-set decomposition where censoring occurs (mean jobs per run). '
              'Two non-zero -only counts mean a TWO-WAY EXCHANGE -- each policy starts '
              'jobs the other strands and neither started set contains the other, so '
              '"b starts more jobs than a" is NOT "b starts a superset of a\'s jobs". '
              'One zero -only count means the sets are nested.')
        for _, r in censored.iterrows():
            if r['n_only_a'] > 0.0 and r['n_only_b'] > 0.0:
                shape = 'two-way exchange'
            elif r['n_only_a'] > 0.0 or r['n_only_b'] > 0.0:
                shape = 'nested: one started set contains the other'
            else:
                shape = 'identical started sets'
            print(f"  {r['scenario']:<16s} {r['policy_a']} vs {r['policy_b']}: "
                  f"common {r['n_common']:.1f} | {r['policy_a']}-only {r['n_only_a']:.1f} "
                  f"| {r['policy_b']}-only {r['n_only_b']:.1f}  ({shape})")

    print('\nHOW TO STATE THIS RESULT. The selection effect is real, is confined to the '
          'rows listed above, and runs in the direction that PENALISED the learned '
          'policies rather than flattering them: every non-zero gap is negative. That '
          'is not the same as showing the censoring concern is unfounded -- the common '
          'set adjudicates only the jobs both policies ran, while the jobs in the '
          'exchange are described and left unadjudicated, because no policy-free wait '
          'exists for a job one policy never started.')
    print('\nSaved:', csv_path)


if __name__ == '__main__':
    main()

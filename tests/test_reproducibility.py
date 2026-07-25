"""Reproducibility and anti-leakage invariants.

Every headline number in this repository is the mean of a set of *seeded*
simulation runs. That makes two properties load-bearing, and neither of them is
checked anywhere in the source:

  1. REPRODUCIBILITY - the same seed must regenerate the same workload, and two
     different seeds must give genuinely different workloads. If the first
     breaks, no published figure can be re-derived; if the second breaks, the
     "20 independent runs" behind every confidence interval are 20 copies of one
     run and every CI is a fiction.

  2. NON-LEAKAGE - the workloads a benchmark evaluates on must not be the
     workloads the wait model was trained on. This repo has already shipped that
     bug once (the synthetic benchmark used to evaluate on `42 + run`, the
     dataset generator's own seeds) and it was fixed by moving evaluation to the
     `1000 + run` family. Nothing stops it recurring; the tests below make it
     fail loudly.

The seed-family tests deliberately parse the *source text* of the two modules
rather than importing them. Importing `multi_scheduler_benchmark` unpickles
`03_models/wait_model_v2.pkl` at module scope, so an import-based leakage guard
would silently skip on a fresh clone - precisely the checkout where a
freshly-introduced seed collision is most likely to go unnoticed.
"""

import os
import random
import re

import numpy as np
import pytest

from conftest import PROJECT_ROOT, require

# The three named seed families are compared over far more indices than any
# experiment uses (production is NUM_RUNS = 20). 900 is chosen so the assertion
# still has real teeth without being vacuously true: the 42-base and 1000-base
# families do eventually collide, at 958 runs, and
# `test_seed_family_headroom_is_large...` below pins that headroom explicitly
# instead of hiding it.
GENEROUS_RANGE = 900

# The repo-wide scan for scripts that reuse the TRAINING family needs a tighter
# window than GENEROUS_RANGE: several legitimate, deliberately-separate
# evaluation families live below 942 (fairness_analysis uses 800 + run), and a
# 900-wide training window would flag them as leakage when they are nothing of
# the kind. 10x the dataset generator's declared NUM_RUNS is the honest span -
# generous enough that a copy-pasted `42 + i` is caught, tight enough that a
# deliberately-chosen distinct base is not.
TRAINING_SPAN_FACTOR = 10


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_global_rngs():
    """Restore the global `random` / `numpy.random` state around every test.

    These tests reseed the global streams on purpose. Without this, test order
    would leak entropy between tests and a failure here could be caused by a
    neighbouring test rather than by the code under test.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    yield
    random.setstate(py_state)
    np.random.set_state(np_state)


@pytest.fixture(scope='module')
def msb():
    """`04_scheduler/multi_scheduler_benchmark`, or skip if untrained.

    The module unpickles the v2 wait model at import time, so it is an
    artefact-dependent import, not a pure one.
    """
    require('03_models/wait_model_v2.pkl',
            'unpickled at import by multi_scheduler_benchmark')
    import multi_scheduler_benchmark
    return multi_scheduler_benchmark


def _source(rel_path):
    with open(os.path.join(PROJECT_ROOT, rel_path), encoding='utf-8') as fh:
        return fh.read()


def _one_int(pattern, text, what):
    """Extract a single integer literal from source, asserting it is unambiguous.

    Used instead of hard-coding seed bases in the test: if someone edits a seed
    in the source, the test reads the NEW value and re-checks the invariant,
    rather than comparing against a stale copy that has silently drifted.
    """
    found = set(re.findall(pattern, text, re.MULTILINE))
    assert len(found) == 1, (
        f'expected exactly one {what} in the source, found {sorted(found)!r} '
        f'via {pattern!r} - the seed convention changed; update this test '
        f'deliberately after checking the disjointness invariant still holds')
    return int(found.pop())


TRAIN_SRC_REL = os.path.join('02_data', 'generate_improved_dataset.py')
EVAL_SRC_REL = os.path.join('04_scheduler', 'multi_scheduler_benchmark.py')


def seed_bases():
    """The three seed families the project's credibility depends on.

    training   - `02_data/generate_improved_dataset.py`: the workloads that
                 become `improved_wait_dataset.csv` and hence `wait_model_v2`.
    evaluation - `04_scheduler/multi_scheduler_benchmark.py`: the workloads
                 every published scheduler comparison is measured on.
    estimate   - the dedicated RNG that attaches user runtime estimates.
    """
    train_src = _source(TRAIN_SRC_REL)
    eval_src = _source(EVAL_SRC_REL)
    return {
        'training': _one_int(r'random\.seed\((\d+)\s*\+\s*i\)', train_src,
                             'per-run training seed base'),
        'evaluation': _one_int(r'^\s*random\.seed\((\d+)\s*\+\s*run\)', eval_src,
                               'per-run evaluation seed base'),
        'estimate': _one_int(r'^EST_SEED_BASE\s*=\s*(\d+)', eval_src,
                             'EST_SEED_BASE'),
    }


def job_tuples(jobs):
    """Full identity of a generated workload, not just its length.

    A count-only comparison passes even when the arrival/size/runtime draws have
    completely changed, which is exactly the regression these tests exist to
    catch.
    """
    return [(j.job_id, j.arrival_time, j.num_gpus, j.runtime, j.priority_score)
            for j in jobs]


# ---------------------------------------------------------------------------
# (a) Seed determinism
# ---------------------------------------------------------------------------

def test_generate_jobs_is_bit_identical_for_a_repeated_seed(msb):
    """INVARIANT: seed -> workload is a pure function of the seed.

    Re-seeding the global streams to the same value must reproduce the *same*
    jobs, tuple for tuple. This is what lets any reader re-derive a published
    number: `random.seed(1000 + run); generate_jobs()` is the only record of
    what workload run `run` actually was - the workloads themselves are never
    committed. If this breaks, every CSV in 05_results becomes unfalsifiable.
    """
    random.seed(1000)
    np.random.seed(1000)
    first = job_tuples(msb.generate_jobs(12))

    random.seed(1000)
    np.random.seed(1000)
    second = job_tuples(msb.generate_jobs(12))

    assert first == second, (
        'the same seed produced a different workload; no published number in '
        '05_results can be re-derived from its recorded seed')
    # Guard against the degenerate way this could pass: an empty or constant
    # workload compares equal to itself but carries no information.
    assert len(first) == 12
    assert len({t[1:] for t in first}) > 1, 'workload has no variety'


def test_generate_jobs_differs_across_the_per_run_seeds(msb):
    """INVARIANT: consecutive run seeds produce genuinely different workloads.

    The benchmark's error bars come from treating the 20 runs as independent
    samples. If `1000 + run` collapsed to the same workload for every run (an
    off-by-one that reseeded with a constant, say), the paired t-tests and TOST
    intervals in 05_results/schedulers would report a spuriously tiny variance
    and every significance claim would be wrong.
    """
    workloads = {}
    for run in range(5):
        random.seed(1000 + run)
        np.random.seed(1000 + run)
        workloads[run] = job_tuples(msb.generate_jobs(12))

    distinct = {tuple(w) for w in workloads.values()}
    assert len(distinct) == len(workloads), (
        'two per-run evaluation seeds produced identical workloads; the runs '
        'are not independent samples')


def test_generate_jobs_respects_its_declared_ranges(msb):
    """INVARIANT: generated jobs stay inside the documented workload envelope.

    arrival in [0, SIM_TIME//2], gpus in [1, 8], runtime in [5, 20], unique ids,
    arrival-sorted. The wait model is trained on exactly this envelope
    (`02_data/generate_improved_dataset.py` draws from the same ranges), so a
    silent widening here would put the benchmark out of distribution - the model
    would be extrapolating - without any error being raised.
    """
    random.seed(1000)
    np.random.seed(1000)
    jobs = msb.generate_jobs(30)

    arrivals = [j.arrival_time for j in jobs]
    assert arrivals == sorted(arrivals), 'generate_jobs must return arrival-sorted jobs'
    assert all(0 <= j.arrival_time <= msb.SIM_TIME // 2 for j in jobs)
    assert all(1 <= j.num_gpus <= 8 for j in jobs)
    assert all(5 <= j.runtime <= 20 for j in jobs)
    assert {j.job_id for j in jobs} == set(range(30)), 'job ids must be unique'


# ---------------------------------------------------------------------------
# (b) assign_estimates must not perturb the global stream
# ---------------------------------------------------------------------------

def test_assign_estimates_does_not_disturb_the_global_random_stream(msb):
    """INVARIANT: runtime estimates are drawn from a *dedicated* RNG.

    `main()` calls `assign_estimates(jobs, random.Random(EST_SEED_BASE + run))`
    between generating the workload and running the schedulers. If that call
    consumed from the global `random` stream instead, every downstream draw
    would shift and every pre-v3.3 scheduler number in
    05_results/schedulers/*.csv would silently change value - the historical
    results would stop reproducing without a single line of scheduler code
    having been touched. This property is why those numbers stayed bit-identical
    when runtime estimates were introduced.

    Method: measure the global stream's next two draws with nothing in between,
    then again with `assign_estimates` in between. They must agree exactly.
    """
    random.seed(4242)
    jobs = msb.generate_jobs(8)

    random.seed(777)
    baseline = (random.random(), random.random())

    random.seed(777)
    first = random.random()
    msb.assign_estimates(jobs, random.Random(msb.EST_SEED_BASE))
    second = random.random()

    assert (first, second) == baseline, (
        'assign_estimates consumed from the global random stream; every '
        'historical scheduler result would shift')


def test_assign_estimates_does_not_disturb_the_global_numpy_stream(msb):
    """INVARIANT: the same isolation holds for numpy's global stream.

    `main()` seeds `np.random` alongside `random`, and the MLP baseline draws
    from it. Perturbing it here would move the ML schedulers' numbers only,
    which is the most misleading possible failure mode: it looks like a real
    effect of the estimates rather than like contamination.
    """
    random.seed(4242)
    jobs = msb.generate_jobs(8)

    np.random.seed(31337)
    before = np.random.get_state()
    msb.assign_estimates(jobs, random.Random(msb.EST_SEED_BASE))
    after = np.random.get_state()

    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_assign_estimates_is_a_pure_function_of_its_rng_seed(msb):
    """INVARIANT: same Random seed -> same estimates; different seed -> different.

    The estimate-quality sweep (`04_scheduler/estimate_sensitivity.py`) reuses
    `EST_SEED_BASE + run` across every over-factor C so that the sweep varies
    estimate QUALITY and not the noise realisation. That coupling argument is
    only valid if the draw is reproducible from the seed alone.
    """
    random.seed(4242)
    jobs = msb.generate_jobs(10)

    msb.assign_estimates(jobs, random.Random(msb.EST_SEED_BASE))
    first = [j.est_runtime for j in jobs]
    msb.assign_estimates(jobs, random.Random(msb.EST_SEED_BASE))
    second = [j.est_runtime for j in jobs]
    msb.assign_estimates(jobs, random.Random(msb.EST_SEED_BASE + 1))
    other = [j.est_runtime for j in jobs]

    assert first == second, 'estimates are not reproducible from the rng seed'
    assert first != other, 'estimates ignore the rng seed'


def test_estimates_stay_inside_the_f_model_envelope(msb):
    """INVARIANT: est = runtime * (1 + (C-1)*u), u~U(0,1)  =>  est in [rt, C*rt].

    The f-model (Mu'alem & Feitelson 2001) says users OVER-estimate. An estimate
    BELOW the true runtime would let EASY backfill plan a reservation it cannot
    honour, breaking the no-delay guarantee the whole backfill baseline rests
    on. The upper bound pins the documented C = 5 regime.
    """
    random.seed(4242)
    jobs = msb.generate_jobs(20)
    msb.assign_estimates(jobs, random.Random(msb.EST_SEED_BASE))

    c = msb.EST_OVER_FACTOR
    for j in jobs:
        assert j.runtime <= j.est_runtime <= j.runtime * c + 1e-9, j


# ---------------------------------------------------------------------------
# (c) The leakage guard
# ---------------------------------------------------------------------------

def test_evaluation_seeds_are_disjoint_from_training_seeds():
    """INVARIANT: benchmark workloads are never training workloads. LEAKAGE GUARD.

    `02_data/generate_improved_dataset.py` builds the wait-model training set
    from workloads seeded `42 + i`. `04_scheduler/multi_scheduler_benchmark.py`
    measures PROACTIVE against every baseline on workloads seeded `1000 + run`.
    Both call the SAME generator shape with the SAME parameters (110 jobs,
    arrival U[0,150], gpus U[1,8], runtime U[5,20]), so an overlap between the
    two seed families is not a near-miss - it reproduces byte-for-byte identical
    workloads. The benchmark would then be scoring the XGBoost model on the
    exact queues it memorised, and the reported wait-time reduction would be a
    measurement of overfitting rather than of scheduling.

    This repository has shipped that bug before. The fix was to move evaluation
    off the `42 + i` family; this test is the thing that keeps it fixed.
    """
    bases = seed_bases()
    families = {name: set(range(base, base + GENEROUS_RANGE))
                for name, base in bases.items()}

    for a, b in (('training', 'evaluation'),
                 ('training', 'estimate'),
                 ('evaluation', 'estimate')):
        overlap = families[a] & families[b]
        assert not overlap, (
            f'{a} seeds and {b} seeds overlap at {sorted(overlap)[:5]}... '
            f'(bases {bases[a]} and {bases[b]}). An overlap between training '
            f'and evaluation means the benchmark is evaluating the model on '
            f'the model\'s own training workloads.')


def test_seed_family_headroom_is_large_relative_to_the_run_count():
    """INVARIANT: the seed families have room to grow before they collide.

    Disjointness at today's NUM_RUNS is necessary but not sufficient: someone
    raising NUM_RUNS to get tighter confidence intervals must not walk the
    evaluation family into the training family. `42 + i` and `1000 + run` first
    collide at run 958, ~48x the production run count, so the margin is real
    rather than incidental. Asserting the RATIO (not the literal 958) means the
    test still protects the property if the bases are ever re-chosen.
    """
    bases = seed_bases()
    num_runs = _one_int(r'^NUM_RUNS\s*=\s*(\d+)', _source(EVAL_SRC_REL), 'NUM_RUNS')

    headroom = abs(bases['evaluation'] - bases['training'])
    assert headroom >= 10 * num_runs, (
        f'only {headroom} seeds separate the training base {bases["training"]} '
        f'from the evaluation base {bases["evaluation"]}, with NUM_RUNS = '
        f'{num_runs}; raising NUM_RUNS a little would cause leakage')
    assert abs(bases['estimate'] - bases['evaluation']) >= 10 * num_runs


def test_no_new_scheduler_script_evaluates_on_the_training_seed_family():
    """INVARIANT: no script that CONSUMES the trained model evaluates on 42 + i.

    A new experiment added by copy-pasting an older script is the realistic way
    leakage returns. This sweeps every per-run `random.seed(<base> + <var>)` in
    04_scheduler and asserts its base lies outside the training family's range.

    NOTE - THIS TEST DOCUMENTS A REAL, PRE-EXISTING DEFECT rather than passing
    cleanly. `04_scheduler/proactive_scheduler.py` and
    `04_scheduler/proactive_Schedule_v2.py` both load `wait_model_v2.pkl` and
    then evaluate it on `random.seed(42 + i)` workloads for i in range(10) - the
    first ten of the dataset generator's own twenty training seeds, drawn by a
    character-identical `generate_jobs(110, 300)`. Those two scripts therefore
    report the model's performance on its own training data. Neither is in
    run_all_experiments.sh and no committed CSV under 05_results comes from
    them, so the published numbers are unaffected - which is why this test pins
    the CURRENT behaviour with an explicit known-bad list instead of failing.
    Any NEW script joining that list breaks the test, which is the point.
    Repairing the two scripts means moving them to the 1000+ family and deleting
    them from `known_leaky` here.
    """
    bases = seed_bases()
    train_runs = _one_int(r'NUM_RUNS\s*=\s*(\d+)', _source(TRAIN_SRC_REL),
                          'dataset-generator NUM_RUNS')
    span = TRAINING_SPAN_FACTOR * train_runs
    training = set(range(bases['training'], bases['training'] + span))

    # Known-bad, quarantined: not run by the pipeline, no committed artefact.
    known_leaky = {'proactive_scheduler.py', 'proactive_Schedule_v2.py'}

    sched_dir = os.path.join(PROJECT_ROOT, '04_scheduler')
    pattern = re.compile(r'random\.seed\((\d+)\s*\+\s*\w+\)')
    offenders = {}
    for name in sorted(os.listdir(sched_dir)):
        if not name.endswith('.py'):
            continue
        with open(os.path.join(sched_dir, name), encoding='utf-8') as fh:
            for base in {int(b) for b in pattern.findall(fh.read())}:
                if base in training:
                    offenders.setdefault(name, set()).add(base)

    assert set(offenders) == known_leaky, (
        f'the seed-leakage set changed: {sorted(offenders)} vs the known-bad '
        f'{sorted(known_leaky)}. A script under 04_scheduler seeds its '
        f'workloads from the training family ({bases["training"]} + i), so it '
        f'evaluates the wait model on the workloads the model was trained on.')


# ---------------------------------------------------------------------------
# (d) vizstyle: colour follows the entity
# ---------------------------------------------------------------------------

def test_color_of_is_stable_regardless_of_call_order():
    """INVARIANT: colour is a function of the POLICY, not of its rank in a chart.

    This is the exact defect vizstyle was written to fix: figures used to take
    colours from matplotlib's cycle, so a scheduler was blue in a panel sorted
    by wait and green in the adjacent panel sorted by slowdown, and a reader
    could not follow one policy across a figure. Every panel of
    scheduler_comparison.png sorts by a different metric, so the mapping must
    survive being called in any order and interleaved with other policies.
    """
    import vizstyle

    policies = ['PROACTIVE', 'SMALLEST', 'FIFO', 'SRPT', 'NN', 'CONS_BF']
    for mode in ('light', 'dark'):
        forward = {p: vizstyle.color_of(p, mode) for p in policies}
        # same lookups in reverse, interleaved with unrelated ones
        backward = {}
        for p in reversed(policies):
            vizstyle.color_of('HRRN', mode)
            backward[p] = vizstyle.color_of(p, mode)
            vizstyle.color_of('PRIORITY', 'light' if mode == 'dark' else 'dark')
        assert forward == backward

        # The encoding must actually distinguish the three roles, otherwise a
        # stable-but-constant mapping would satisfy the check above.
        assert vizstyle.color_of('PROACTIVE', mode) != vizstyle.color_of('SMALLEST', mode)
        assert vizstyle.color_of('SMALLEST', mode) != vizstyle.color_of('FIFO', mode)


def test_color_of_is_case_insensitive_and_falls_back_to_baseline():
    """INVARIANT: the same policy spelled differently gets the same colour.

    Results CSVs store scheduler names upper-cased; the scheduler loops inside
    the benchmarks use lower-cased keys ('proactive'). A case-sensitive lookup
    would silently demote the ML policy to the recessive baseline grey in some
    figures and not others.
    """
    import vizstyle

    assert vizstyle.color_of('proactive') == vizstyle.color_of('PROACTIVE')
    assert vizstyle.color_of('Smallest') == vizstyle.color_of('SMALLEST')
    # Unknown policies recede to the baseline colour rather than raising - but
    # see the POLICY_ROLE coverage test below: that fallback must never be
    # reached by a policy that actually appears in a results CSV.
    assert vizstyle.color_of('NOT_A_POLICY') == vizstyle.color_of('FIFO')


def test_every_policy_role_maps_to_a_defined_role():
    """INVARIANT: POLICY_ROLE values are keys of ROLE_KEY (and of ROLE_LABEL).

    `color_of` indexes `ROLE_KEY[role_of(policy)]` with no `.get` fallback, so a
    typo'd role in POLICY_ROLE ('control ' with a trailing space, or 'ML')
    raises KeyError deep inside a plotting run - after a long benchmark has
    already executed and just before it would have written its figures.
    Checking the table directly turns that into an instant failure.
    """
    import vizstyle

    bad = {p: r for p, r in vizstyle.POLICY_ROLE.items() if r not in vizstyle.ROLE_KEY}
    assert not bad, f'POLICY_ROLE entries with unknown roles: {bad}'

    missing_label = {r for r in vizstyle.POLICY_ROLE.values()
                     if r not in vizstyle.ROLE_LABEL}
    assert not missing_label, f'roles with no legend label: {missing_label}'

    # Every ROLE_KEY must resolve in BOTH modes, or `color_of` fails in dark
    # only - a defect that surfaces solely in the -dark.png half of each pair.
    for mode in ('light', 'dark'):
        for key in vizstyle.ROLE_KEY.values():
            assert key in vizstyle.PALETTE[mode], (key, mode)


@pytest.mark.artefact
@pytest.mark.parametrize('csv_rel', [
    os.path.join('05_results', 'schedulers', 'multi_scheduler_benchmark.csv'),
    os.path.join('05_results', 'trace_schedulers', 'trace_scheduler_summary.csv'),
])
def test_every_scheduler_in_the_results_csvs_has_a_policy_role(csv_rel):
    """INVARIANT: no policy in a published table renders in the default colour.

    `role_of` returns 'baseline' for anything it does not know, so adding a new
    scheduler to a benchmark without adding it to POLICY_ROLE does not raise -
    it quietly draws the new policy in the recessive grey reserved for classical
    baselines. If that new policy is an ML variant, the figure then makes the
    opposite claim to the data underneath it. The two CSVs checked here feed the
    manuscript's synthetic and trace-driven scheduler-comparison figures.
    """
    import csv as csvmod

    import vizstyle

    path = require(csv_rel, 'scheduler results table')
    with open(path, newline='', encoding='utf-8') as fh:
        names = {row['scheduler'].strip() for row in csvmod.DictReader(fh)
                 if row.get('scheduler')}

    assert names, f'{csv_rel} has no scheduler column values'
    unmapped = sorted(n for n in names if n.upper() not in vizstyle.POLICY_ROLE)
    assert not unmapped, (
        f'{csv_rel} contains schedulers with no vizstyle.POLICY_ROLE entry: '
        f'{unmapped} - they will silently render in the baseline grey')

    # Labels too: an unmapped policy falls back to its raw CSV key, so the
    # figure would show 'CONS_BF_USEREST' instead of 'Conservative backfill'.
    unlabelled = sorted(n for n in names if n.upper() not in vizstyle.POLICY_LABEL)
    assert not unlabelled, (
        f'{csv_rel} contains schedulers with no vizstyle.POLICY_LABEL entry: '
        f'{unlabelled} - the figure will show the raw key as a tick label')

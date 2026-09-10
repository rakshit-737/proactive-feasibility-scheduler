"""The wait-budget sweep's STARVATION DEFINITION, pinned so it cannot drift back.

WHAT THIS GUARDS
----------------
`04_scheduler/fairness_budget_sweep.py` once counted a job as starved when its wait
exceeded 3x the RUN'S MEAN WAIT -- a distribution-relative rule found nowhere else in
the repository. Every other producer (`04_scheduler/fairness_analysis.py`,
`phases_22_30/phase_27_fairness` SLA-2) uses the per-job rule: starved when the wait
exceeds 3x the job's OWN runtime. The two are not interchangeable. The per-job rule
has a fixed yardstick per job, so short jobs starve sooner; the distribution-relative
one moves with the workload, so a policy that makes *everyone* wait longer can report
*less* starvation. The sweep was corrected to the per-job rule, but until this module
existed no test anywhere imported that file, so the correction could be reverted
silently -- and, because the correction also INVERTS the column's trend across the
sweep (see the module header), a reverted file would look more like the published
numbers, not less.

Four invariants are protected:

  1. THE RULE IS PER-JOB. One hand-built case separates the two definitions with a
     single number: two jobs whose waits are equal (so both sit exactly at the mean)
     but whose runtimes differ by 8x. The per-job rule counts one; the old rule counts
     none. Anything that answers 0 is running the old definition.
  2. THE BOUNDARY IS STRICT. `wait == 3 * runtime` is not starvation; the comparison
     is `>`, not `>=`, matching fairness_analysis.py exactly.
  3. THE WAIT METRICS STAY DECOUPLED. mean/p95/max/gini are functions of the waits
     alone. Runtimes entered wait_metrics() only when starvation became per-job, and
     nothing else may ever start reading them.
  4. THE ARTEFACT DECLARES ITS DEFINITION. The multiple is a named constant, and the
     CSV column spells the rule out, so a stale committed sweep and a correctly
     regenerated one are distinguishable by inspection.

No simulation, no benchmark, no artefact written: every test here calls a pure helper
with a few hand-built tuples. The module under test loads the trained wait model at
IMPORT time, so the import is fixtured behind conftest.require() -- the same skip the
rest of this suite gives artefact-dependent tests on a fresh clone.
"""

import numpy as np
import pytest

# fairness_budget_sweep unpickles this at module import time (the sweep scores queues
# with the trained model), so it must be require()d BEFORE the import, not inside a
# test body.
_MODEL_ARTEFACT = '03_models/wait_model_v2.pkl'


@pytest.fixture(scope='module')
def fbs():
    """The module under test, or a clean skip when the model has not been trained."""
    from conftest import require
    require(_MODEL_ARTEFACT, 'trained wait model, loaded by the sweep at import time')
    import fairness_budget_sweep
    return fairness_budget_sweep


def _old_rule_count(completed):
    """The SUPERSEDED definition, reimplemented here as a control: starved when the
    wait exceeds 3x the run's mean wait. Tests below assert that the live code and
    this disagree on the discriminating case, which is what makes them able to tell
    the two definitions apart at all."""
    mean_wait = float(np.mean([wait for wait, _ in completed]))
    return sum(1 for wait, _ in completed if wait > 3 * mean_wait)


# ─────────────────────────────────────────────────────────────────────────────
# (1) The rule is per-job, not distribution-relative
# ─────────────────────────────────────────────────────────────────────────────

def test_starvation_is_measured_against_each_jobs_own_runtime(fbs):
    """INVARIANT: a job is starved when its wait exceeds 3x ITS OWN runtime.

    The case is chosen so the two candidate definitions cannot both be right:

        job A: waited 20 ticks for a  5-tick job -> 4.0x its own runtime -> STARVED
        job B: waited 20 ticks for a 40-tick job -> 0.5x its own runtime -> fine

    Both waits are 20, so both sit exactly ON the run's mean wait; under the old
    distribution-relative rule (wait > 3x the mean = 60) NEITHER job counts and the
    answer is 0. Under the per-job rule the answer is 1. There is no third value, so
    this single assertion identifies which definition the module is running.
    """
    completed = [(20, 5), (20, 40)]

    assert fbs.wait_metrics(completed)['starvation'] == 1, (
        'expected exactly the 4x-its-own-runtime job to count as starved; a 0 here '
        'means the distribution-relative rule (wait > 3x the mean wait) is back')

    # The control: the superseded rule really does answer differently on this input,
    # so the assertion above is not passing for some incidental reason.
    assert _old_rule_count(completed) == 0


def test_which_job_starves_depends_only_on_its_own_runtime(fbs):
    """INVARIANT (the same fact, from the other side): holding a job's wait fixed,
    shrinking ITS runtime is what makes it starve -- no property of the rest of the
    distribution is consulted.

    Under the old rule, adding an enormous wait to the run would lift the mean and
    make the short job STOP counting. Here the short job keeps counting no matter what
    company it is in, which is precisely the per-job semantics.
    """
    short_job_alone = fbs.wait_metrics([(20, 5)])['starvation']
    # a colossal outlier that would drag the old rule's threshold far above 20
    with_outlier = fbs.wait_metrics([(20, 5), (4000, 1)])['starvation']

    assert short_job_alone == 1
    # the short job still starves; the outlier (4000 > 3) starves too
    assert with_outlier == 2, (
        'a job stopped/started starving because of OTHER jobs in the run: the '
        'threshold is no longer per-job')


def test_the_starvation_multiple_is_load_bearing(fbs, monkeypatch):
    """INVARIANT: STARVATION_RUNTIME_MULTIPLE is the number the comparison actually
    uses -- it is not a decorative constant sitting next to a hard-coded 3.

    Without this, someone could keep the well-named constant and still inline the
    threshold, and finding 1's rename would be cosmetic.
    """
    completed = [(20, 5), (20, 40)]
    assert fbs.wait_metrics(completed)['starvation'] == 1

    # 0.4x: thresholds become 2 and 16, and the wait of 20 clears both
    monkeypatch.setattr(fbs, 'STARVATION_RUNTIME_MULTIPLE', 0.4)
    assert fbs.wait_metrics(completed)['starvation'] == 2

    # 10x: thresholds become 50 and 400, and nothing clears either
    monkeypatch.setattr(fbs, 'STARVATION_RUNTIME_MULTIPLE', 10.0)
    assert fbs.wait_metrics(completed)['starvation'] == 0


# ─────────────────────────────────────────────────────────────────────────────
# (2) The boundary is strict
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('wait, expected', [
    (29.999, 0),      # just under 3x
    (30, 0),          # EXACTLY 3x -- not starved: the comparison is >, not >=
    (30.001, 1),      # just over 3x
    (31, 1),
])
def test_exactly_three_times_own_runtime_is_not_starved(fbs, wait, expected):
    """INVARIANT: the threshold is exclusive. `wait > 3 * runtime`, so a job that
    waited exactly 3x its runtime is NOT counted.

    fairness_analysis.py uses `>` too, so flipping this to `>=` here would silently
    reintroduce the divergence between producers that the round-one fix removed --
    and off-by-one at the boundary is the cheapest way to do that.
    """
    assert fbs.wait_metrics([(wait, 10)])['starvation'] == expected


# ─────────────────────────────────────────────────────────────────────────────
# (3) The wait-distribution metrics never read the runtimes
# ─────────────────────────────────────────────────────────────────────────────

WAIT_ONLY_METRICS = ('mean_wait', 'p95_wait', 'max_wait', 'gini')


def test_wait_distribution_metrics_are_unchanged_by_the_runtimes(fbs):
    """INVARIANT: mean_wait, p95_wait, max_wait and gini are functions of the WAITS
    alone. The runtimes entered wait_metrics() only to make starvation per-job, and
    no future edit may quietly couple them to the rest of the row.

    Same waits, two very different runtime vectors: the four distribution metrics must
    come back bit-identical. The starvation count must NOT -- otherwise the runtimes
    are being ignored altogether and this test would pass vacuously.
    """
    waits = [1, 4, 9, 16, 40]
    long_runtimes = [(w, 100) for w in waits]     # nothing waits > 300
    short_runtimes = [(w, 1) for w in waits]      # everything over 3 starves

    a = fbs.wait_metrics(long_runtimes)
    b = fbs.wait_metrics(short_runtimes)

    for key in WAIT_ONLY_METRICS:
        assert a[key] == b[key], (
            f'{key} changed when only the runtimes changed: a wait-distribution '
            f'metric has been coupled to job runtimes')

    # non-vacuity: the runtimes did reach the one metric that is allowed to see them
    assert a['starvation'] == 0
    assert b['starvation'] == 4          # waits 4, 9, 16, 40 all exceed 3 * 1
    assert set(a) == set(WAIT_ONLY_METRICS) | {'starvation'}, (
        'wait_metrics grew or lost a key; decide deliberately whether the new one '
        'may read runtimes')


# ─────────────────────────────────────────────────────────────────────────────
# (4) The definition travels with the code and with the data
# ─────────────────────────────────────────────────────────────────────────────

def test_the_constant_is_named_for_what_it_measures(fbs):
    """INVARIANT: the multiple is exposed as STARVATION_RUNTIME_MULTIPLE == 3.0, and
    the old, definition-free name STARVATION_FACTOR is gone.

    The old name described no yardstick, which is how a threshold measured against the
    mean wait and one measured against a job's runtime came to share a spelling. The
    rename is part of the fix, so reverting it must fail a test.
    """
    assert fbs.STARVATION_RUNTIME_MULTIPLE == 3.0
    assert not hasattr(fbs, 'STARVATION_FACTOR'), (
        'STARVATION_FACTOR is back: the constant no longer says what the threshold '
        'is a multiple OF')


def test_csv_column_name_states_the_definition(fbs):
    """INVARIANT: the sweep's starvation column is 'starved_jobs_wait_gt_3x_own_runtime',
    not a bare 'starvation'.

    Both definitions produce a plausible column of counts, and the reviewer measured
    that their TRENDS across the budget sweep are opposite (old: rises with looser
    budgets; new: falls). A bare header therefore makes a stale committed CSV and a
    correctly regenerated one indistinguishable by inspection, and invites reading the
    correct regeneration as a bug. The header is the provenance.
    """
    assert fbs.STARVATION_COLUMN == 'starved_jobs_wait_gt_3x_own_runtime'
    # the name must keep encoding the multiple it was built from
    assert f'{int(fbs.STARVATION_RUNTIME_MULTIPLE)}x_own_runtime' in fbs.STARVATION_COLUMN
    assert fbs.STARVATION_RUNTIME_MULTIPLE == int(fbs.STARVATION_RUNTIME_MULTIPLE), (
        'a fractional multiple no longer matches the "3x" spelled in the column name')


def test_summarized_row_writes_the_self_describing_column(fbs):
    """INVARIANT: the row that becomes budget_sweep.csv carries the starvation count
    under STARVATION_COLUMN and under no other name.

    summarize_runs() is the only writer of that row, so asserting its schema here pins
    the CSV header without running the 20-run sweep or touching a tracked artefact.
    The improvement-vs-FIFO arithmetic is checked alongside it so that extracting this
    helper out of main() cannot have changed a published number.
    """
    runs = [
        {'mean_wait': 10.0, 'p95_wait': 25.0, 'max_wait': 30.0, 'gini': 0.4,
         'starvation': 4},
        {'mean_wait': 20.0, 'p95_wait': 35.0, 'max_wait': 50.0, 'gini': 0.6,
         'starvation': 6},
    ]
    fifo_mean_waits = np.array([20.0, 40.0])       # each run's FIFO reference

    row = fbs.summarize_runs(60, runs, fifo_mean_waits)

    assert fbs.STARVATION_COLUMN in row
    assert 'starvation' not in row, (
        "budget_sweep.csv is writing a bare 'starvation' column again: the meaning of "
        'that header changed, so it can no longer identify the number under it')
    assert row[fbs.STARVATION_COLUMN] == pytest.approx(5.0)

    # the rest of the schema, and the paired-improvement arithmetic, unchanged
    assert row['budget'] == '60'
    assert row['mean_wait'] == pytest.approx(15.0)
    assert row['max_wait'] == pytest.approx(40.0)
    assert row['gini'] == pytest.approx(0.5)
    # both runs improved on FIFO by exactly half their reference wait
    assert row['mean_improvement_vs_fifo_pct'] == pytest.approx(50.0)
    assert fbs.budget_label(None) == 'None'
    assert list(row) == ['budget', 'mean_wait', 'mean_wait_std', 'p95_wait', 'max_wait',
                         'gini', fbs.STARVATION_COLUMN, 'mean_improvement_vs_fifo_pct']

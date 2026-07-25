"""Queue-ordering policy helpers: SJF, HRRN, smallest-size, priority.

These four modules are the classical baselines the whole benchmark is measured
against. They are pure functions -- queue in, reordered queue out -- which makes
them cheap to test exhaustively and makes any defect in them silently
contaminate every number in the paper.

Three families of invariant are protected here:

  1. PERMUTATION. An ordering helper may reorder, never add, drop or duplicate.
     A dropped job disappears from the completed-job list, which shortens the
     wait-time sample the results are averaged over -- the benchmark would still
     produce a plausible-looking number, just a wrong one. This is the single
     worst failure mode of a scheduler helper, so it is checked generically over
     every ordering function rather than per policy.

  2. THE KEY EACH POLICY CLAIMS TO USE. sjf.order_queue is a perfect-information
     oracle (true runtime); sjf.order_queue_estimated is the deployable policy
     (user estimate). If the two ever collapsed onto the same key, the paper's
     oracle-vs-deployable gap -- one of its headline results -- would silently
     become zero. Same for the HRRN pair.

  3. TOTAL, DETERMINISTIC TIE-BREAKING. Every helper breaks ties by
     (arrival_time, job_id), and job ids are unique, so the induced order is a
     total order: it cannot depend on the incoming list order. If it could, the
     benchmark's results would depend on dict/list iteration order and the
     "bit-identical across runs" reproducibility claim would be false.

No artefacts, no simulation, no I/O: this module is pure arithmetic and runs in
milliseconds.
"""

import random

import pytest

import hrrn_scheduler
import priority_scheduler
import size_scheduler
import sjf_scheduler

# Every queue-ordering helper under test, adapted to one common signature
# (queue, current_time). sjf.* and size.* take current_time as an ignored
# optional argument; hrrn.* and priority.* require it. Keeping them in one
# table is what lets the generic invariants below be stated once.
ORDERINGS = [
    ('sjf.order_queue', sjf_scheduler.order_queue),
    ('sjf.order_queue_estimated', sjf_scheduler.order_queue_estimated),
    ('hrrn.order_queue', hrrn_scheduler.order_queue),
    ('hrrn.order_queue_estimated', hrrn_scheduler.order_queue_estimated),
    ('size.order_queue', size_scheduler.order_queue),
    ('priority.order_queue', priority_scheduler.order_queue),
]

ORDERING_PARAMS = [pytest.param(fn, id=name) for name, fn in ORDERINGS]

# A dispatch instant well after every arrival used below, so `waited` is
# positive for every job and the aging terms are actually exercised.
NOW = 50


def _varied_queue(SimpleJob, n=12, seed=20260721):
    """A small heterogeneous queue with deliberate ties in every sort key.

    Values are drawn from short lists with repeats so that runtime, est_runtime,
    num_gpus, priority_score AND arrival_time all collide between some pair of
    jobs -- otherwise a broken tie-break would never be reached.
    """
    rng = random.Random(seed)
    return [
        SimpleJob(
            job_id=i,
            arrival_time=rng.choice([0, 0, 3, 3, 7, 12]),
            num_gpus=rng.choice([1, 1, 2, 4, 4, 8]),
            runtime=rng.choice([1, 2, 2, 5, 5, 5, 13]),
            est_runtime=rng.choice([1, 2, 5, 5, 20]),
            priority_score=rng.choice([1, 1, 3, 7]),
        )
        for i in range(n)
    ]


def _snapshot(jobs):
    return [(j.job_id, j.arrival_time, j.num_gpus, j.runtime, j.est_runtime,
             j.priority_score) for j in jobs]


def _ids(jobs):
    return [j.job_id for j in jobs]


# --------------------------------------------------------------------------
# (a) Every ordering function returns a permutation of its input.
# --------------------------------------------------------------------------

@pytest.mark.parametrize('order', ORDERING_PARAMS)
@pytest.mark.parametrize('now', [0, 50, 5000])
def test_ordering_returns_a_permutation(order, now, make_job):
    """INVARIANT: ordering is a permutation -- no job invented, dropped or cloned.

    Checked as three separate facts because they fail differently: same length
    (nothing dropped/added), same *set* of ids with no duplicates (nothing
    cloned), and same object identities (the returned jobs are the very objects
    the dispatch loop will mutate -- returning copies would make start_time /
    end_time assignments invisible to the caller and silently zero the results).
    """
    queue = _varied_queue(make_job)
    out = order(queue, now)

    assert len(out) == len(queue)
    assert len(set(_ids(out))) == len(out), 'a job id appears twice in the output'
    assert sorted(_ids(out)) == sorted(_ids(queue))
    # Identity, not equality: the dispatch loop relies on getting the same objects.
    assert {id(j) for j in out} == {id(j) for j in queue}


@pytest.mark.parametrize('order', ORDERING_PARAMS)
def test_ordering_does_not_mutate_its_input(order, make_job):
    """INVARIANT: ordering is side-effect free.

    The benchmark calls these helpers once per tick on the live queue list and
    then keeps using that list. An in-place sort (or any attribute write) would
    make a policy's behaviour depend on how many times it had been called --
    which is exactly the kind of hidden state that makes results unreproducible.
    """
    queue = _varied_queue(make_job)
    before_order = _ids(queue)
    before_attrs = _snapshot(queue)

    out = order(queue, NOW)

    assert out is not queue, 'helper returned the caller list itself, not a new list'
    assert _ids(queue) == before_order, 'input list was reordered in place'
    assert _snapshot(queue) == before_attrs, 'helper mutated job attributes'


@pytest.mark.parametrize('order', ORDERING_PARAMS)
def test_ordering_handles_degenerate_queues(order, make_job):
    """INVARIANT: empty and singleton queues are legal, not edge-case crashes.

    Both occur constantly in a real run (idle ticks, last job in the trace); a
    helper that raised on them would abort a benchmark mid-flight.
    """
    assert order([], NOW) == []
    solo = make_job(job_id=0, arrival_time=1, num_gpus=2, runtime=3)
    assert _ids(order([solo], NOW)) == [0]


# --------------------------------------------------------------------------
# (b) SJF: the oracle sorts by true runtime, the deployable policy by estimate.
# --------------------------------------------------------------------------

def test_sjf_true_and_estimated_use_different_keys(make_job):
    """INVARIANT: order_queue reads .runtime, order_queue_estimated reads .est_runtime.

    This is the entire difference between the SJF *oracle* (perfect information,
    an upper bound nobody can deploy) and SJF-EST (what a real site could run).
    The estimates here are deliberately anti-correlated with the truth, so the
    two helpers must return exactly reversed orders. If a refactor made either
    helper read the other attribute, both rows of the results table would move
    to the same value and the oracle gap would vanish unnoticed.
    """
    # runtime descending, est_runtime ascending -> the two keys disagree maximally.
    jobs = [
        make_job(job_id=1, arrival_time=0, runtime=30, est_runtime=1),
        make_job(job_id=2, arrival_time=0, runtime=20, est_runtime=2),
        make_job(job_id=3, arrival_time=0, runtime=10, est_runtime=3),
    ]

    by_true = sjf_scheduler.order_queue(jobs, NOW)
    by_est = sjf_scheduler.order_queue_estimated(jobs, NOW)

    assert _ids(by_true) == [3, 2, 1], 'oracle SJF must sort by TRUE runtime'
    assert _ids(by_est) == [1, 2, 3], 'SJF-EST must sort by est_runtime'
    assert _ids(by_true) != _ids(by_est)
    # State the keys as monotonicity too, so the test still means something if
    # the fixture values are ever changed.
    assert [j.runtime for j in by_true] == sorted(j.runtime for j in jobs)
    assert [j.est_runtime for j in by_est] == sorted(j.est_runtime for j in jobs)


def test_sjf_shorter_job_first_regardless_of_arrival(make_job):
    """INVARIANT: SJF is not FCFS -- runtime dominates arrival order.

    A late-arriving short job must overtake an early long one; that overtaking
    is the whole mechanism by which SJF beats FIFO on mean wait.
    """
    long_early = make_job(job_id=1, arrival_time=0, runtime=100)
    short_late = make_job(job_id=2, arrival_time=40, runtime=1)
    assert _ids(sjf_scheduler.order_queue([long_early, short_late], NOW)) == [2, 1]


# --------------------------------------------------------------------------
# (c) HRRN: response ratio = (wait + service) / service.
# --------------------------------------------------------------------------

@pytest.mark.parametrize('order,service_attr', [
    pytest.param(hrrn_scheduler.order_queue, 'runtime', id='hrrn.order_queue'),
    pytest.param(hrrn_scheduler.order_queue_estimated, 'est_runtime',
                 id='hrrn.order_queue_estimated'),
])
def test_hrrn_equal_service_longest_wait_first(order, service_attr, make_job):
    """INVARIANT: with equal service times, HRRN dispatches the longest waiter.

    This is HRRN's anti-starvation half: ratio = 1 + wait/service, so among
    equals the queue degenerates to FCFS and nobody is overtaken forever. If it
    broke, HRRN's headline claim (starvation strongly discouraged) would be
    false while its mean-wait numbers still looked fine.
    """
    jobs = [make_job(job_id=i, arrival_time=arr, runtime=10, est_runtime=10)
            for i, arr in [(1, 0), (2, 5), (3, 10)]]
    out = order(list(reversed(jobs)), 100)  # feed it worst-case ordered
    assert _ids(out) == [1, 2, 3], 'earliest arrival = longest wait must go first'
    assert [j.arrival_time for j in out] == sorted(j.arrival_time for j in jobs)


@pytest.mark.parametrize('order,service_attr', [
    pytest.param(hrrn_scheduler.order_queue, 'runtime', id='hrrn.order_queue'),
    pytest.param(hrrn_scheduler.order_queue_estimated, 'est_runtime',
                 id='hrrn.order_queue_estimated'),
])
def test_hrrn_equal_wait_shortest_service_first(order, service_attr, make_job):
    """INVARIANT: with equal waits, HRRN dispatches the shortest service time.

    This is HRRN's SJF-like half: for a fixed wait, ratio 1 + wait/service is
    decreasing in service. Losing it would turn HRRN into plain FCFS, which is
    the baseline it is supposed to beat.
    """
    jobs = []
    for i, service in [(1, 20), (2, 10), (3, 5)]:
        kwargs = {'runtime': service} if service_attr == 'runtime' else {
            'runtime': 7, 'est_runtime': service}
        jobs.append(make_job(job_id=i, arrival_time=0, **kwargs))

    out = order(jobs, 100)
    assert _ids(out) == [3, 2, 1]
    assert [getattr(j, service_attr) for j in out] == sorted(
        getattr(j, service_attr) for j in jobs)


def test_hrrn_estimated_uses_est_runtime_not_runtime(make_job):
    """INVARIANT: the deployable HRRN variant reads est_runtime, the oracle reads runtime.

    Same stake as the SJF pair: HRRN and HRRN-EST are reported as separate rows,
    and the trace-driven benchmark feeds est_runtime from the SWF requested-time
    field. Anti-correlated fixtures force the two helpers to disagree, so a
    copy-paste slip between the two lambdas in hrrn_scheduler._order is caught.
    """
    jobs = [
        make_job(job_id=1, arrival_time=0, runtime=1, est_runtime=100),
        make_job(job_id=2, arrival_time=0, runtime=100, est_runtime=1),
    ]
    # equal waits -> shorter service wins, and "service" is the only difference
    assert _ids(hrrn_scheduler.order_queue(jobs, 100)) == [1, 2]
    assert _ids(hrrn_scheduler.order_queue_estimated(jobs, 100)) == [2, 1]


def test_hrrn_order_is_a_live_function_of_current_time(make_job):
    """INVARIANT: HRRN aging is real -- the order genuinely changes as time passes.

    Two jobs, fixed attributes, only `current_time` differs between the two
    calls, and the order flips. That is the property that distinguishes an aging
    policy from a static key, and (see the priority test below) it is NOT shared
    by priority_scheduler.
    """
    long_job = make_job(job_id=1, arrival_time=0, runtime=100)
    short_job = make_job(job_id=2, arrival_time=99, runtime=1)
    queue = [long_job, short_job]

    # t=99: ratios 199/100 = 1.99 vs 1/1 = 1.0  -> the aged long job leads.
    assert _ids(hrrn_scheduler.order_queue(queue, 99)) == [1, 2]
    # t=101: ratios 201/100 = 2.01 vs 3/1 = 3.0 -> the short job has overtaken.
    assert _ids(hrrn_scheduler.order_queue(queue, 101)) == [2, 1]


# --------------------------------------------------------------------------
# (d) size_scheduler: ascending requested size.
# --------------------------------------------------------------------------

def test_size_orders_ascending_by_num_gpus(make_job):
    """INVARIANT: smallest requested size first, ignoring runtime and priority.

    size_scheduler is the CONTROL for the ranking-degeneracy result: the claim
    is that the learned wait-model's ranking is a permutation of the requested-
    size ranking. That control is only meaningful if this policy really is
    "order by num_gpus and nothing else", so the fixture deliberately makes
    runtime and priority_score point the opposite way.
    """
    jobs = [
        make_job(job_id=1, arrival_time=0, num_gpus=8, runtime=1, priority_score=1),
        make_job(job_id=2, arrival_time=0, num_gpus=1, runtime=100, priority_score=9),
        make_job(job_id=3, arrival_time=0, num_gpus=4, runtime=50, priority_score=5),
        make_job(job_id=4, arrival_time=0, num_gpus=2, runtime=70, priority_score=7),
    ]
    out = size_scheduler.order_queue(jobs, NOW)
    assert [j.num_gpus for j in out] == [1, 2, 4, 8]
    assert _ids(out) == [2, 4, 3, 1]


def test_size_ignores_current_time(make_job):
    """INVARIANT: size ordering is time-invariant (it has no aging term).

    Stated explicitly so that anyone adding aging to this policy has to delete a
    test -- silently making the control time-dependent would invalidate the
    ranking-degeneracy comparison, which is taken at single dispatch instants.
    """
    jobs = _varied_queue(make_job)
    assert _ids(size_scheduler.order_queue(jobs, 0)) == \
        _ids(size_scheduler.order_queue(jobs, 100000))


# --------------------------------------------------------------------------
# priority_scheduler: lower score first, with an aging discount.
# --------------------------------------------------------------------------

def test_priority_lower_score_first(make_job):
    """INVARIANT: in priority_scheduler a LOWER priority_score means dispatch sooner.

    The sign convention is the opposite of the everyday meaning of "priority"
    and is only documented in a one-line comment in the source; inverting it
    would still produce a complete, plausible benchmark run.
    """
    jobs = [make_job(job_id=i, arrival_time=0, priority_score=p)
            for i, p in [(1, 9), (2, 1), (3, 5)]]
    out = priority_scheduler.order_queue(jobs, NOW)
    assert _ids(out) == [2, 3, 1]
    assert [j.priority_score for j in out] == [1, 5, 9]


def test_priority_aging_never_reorders_two_queued_jobs(make_job):
    """DOCUMENTED CURRENT BEHAVIOUR (suspected defect), not an endorsement.

    priority_scheduler advertises "aging", but its key is
        priority_score - 0.03 * (current_time - arrival_time)
    which expands to
        (priority_score + 0.03 * arrival_time) - 0.03 * current_time.
    The `- 0.03 * current_time` term is IDENTICAL for every job in the queue, so
    it cancels out of every pairwise comparison. The induced order is therefore
    the static key (priority_score + 0.03 * arrival_time) and does not change as
    jobs wait: a low-priority job can never overtake a high-priority one by
    waiting, no matter how long it waits. The aging term acts as a one-off
    arrival-time bonus, not as anti-starvation aging.

    (The clamp `max(0, ...)` does not rescue this: queued jobs always have
    arrival_time <= current_time, so the clamp is never active in the dispatch
    loop.)

    This test pins the behaviour as it is today. If someone fixes the policy so
    aging really does bite, this test SHOULD fail and be rewritten -- that is
    the point of it.
    """
    # id=1 is very low priority (high score) and has waited since t=0;
    # id=2 is top priority and arrives at t=90.
    old_unimportant = make_job(job_id=1, arrival_time=0, priority_score=10)
    fresh_important = make_job(job_id=2, arrival_time=90, priority_score=1)
    queue = [old_unimportant, fresh_important]

    # Even after ~11 simulated days of waiting the order is byte-identical.
    for now in (90, 100, 1000, 100000, 1000000):
        assert _ids(priority_scheduler.order_queue(queue, now)) == [2, 1], (
            f'order changed at current_time={now}; if aging was fixed, '
            'rewrite this test')


# --------------------------------------------------------------------------
# (e) Tie-breaking: deterministic, total, independent of input order.
# --------------------------------------------------------------------------

@pytest.mark.parametrize('order', ORDERING_PARAMS)
def test_ties_break_by_arrival_then_job_id(order, make_job):
    """INVARIANT: equal primary keys break by arrival_time, then job_id.

    The queue here is uniform in every policy key (same runtime, est_runtime,
    num_gpus, priority_score), so EVERY helper is forced into its tie-break path
    and every helper must produce the same answer: sorted by (arrival, id).
    Because job ids are unique, that key is total -- there is no residual
    ambiguity left for Python's sort stability (i.e. for insertion order) to
    resolve. Without a total key the benchmark's output would depend on the
    order jobs happen to sit in the queue list, and the reproducibility claim
    would be false.
    """
    # Two jobs per arrival time, ids interleaved so neither id order nor arrival
    # order alone can accidentally produce the right answer.
    spec = [(5, 0), (2, 0), (6, 3), (1, 3), (4, 8), (3, 8)]
    jobs = [make_job(job_id=jid, arrival_time=arr, num_gpus=2, runtime=10,
                     est_runtime=10, priority_score=4) for jid, arr in spec]
    expected = [jid for jid, _ in sorted(spec, key=lambda s: (s[1], s[0]))]

    assert _ids(order(jobs, NOW)) == expected


@pytest.mark.parametrize('order', ORDERING_PARAMS)
def test_output_order_is_independent_of_input_order(order, make_job):
    """INVARIANT: shuffling the input cannot change the output.

    The strongest statement of determinism available for a pure ordering
    function, and it is what makes "results are bit-identical across runs" true:
    if the sort key were not total, Python's stable sort would leak the incoming
    list order into the dispatch decision, and the benchmark's numbers would
    depend on an implementation detail of how the queue list was built.

    Seeded shuffles only -- a flaky ordering test would be worse than none.
    """
    jobs = _varied_queue(make_job)
    reference = _ids(order(jobs, NOW))

    for seed in range(25):
        shuffled = list(jobs)
        random.Random(seed).shuffle(shuffled)
        assert _ids(order(shuffled, NOW)) == reference, \
            f'output depended on input order (shuffle seed {seed})'

    # ...and the reverse of the sorted order, the adversarial case for a stable
    # sort with an incomplete key.
    assert _ids(order(list(reversed(order(jobs, NOW))), NOW)) == reference


@pytest.mark.parametrize('order', ORDERING_PARAMS)
def test_repeated_calls_are_idempotent(order, make_job):
    """INVARIANT: ordering an already-ordered queue is a no-op (the sort is a fixpoint).

    A helper whose output depends on its own previous output would drift over
    the ticks of a run; combined with the shuffle test above this pins the
    ordering down to a single well-defined function of the job attributes.
    """
    jobs = _varied_queue(make_job)
    once = order(jobs, NOW)
    twice = order(once, NOW)
    assert _ids(twice) == _ids(once)

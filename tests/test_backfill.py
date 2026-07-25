"""Tests for `04_scheduler/backfill_scheduler.py` — the EASY and conservative
backfill dispatchers.

This is the most correctness-critical module in the repository: every scheduler
comparison in RESULTS.md is only meaningful if the classical baselines really
implement the classical rules. CHANGELOG.md v3.3 claims two property tests that
were never committed ("300-trial property test that a reserved head is never
delayed", helper "property-tested against a brute-force oracle"); this module
makes those claims true.

The three invariants that matter, and why:

  1. THE EASY GUARANTEE (Mu'alem & Feitelson 2001). Backfilling is only
     defensible because it cannot delay the reserved head job. If it could, EASY
     would be an unbounded-starvation policy and its wait-time numbers would be
     bought with an unadmitted cost. `test_easy_guarantee_*` is the randomised
     proof that it cannot.

  2. THE SECOND BACKFILL CONDITION. Canonical EASY admits a candidate under
     EITHER "finishes by the shadow time" OR "fits in the surplus capacity
     beyond the head's requirement". v3.2 shipped only the first condition,
     which under-backfills and overstates the price of a reservation by ~4x.
     `test_easy_second_condition_*` pins the fix so it cannot silently regress.

  3. NO OVER-ALLOCATION. A dispatcher that hands out more GPUs than exist would
     produce a fictitious utilisation curve and fictitious wait times.

Everything here is deterministic (every RNG is explicitly seeded), needs no
generated artefact, writes no file, and runs in well under a second.
"""

import random

import pytest

from backfill_scheduler import (
    _profile_reserve,   # private, but it carries the conservative guarantee — see below
    compute_reservation,
    conservative_backfill_dispatch,
    easy_backfill_dispatch,
)

# Trial counts. Kept at the size the CHANGELOG claims where that is cheap; the
# suite must stay fast enough that nobody is tempted to skip it.
EASY_TRIALS = 300
ORACLE_TRIALS = 800
CONTRACT_TRIALS = 200


# ─────────────────────────────────────────────────────────────────────────────
# Instance builders
# ─────────────────────────────────────────────────────────────────────────────

def _blocked_head_instance(rng, make_job, make_cluster, t=0):
    """A cluster/queue where the head job provably does NOT fit at `t`.

    Forcing `free < head.num_gpus` is what makes the EASY guarantee testable:
    phase 1 of the dispatcher then starts nothing, so the head the dispatcher
    reserves for is `queue[0]` — the same job (and the same reservation) the
    test computes up front.

    Running jobs are allocated at `t` with runtime >= 1, so every one of them
    has `end_time > t`. That is exactly the precondition `compute_reservation`
    documents: the simulators release finished jobs before dispatching.
    """
    capacity = rng.randint(4, 32)
    free = rng.randint(0, capacity - 1)      # < capacity, so >= 1 GPU is busy
    cluster = make_cluster(capacity, 1)

    running = []
    remaining = capacity - free
    while remaining > 0:
        gpus = rng.randint(1, remaining)
        job = make_job(f'R{len(running)}', arrival_time=t - 1, num_gpus=gpus,
                       runtime=rng.randint(1, 15))
        assert cluster.allocate(job, t)
        running.append(job)
        remaining -= gpus

    # head needs more than is free but no more than the whole cluster, so its
    # reservation is finite.
    head = make_job('HEAD', arrival_time=t, num_gpus=rng.randint(free + 1, capacity),
                    runtime=rng.randint(1, 20))
    queue = [head]
    for k in range(rng.randint(0, 6)):
        queue.append(make_job(f'Q{k}', arrival_time=t,
                              num_gpus=rng.randint(1, capacity),
                              runtime=rng.randint(1, 25)))
    return cluster, queue, running, head, capacity


def _mixed_instance(rng, make_job, make_cluster, t=0):
    """An unconstrained instance: the head may or may not fit, queued jobs may
    even be larger than the whole cluster (the 'can never run' branch)."""
    capacity = rng.randint(4, 24)
    free = rng.randint(0, capacity)
    cluster = make_cluster(capacity, 1)

    running = []
    remaining = capacity - free
    while remaining > 0:
        gpus = rng.randint(1, remaining)
        job = make_job(f'R{len(running)}', arrival_time=t - 1, num_gpus=gpus,
                       runtime=rng.randint(1, 15))
        assert cluster.allocate(job, t)
        running.append(job)
        remaining -= gpus

    queue = [make_job(f'Q{k}', arrival_time=t,
                      num_gpus=rng.randint(1, capacity + 3),
                      runtime=rng.randint(1, 20))
             for k in range(rng.randint(1, 8))]
    return cluster, queue, running, capacity


def _held_at(jobs, shadow):
    """GPUs still held at tick `shadow` by `jobs`.

    A job whose `end_time == shadow` releases AT that tick (the simulators run
    release -> arrivals -> dispatch), so it does NOT hold capacity at `shadow`.
    This is the same convention `compute_reservation` uses when it accumulates
    releases up to and including the shadow tick.
    """
    return sum(j.num_gpus for j in jobs if j.end_time is not None and j.end_time > shadow)


# ─────────────────────────────────────────────────────────────────────────────
# (a) The EASY guarantee, as a randomised property
# ─────────────────────────────────────────────────────────────────────────────

def test_easy_guarantee_reserved_head_is_never_delayed(make_job, make_cluster):
    """INVARIANT: after an EASY dispatch tick, the reserved head job can still
    start at its reservation (shadow) time.

    This is the entire justification for backfilling. Concretely: the capacity
    believed free at the shadow tick — total capacity minus the GPUs held by
    every job (still-running or just backfilled) whose end_time is strictly
    after the shadow — must remain >= head.num_gpus.

    Perfect estimates (`est_runtime_of=None`) are used so the scheduler's
    beliefs equal reality and any violation is a real violation, not an
    artefact of a bad user estimate.

    This is the "300-trial property test that a reserved head is never delayed"
    claimed in CHANGELOG.md v3.3.
    """
    rng = random.Random(20260721)
    t = 0
    checked_with_backfill = 0

    for trial in range(EASY_TRIALS):
        cluster, queue, running, head, capacity = _blocked_head_instance(
            rng, make_job, make_cluster, t)

        # The reservation the dispatcher will make, computed independently and
        # BEFORE anything is dispatched.
        shadow, extra = compute_reservation(cluster, running, head, t)
        assert shadow != float('inf'), 'head fits in the cluster, so it must be reservable'
        assert shadow > t, 'the head does not fit now, so its reservation is in the future'
        assert extra >= 0

        queue_snapshot = list(queue)
        started = easy_backfill_dispatch(cluster, queue, running, t)

        assert head not in started, 'head does not fit; it must not have been started'
        assert all(j in queue_snapshot for j in started)
        if started:
            checked_with_backfill += 1

        # The guarantee.
        free_at_shadow = capacity - _held_at(running + started, shadow)
        assert free_at_shadow >= head.num_gpus, (
            f'trial {trial}: EASY delayed its own reservation — {free_at_shadow} GPUs '
            f'free at shadow t={shadow}, head needs {head.num_gpus}')

        # Sanity: the tick itself never over-allocates, and the surplus budget
        # handed to condition-(b) backfills is never overspent.
        assert cluster.total_free_gpus() >= 0
        assert free_at_shadow - head.num_gpus <= extra

    # Guard against a vacuous pass: if the generator never produced an instance
    # where backfilling actually happened, the property proves nothing.
    assert checked_with_backfill > EASY_TRIALS // 4, (
        f'only {checked_with_backfill}/{EASY_TRIALS} trials backfilled anything')


def test_compute_reservation_matches_a_brute_force_belief_oracle(make_job, make_cluster):
    """INVARIANT: (shadow, extra) is EXACTLY right, not merely safe.

    Stronger than "the head is not delayed", and it is the stronger statement
    the paper needs. `shadow` must be the EARLIEST tick at which the head's
    GPUs are free — a late shadow would make EASY look needlessly conservative,
    an early one would break the guarantee. `extra` must be exactly the surplus
    at that tick — too small under-backfills (that was the v3.2 defect), too
    large delays the head.

    The oracle recomputes both by scanning ticks and asking "how many GPUs are
    held by jobs whose end_time is after this tick", with no reference to the
    implementation's sorted-accumulation strategy.
    """
    rng = random.Random(97531)
    t = 0
    for trial in range(400):
        cluster, queue, running, head, capacity = _blocked_head_instance(
            rng, make_job, make_cluster, t)
        shadow, extra = compute_reservation(cluster, running, head, t)

        horizon = max(j.end_time for j in running)
        want_shadow = next(tau for tau in range(t, horizon + 1)
                           if capacity - _held_at(running, tau) >= head.num_gpus)
        assert shadow == want_shadow, f'trial {trial}: shadow {shadow} != earliest {want_shadow}'
        assert extra == capacity - _held_at(running, shadow) - head.num_gpus, (
            f'trial {trial}: surplus at the shadow tick is misreported')


def test_easy_guarantee_holds_when_backfill_scan_is_reordered(make_job, make_cluster):
    """INVARIANT: the guarantee is a property of the RULE, not of the scan order.

    `proactive_bf` in multi_scheduler_benchmark.py passes a `backfill_order`
    callable that reorders the backfill scan by predicted wait. If the guarantee
    depended on scanning in arrival order, that scheduler would silently starve
    its head job and its published numbers would be wrong.
    """
    rng = random.Random(4242)
    t = 0
    for trial in range(100):
        cluster, queue, running, head, capacity = _blocked_head_instance(
            rng, make_job, make_cluster, t)
        shadow, _ = compute_reservation(cluster, running, head, t)

        # Deterministic but deliberately perverse: biggest-and-longest first,
        # i.e. the order most likely to eat the reservation.
        def worst_first(jobs, active, _rng=rng):
            return sorted(jobs, key=lambda j: (-j.num_gpus, -j.runtime, j.job_id))

        started = easy_backfill_dispatch(cluster, queue, running, t,
                                         backfill_order=worst_first)
        assert head not in started
        free_at_shadow = capacity - _held_at(running + started, shadow)
        assert free_at_shadow >= head.num_gpus, f'trial {trial}: reordered scan delayed the head'


# ─────────────────────────────────────────────────────────────────────────────
# (b) + (c) The canonical two-condition backfill rule
# ─────────────────────────────────────────────────────────────────────────────

def _surplus_scenario(make_job, make_cluster):
    """A hand-built instance with a known reservation of (shadow=2, extra=1).

        capacity 10 = R_short(2 GPUs, ends t=2) + R_long(6 GPUs, ends t=5) + 2 free
        head needs 3 -> does not fit now (2 free)
        at t=2 R_short releases -> 4 believed free -> shadow = 2, extra = 4 - 3 = 1
    """
    t = 0
    cluster = make_cluster(10, 1)
    r_short = make_job('R_short', num_gpus=2, runtime=2)   # ends at t=2
    r_long = make_job('R_long', num_gpus=6, runtime=5)     # ends at t=5
    assert cluster.allocate(r_short, t)
    assert cluster.allocate(r_long, t)
    running = [r_short, r_long]
    head = make_job('HEAD', num_gpus=3, runtime=4)
    assert cluster.total_free_gpus() == 2 < head.num_gpus
    return t, cluster, running, head


def test_easy_second_condition_starts_a_job_that_outlives_the_shadow(make_job, make_cluster):
    """INVARIANT: condition (b) of canonical EASY is implemented and effective.

    A candidate that does NOT finish by the shadow time may still start if it
    needs no more than the SURPLUS capacity available at the shadow tick beyond
    the head's requirement — the head still starts on time, so nothing is lost.

    Omitting this condition was a real defect in v3.2: it under-backfills and
    makes a reservation look ~4x more expensive than it is, which would have
    changed the paper's conclusion about EASY. This test fails if condition (b)
    is ever removed again.
    """
    t, cluster, running, head = _surplus_scenario(make_job, make_cluster)
    shadow, extra = compute_reservation(cluster, running, head, t)
    assert (shadow, extra) == (2, 1)

    # 1 GPU (<= extra=1) but runs to t=50, far past the shadow: admitted only
    # by condition (b).
    long_small = make_job('LONG_SMALL', num_gpus=1, runtime=50)
    queue = [head, long_small]

    started = easy_backfill_dispatch(cluster, queue, running, t)

    assert started == [long_small], (
        'canonical EASY must backfill a job that fits in the shadow-time surplus '
        'even though it outlives the shadow (Mu\'alem & Feitelson 2001, cond. b)')
    # ...and the head is still able to start at t=2: 10 - (6 held by R_long
    # + 1 held by LONG_SMALL) = 3 == head.num_gpus. Exactly tight, by design.
    assert 10 - _held_at(running + started, shadow) == head.num_gpus


def test_easy_does_not_start_a_job_that_neither_ends_by_shadow_nor_fits_surplus(
        make_job, make_cluster):
    """INVARIANT: condition (b) is a bounded budget, not a free pass.

    The candidate here fits in the currently free GPUs (2 free, needs 2), but it
    runs past the shadow AND needs more than the surplus (2 > extra=1). Starting
    it would leave only 10 - 6 - 2 = 2 GPUs free at the shadow tick while the
    head needs 3 — i.e. it would delay the reservation. It must be refused.
    """
    t, cluster, running, head = _surplus_scenario(make_job, make_cluster)
    shadow, extra = compute_reservation(cluster, running, head, t)
    assert (shadow, extra) == (2, 1)

    long_big = make_job('LONG_BIG', num_gpus=2, runtime=50)
    assert cluster.total_free_gpus() >= long_big.num_gpus, 'it does fit right now'
    queue = [head, long_big]

    started = easy_backfill_dispatch(cluster, queue, running, t)

    assert started == [], 'starting LONG_BIG would push the head past its reservation'
    assert long_big.start_time is None
    # Show the counterfactual the rule is protecting against: had LONG_BIG been
    # started it would end at t=50, so at the shadow tick it would still hold
    # 2 GPUs on top of R_long's 6, leaving 10 - 8 = 2 < 3 for the head.
    would_hold = _held_at(running, shadow) + long_big.num_gpus
    assert 10 - would_hold < head.num_gpus


def test_easy_first_condition_starts_a_job_that_ends_by_the_shadow(make_job, make_cluster):
    """INVARIANT (control for the two tests above): condition (a) still works.

    Same 2-GPU candidate as the refused one, but with runtime 2 so it ends
    exactly AT the shadow tick. A job ending at the shadow releases its GPUs in
    time for the head, so `t + runtime <= shadow` is admissible with `<=`, not
    `<`. Off-by-one here would silently disable most legitimate backfilling.
    """
    t, cluster, running, head = _surplus_scenario(make_job, make_cluster)
    shadow, _ = compute_reservation(cluster, running, head, t)

    ends_at_shadow = make_job('ENDS_AT_SHADOW', num_gpus=2, runtime=2)
    started = easy_backfill_dispatch(cluster, [head, ends_at_shadow], running, t)

    assert started == [ends_at_shadow]
    assert ends_at_shadow.end_time == shadow
    assert 10 - _held_at(running + started, shadow) >= head.num_gpus


def test_compute_reservation_is_infinite_for_a_job_larger_than_the_cluster(
        make_job, make_cluster):
    """INVARIANT: an unschedulable head does not freeze the cluster.

    A job needing more GPUs than exist can never be reserved. The documented
    behaviour is `(inf, 0)`, which makes every backfill candidate satisfy
    condition (a) — the queue keeps draining behind the impossible job instead
    of deadlocking. Worth pinning: the natural "safe" alternative (refuse all
    backfill) would be a livelock.
    """
    t = 0
    cluster = make_cluster(8, 1)
    running = [make_job('R', num_gpus=8, runtime=3)]
    assert cluster.allocate(running[0], t)
    impossible = make_job('TOO_BIG', num_gpus=9, runtime=1)

    assert compute_reservation(cluster, running, impossible, t) == (float('inf'), 0)


# ─────────────────────────────────────────────────────────────────────────────
# (d) `_profile_reserve` against a brute-force oracle
# ─────────────────────────────────────────────────────────────────────────────

def _capacity_at(times, free, tau):
    """Capacity of the piecewise-constant profile at tick `tau` (>= times[0]).

    Segment i covers [times[i], times[i+1]); the last segment runs to infinity.
    """
    cap = free[0]
    for k, tk in enumerate(times):
        if tk > tau:
            break
        cap = free[k]
    return cap


def _oracle_reserve(times, free, t, need, dur):
    """Brute force: the earliest integer start >= t at which `need` capacity is
    continuously available for `dur` ticks. Independent of the implementation —
    it scans ticks one at a time over the untouched profile.
    """
    horizon = times[-1]          # free[-1] >= need is a precondition, so this start always works
    for s in range(t, horizon + 1):
        if all(_capacity_at(times, free, tau) >= need for tau in range(s, s + dur)):
            return s
    raise AssertionError('oracle found no feasible start; precondition free[-1] >= need broken')


def _random_profile(rng, t, need):
    """A random piecewise-constant profile with integer breakpoints.

    The final (infinite) segment is forced to hold at least `need`. That is not
    a convenience: it is the precondition `conservative_backfill_dispatch`
    guarantees (its last segment is always full cluster capacity, and jobs
    bigger than capacity are filtered out beforehand). See
    `test_profile_reserve_crashes_when_the_final_segment_is_too_small`.
    """
    times, free = [t], [rng.randint(0, 10)]
    cur = t
    for _ in range(rng.randint(0, 6)):
        cur += rng.randint(1, 6)
        times.append(cur)
        free.append(rng.randint(0, 12))
    free[-1] = max(free[-1], need)
    return times, free


def test_profile_reserve_matches_a_brute_force_oracle(make_job):
    """INVARIANT: `_profile_reserve` returns the EARLIEST feasible start.

    Conservative backfill's whole guarantee — no job is ever delayed by a job
    behind it — rests on this helper. If it returned a start that was merely
    feasible rather than earliest, CONS-BF would look worse than it is; if it
    returned an INfeasible start, the reservations would be lies and the
    guarantee void. Checked against an independent tick-by-tick scan.

    The helper mutates the profile in place, so it is handed a deep copy and the
    oracle reads the pristine original.

    This is the "property-tested against a brute-force oracle" claim of
    CHANGELOG.md v3.3.
    """
    rng = random.Random(31337)
    nontrivial = 0

    for trial in range(ORACLE_TRIALS):
        need = rng.randint(1, 8)
        dur = rng.randint(1, 8)
        t = rng.choice([0, 0, 5, 17])
        times, free = _random_profile(rng, t, need)

        mut_times, mut_free = list(times), list(free)
        got = _profile_reserve(mut_times, mut_free, t, need, dur)
        want = _oracle_reserve(times, free, t, need, dur)

        assert got == want, (
            f'trial {trial}: _profile_reserve returned {got}, oracle says {want} '
            f'(times={times}, free={free}, need={need}, dur={dur})')
        if got != t:
            nontrivial += 1

        # The returned start must actually be feasible on the ORIGINAL profile.
        assert all(_capacity_at(times, free, tau) >= need for tau in range(got, got + dur))

        # The in-place mutation must subtract exactly the reservation: the new
        # profile equals the old one minus `need` on [got, got+dur) and is
        # unchanged everywhere else. Later queued jobs are placed on this
        # mutated profile, so a wrong subtraction corrupts every reservation
        # after it.
        assert mut_times == sorted(set(mut_times)), 'breakpoints must stay sorted and unique'
        assert len(mut_times) == len(mut_free)
        for tau in range(t, times[-1] + dur + 3):
            expected = _capacity_at(times, free, tau) - (need if got <= tau < got + dur else 0)
            assert _capacity_at(mut_times, mut_free, tau) == expected, (
                f'trial {trial}: profile corrupted at tick {tau}')

    # Guard against a vacuous pass: most instances must exercise the search,
    # not just return `t` immediately.
    assert nontrivial > ORACLE_TRIALS // 5, f'only {nontrivial}/{ORACLE_TRIALS} needed a search'


def test_profile_reserve_crashes_when_the_final_segment_is_too_small():
    """DOCUMENTS CURRENT (fragile) BEHAVIOUR — not an endorsement.

    `_profile_reserve` assumes the last, infinite segment can always host the
    request. If it cannot, the search index walks off the end of `times` and the
    helper raises IndexError instead of reporting "never feasible".

    Its only caller, `conservative_backfill_dispatch`, upholds the precondition
    (it skips jobs bigger than total capacity, and the last segment is always
    total capacity), so this is not currently reachable in production. It is
    pinned here so that a future caller — or a refactor of that capacity filter
    — meets a failing test rather than an IndexError from inside a simulation.
    """
    times, free = [0, 5], [1, 2]
    with pytest.raises(IndexError):
        _profile_reserve(times, free, 0, need=4, dur=3)


# ─────────────────────────────────────────────────────────────────────────────
# (e) Conservative backfill never over-allocates
# ─────────────────────────────────────────────────────────────────────────────

def test_conservative_backfill_never_over_allocates(make_job, make_cluster):
    """INVARIANT: the cluster never hands out more GPUs than it has.

    After a conservative dispatch tick, the GPUs held by running plus
    newly-started jobs must not exceed capacity, free GPUs must not go negative,
    and every started job must actually have been allocated (start/end times
    set) and have fitted within capacity. An over-allocating dispatcher would
    report utilisation > 100% and wait times that no real cluster could deliver.
    """
    rng = random.Random(8675309)
    t = 0
    started_anything = 0

    for trial in range(CONTRACT_TRIALS):
        cluster, queue, running, capacity = _mixed_instance(rng, make_job, make_cluster, t)
        started = conservative_backfill_dispatch(cluster, queue, running, t)

        held = sum(j.num_gpus for j in running) + sum(j.num_gpus for j in started)
        assert held <= capacity, f'trial {trial}: over-allocated {held}/{capacity} GPUs'
        assert cluster.total_free_gpus() == capacity - held
        assert cluster.total_free_gpus() >= 0
        assert all(n >= 0 for n in cluster.nodes)

        for job in started:
            assert job.num_gpus <= capacity, 'a job larger than the cluster was started'
            assert job.start_time == t
            assert job.end_time == t + job.runtime
            assert job.allocated_nodes, 'started job was never actually allocated'
        for job in queue:
            if job not in started:
                assert job.start_time is None, 'unstarted job must not be allocated'
        if started:
            started_anything += 1

    assert started_anything > CONTRACT_TRIALS // 4, 'generator never dispatched anything'


def test_conservative_backfill_refuses_a_job_that_would_delay_an_earlier_one(
        make_job, make_cluster):
    """INVARIANT: conservative backfill protects EVERY queued job, not just the head.

    Setup: capacity 4, 2 GPUs free, one running job releasing 2 more at t=3.
      J1 (4 GPUs, 5 ticks) is reserved at t=3 — the first tick it fits.
      J2 (2 GPUs, 5 ticks) FITS RIGHT NOW, but running it to t=5 would leave J1
      only 2 GPUs at t=3 and push J1 to t=5.
    Conservative backfill must therefore start nothing. Under EASY, J2 would
    also be refused (it is the head's own blocker), but the distinguishing
    property is that this holds for a job at ANY queue position, so this test
    would catch a regression to a head-only reservation.
    """
    t = 0
    cluster = make_cluster(4, 1)
    r = make_job('R', num_gpus=2, runtime=3)     # releases 2 GPUs at t=3
    assert cluster.allocate(r, t)
    j1 = make_job('J1', num_gpus=4, runtime=5)
    j2 = make_job('J2', num_gpus=2, runtime=5)
    assert cluster.total_free_gpus() == 2 >= j2.num_gpus, 'J2 fits right now'

    started = conservative_backfill_dispatch(cluster, [j1, j2], [r], t)

    assert started == [], 'J2 fits now but would delay J1 from t=3 to t=5'
    assert cluster.total_free_gpus() == 2


def test_conservative_backfill_starts_a_job_that_fits_behind_a_reservation(
        make_job, make_cluster):
    """INVARIANT (control for the test above): the conservative rule is not
    simply "block everything behind a blocked job".

    Identical setup, except J2 needs 2 GPUs for only 3 ticks — it ends exactly
    when J1's reservation begins, so J1 still starts at t=3. It must run now.
    Without this control, the previous test would also pass on a broken
    implementation that never backfills at all.
    """
    t = 0
    cluster = make_cluster(4, 1)
    r = make_job('R', num_gpus=2, runtime=3)
    assert cluster.allocate(r, t)
    j1 = make_job('J1', num_gpus=4, runtime=5)
    j2 = make_job('J2', num_gpus=2, runtime=3)

    started = conservative_backfill_dispatch(cluster, [j1, j2], [r], t)

    assert started == [j2]
    assert j2.end_time == 3


# ─────────────────────────────────────────────────────────────────────────────
# (f) The dispatcher contract the simulators rely on
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('dispatch', [easy_backfill_dispatch, conservative_backfill_dispatch],
                         ids=['easy', 'conservative'])
def test_dispatchers_do_not_mutate_the_queue_and_return_only_queued_jobs(
        dispatch, make_job, make_cluster):
    """INVARIANT: a dispatcher reports, it does not bookkeep.

    Both simulators call `queue.remove(job)` for every returned job. That is
    only sound if (1) the dispatcher left the caller's list untouched — a
    dispatcher that popped jobs itself would make the caller's remove() raise or
    silently drop the wrong job — and (2) every returned job really came from
    that list, by identity, not an equal-looking copy. The same applies to the
    running-jobs list, which the caller extends afterwards.
    """
    rng = random.Random(11235)
    t = 0
    for trial in range(CONTRACT_TRIALS):
        cluster, queue, running, _cap = _mixed_instance(rng, make_job, make_cluster, t)
        queue_ids = [id(j) for j in queue]
        running_ids = [id(j) for j in running]

        started = dispatch(cluster, queue, running, t)

        assert [id(j) for j in queue] == queue_ids, 'dispatcher mutated the caller\'s queue'
        assert [id(j) for j in running] == running_ids, 'dispatcher mutated the running list'
        assert len(set(id(j) for j in started)) == len(started), 'a job was returned twice'
        assert set(id(j) for j in started) <= set(queue_ids), (
            f'trial {trial}: dispatcher returned a job that was not in the queue')

        # And the caller's documented follow-up must work.
        for job in started:
            queue.remove(job)
            running.append(job)

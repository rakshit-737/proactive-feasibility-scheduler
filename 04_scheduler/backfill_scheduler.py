"""Backfill dispatch helpers: EASY and conservative (reservation-per-job).

EASY (Extensible Argonne Scheduling sYstem) rule per tick:
  1. Start head jobs (queue[0] under 'order', default arrival order) while
     they fit in the currently free GPUs.
  2. When the head does NOT fit, compute its reservation (shadow) time: the
     earliest future tick at which enough GPUs will be free, derived from
     the running jobs' (estimated) end times.
  3. Backfill pass over the REMAINING queue, canonical two-condition rule
     (Mu'alem & Feitelson 2001): a candidate that fits now may start iff
       (a) it (believes it) finishes by the shadow time, OR
       (b) it needs no more than the EXTRA processors — those believed free
           at the shadow time beyond the head's requirement (candidates
           admitted under (b) consume the extra budget).
     Either way the reserved head job is never delayed (the EASY guarantee).

Conservative backfill strengthens the guarantee: EVERY queued job gets a
reservation (earliest start on a future free-capacity profile, in queue
order), so no job is ever delayed by any job behind it in the queue.

RUNTIME ESTIMATES: by default both policies use each job's TRUE runtime —
the strongest-possible perfect-estimate baseline. Pass `est_runtime_of` (a
callable job -> believed runtime) to model realistic user estimates: the
scheduler then plans with estimated durations and estimated release times
(start + estimate) while jobs still physically run for their true runtime.
With over-estimates (the empirically dominant case) planned releases are
later than reality, so reservations are pessimistic — the behaviour studied
in the backfill literature (Mu'alem & Feitelson 2001).
"""


def compute_reservation(cluster, running_jobs, head_job, t, est_end=None):
    """(shadow_time, extra) for the blocked head job.

    shadow_time — earliest future tick at which head_job's GPUs are believed
    free: sorts running jobs by (estimated) end time and accumulates released
    GPUs onto the current free count. Assumes the caller has already released
    jobs whose end_time == t (the simulators do release -> arrivals ->
    dispatch), so every running job has true end_time > t; estimated ends are
    clamped to t+1 so under-estimates cannot place a release in the past.

    extra — GPUs believed free AT the shadow time beyond the head's need
    (releases at exactly the shadow tick included). This is the budget the
    canonical second backfill condition may hand to candidates that run past
    the shadow without delaying the head.
    """
    if est_end is None:
        est_end = lambda j: j.end_time
    free = cluster.total_free_gpus()
    if free >= head_job.num_gpus:
        return t, free - head_job.num_gpus
    shadow = None
    for job in sorted(running_jobs, key=lambda j: max(est_end(j), t + 1)):
        free += job.num_gpus
        if shadow is None and free >= head_job.num_gpus:
            shadow = max(est_end(job), t + 1)
        elif shadow is not None:
            if max(est_end(job), t + 1) > shadow:
                free -= job.num_gpus       # releases after the shadow tick
                break
    if shadow is not None:
        return shadow, free - head_job.num_gpus
    # Head can never fit even on an empty cluster (job larger than total
    # capacity) — cannot reserve; no backfill window constraint applies.
    return float('inf'), 0


def easy_backfill_dispatch(cluster, queue, running_jobs, t, order=None,
                           backfill_order=None, pre_dispatch_hook=None,
                           est_runtime_of=None):
    """Run one tick of EASY-backfill dispatch. Returns the started jobs.

    Parameters
    ----------
    cluster        : Cluster with per-node free-GPU lists (mutated by allocate)
    queue          : waiting jobs; NOT mutated here — the caller removes the
                     returned started jobs from its queue and adds them to
                     its running list
    running_jobs   : currently running jobs (read-only; end_time known)
    t              : current tick
    order          : optional callable(jobs) -> ordered list; defines the
                     head-of-queue sequence (default: arrival order, i.e.
                     the queue as given)
    backfill_order : optional callable(jobs, active_jobs) -> ordered list;
                     scan order for the backfill pass over the remaining
                     queue (default: same order as the head sequence)
    pre_dispatch_hook : optional callable(job, cluster, pending, active)
                     invoked just before each allocation, with the
                     PRE-allocation cluster state (used by the benchmark to
                     score wait-time predictions at dispatch decisions)
    est_runtime_of : optional callable(job) -> believed runtime; default is
                     the true runtime (perfect estimates). Used for both the
                     reservation (running ends = start + estimate) and the
                     backfill window check.
    """
    rt = est_runtime_of if est_runtime_of is not None else (lambda j: j.runtime)
    est_end = ((lambda j: j.start_time + rt(j)) if est_runtime_of is not None
               else (lambda j: j.end_time))
    pending = list(queue) if order is None else list(order(list(queue)))
    active = list(running_jobs)   # running + jobs started this call
    started = []

    # ── Phase 1: start head jobs while they fit ─────────────────────────
    while pending:
        head = pending[0]
        if cluster.total_free_gpus() < head.num_gpus:
            break
        if pre_dispatch_hook is not None:
            pre_dispatch_hook(head, cluster, pending, active)
        if not cluster.allocate(head, t):
            break   # defensive: greedy allocate should not fail after fit check
        started.append(head)
        active.append(head)
        pending.pop(0)

    if not pending:
        return started

    # ── Phase 2: reservation for the blocked head ───────────────────────
    head = pending[0]
    reservation, extra = compute_reservation(cluster, active, head, t,
                                             est_end=est_end)

    # ── Phase 3: backfill scan over the remaining queue ─────────────────
    # Canonical EASY two-condition rule (Mu'alem & Feitelson 2001): a job
    # that fits now may jump the head if
    #   (a) it (believes it) finishes by the reservation — the capacity
    #       promised to the head at the shadow tick is untouched; or
    #   (b) it fits within the EXTRA GPUs beyond the head's need at the
    #       shadow tick — it may run past the shadow, consuming the extra
    #       budget, and the head still starts on time.
    # Either way the head is never delayed under the scheduler's beliefs.
    rest = pending[1:]
    if backfill_order is not None:
        rest = list(backfill_order(rest, active))
    for job in rest:
        if cluster.total_free_gpus() < job.num_gpus:
            continue
        ends_by_shadow = t + rt(job) <= reservation
        if not ends_by_shadow and job.num_gpus > extra:
            continue
        if pre_dispatch_hook is not None:
            pre_dispatch_hook(job, cluster, pending, active)
        if cluster.allocate(job, t):
            if not ends_by_shadow:
                extra -= job.num_gpus
            started.append(job)
            active.append(job)
            pending.remove(job)   # keep hook's queue view accurate

    return started


# ─────────────────────────────────────────────────────────────────────────────
# Conservative backfill: a reservation for EVERY queued job
# ─────────────────────────────────────────────────────────────────────────────

def _profile_reserve(times, free, t, need, dur):
    """Find the earliest start >= t where `need` GPUs are free for `dur`
    ticks on the piecewise-constant profile (times[i] -> free capacity on
    [times[i], times[i+1]), last segment extends to infinity), then subtract
    the reservation from the profile in place. Returns the start tick.

    Capacity only increases at breakpoints once reservations are subtracted
    left-to-right in queue order, so the earliest feasible start is always a
    breakpoint (or t itself, which is breakpoint 0).
    """
    idx = 0
    while True:
        s = times[idx]
        end = s + dur
        j = idx
        feasible = True
        while j < len(times) and times[j] < end:
            if free[j] < need:
                feasible = False
                idx = j + 1     # restart the search after the blocking segment
                break
            j += 1
        if not feasible:
            continue

        # subtract the reservation over [s, end): split segments at s and end
        for cut in (s, end):
            for k in range(len(times)):
                if times[k] == cut:
                    break
                if times[k] > cut:
                    times.insert(k, cut)
                    free.insert(k, free[k - 1])
                    break
            else:
                times.append(cut)
                free.append(free[-1])
        for k in range(len(times)):
            if s <= times[k] < end:
                free[k] -= need
        return s


def conservative_backfill_dispatch(cluster, queue, running_jobs, t,
                                   pre_dispatch_hook=None, est_runtime_of=None):
    """Run one tick of conservative-backfill dispatch. Returns started jobs.

    Every queued job, scanned in queue (arrival) order, is assigned the
    earliest start on the believed future free-capacity profile; a job whose
    reserved start is `t` is dispatched now. Because each job's reservation
    is carved out of the profile before later jobs are placed, no queued job
    can ever be delayed by a job behind it — the conservative guarantee
    (contrast EASY, which only protects the queue head).

    Same contract as easy_backfill_dispatch: `queue` is not mutated; the
    caller removes the returned jobs. Capacity-level reservations match this
    simulator's allocator, which only requires total free GPUs (jobs may
    span nodes), so a capacity-feasible reservation is always allocatable.
    """
    rt = est_runtime_of if est_runtime_of is not None else (lambda j: j.runtime)
    est_end = ((lambda j: j.start_time + rt(j)) if est_runtime_of is not None
               else (lambda j: j.end_time))
    capacity = cluster.total_free_gpus() + sum(j.num_gpus for j in running_jobs)

    # future free-capacity profile from the running jobs' believed releases
    times, free = [t], [cluster.total_free_gpus()]
    for job in sorted(running_jobs, key=lambda j: max(est_end(j), t + 1)):
        e = max(est_end(job), t + 1)
        if times[-1] == e:
            free[-1] += job.num_gpus
        else:
            times.append(e)
            free.append(free[-1] + job.num_gpus)

    started = []
    for job in list(queue):
        if job.num_gpus > capacity:
            continue    # can never run even on an empty cluster
        start = _profile_reserve(times, free, t, job.num_gpus, rt(job))
        if start == t:
            if pre_dispatch_hook is not None:
                pre_dispatch_hook(job, cluster, queue, running_jobs)
            if cluster.allocate(job, t):
                started.append(job)
    return started

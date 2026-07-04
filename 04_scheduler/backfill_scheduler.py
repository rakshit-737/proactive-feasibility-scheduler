"""EASY-backfill dispatch helper.

Implements the classic EASY (Extensible Argonne Scheduling sYstem) backfill
rule per tick:
  1. Start head jobs (queue[0] under 'order', default arrival order) while
     they fit in the currently free GPUs.
  2. When the head does NOT fit, compute its reservation time: the earliest
     future tick at which enough GPUs will be free, derived from the running
     jobs' known end_time values.
  3. Backfill pass over the REMAINING queue: a job may start now iff it fits
     now AND t + job.runtime <= reservation, i.e. it can never delay the
     reserved head job (this is the EASY guarantee / anti-starvation rule).

NOTE ON RUNTIME ESTIMATES: this implementation uses each job's TRUE runtime
as its estimate. Real EASY deployments rely on user-supplied (often inflated)
runtime estimates, so this is the strongest-possible perfect-estimate
baseline — deliberately conservative for our comparison: any advantage a
competing scheduler shows here is against backfill at its best.
"""


def compute_reservation(cluster, running_jobs, head_job, t):
    """Earliest future tick at which head_job's GPUs will be free.

    Sorts running jobs by end_time and accumulates released GPUs onto the
    current free count. Assumes the caller has already released jobs whose
    end_time == t (the simulators do release -> arrivals -> dispatch), so
    every running job has end_time > t.
    """
    free = cluster.total_free_gpus()
    if free >= head_job.num_gpus:
        return t
    for job in sorted(running_jobs, key=lambda j: j.end_time):
        free += job.num_gpus
        if free >= head_job.num_gpus:
            return job.end_time
    # Head can never fit even on an empty cluster (job larger than total
    # capacity) — cannot reserve; no backfill window constraint applies.
    return float('inf')


def easy_backfill_dispatch(cluster, queue, running_jobs, t, order=None,
                           backfill_order=None, pre_dispatch_hook=None):
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
    """
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
    reservation = compute_reservation(cluster, active, head, t)

    # ── Phase 3: backfill scan over the remaining queue ─────────────────
    # EASY rule: a job may jump the head only if it fits now AND finishes
    # by the head's reservation, so the head is never delayed. Backfilled
    # jobs end at or before the reservation, so the GPU count promised to
    # the head at that tick is preserved and no recomputation is needed.
    rest = pending[1:]
    if backfill_order is not None:
        rest = list(backfill_order(rest, active))
    for job in rest:
        if cluster.total_free_gpus() >= job.num_gpus and t + job.runtime <= reservation:
            if pre_dispatch_hook is not None:
                pre_dispatch_hook(job, cluster, pending, active)
            if cluster.allocate(job, t):
                started.append(job)
                active.append(job)
                pending.remove(job)   # keep hook's queue view accurate

    return started

"""Static-priority queue ordering helper (no aging).

The policy ranks queued jobs by a STATIC key: priority_score blended with a
fixed arrival-time bonus, so an earlier arrival buys a small, permanent head
start. Lower key dispatches first. The position a job is assigned the instant
it joins the queue is the position it keeps -- waiting never improves it.

Named honestly after the earlier "priority + aging" label was found to be
wrong: the current_time term in the key cancels pairwise (see the expansion in
order_queue). hrrn_scheduler is this repository's genuinely aging baseline; the
value of THIS policy as a baseline is precisely that it does not age.
"""

def order_queue(queue, current_time):
    def priority(job):
        waited = max(0, current_time - job.arrival_time)
        # Lower score gets higher priority.
        #
        # WHY this is a static order, not aging. Expand the first key element:
        #     priority_score - 0.03 * (current_time - arrival_time)
        #   = (priority_score + 0.03 * arrival_time) - 0.03 * current_time
        # The `- 0.03 * current_time` term is the SAME additive constant for
        # every job in the queue at any one instant, so it cancels out of every
        # pairwise comparison. The effective key is the time-independent
        # (priority_score + 0.03 * arrival_time): the induced order is
        # time-invariant and a queued job can NEVER overtake another by having
        # waited. This is NOT an anti-starvation mechanism. (The max(0, ...)
        # clamp never fires during dispatch -- a queued job always has
        # arrival_time <= current_time -- so it does not rescue the term.)
        # Kept exactly as written: the published numbers are this policy's, and
        # it is relabelled STATIC_PRIORITY rather than silently repaired.
        return (job.priority_score - 0.03 * waited, job.arrival_time, job.job_id)

    return sorted(queue, key=priority)

"""Smallest-Job-First (by requested size) queue ordering.

This is not a classical baseline chosen for its own sake -- it is the CONTROL
for the ranking-degeneracy result (see ranking_degeneracy.py).

At any single dispatch instant every queued job observes the SAME cluster
state, so in the wait-model feature vector only the per-job features vary,
and each of those is a deterministic function of the job's requested size.
The learned wait-time score is therefore a function of requested size alone,
and the queue order it induces is a permutation of the size order. If that
argument holds, an ML-free "order by requested size" policy must reproduce
the ML scheduler's decisions -- which is exactly what this policy tests.
"""

def order_queue(queue, current_time=None):
    """Smallest requested size first; ties broken by arrival then id."""
    return sorted(queue, key=lambda job: (job.num_gpus, job.arrival_time, job.job_id))

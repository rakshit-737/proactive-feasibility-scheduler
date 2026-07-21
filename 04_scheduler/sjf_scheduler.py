"""Shortest Job First queue ordering helpers.

order_queue sorts by TRUE runtime — a perfect-information oracle (no real
scheduler knows runtimes exactly). order_queue_estimated sorts by the job's
est_runtime attribute (user-supplied runtime estimate), the deployable
version of SJF; estimate generation lives with the benchmark's workload
generator so estimates are paired across schedulers.
"""

def order_queue(queue, current_time=None):
    return sorted(queue, key=lambda job: (job.runtime, job.arrival_time, job.job_id))


def order_queue_estimated(queue, current_time=None):
    return sorted(queue, key=lambda job: (job.est_runtime, job.arrival_time, job.job_id))

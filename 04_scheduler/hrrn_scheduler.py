"""Highest Response Ratio Next (HRRN) queue ordering helper.

Classic starvation-aware policy (Brinch Hansen, 1971): dispatch the job with
the highest response ratio (wait + expected_runtime) / expected_runtime.
Short jobs start with an SJF-like advantage, but the ratio of every waiting
job grows without bound over time, so aging is built into the ratio rather
than bolted on as a tunable weight. NOTE: in this benchmark's dispatch loop
(any fitting job may start, no head reservation) the growing ratio moves a
long job to the FRONT of the scan but cannot guarantee it a start time the
way EASY's reservation does — starvation is strongly discouraged, not
formally impossible.

order_queue uses the TRUE runtime as expected_runtime, mirroring the SJF
oracle baseline (the benchmark's SJF-EST covers the estimated-runtime
regime). order_queue_estimated is the deployable variant: it uses the job's
`est_runtime` attribute (a user-supplied estimate), which is what the
trace-driven benchmark feeds it from the SWF requested-time field.
"""

def _order(queue, current_time, service_of):
    def response_ratio(job):
        waited = max(0, current_time - job.arrival_time)
        service = max(service_of(job), 1e-9)
        return (service + waited) / service

    # highest ratio first; ties broken by arrival then id for determinism
    return sorted(queue, key=lambda j: (-response_ratio(j), j.arrival_time, j.job_id))


def order_queue(queue, current_time):
    return _order(queue, current_time, lambda j: j.runtime)


def order_queue_estimated(queue, current_time):
    return _order(queue, current_time, lambda j: j.est_runtime)

"""Shortest Job First queue ordering helper."""

def order_queue(queue, current_time=None):
    return sorted(queue, key=lambda job: (job.runtime, job.arrival_time, job.job_id))

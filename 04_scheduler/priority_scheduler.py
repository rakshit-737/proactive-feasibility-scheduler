"""Priority scheduler queue ordering helper with aging."""

def order_queue(queue, current_time):
    def priority(job):
        waited = max(0, current_time - job.arrival_time)
        # Lower score gets higher priority.
        return (job.priority_score - 0.03 * waited, job.arrival_time, job.job_id)

    return sorted(queue, key=priority)

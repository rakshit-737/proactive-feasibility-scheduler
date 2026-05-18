"""Simple neural-network scheduler score helper."""

import numpy as np


def build_feature_vector(job, cluster, queue, running_jobs):
    total_free = cluster.total_free_gpus()
    return np.array([
        job.num_gpus,
        total_free,
        len(queue),
        len(running_jobs),
        max(cluster.nodes),
        float(np.var(cluster.nodes)),
        int(total_free >= job.num_gpus),
        min(total_free / (job.num_gpus + 1e-6), 1.0),
        float(np.std(cluster.nodes)),
        sum(q.num_gpus for q in queue) / (total_free + 1),
        sum(1 for n in cluster.nodes if n >= job.num_gpus) / cluster.num_nodes,
        total_free / cluster.num_nodes,
    ])


def order_queue(queue, cluster, running_jobs, predictor):
    scored = []
    for job in queue:
        x = build_feature_vector(job, cluster, queue, running_jobs).reshape(1, -1)
        pred_wait = float(predictor.predict(x)[0])
        scored.append((pred_wait, job))
    scored.sort(key=lambda x: (x[0], x[1].arrival_time, x[1].job_id))
    return [j for _, j in scored]

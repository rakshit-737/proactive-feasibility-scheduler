import random
import numpy as np
import pandas as pd
import pickle

with open("wait_model.pkl", "rb") as f:
    model = pickle.load(f)

class Job:
    def __init__(self, job_id, arrival_time, num_gpus, runtime):
        self.job_id = job_id
        self.arrival_time = arrival_time
        self.num_gpus = num_gpus
        self.runtime = runtime
        self.start_time = None
        self.end_time = None
        self.allocated_nodes = []

class Cluster:
    def __init__(self, num_nodes, gpus_per_node):
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.nodes = [gpus_per_node] * num_nodes

    def total_free_gpus(self):
        return sum(self.nodes)

    def allocate(self, job, current_time):
        required = job.num_gpus
        allocation = []
        for i in range(self.num_nodes):
            if required <= 0:
                break
            available = self.nodes[i]
            if available > 0:
                used = min(available, required)
                self.nodes[i] -= used
                allocation.append((i, used))
                required -= used
        if required == 0:
            job.start_time = current_time
            job.end_time = current_time + job.runtime
            job.allocated_nodes = allocation
            return True
        else:
            for node_id, used in allocation:
                self.nodes[node_id] += used
            return False

    def release(self, job):
        for node_id, used in job.allocated_nodes:
            self.nodes[node_id] += used

def generate_jobs(num_jobs, max_time):
    jobs = []
    for i in range(num_jobs):
        arrival_time = random.randint(0, max_time // 2)
        num_gpus = random.randint(1, 8)
        runtime = random.randint(5, 20)
        jobs.append(Job(i, arrival_time, num_gpus, runtime))
    return sorted(jobs, key=lambda x: x.arrival_time)

def get_features(job, cluster, queue, running_jobs):
    total_free = cluster.total_free_gpus()
    smaller_jobs = sum(1 for q in queue if q.num_gpus <= job.num_gpus)
    demand_ratio = job.num_gpus / (total_free + 1)
    return [
        job.num_gpus,
        total_free,
        len(queue),
        len(running_jobs),
        max(cluster.nodes),
        np.var(cluster.nodes),
        smaller_jobs,
        demand_ratio
    ]

def run_baseline(jobs_input):
    SIM_TIME = 300
    NUM_NODES = 8
    GPUS_PER_NODE = 4

    cluster = Cluster(NUM_NODES, GPUS_PER_NODE)
    jobs = [Job(j.job_id, j.arrival_time, j.num_gpus, j.runtime) for j in jobs_input]

    queue = []
    running_jobs = []
    completed_jobs = []
    gpu_usage = []
    total_capacity = NUM_NODES * GPUS_PER_NODE

    for t in range(SIM_TIME):
        for job in running_jobs[:]:
            if job.end_time == t:
                cluster.release(job)
                running_jobs.remove(job)
                completed_jobs.append(job)

        for job in jobs:
            if job.arrival_time == t:
                queue.append(job)

        for job in queue[:]:
            if cluster.total_free_gpus() >= job.num_gpus:
                success = cluster.allocate(job, t)
                if success:
                    running_jobs.append(job)
                    queue.remove(job)

        gpu_usage.append((total_capacity - cluster.total_free_gpus()) / total_capacity)

    wait_times = [j.start_time - j.arrival_time for j in completed_jobs if j.start_time is not None]
    return np.mean(wait_times), np.mean(gpu_usage), len(completed_jobs)

def run_proactive(jobs_input):
    SIM_TIME = 300
    NUM_NODES = 8
    GPUS_PER_NODE = 4

    cluster = Cluster(NUM_NODES, GPUS_PER_NODE)
    jobs = [Job(j.job_id, j.arrival_time, j.num_gpus, j.runtime) for j in jobs_input]

    queue = []
    running_jobs = []
    completed_jobs = []
    gpu_usage = []
    total_capacity = NUM_NODES * GPUS_PER_NODE

    for t in range(SIM_TIME):
        for job in running_jobs[:]:
            if job.end_time == t:
                cluster.release(job)
                running_jobs.remove(job)
                completed_jobs.append(job)

        for job in jobs:
            if job.arrival_time == t:
                queue.append(job)

        if queue:
            # predict wait time for each job and sort by predicted wait ascending
            scored = []
            for job in queue:
                features = get_features(job, cluster, queue, running_jobs)
                predicted_wait = model.predict([features])[0]
                scored.append((predicted_wait, job))

            scored.sort(key=lambda x: x[0])
            queue = [job for _, job in scored]

        for job in queue[:]:
            if cluster.total_free_gpus() >= job.num_gpus:
                success = cluster.allocate(job, t)
                if success:
                    running_jobs.append(job)
                    queue.remove(job)
                    break

        gpu_usage.append((total_capacity - cluster.total_free_gpus()) / total_capacity)

    wait_times = [j.start_time - j.arrival_time for j in completed_jobs if j.start_time is not None]
    return np.mean(wait_times), np.mean(gpu_usage), len(completed_jobs)

# run comparison
NUM_RUNS = 10
baseline_waits, baseline_utils = [], []
proactive_waits, proactive_utils = [], []

for i in range(NUM_RUNS):
    jobs = generate_jobs(110, 300)

    bwait, butil, bjobs = run_baseline(jobs)
    pwait, putil, pjobs = run_proactive(jobs)

    baseline_waits.append(bwait)
    baseline_utils.append(butil)
    proactive_waits.append(pwait)
    proactive_utils.append(putil)

    print(f"Run {i+1}: Baseline wait={bwait:.2f} util={butil:.2f} | Proactive wait={pwait:.2f} util={putil:.2f}")

print("\n=== Final Results (avg over 10 runs) ===")
print(f"Baseline   — Avg Wait: {np.mean(baseline_waits):.2f} | Avg Utilization: {np.mean(baseline_utils):.2f}")
print(f"Proactive  — Avg Wait: {np.mean(proactive_waits):.2f} | Avg Utilization: {np.mean(proactive_utils):.2f}")
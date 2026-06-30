import random
import numpy as np

# ----------------------------
# Job Class
# ----------------------------

class Job:
    def __init__(self, job_id, arrival_time, num_gpus, runtime):
        self.job_id = job_id
        self.arrival_time = arrival_time
        self.num_gpus = num_gpus
        self.runtime = runtime
        self.start_time = None
        self.end_time = None
        self.allocated_nodes = []


# ----------------------------
# Cluster Class
# ----------------------------

class Cluster:
    def __init__(self, num_nodes, gpus_per_node):
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.nodes = [gpus_per_node] * num_nodes  # available GPUs per node

    def total_free_gpus(self):
        return sum(self.nodes)

    def allocate(self, job, current_time):
        """
        Try to allocate GPUs across nodes.
        Simple greedy allocation.
        """
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
            # Rollback allocation if failed
            for node_id, used in allocation:
                self.nodes[node_id] += used
            return False

    def release(self, job):
        for node_id, used in job.allocated_nodes:
            self.nodes[node_id] += used


# ----------------------------
# Simulation
# ----------------------------

def generate_jobs(num_jobs, max_time):
    jobs = []
    for i in range(num_jobs):
        arrival_time = random.randint(0, max_time // 2)
        num_gpus = random.randint(1, 8)
        runtime = random.randint(5, 20)
        jobs.append(Job(i, arrival_time, num_gpus, runtime))
    return sorted(jobs, key=lambda x: x.arrival_time)


def simulate():
    SIM_TIME = 200
    NUM_NODES = 8
    GPUS_PER_NODE = 4
    NUM_JOBS = 50

    cluster = Cluster(NUM_NODES, GPUS_PER_NODE)
    jobs = generate_jobs(NUM_JOBS, SIM_TIME)

    queue = []
    running_jobs = []
    completed_jobs = []

    total_gpu_capacity = NUM_NODES * GPUS_PER_NODE
    gpu_usage_over_time = []

    for t in range(SIM_TIME):

        # Release finished jobs
        for job in running_jobs[:]:
            if job.end_time == t:
                cluster.release(job)
                running_jobs.remove(job)
                completed_jobs.append(job)

        # Add arriving jobs to queue
        for job in jobs:
            if job.arrival_time == t:
                queue.append(job)

        # Try scheduling jobs (FIFO)
        for job in queue[:]:
            if cluster.total_free_gpus() >= job.num_gpus:
                success = cluster.allocate(job, t)
                if success:
                    running_jobs.append(job)
                    queue.remove(job)

        # Track utilization
        used_gpus = total_gpu_capacity - cluster.total_free_gpus()
        gpu_usage_over_time.append(used_gpus / total_gpu_capacity)

    # Metrics

    wait_times = [
        job.start_time - job.arrival_time
        for job in completed_jobs
        if job.start_time is not None
    ]

    avg_wait_time = np.mean(wait_times) if wait_times else 0
    avg_utilization = np.mean(gpu_usage_over_time)

    print("Simulation Complete")
    print(f"Completed Jobs: {len(completed_jobs)}")
    print(f"Average Waiting Time: {avg_wait_time:.2f}")
    print(f"Average GPU Utilization: {avg_utilization:.2f}")


if __name__ == "__main__":
    simulate()

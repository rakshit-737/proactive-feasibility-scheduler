"""Pytest configuration and shared fixtures.

The repository predates any packaging: its modules live in numbered directories
(`01_simulation`, `04_scheduler`, ...) whose names are not valid Python
identifiers, so they cannot be imported as packages. Every script works around
this with `sys.path.insert(0, os.path.dirname(__file__))`. The tests do the same
thing once, here, so a test module can simply `import backfill_scheduler`.

Design rules for this suite:
  * FAST. No test may run a full benchmark. Anything that needs a real
    simulation uses a handful of jobs, not the 110-job / 300-tick production
    configuration.
  * NO NETWORK, NO WRITES to tracked artefacts. Tests must never regenerate a
    committed CSV or PNG — a test run must leave `git status` clean.
  * Artefact-dependent tests SKIP rather than fail when a generated input is
    missing, so a fresh clone can run the suite before the pipeline has run.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Order matters only in that every directory holding an importable module is on
# the path; the numbered dirs contain no name collisions with each other.
for _sub in ('', '01_simulation', '02_data', '03_models', '04_scheduler',
             '05_results'):
    _p = os.path.join(PROJECT_ROOT, _sub) if _sub else PROJECT_ROOT
    if _p not in sys.path:
        sys.path.insert(0, _p)


def require(path, what=None):
    """Skip the calling test unless a repo-relative artefact exists."""
    full = os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(full):
        pytest.skip(f'missing generated artefact: {path}'
                    + (f' ({what})' if what else '')
                    + ' — run run_all_experiments.sh first')
    return full


@pytest.fixture(scope='session')
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope='session')
def wait_model():
    """The trained v2 wait model, or skip if the pipeline has not been run."""
    import pickle
    path = require('03_models/wait_model_v2.pkl', 'trained wait model')
    with open(path, 'rb') as f:
        return pickle.load(f)


class SimpleJob:
    """Minimal job matching the duck-type every scheduler helper expects.

    The production code has ~17 near-duplicate `Job` classes; the helpers only
    ever touch these attributes, so tests construct this instead of importing
    one arbitrary copy and implying it is canonical.
    """

    __slots__ = ('job_id', 'arrival_time', 'num_gpus', 'runtime', 'est_runtime',
                 'priority_score', 'start_time', 'end_time', 'allocated_nodes',
                 'remaining')

    def __init__(self, job_id, arrival_time=0, num_gpus=1, runtime=1,
                 est_runtime=None, priority_score=1):
        self.job_id = job_id
        self.arrival_time = arrival_time
        self.num_gpus = num_gpus
        self.runtime = runtime
        self.est_runtime = runtime if est_runtime is None else est_runtime
        self.priority_score = priority_score
        self.start_time = None
        self.end_time = None
        self.allocated_nodes = []
        self.remaining = float(runtime)

    def __repr__(self):
        return (f'SimpleJob(id={self.job_id}, arr={self.arrival_time}, '
                f'gpus={self.num_gpus}, rt={self.runtime})')


class SimpleCluster:
    """Capacity-only cluster exposing the interface the dispatch helpers use."""

    def __init__(self, capacity, num_nodes=1):
        self.num_nodes = num_nodes
        self.gpus_per_node = capacity // num_nodes
        self.capacity = capacity
        self.nodes = [self.gpus_per_node] * num_nodes

    def total_free_gpus(self):
        return sum(self.nodes)

    def allocate(self, job, t):
        need = job.num_gpus
        alloc = []
        for i in range(self.num_nodes):
            if need <= 0:
                break
            free = self.nodes[i]
            if free > 0:
                used = min(free, need)
                self.nodes[i] -= used
                alloc.append((i, used))
                need -= used
        if need == 0:
            job.start_time = t
            job.end_time = t + job.runtime
            job.allocated_nodes = alloc
            return True
        for nid, used in alloc:
            self.nodes[nid] += used
        return False

    def release(self, job):
        for nid, used in job.allocated_nodes:
            self.nodes[nid] += used
        job.allocated_nodes = []


@pytest.fixture
def make_job():
    return SimpleJob


@pytest.fixture
def make_cluster():
    return SimpleCluster

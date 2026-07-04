# build_real_trace_datasets.py
#
# Parse the two REAL SWF traces (Parallel Workloads Archive, cleaned versions)
# and reconstruct, for every kept job, the aggregate cluster state that existed
# at its submit instant by chronologically replaying the RECORDED schedule
# (start = submit + wait, end = start + run).  Only HONESTLY RECONSTRUCTABLE
# aggregate features are emitted -- the SWF logs contain no per-node placement,
# so no node-level features are fabricated here.
#
# Filter (documented):
#   * procs = allocated_procs (field 5), falling back to requested_procs
#     (field 8) when allocated <= 0 (LANL records -1 for requested sometimes,
#     SDSC the reverse).
#   * keep rows with procs > 0, run_time > 0, wait_time >= 0 and
#     status in (0, 1, -1)  -> completed (1), failed (0) and unknown (-1) jobs
#     are kept because they all really occupied the machine for run_time
#     seconds; cancelled jobs (5) and partial checkpoint segments (2,3,4) are
#     dropped.  In the traces on disk: LANL has statuses {0,1} only, SDSC has
#     {1,5} (cancelled jobs are dropped there).
#
# NOTE on negative free capacity: the LANL CM-5 header says "many jobs,
# especially smaller ones, ran in non-dedicated mode", so the recorded
# schedule can overlap beyond MaxProcs and reconstructed total_free can go
# NEGATIVE.  We clamp total_free at 0, count how often the raw value was
# negative, print the percentage and keep a neg_free_flag column so
# downstream scripts can report it.
#
# Outputs: real_trace_dataset_lanl.csv, real_trace_dataset_sdsc.csv
#
# Usage:
#   python build_real_trace_datasets.py               # full traces (default)
#   python build_real_trace_datasets.py --max-lines N # smoke-test on first N
#                                                     # data lines per trace

import argparse
import heapq
import os
import random

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, '02_data')

TRACES = [
    # (short name, swf file, output csv)
    ('lanl', 'LANL-CM5-1994-4.1-cln.swf', 'real_trace_dataset_lanl.csv'),
    ('sdsc', 'SDSC-SP2-1998-4.2-cln.swf', 'real_trace_dataset_sdsc.csv'),
]

KEEP_STATUS = (0, 1, -1)   # completed / failed / unknown; drop cancelled+partial


def parse_swf(path, max_lines=None):
    """Parse an SWF file. Returns (capacity, list-of-job-dicts, filter stats)."""
    capacity = None
    max_nodes = None
    jobs = []
    stats = {'total': 0, 'bad_procs': 0, 'bad_runtime': 0,
             'bad_wait': 0, 'bad_status': 0, 'kept': 0}

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        n_data = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(';'):
                # header comments -- pull MaxProcs / MaxNodes
                low = line.lower()
                if low.startswith('; maxprocs:'):
                    capacity = int(line.split(':')[1])
                elif low.startswith('; maxnodes:'):
                    max_nodes = int(line.split(':')[1])
                continue
            if max_lines is not None and n_data >= max_lines:
                break
            n_data += 1
            parts = line.split()
            if len(parts) < 11:
                continue
            stats['total'] += 1
            job_id = int(parts[0])
            submit = int(parts[1])
            wait = int(parts[2])
            run = int(float(parts[3]))
            alloc = int(parts[4])
            req = int(parts[7])
            status = int(parts[10])

            procs = alloc if alloc > 0 else req   # fallback to requested
            if procs <= 0:
                stats['bad_procs'] += 1
                continue
            if run <= 0:
                stats['bad_runtime'] += 1
                continue
            if wait < 0:
                stats['bad_wait'] += 1
                continue
            if status not in KEEP_STATUS:
                stats['bad_status'] += 1
                continue
            stats['kept'] += 1
            jobs.append({'job_id': job_id, 'submit': submit, 'wait': wait,
                         'run': run, 'procs': procs})

    if capacity is None:
        capacity = max_nodes   # fall back to MaxNodes if MaxProcs missing
    if capacity is None:
        raise ValueError(f'No MaxProcs/MaxNodes header found in {path}')
    return capacity, jobs, stats


def replay_trace_features(jobs, capacity):
    """Chronological replay of the recorded schedule.

    For each job, at its submit instant, compute aggregate cluster features
    from the previously submitted jobs only (no lookahead into the job's own
    placement).  O(n log n) via two heaps:
      * queued heap keyed by recorded start time (submitted, not yet started)
      * running heap keyed by recorded end time (started, not yet finished)
    Occupancy convention: a job holds its procs during [start, end).
    """
    jobs = sorted(jobs, key=lambda j: (j['submit'], j['job_id']))

    queued = []          # (start_time, end_time, procs) -- submitted, not started
    running = []         # (end_time, procs)             -- started, not finished
    used_procs = 0
    queued_procs = 0
    neg_free_events = 0

    rows = []
    for job in jobs:
        t = job['submit']

        # 1) move previously-queued jobs whose recorded start has passed
        while queued and queued[0][0] <= t:
            _start, end, procs = heapq.heappop(queued)
            queued_procs -= procs
            heapq.heappush(running, (end, procs))
            used_procs += procs

        # 2) release running jobs whose recorded end has passed ([start,end))
        while running and running[0][0] <= t:
            _end, procs = heapq.heappop(running)
            used_procs -= procs

        # 3) features at the submit instant (self NOT yet in the queue)
        raw_free = capacity - used_procs
        if raw_free < 0:
            neg_free_events += 1
        total_free = max(raw_free, 0)          # clamp (non-dedicated overlap)

        procs = job['procs']
        queue_length = len(queued)
        running_jobs = len(running)
        can_fit_now = int(total_free >= procs)
        fit_ratio = min(total_free / procs, 1.0)
        queue_pressure = queued_procs / (total_free + 1)
        free_frac = total_free / capacity

        wait = job['wait']
        rows.append({
            'job_id': job['job_id'],
            'submit_time': t,
            'job_procs': procs,
            'total_free': total_free,
            'queue_length': queue_length,
            'running_jobs': running_jobs,
            'can_fit_now': can_fit_now,
            'fit_ratio': fit_ratio,
            'queue_pressure': queue_pressure,
            'free_frac': free_frac,
            'neg_free_flag': int(raw_free < 0),
            'run_time': job['run'],
            'wait_time': wait,
            'wait_log_minutes': np.log1p(wait / 60.0),
        })

        # 4) now enqueue this job into the recorded schedule
        start = t + wait
        end = start + job['run']
        heapq.heappush(queued, (start, end, procs))
        queued_procs += procs

    return pd.DataFrame(rows), neg_free_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-lines', type=int, default=None,
                    help='parse only the first N data lines per trace (smoke test)')
    args = ap.parse_args()

    random.seed(42)      # run_index 0 -> seed 42 (repo convention); replay is
    np.random.seed(42)   # deterministic anyway, seeded for uniformity

    for name, swf_file, out_file in TRACES:
        swf_path = os.path.join(DATA_DIR, swf_file)
        out_path = os.path.join(DATA_DIR, out_file)
        print(f'\n=== {name.upper()} : {swf_file} ===')

        capacity, jobs, stats = parse_swf(swf_path, max_lines=args.max_lines)
        print(f'Capacity (MaxProcs from header): {capacity}')
        print(f"Parsed rows          : {stats['total']}")
        print(f"  dropped procs<=0   : {stats['bad_procs']}")
        print(f"  dropped run<=0     : {stats['bad_runtime']}")
        print(f"  dropped wait<0     : {stats['bad_wait']}")
        print(f"  dropped status     : {stats['bad_status']} (not in {KEEP_STATUS})")
        kept_pct = 100.0 * stats['kept'] / max(stats['total'], 1)
        print(f"  KEPT               : {stats['kept']} ({kept_pct:.1f}%)")

        df, neg_events = replay_trace_features(jobs, capacity)
        neg_pct = 100.0 * neg_events / max(len(df), 1)
        print(f'Negative reconstructed free-capacity events (clamped to 0): '
              f'{neg_events} / {len(df)} = {neg_pct:.2f}%')
        if name == 'lanl' and neg_pct > 0:
            print('  (expected: LANL CM-5 header notes many jobs ran in '
                  'non-dedicated mode, so recorded allocations overlap)')

        # sanity: no NaN/inf anywhere
        num = df.select_dtypes(include=[np.number])
        assert not num.isna().any().any(), 'NaN in dataset'
        assert np.isfinite(num.to_numpy(dtype=float)).all(), 'inf in dataset'

        df.to_csv(out_path, index=False, encoding='utf-8')
        print(f'Wrote {out_path}  ({len(df)} rows)')
        print(df[['job_procs', 'total_free', 'queue_length', 'wait_time']]
              .describe().to_string())


if __name__ == '__main__':
    main()

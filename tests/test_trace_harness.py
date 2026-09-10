"""Invariants of the real-trace harness (`04_scheduler/trace_driven_benchmark.py`).

Every headline number in the trace-driven study (`05_results/trace_schedulers/`)
is produced by two functions in that module: `parse_swf_jobs`, which turns a
Parallel Workloads Archive SWF file into the job population, and `simulate` /
`simulate_srpt`, which replay that population under each policy. If either is
wrong, the scheduler ranking is wrong and nothing downstream can detect it —
the simulator has no ground truth to be checked against at run time.

These tests therefore protect two different kinds of correctness:

1.  POPULATION correctness — the right jobs, with the right sizes, and (most
    importantly) with runtime estimates that contain NO information the
    scheduler could not have had at submit time. See the leakage guard,
    `test_missing_estimate_falls_back_to_median_not_true_runtime`.

2.  PHYSICAL correctness — the simulated machine obeys conservation laws: a
    job cannot start before it arrives, cannot use processors that do not
    exist, and cannot finish before it has run. A simulator that violates any
    of these can manufacture an arbitrarily good scheduler.

3.  SHARED-INPUT correctness — the benchmark and `02_data/build_real_trace_
    datasets.py` must read the same trace files and build the same features.
    Both now go through `02_data/swf_io.open_swf`, and the benchmark's
    per-trace model is trained on `replay_trace_features` output rather than on
    a cached CSV, so the two scripts can no longer disagree about their input.

Everything here runs on hand-built job lists of five jobs or fewer and on SWF
files of eight rows or fewer. Nothing reads or writes a tracked artefact.
"""

import gzip
import os

import numpy as np
import pandas as pd
import pytest

# `trace_driven_benchmark` pulls in XGBRegressor at module load for the
# PROACTIVE policies. None of the tests below exercise the ML policies, but the
# import is unconditional, so the whole module is skipped without xgboost.
tdb = pytest.importorskip(
    'trace_driven_benchmark',
    reason='trace_driven_benchmark imports xgboost (XGBRegressor) at module load',
)

# The dataset builder: producer of 02_data/real_trace_dataset_*.csv, and the
# source of the feature replay the benchmark now always runs in memory.
import build_real_trace_datasets as brtd  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: writing miniature SWF files
# ─────────────────────────────────────────────────────────────────────────────

def _swf_row(job_id, submit, wait, run, alloc, req_procs, req_time, status):
    """One SWF data line in the real Standard Workload Format column order.

    Fields the parser reads are at indices 0,1,2,3,4,7,8,10; the rest are
    written as the archive's "not recorded" sentinel -1 so the row is a
    faithful 18-column SWF record rather than a shape the parser happens to
    tolerate.
    """
    fields = [
        job_id,      # 1  job number
        submit,      # 2  submit time
        wait,        # 3  wait time
        run,         # 4  run time
        alloc,       # 5  allocated processors
        -1,          # 6  average CPU time used
        -1,          # 7  used memory
        req_procs,   # 8  requested processors
        req_time,    # 9  requested time  <- the REAL user runtime estimate
        -1,          # 10 requested memory
        status,      # 11 status
        -1, -1, -1, -1, -1, -1, -1,   # 12-18 user/group/exe/queue/part/prec/think
    ]
    return ' '.join(str(f) for f in fields)


def _swf_text(rows, max_procs=64, max_nodes=None):
    header = ['; Version: 2.2', '; Computer: pytest rig']
    if max_nodes is not None:
        header.append(f'; MaxNodes: {max_nodes}')
    header.append(f'; MaxProcs: {max_procs}')
    return '\n'.join(header + list(rows)) + '\n'


def _write_swf(tmp_path, rows, name='mini.swf', **kwargs):
    path = tmp_path / name
    path.write_text(_swf_text(rows, **kwargs), encoding='utf-8')
    return str(path)


# A fixed eight-row trace exercising every branch of the parser's filter.
# Exactly one row is dropped per rejection reason, and the filter is applied in
# the source's own order (procs -> runtime -> wait -> status), so each row here
# violates exactly one rule and is attributable to a single counter.
FILTER_ROWS = [
    # kept: allocated (4) disagrees with requested (99) -> allocated must win
    _swf_row(1, 100, 10, 50, alloc=4, req_procs=99, req_time=100, status=1),
    _swf_row(2, 200, 0, 60, alloc=8, req_procs=8, req_time=200, status=0),
    _swf_row(3, 300, 5, 70, alloc=2, req_procs=2, req_time=300, status=-1),
    # dropped: status 5 == cancelled (never actually occupied the machine)
    _swf_row(4, 400, 0, 80, alloc=4, req_procs=4, req_time=400, status=5),
    # dropped: zero runtime
    _swf_row(5, 500, 0, 0, alloc=4, req_procs=4, req_time=100, status=1),
    # dropped: no processors recorded at all (allocated 0 AND requested 0)
    _swf_row(6, 600, 0, 50, alloc=0, req_procs=0, req_time=100, status=1),
    # dropped: negative wait
    _swf_row(7, 700, -1, 50, alloc=4, req_procs=4, req_time=100, status=1),
    # kept: allocated not recorded (-1) -> fall back to requested (16)
    _swf_row(8, 800, 0, 50, alloc=-1, req_procs=16, req_time=400, status=1),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: driving the simulator and observing what it actually did
# ─────────────────────────────────────────────────────────────────────────────

def _capture_summarize(monkeypatch):
    """Spy on `tdb.summarize` to recover the per-job ground truth.

    `simulate` returns only aggregate metrics, so the physical invariants
    (start >= arrival, capacity never exceeded, every job completed) are not
    observable from its return value. `summarize` is the single point where the
    finished job objects, the integrated utilisation area and the horizon are
    handed over, so wrapping it — without touching the source — exposes exactly
    the state the tests need to audit.
    """
    captured = {}
    original = tdb.summarize

    def spy(completed, util_area, horizon, capacity, policy, window_meta,
            preemptions=0):
        captured['completed'] = list(completed)
        captured['util_area'] = float(util_area)
        captured['horizon'] = float(horizon)
        return original(completed, util_area, horizon, capacity, policy,
                        window_meta, preemptions)

    monkeypatch.setattr(tdb, 'summarize', spy)
    return captured


def _peak_concurrency(jobs):
    """Peak simultaneously-held processors, recomputed from [start, end)."""
    events = []
    for j in jobs:
        events.append((j.start_time, j.num_gpus))
        events.append((j.end_time, -j.num_gpus))
    # At an identical instant the simulator releases completions before it
    # dispatches, so releases (negative delta) must sort first.
    events.sort(key=lambda e: (e[0], e[1]))
    current = peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def _jobs(specs):
    """(job_id, arrival, procs, runtime, est_runtime, measured) -> TraceJobs.

    Sorted by (arrival, id) because `simulate` admits arrivals by walking the
    list once and assumes it is already in arrival order (`make_jobs` sorts it
    in production; the simulator itself does not re-sort defensively).
    """
    jobs = [tdb.TraceJob(jid, arr, procs, run, est, measured)
            for jid, arr, procs, run, est, measured in specs]
    jobs.sort(key=lambda j: (j.arrival_time, j.job_id))
    return jobs


# Five jobs on an 8-processor machine, in seconds. Sized so that the queue
# genuinely blocks (job 2 needs 6 of 8), a small job can backfill behind it
# (job 3 needs 2), and a full-machine job arrives mid-flight (job 4 needs 8).
# Estimates deliberately disagree with true runtimes in both directions so the
# estimate-driven policies take a different path from the oracle ones.
WORKLOAD = [
    # id, arrival, procs, runtime, est_runtime, measured
    (1, 0, 4, 600, 600, True),
    (2, 0, 6, 300, 300, True),
    (3, 0, 2, 480, 1200, True),    # over-estimates by 2.5x
    (4, 180, 8, 240, 240, True),
    (5, 180, 1, 180, 6000, True),  # wildly over-estimates
]
CAPACITY = 8

# The non-ML policies. PROACTIVE / PROACTIVE_EST need a fitted XGBoost model
# and are covered by the degeneracy tests, not here.
NON_ML_POLICIES = ['FCFS', 'FIFO_STRICT', 'SJF_ORACLE', 'SMALLEST_FIRST',
                   'EASY_USEREST']


# ─────────────────────────────────────────────────────────────────────────────
# (a) Trace parsing: the job population
# ─────────────────────────────────────────────────────────────────────────────

def test_capacity_comes_from_the_maxprocs_header(tmp_path):
    """Machine size is read from the trace, never assumed.

    Every offered-load figure and every "does this job fit" decision divides by
    this number, so a wrong capacity silently rescales the entire study. The
    file also carries a *different* MaxNodes value: MaxProcs is the processor
    count and must win, MaxNodes is only a fallback for traces that omit it.
    """
    path = _write_swf(tmp_path, FILTER_ROWS, max_procs=64, max_nodes=8)
    capacity, df, _ = tdb.parse_swf_jobs(path)

    assert capacity == 64
    # Sanity: the parser did not quietly reinterpret MaxNodes as the capacity.
    assert capacity != 8
    assert len(df) > 0


def test_cancelled_jobs_are_dropped_and_completed_failed_unknown_are_kept(tmp_path):
    """Only jobs that really occupied the machine enter the simulation.

    SWF status 5 is "cancelled": the job never ran, so replaying it would
    invent load that the real machine never carried. Statuses 0 (failed),
    1 (completed) and -1 (unknown) all consumed `run_time` seconds of real
    machine time and must be kept — dropping them would understate congestion
    and flatter every policy equally but not equally *much*.
    """
    path = _write_swf(tmp_path, FILTER_ROWS)
    _, df, stats = tdb.parse_swf_jobs(path)

    kept_ids = set(df['job_id'])
    assert 4 not in kept_ids, 'cancelled (status 5) job survived the filter'
    # jobs 1 (status 1), 2 (status 0), 3 (status -1) are the keep-list statuses
    assert {1, 2, 3}.issubset(kept_ids)
    assert stats['bad_status'] == 1


def test_unusable_rows_are_dropped_and_counted(tmp_path):
    """Rows the simulator cannot physically replay are removed, not repaired.

    A zero/negative runtime, a job with no processor count, or a negative wait
    are all corrupt records. Silently coercing them (to 1 second, to 1
    processor, to wait 0) would fabricate data; the parser drops them and
    reports how many, which is what makes the "kept %" line in the run log an
    auditable number rather than a decoration.
    """
    path = _write_swf(tmp_path, FILTER_ROWS)
    _, df, stats = tdb.parse_swf_jobs(path)

    kept_ids = set(df['job_id'])
    assert 5 not in kept_ids and stats['bad_runtime'] == 1   # runtime <= 0
    assert 6 not in kept_ids and stats['bad_procs'] == 1     # procs <= 0
    assert 7 not in kept_ids and stats['bad_wait'] == 1      # wait < 0

    # Accounting must close: every counted row is either kept or attributed to
    # exactly one rejection reason.
    assert stats['total'] == 8
    assert stats['kept'] == len(df) == 4
    dropped = (stats['bad_procs'] + stats['bad_runtime']
               + stats['bad_wait'] + stats['bad_status'])
    assert stats['kept'] + dropped == stats['total']


def test_allocated_procs_preferred_with_requested_as_fallback(tmp_path):
    """Job size is what the machine actually gave, falling back to the ask.

    The two traces disagree about which field they populate (LANL sometimes
    records -1 for requested, SDSC the reverse), so the parser must take
    allocated processors when present and requested only when allocated is
    <= 0. Getting this backwards changes every job's footprint and therefore
    every wait time in the study.
    """
    path = _write_swf(tmp_path, FILTER_ROWS)
    _, df, _ = tdb.parse_swf_jobs(path)
    procs = dict(zip(df['job_id'], df['procs']))

    # job 1: allocated 4, requested 99 -> the allocation is the truth
    assert procs[1] == 4
    # job 8: allocated -1 (not recorded), requested 16 -> fall back to the ask
    assert procs[8] == 16


# ─────────────────────────────────────────────────────────────────────────────
# (b) THE LEAKAGE GUARD
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_estimate_falls_back_to_median_not_true_runtime(tmp_path):
    """A job with no recorded user estimate must NOT be handed its own runtime.

    THIS IS THE MOST IMPORTANT TEST IN THIS MODULE. SWF field 9 is what the
    user typed at submit; -1 means the trace recorded nothing. The only
    information available at that instant is the rest of the trace, so the
    fallback is the trace's MEDIAN requested time.

    If the fallback were ever changed to the job's true runtime, every
    estimate-driven policy (SJF_USEREST, HRRN_USEREST, EASY_USEREST,
    CONS_BF_USEREST, PROACTIVE_EST) would receive a perfect oracle on exactly
    those jobs — 9% of LANL — while still being reported as "deployable".
    Their wait and slowdown numbers would improve for a reason that cannot
    exist in production, and no downstream check would notice: the simulation
    would still be internally consistent, just measuring a scheduler nobody
    can build.

    The row below makes the two candidate fallbacks impossible to confuse: its
    true runtime is 99999 s while the trace median request is 200 s.
    """
    rows = [
        _swf_row(1, 100, 10, 50, alloc=4, req_procs=4, req_time=100, status=1),
        _swf_row(2, 200, 0, 60, alloc=4, req_procs=4, req_time=200, status=1),
        _swf_row(3, 300, 5, 70, alloc=4, req_procs=4, req_time=300, status=1),
        # No user estimate (-1), and a true runtime nothing like the median.
        _swf_row(9, 900, 0, 99999, alloc=4, req_procs=4, req_time=-1, status=1),
    ]
    path = _write_swf(tmp_path, rows, name='leak.swf')
    _, df, stats = tdb.parse_swf_jobs(path)

    assert stats['no_estimate'] == 1, 'the missing estimate was not detected'

    present = df[df['req_time'] > 0]
    expected_median = float(present['req_time'].median())
    assert expected_median == 200.0     # median of {100, 200, 300}

    row = df[df['job_id'] == 9].iloc[0]
    assert row['runtime'] == 99999

    # The fallback is the population median ...
    assert row['est_runtime'] == pytest.approx(expected_median)
    # ... and emphatically NOT the job's own runtime.
    assert row['est_runtime'] != pytest.approx(float(row['runtime'])), (
        'LEAKAGE: a job with no recorded user estimate was given its own true '
        'runtime as est_runtime — every estimate-driven policy now has an '
        'oracle on these jobs and all SJF/EASY numbers are inflated.'
    )

    # Jobs that DO carry an estimate keep it verbatim; the median must not
    # smear over them.
    for jid, expected in ((1, 100.0), (2, 200.0), (3, 300.0)):
        got = df[df['job_id'] == jid].iloc[0]['est_runtime']
        assert got == pytest.approx(expected)


# ─────────────────────────────────────────────────────────────────────────────
# (c) The gzip fallback a fresh clone depends on
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_reads_the_gzipped_copy_when_only_it_exists(tmp_path):
    """`.gitignore` excludes `*.swf`, so a fresh clone has ONLY `*.swf.gz`.

    Every caller passes the plain `.swf` path. If `open_swf`'s fallback broke,
    the entire trace pipeline would fail immediately after `git clone` while
    continuing to work on the author's machine (where the expanded file is
    lying around) — the classic irreproducibility failure this repository
    exists to avoid.
    """
    text = _swf_text(FILTER_ROWS)
    gz_path = tmp_path / 'onlygz.swf.gz'
    with gzip.open(str(gz_path), 'wt', encoding='utf-8') as fh:
        fh.write(text)

    plain_path = str(tmp_path / 'onlygz.swf')
    assert not os.path.exists(plain_path), 'the uncompressed file must be absent'

    capacity, df, stats = tdb.parse_swf_jobs(plain_path)
    assert capacity == 64
    assert stats['kept'] == len(df) == 4
    assert set(df['job_id']) == {1, 2, 3, 8}


def test_dataset_builder_also_reads_the_gzipped_copy_when_only_it_exists(tmp_path):
    """The DATASET BUILDER must survive a fresh clone too, not just the benchmark.

    `02_data/build_real_trace_datasets.py` used a bare `open()` on the plain
    `.swf` name while the benchmark had a gzip fallback, so the builder could
    not run at all on a fresh clone — only on a machine where someone had
    already expanded the archive. Both now go through the same
    `02_data/swf_io.open_swf`, and this test is the mirror image of the one
    above: same trace text, same expectations, other reader.
    """
    text = _swf_text(FILTER_ROWS)
    gz_path = tmp_path / 'builder_onlygz.swf.gz'
    with gzip.open(str(gz_path), 'wt', encoding='utf-8') as fh:
        fh.write(text)

    plain_path = str(tmp_path / 'builder_onlygz.swf')
    assert not os.path.exists(plain_path), 'the uncompressed file must be absent'

    capacity, jobs, stats = brtd.parse_swf(plain_path)
    assert capacity == 64
    assert stats['kept'] == len(jobs) == 4
    assert {j['job_id'] for j in jobs} == {1, 2, 3, 8}


def test_a_gz_path_passed_directly_is_decompressed_not_read_as_text(tmp_path):
    """A `.swf.gz` handed in DIRECTLY must be decompressed, not read as UTF-8.

    `open_swf` tested `os.path.exists(path)` before it looked at the suffix, so
    an existing `.gz` argument fell into the plain-text branch and was decoded
    as UTF-8 with `errors='replace'`. That does not raise: gzip's bytes become
    replacement characters, the parser skips every line that fails to split into
    >= 11 numeric fields, and the caller gets ZERO jobs — which reads downstream
    like a quiet trace rather than like a broken read. The header is gone too,
    so capacity would come back as None and `parse_swf_jobs` would blame the
    file for having no MaxProcs line.

    Nothing in the pipeline passes a `.gz` path today, which is exactly why this
    needs a test: the next caller that does gets silence, not an error.
    """
    text = _swf_text(FILTER_ROWS)
    gz_path = tmp_path / 'direct.swf.gz'
    with gzip.open(str(gz_path), 'wt', encoding='utf-8') as fh:
        fh.write(text)

    # The uncompressed sibling must NOT exist: this is about the .gz argument
    # itself being handled, not about the fallback finding a plain file.
    assert not os.path.exists(str(tmp_path / 'direct.swf'))

    # The reader returns real trace text ...
    with tdb.open_swf(str(gz_path)) as fh:
        first = fh.readline()
    assert first.startswith('; Version: 2.2'), (
        'the .gz path was read undecompressed: the caller received bytes '
        'decoded as UTF-8 instead of the trace')

    # ... and the whole parse succeeds identically to the plain-path case.
    capacity, df, stats = tdb.parse_swf_jobs(str(gz_path))
    assert capacity == 64
    assert stats['kept'] == len(df) == 4
    assert set(df['job_id']) == {1, 2, 3, 8}


def test_a_gz_path_is_decompressed_even_when_a_plain_file_sits_beside_it(tmp_path):
    """The SUFFIX decides, not which file happens to exist.

    The bug was an ordering bug, so the fix must not be an ordering coincidence:
    with BOTH files present, a `.gz` argument must still be decompressed. Here
    the plain sibling holds a different trace (a smaller machine), so reading
    the wrong one is visible in the capacity rather than silent.
    """
    gz_path = tmp_path / 'both.swf.gz'
    with gzip.open(str(gz_path), 'wt', encoding='utf-8') as fh:
        fh.write(_swf_text(FILTER_ROWS, max_procs=64))
    _write_swf(tmp_path, FILTER_ROWS, name='both.swf', max_procs=8)

    gz_capacity, _, _ = tdb.parse_swf_jobs(str(gz_path))
    plain_capacity, _, _ = tdb.parse_swf_jobs(str(tmp_path / 'both.swf'))

    assert gz_capacity == 64, 'the .gz argument was served the plain file'
    assert plain_capacity == 8, 'the fixture must make the two distinguishable'


def test_open_swf_reports_a_missing_gz_path_instead_of_falling_back(tmp_path):
    """An absent `.gz` argument raises; it does not silently find something else.

    A caller that names a compressed trace and gets a FileNotFoundError can fix
    its path. One that gets a different file cannot tell that it did.
    """
    with pytest.raises(FileNotFoundError):
        tdb.open_swf(str(tmp_path / 'nothing-here.swf.gz'))


def test_open_swf_reports_a_missing_trace_instead_of_returning_empty(tmp_path):
    """Neither file present is an error, never a silently empty job set.

    An empty DataFrame here would propagate as "0 jobs, all policies tie",
    which reads like a result rather than a broken input.
    """
    with pytest.raises(FileNotFoundError):
        tdb.open_swf(str(tmp_path / 'absent.swf'))


# ─────────────────────────────────────────────────────────────────────────────
# (d) Simulator physics
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('policy', NON_ML_POLICIES)
def test_simulate_conserves_jobs_time_and_capacity(policy, monkeypatch):
    """The simulated machine obeys conservation laws under every policy.

    Four independent invariants, each of which a broken dispatcher could
    violate while still producing plausible-looking metrics:

      * EVERY job completes. A dispatcher that loses a job silently removes
        its (typically large) wait from the mean.
      * No job starts before it arrives. Starting early is time travel and is
        indistinguishable from a policy that "predicts" the future.
      * Peak concurrent processors never exceed capacity — recomputed here
        from the finished jobs' [start, end) intervals rather than trusted
        from the allocator's own bookkeeping. Over-packing manufactures
        throughput that the real machine could not deliver.
      * Wait = turnaround - runtime is non-negative for every job, and mean
        bounded slowdown is >= 1 by its definition (max(ratio, 1)). A negative
        wait or a slowdown below 1 means a job finished before it could have
        run, i.e. the metric is not measuring what it is named after.
    """
    captured = _capture_summarize(monkeypatch)
    jobs = _jobs(WORKLOAD)
    out = tdb.simulate(jobs, policy, CAPACITY, None, {})

    finished = captured['completed']

    # every job completes, and the measured accounting agrees with the input
    assert len(finished) == len(WORKLOAD)
    assert {j.job_id for j in finished} == {spec[0] for spec in WORKLOAD}
    assert out['jobs_measured'] == sum(1 for spec in WORKLOAD if spec[5])

    for j in finished:
        assert j.start_time is not None and j.end_time is not None
        # no job starts before it arrives
        assert j.start_time >= j.arrival_time, (
            f'{policy}: job {j.job_id} started at {j.start_time} but arrived '
            f'at {j.arrival_time}')
        # non-preemptive: a job runs for exactly its runtime, once
        assert j.end_time == j.start_time + j.runtime
        # wait, as the metrics define it, is non-negative
        assert j.end_time - j.arrival_time - j.runtime >= 0

    # capacity is never exceeded at any instant
    assert _peak_concurrency(finished) <= CAPACITY, (
        f'{policy}: peak concurrent processors exceeded capacity {CAPACITY}')

    # bounded slowdown is floored at 1 by construction; a mean below 1 would
    # mean the floor was lost
    assert out['mean_bounded_slowdown'] >= 1.0
    assert out['p95_bounded_slowdown'] >= 1.0
    assert out['mean_wait'] >= 0.0
    assert out['median_wait'] >= 0.0
    assert 0.0 <= out['gpu_util'] <= 1.0


def test_warmup_jobs_occupy_the_machine_but_are_excluded_from_the_metrics(monkeypatch):
    """Warm-up load is real for the scheduler and invisible to the report.

    Each window simulates WARMUP_DAYS before the measured period so no policy
    is scored on a conveniently empty cluster. That only works if warm-up jobs
    (measured=False) do two things at once: hold processors (so measured jobs
    experience realistic contention) and stay out of the reported means (so the
    cold-start period is not scored).

    Here a 10-processor warm-up job runs 0..6000 s and a 10-processor measured
    job arrives at t=300 s. The measured job cannot start until 6000 s, so it
    waits 5700 s — entirely because of a job that never appears in the metrics.
    Remove the warm-up job and the same measured job waits zero.
    """
    warmup_job = (90, 0, 10, 6000, 6000, False)
    measured_job = (91, 300, 10, 600, 600, True)

    captured = _capture_summarize(monkeypatch)
    with_warmup = tdb.simulate(_jobs([warmup_job, measured_job]), 'FCFS', 10,
                               None, {})
    finished = {j.job_id: j for j in captured['completed']}

    # the warm-up job really ran ...
    assert finished[90].start_time == 0
    assert finished[90].end_time == 6000
    # ... and really delayed the measured job ...
    assert finished[91].start_time == 6000
    # ... but only the measured job is counted.
    assert with_warmup['jobs_measured'] == 1
    assert with_warmup['mean_wait'] == pytest.approx(5700.0)

    without_warmup = tdb.simulate(_jobs([measured_job]), 'FCFS', 10, None, {})
    assert without_warmup['jobs_measured'] == 1
    assert without_warmup['mean_wait'] == pytest.approx(0.0)

    # The contrast is the point: identical measured population, different
    # measured wait, because the warm-up job occupied the machine.
    assert with_warmup['mean_wait'] > without_warmup['mean_wait']


# ─────────────────────────────────────────────────────────────────────────────
# (e) Preemptive SRPT and its checkpoint cost
# ─────────────────────────────────────────────────────────────────────────────

def test_simulate_srpt_completes_every_job_and_charges_for_preemption(monkeypatch):
    """SRPT is a bound, not a free lunch: preemption must cost something.

    SRPT_ORACLE is reported as the strongest preemptive classical baseline. If
    preemption were free, it would be an unattainable bound presented as a
    comparable policy. The model charges PREEMPT_OVERHEAD seconds of extra work
    per preemption (checkpoint + restore), so the total processor-seconds the
    machine actually burns must EXCEED the pure work of the job set — that
    excess is the honesty of the bound.

    Scenario (capacity 4): a 4-processor 100 s job starts at t=0; at t=10 a
    4-processor 5 s job arrives with less remaining work, so SRPT preempts the
    long job exactly once.

      pure work            = 4*100 + 4*5              = 420 processor-seconds
      lower bound on span  = 420 / 4                  = 105 s
      burned with overhead = 420 + 4*PREEMPT_OVERHEAD = 660 processor-seconds
    """
    captured = _capture_summarize(monkeypatch)
    jobs = _jobs([
        (1, 0, 4, 100, 100, True),    # long job, holds the whole machine
        (2, 10, 4, 5, 5, True),       # short job arrives and jumps it
    ])
    capacity = 4
    pure_work = sum(j.num_gpus * j.runtime for j in jobs)

    out = tdb.simulate_srpt(jobs, capacity, {})
    finished = captured['completed']

    # every job completes, with no remaining work left over
    assert len(finished) == len(jobs)
    assert out['jobs_measured'] == len(jobs)
    for j in finished:
        assert j.end_time is not None
        assert j.remaining <= 1e-9
        assert j.start_time >= j.arrival_time

    # preemption count is a count: never negative, and this scenario forces one
    assert out['preemptions'] >= 0
    assert out['preemptions'] >= 1, 'the checkpoint path was never exercised'

    # No schedule can finish the work faster than capacity allows ...
    makespan = captured['horizon']
    assert makespan >= pure_work / capacity

    # ... and with a checkpoint penalty the machine burns strictly MORE
    # processor-seconds than the jobs' own work, because preempted jobs have to
    # redo `overhead` seconds each time they are stopped.
    assert captured['util_area'] >= pure_work
    assert captured['util_area'] > pure_work, (
        'preemption was free: SRPT_ORACLE is being reported as a policy while '
        'behaving as an unattainable bound')
    assert captured['util_area'] == pytest.approx(
        pure_work + out['preemptions'] * 4 * tdb.PREEMPT_OVERHEAD)


def test_simulate_srpt_charges_nothing_when_nothing_is_preempted(monkeypatch):
    """The overhead is charged per preemption, not per job or per dispatch.

    Complements the test above: with a single job there is nothing to preempt,
    so the burned processor-seconds must equal the pure work exactly. Together
    the two tests pin the penalty to the preemption events themselves.
    """
    captured = _capture_summarize(monkeypatch)
    jobs = _jobs([(1, 0, 2, 100, 100, True)])
    out = tdb.simulate_srpt(jobs, 4, {})

    assert out['preemptions'] == 0
    assert captured['util_area'] == pytest.approx(2 * 100)
    assert out['mean_wait'] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# (f) FIFO_STRICT really blocks; FCFS really does not
# ─────────────────────────────────────────────────────────────────────────────

def test_fifo_strict_head_blocks_where_fcfs_runs_past_the_head(monkeypatch):
    """The two FIFO baselines must genuinely differ, or the comparison is a lie.

    The repository's historical "FIFO" baseline is FCFS + unrestricted
    first-fit: it scans the whole queue and starts anything that fits, so a
    blocked head does not stop later jobs. FIFO_STRICT is textbook FIFO: the
    head blocks everything behind it. FIFO_STRICT is the honest reference
    against which backfill (EASY, conservative) is supposed to show a gain — if
    the `break` in the strict branch ever degraded to a `continue`, the two
    policies would silently become the same policy and every "backfill helps by
    X%" claim would be measured against the wrong baseline.

    Scenario (capacity 10): job 1 takes 6 processors, leaving 4 free. Job 2 —
    the queue head after job 1 starts — needs 8 and cannot fit. Job 3 needs
    only 3 and COULD run right now.
    """
    workload = [
        (1, 0, 6, 6000, 6000, True),   # starts immediately, leaves 4 free
        (2, 0, 8, 3000, 3000, True),   # blocked head: needs 8, only 4 free
        (3, 0, 3, 600, 600, True),     # would fit in the 4 free processors
    ]
    capacity = 10

    captured = _capture_summarize(monkeypatch)
    fcfs = tdb.simulate(_jobs(workload), 'FCFS', capacity, None, {})
    fcfs_starts = {j.job_id: j.start_time for j in captured['completed']}

    captured = _capture_summarize(monkeypatch)
    strict = tdb.simulate(_jobs(workload), 'FIFO_STRICT', capacity, None, {})
    strict_starts = {j.job_id: j.start_time for j in captured['completed']}

    # Both policies start the first job immediately — the setup is identical.
    assert fcfs_starts[1] == 0 and strict_starts[1] == 0

    # FCFS scans past the blocked head and starts the small job at once.
    assert fcfs_starts[3] == 0, 'FCFS failed to run past a blocked head'

    # FIFO_STRICT does not: job 3 must wait for the head (job 2) to run and
    # finish, i.e. until 6000 + 3000.
    assert strict_starts[3] == 9000, 'FIFO_STRICT did not head-block'
    assert strict_starts[3] > fcfs_starts[3]

    # The head itself is treated identically; only the jobs behind it differ.
    assert fcfs_starts[2] == strict_starts[2] == 6000

    # And the difference shows up in the reported metric, which is what the
    # study actually compares.
    assert strict['mean_wait'] > fcfs['mean_wait']


# ─────────────────────────────────────────────────────────────────────────────
# (g) The in-memory feature rebuild the retrained model is fitted on
# ─────────────────────────────────────────────────────────────────────────────

# A four-job recorded schedule on an 8-processor machine, chosen so the replay
# is exercised without ever oversubscribing the machine (no clamping) and so
# every interesting state appears at least once: a fully free cluster, a job
# that does not fit, a non-empty queue with real pressure, and a completion
# retired in the same step as a start.
#
#   job 1  [0, 600)  4 procs        job 3  [300, 540)  4 procs
#   job 2  [0, 300)  2 procs        job 4  [600, 720)  2 procs
#
# Peak concurrency is 8 == capacity over [300, 540).
REPLAY_JOBS = [
    {'job_id': 1, 'submit': 0,   'wait': 0,   'run': 600, 'procs': 4},
    {'job_id': 2, 'submit': 0,   'wait': 0,   'run': 300, 'procs': 2},
    {'job_id': 3, 'submit': 120, 'wait': 180, 'run': 240, 'procs': 4},
    {'job_id': 4, 'submit': 200, 'wait': 400, 'run': 120, 'procs': 2},
]
REPLAY_CAPACITY = 8


def test_in_memory_rebuild_supplies_exactly_the_columns_the_model_is_trained_on():
    """`train_trace_model` no longer has a cached CSV to fall back to.

    The benchmark used to read `02_data/real_trace_dataset_<trace>.csv` when it
    happened to exist and replay the trace only otherwise. That file is
    gitignored and nothing validated it against the current trace or capacity,
    so a stale copy on one machine silently trained a different model — and
    published different numbers — from a fresh clone. The cached branch is gone,
    which makes `replay_trace_features` the single source of training features
    and makes its output schema load-bearing: a renamed or dropped column here
    is a KeyError at training time, and a *silently* changed one is worse.
    """
    df, neg_events = brtd.replay_trace_features(REPLAY_JOBS, REPLAY_CAPACITY)

    # One row per job, in chronological order -- the replay sorts by
    # (submit, job_id) and must not drop or duplicate anyone.
    assert len(df) == len(REPLAY_JOBS)
    assert list(df['job_id']) == [1, 2, 3, 4]

    # Exactly what train_trace_model selects: the eight base features, the
    # chronological-split key, the join key for the estimate variant, and the
    # target.
    required = set(tdb.BASE_FEATURES) | {'job_id', 'submit_time', 'wait_time'}
    missing = required - set(df.columns)
    assert not missing, f'the model would be trained without {sorted(missing)}'

    # The recorded schedule fits inside the machine, so the negative-free clamp
    # (LANL's non-dedicated-mode overlap) is NOT what this test is measuring.
    assert neg_events == 0
    assert (df['neg_free_flag'] == 0).all()

    # Targets and split keys round-trip from the input, unaltered.
    assert list(df['submit_time']) == [j['submit'] for j in REPLAY_JOBS]
    assert list(df['wait_time']) == [j['wait'] for j in REPLAY_JOBS]

    rows = {int(r['job_id']): r for _, r in df.iterrows()}

    # Job 1 submits into an empty cluster.
    assert rows[1]['total_free'] == REPLAY_CAPACITY
    assert rows[1]['running_jobs'] == 0 and rows[1]['queue_length'] == 0
    assert rows[1]['can_fit_now'] == 1 and rows[1]['free_frac'] == 1.0

    # Job 3 asks for 4 processors with only 2 free: it demonstrably does not
    # fit, and fit_ratio is the un-clipped shortfall.
    assert rows[3]['total_free'] == 2
    assert rows[3]['can_fit_now'] == 0
    assert rows[3]['fit_ratio'] == pytest.approx(0.5)

    # Job 4 submits while job 3 is still queued -- the only instant with a
    # non-empty queue, so it is the only one that exercises queue_pressure
    # (queued processors that are NOT the scored job, over free + 1).
    assert rows[4]['queue_length'] == 1
    assert rows[4]['running_jobs'] == 2
    assert rows[4]['total_free'] == 2
    assert rows[4]['queue_pressure'] == pytest.approx(4.0 / 3.0)


def test_in_memory_rebuild_is_deterministic():
    """Two replays of one job list must be bit-identical.

    With the cached CSV removed, the features are rebuilt on every run of the
    benchmark, so reproducibility of the published scheduler numbers now rests
    on this replay being a pure function of (jobs, capacity). It uses heaps
    keyed on (start, end, procs) and (end, procs); if two entries ever tied on
    a key that does not fully order them, the pop order — and therefore the
    features — could vary between runs. `check_exact=True` because a float
    tolerance would hide exactly that.
    """
    first, first_neg = brtd.replay_trace_features(REPLAY_JOBS, REPLAY_CAPACITY)
    second, second_neg = brtd.replay_trace_features(REPLAY_JOBS, REPLAY_CAPACITY)

    assert first_neg == second_neg
    pd.testing.assert_frame_equal(first, second, check_exact=True)


# ─────────────────────────────────────────────────────────────────────────────
# (h) The deleted cached-features branch must stay deleted
# ─────────────────────────────────────────────────────────────────────────────

# The exact filenames the removed branch read. Both trace keys are planted so a
# resurrection keyed on either name picks up the trap, and the bare name is
# planted for a resurrection that drops the key altogether (which it must, now
# that `train_trace_model` has no trace_key parameter to key on).
_CACHE_NAMES = ('real_trace_dataset_sdsc.csv', 'real_trace_dataset_lanl.csv',
                'real_trace_dataset.csv')

# Six jobs on a 16-processor machine, in submit order. The split below puts the
# first four before it and the last two after, so the pre-split row count (4) is
# a number the planted cache cannot accidentally match.
CACHE_TRAP_ROWS = [
    _swf_row(1, 0,    0,   600, alloc=4,  req_procs=4,  req_time=600, status=1),
    _swf_row(2, 100,  0,   300, alloc=2,  req_procs=2,  req_time=300, status=1),
    _swf_row(3, 200,  180, 240, alloc=8,  req_procs=8,  req_time=240, status=1),
    _swf_row(4, 300,  400, 120, alloc=2,  req_procs=2,  req_time=120, status=1),
    _swf_row(5, 5000, 0,   600, alloc=4,  req_procs=4,  req_time=600, status=1),
    _swf_row(6, 5100, 60,  300, alloc=2,  req_procs=2,  req_time=300, status=1),
]
CACHE_TRAP_SPLIT = 1000        # jobs 1-4 are pre-split; jobs 5-6 are not


def _plant_wrong_cache(directory, n_rows):
    """Write a syntactically perfect, semantically WRONG cached feature file.

    Same column names `train_trace_model` selects, so a resurrected cached
    branch would load it without complaint; different row count and absurd
    feature values, so using it is impossible to miss.
    """
    planted = pd.DataFrame({
        'job_id': range(1000, 1000 + n_rows),
        'submit_time': [0] * n_rows,              # all "pre-split"
        'job_procs': [999] * n_rows,
        'total_free': [-777] * n_rows,
        'queue_length': [555] * n_rows,
        'running_jobs': [444] * n_rows,
        'can_fit_now': [0] * n_rows,
        'fit_ratio': [9.9] * n_rows,
        'queue_pressure': [123.4] * n_rows,
        'free_frac': [-5.0] * n_rows,
        'neg_free_flag': [1] * n_rows,
        'run_time': [1] * n_rows,
        'wait_time': [86400] * n_rows,            # a day of wait, on every row
        'wait_log_minutes': [7.3] * n_rows,
        'est_runtime': [1.0] * n_rows,
    })
    for name in _CACHE_NAMES:
        planted.to_csv(os.path.join(str(directory), name), index=False)
    return planted


def _expected_pre_split_rows(capacity, df, split_time):
    """The in-memory replay's own answer, computed the way the module does."""
    jobs = [{'job_id': int(r.job_id), 'submit': int(r.submit),
             'wait': int(r.recorded_wait), 'run': int(r.runtime),
             'procs': int(r.procs)} for r in df.itertuples(index=False)]
    replayed, _ = brtd.replay_trace_features(jobs, capacity)
    return int((replayed['submit_time'] < split_time).sum())


def test_train_trace_model_ignores_a_planted_cached_feature_file(tmp_path,
                                                                 monkeypatch):
    """INVARIANT: `train_trace_model` reads the trace it was handed, never a
    file lying around in 02_data.

    An earlier version loaded `02_data/real_trace_dataset_<trace>.csv` whenever
    that file happened to exist and replayed the trace only otherwise. The file
    is gitignored and nothing validated it against the current trace or
    capacity, so a developer holding a stale copy trained a different model --
    and published different numbers -- from anyone on a fresh clone. The branch
    was deleted, but nothing FAILED if it came back: the other tests in this
    module call `replay_trace_features` directly and none of them called
    `train_trace_model` at all, so a resurrected cache branch left the suite
    green.

    This test is the trap. A cache file with the right columns and wrong
    contents sits in the directory the branch used to read, and the model must
    come out of the six-job trace regardless: same training-row count as the
    in-memory replay, and predictions identical to a run with no file planted.
    Row count alone would not be enough -- a cache with a coincidentally equal
    length would slip through -- so the trained model is compared too.
    """
    path = _write_swf(tmp_path, CACHE_TRAP_ROWS, name='cachetrap.swf',
                      max_procs=16)
    capacity, df, _ = tdb.parse_swf_jobs(path)
    assert capacity == 16 and len(df) == len(CACHE_TRAP_ROWS)

    expected_rows = _expected_pre_split_rows(capacity, df, CACHE_TRAP_SPLIT)
    assert expected_rows == 4, 'the fixture must straddle the split'

    # Baseline: train with an EMPTY data directory, i.e. nothing to be tempted by.
    clean_dir = tmp_path / 'clean_data'
    clean_dir.mkdir()
    monkeypatch.setattr(tdb, 'DATA_DIR', str(clean_dir))
    monkeypatch.setattr(brtd, 'DATA_DIR', str(clean_dir))
    clean_model, clean_rows = tdb.train_trace_model(
        CACHE_TRAP_SPLIT, df, False, capacity)
    assert clean_rows == expected_rows

    # Now plant the trap and train again from the identical trace.
    trap_dir = tmp_path / 'trap_data'
    trap_dir.mkdir()
    planted = _plant_wrong_cache(trap_dir, n_rows=37)
    assert len(planted) != expected_rows, 'the trap must be distinguishable'
    monkeypatch.setattr(tdb, 'DATA_DIR', str(trap_dir))
    monkeypatch.setattr(brtd, 'DATA_DIR', str(trap_dir))
    for name in _CACHE_NAMES:
        assert os.path.exists(os.path.join(str(trap_dir), name))

    trap_model, trap_rows = tdb.train_trace_model(
        CACHE_TRAP_SPLIT, df, False, capacity)

    assert trap_rows == expected_rows, (
        f'trained on {trap_rows} rows with a cache file present but '
        f'{expected_rows} without it: the cached-features branch is back, and '
        f'the published model depends on whatever CSV is lying around in '
        f'02_data')

    # And not merely the same COUNT: the same model. A cache whose length
    # happened to match would still change every prediction.
    probe = np.array([[4.0, 8.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
                      [8.0, 2.0, 3.0, 2.0, 0.0, 0.25, 2.0, 0.125]])
    np.testing.assert_allclose(trap_model.predict(probe),
                               clean_model.predict(probe), rtol=1e-9, atol=0,
                               err_msg='a planted cache file changed the '
                                       'trained model')

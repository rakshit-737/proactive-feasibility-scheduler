"""The repository's HEADLINE RESEARCH CLAIM, turned into executable assertions.

THE CLAIM (04_scheduler/ranking_degeneracy.py, RESULTS.md v3.4, CHANGELOG.md)
----------------------------------------------------------------------------
The "proactive" scheduler ranks the waiting queue by a learned wait-time score.
At any single dispatch instant every queued job observes the SAME cluster: total
free capacity, queue length, running-job count, fragmentation, and so on are
properties of the machine, not of the job. The only feature that can differ
between two jobs waiting side by side is their requested size (and the handful
of features that are themselves deterministic functions of that size once the
cluster state is fixed). Therefore, holding the instant fixed,

    predicted_wait(job | state) = g_state(job.num_gpus)

is a function of requested SIZE ALONE. Two consequences follow, and both are the
project's advertised findings:

  * two co-queued jobs with equal size receive byte-identical feature rows and
    hence identical scores (feature_variation.csv:
    equal_size_diff_pred_violations == 0 across every setting); and
  * when every queued job has the same size the learned policy cannot reorder
    anything, so it collapses to the tie-break order (arrival, then id) -- i.e.
    to FCFS. The ML pipeline buys nothing a one-line sort key would not.

If any of that ever stops holding, the paper's central claim is false and the
tests in this module fail. These tests also make honest the CHANGELOG's
previously-uncommitted assertions about property tests.

Every test here is fast (a handful of jobs, no simulation loop, no benchmark)
and pure (no artefact is written; artefact READS go through require()).
"""

import numpy as np
import pytest

# Repo-relative path to the trained model. Importing multi_scheduler_benchmark
# opens this at module import time, so we must require() it before importing.
_MODEL_ARTEFACT = '03_models/wait_model_v2.pkl'

# The seven features that describe ONLY the cluster, never the job. Under the
# claim these are identical for every queued job at a fixed instant, so they can
# shift every score by the same amount but can never reorder the queue. This is
# structural knowledge of 04_scheduler/multi_scheduler_benchmark.get_features
# and is what feature_variation.csv independently measured as "0.0% varying".
# queue_pressure is NOT here: since the train/serve-skew fix it subtracts the
# scored job's own size, so it varies across the queue — but only as a
# deterministic function of size given the state, which preserves degeneracy.
NON_SIZE_FEATURES = frozenset({
    'total_free', 'queue_length', 'running_jobs', 'max_free_node',
    'variance_free', 'fragmentation', 'avg_free_per_node',
})


def _varied_synthetic_state(make_job, make_cluster):
    """A cluster + queue engineered so that EVERY size-derived feature actually
    takes more than one value across the queue.

    An 8-node cluster with only [4, 1, 0, ...] free (5 free GPUs total) and job
    sizes spanning {1, 2, 4, 8} forces can_fit_now (size 8 does not fit),
    gpu_fit_ratio (5/8 < 1) and node_availability (nodes able to hold the job
    range over 2, 1, 0) to differ across rows. That isolation is what lets the
    constant columns be EXACTLY the seven non-size features -- if a size-derived
    feature were accidentally constant here the count would not match, so the
    test would (correctly) refuse to pass vacuously.

    The queue deliberately contains TWO size-4 jobs (ids 2 and 3) so the
    equal-size-identical-row property has something to bite on.
    """
    cluster = make_cluster(capacity=32, num_nodes=8)
    cluster.nodes = [4, 1, 0, 0, 0, 0, 0, 0]          # 5 GPUs free, fragmented
    queue = [
        make_job(job_id=0, arrival_time=10, num_gpus=1),
        make_job(job_id=1, arrival_time=20, num_gpus=2),
        make_job(job_id=2, arrival_time=30, num_gpus=4),
        make_job(job_id=3, arrival_time=40, num_gpus=4),   # same size as id 2
        make_job(job_id=4, arrival_time=50, num_gpus=8),
    ]
    return cluster, queue


def _feature_matrix(msb, cluster, queue):
    return np.array([msb.get_features(j, cluster, queue, running=[]) for j in queue])


# ─────────────────────────────────────────────────────────────────────────────
# (a) Equal size => identical features => identical score (the real model)
# ─────────────────────────────────────────────────────────────────────────────

def test_equal_size_jobs_get_identical_features_and_scores(wait_model, make_job,
                                                           make_cluster):
    """INVARIANT: at a fixed dispatch instant, two queued jobs with the same
    requested size produce byte-identical feature vectors and therefore
    identical predicted-wait scores.

    This is the atomic fact the whole degeneracy result rests on. If it breaks,
    the learned score is NOT a pure function of size and the paper's central
    claim is wrong. Uses the actual trained model (via the wait_model fixture,
    which skips on a fresh clone) and the production get_features.
    """
    from conftest import require
    require(_MODEL_ARTEFACT, 'trained wait model')
    import multi_scheduler_benchmark as msb

    # The model's training feature order must equal the order get_features emits,
    # otherwise "column 0 is size" (and every claim built on it) is meaningless.
    assert list(wait_model['features']) == list(msb.FEATURES)

    cluster, queue = _varied_synthetic_state(make_job, make_cluster)
    x = _feature_matrix(msb, cluster, queue)

    # jobs 2 and 3 both request 4 GPUs -> their rows must be identical
    row_of = {j.job_id: x[i] for i, j in enumerate(queue)}
    assert np.array_equal(row_of[2], row_of[3]), (
        'two equal-size jobs produced different feature rows: the score is not '
        'a function of size alone')

    model = wait_model['model']
    pred = model.predict(x)
    pred_of = {j.job_id: float(pred[i]) for i, j in enumerate(queue)}
    assert pred_of[2] == pred_of[3], (
        'equal-size jobs received different scores -- '
        'equal_size_diff_pred_violations should be 0')

    # And the model must genuinely be a function of size: distinct sizes here
    # should not all collapse to one score (else the test could pass vacuously
    # on a constant predictor).
    assert len({round(pred_of[i], 9) for i in (0, 1, 4)}) > 1


# ─────────────────────────────────────────────────────────────────────────────
# (b) Non-size features are constant across the queue; count matches the CSV
# ─────────────────────────────────────────────────────────────────────────────

def test_non_size_features_are_constant_across_the_queue(wait_model, make_job,
                                                         make_cluster):
    """INVARIANT: exactly the cluster-describing features take a single value for
    every job in the queue; only size-derived features vary. This is the
    mechanism behind degeneracy -- constant columns cannot reorder a ranking,
    and the varying columns are all deterministic functions of requested size
    given the (shared) cluster state.

    We identify the constant columns EMPIRICALLY (column ptp == 0 across the
    queue) and assert they are exactly the seven non-size features. When
    05_results/degeneracy/feature_variation.csv exists we additionally
    cross-check that the count and the identity match what that artefact
    reported as 0.0%-varying for the synthetic setting; otherwise we assert the
    structural property directly.
    """
    from conftest import require
    require(_MODEL_ARTEFACT, 'trained wait model')
    import multi_scheduler_benchmark as msb

    cluster, queue = _varied_synthetic_state(make_job, make_cluster)
    x = _feature_matrix(msb, cluster, queue)

    ptp = x.max(axis=0) - x.min(axis=0)
    constant = {name for name, spread in zip(msb.FEATURES, ptp)
                if abs(spread) <= 1e-12}
    varying = {name for name, spread in zip(msb.FEATURES, ptp)
               if abs(spread) > 1e-12}

    # structural claim: the constant columns are exactly the non-size features
    assert constant == NON_SIZE_FEATURES, (
        f'constant columns {sorted(constant)} != non-size features '
        f'{sorted(NON_SIZE_FEATURES)}')
    # sanity: the size-derived columns really did vary in this state
    assert 'job_gpu' in varying and 'can_fit_now' in varying \
        and 'gpu_fit_ratio' in varying and 'node_availability' in varying \
        and 'queue_pressure' in varying

    # cross-check against the committed artefact, if present
    import csv
    import os
    from conftest import PROJECT_ROOT
    csv_path = os.path.join(PROJECT_ROOT, '05_results', 'degeneracy',
                            'feature_variation.csv')
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as f:
            zero_in_csv = {
                row['feature'] for row in csv.DictReader(f)
                if row['setting'].startswith('synthetic')
                and float(row['pct_instants_varying_across_queue']) == 0.0
            }
        assert zero_in_csv == constant, (
            f'feature_variation.csv reports 0.0%-varying = {sorted(zero_in_csv)} '
            f'but the live feature builder makes {sorted(constant)} constant')
        assert len(zero_in_csv) == 7
    else:
        # No artefact (fresh clone): the structural fact still holds directly.
        assert len(constant) == 7


# ─────────────────────────────────────────────────────────────────────────────
# (c) Consequence: an all-equal-size queue degenerates to FCFS
# ─────────────────────────────────────────────────────────────────────────────

def test_equal_size_queue_degenerates_to_fcfs(wait_model, make_job, make_cluster):
    """INVARIANT: when every queued job requests the same size, the learned
    policy's induced order is exactly the tie-break order (arrival, then id) --
    the ML scheduler collapses to FCFS and reorders nothing.

    This is the headline consequence: on a same-size queue the trained model,
    the SHAP explainer, the drift monitor and the whole pipeline reduce to a
    one-line `sort by (arrival, id)`. We drive the REAL production ranking
    (msb.rank_queue with scheduler='proactive', which calls the model) rather
    than reimplementing it, and hand it a deliberately shuffled arrival order so
    "already sorted" cannot make the test pass by accident.
    """
    from conftest import require
    require(_MODEL_ARTEFACT, 'trained wait model')
    import multi_scheduler_benchmark as msb

    cluster = make_cluster(capacity=32, num_nodes=8)
    cluster.nodes = [4, 1, 0, 0, 0, 0, 0, 0]
    # every job requests 4 GPUs; arrivals/ids intentionally out of order
    queue = [
        make_job(job_id=7, arrival_time=30, num_gpus=4),
        make_job(job_id=2, arrival_time=5, num_gpus=4),
        make_job(job_id=9, arrival_time=12, num_gpus=4),
        make_job(job_id=4, arrival_time=5, num_gpus=4),   # ties id-2 on arrival
        make_job(job_id=1, arrival_time=40, num_gpus=4),
    ]

    ordered = msb.rank_queue(queue, t=50, cluster=cluster, running=[],
                             scheduler='proactive', nn_model=None)
    fcfs = sorted(queue, key=lambda j: (j.arrival_time, j.job_id))

    assert [j.job_id for j in ordered] == [j.job_id for j in fcfs], (
        'a same-size queue was NOT ordered by (arrival, id): the learned policy '
        'did not degenerate to FCFS as the claim requires')
    # explicit: the FCFS order here is 2, 4, 9, 7, 1 (arrival then id tie-break)
    assert [j.job_id for j in ordered] == [2, 4, 9, 7, 1]


# ─────────────────────────────────────────────────────────────────────────────
# (d) Same property for the trace feature builder tdb.build_feature_matrix
# ─────────────────────────────────────────────────────────────────────────────

def test_trace_feature_builder_equal_size_rows_identical(make_job, make_cluster):
    """INVARIANT: the real-trace feature builder shares the degeneracy property
    -- two equal-size jobs get identical (size-only) feature rows.

    The 8-feature per-trace vector is a different code path (build_real_trace_
    datasets reconstruction), so the claim "holds in all three substrates"
    (CHANGELOG v3.2) needs its own check. Importing trace_driven_benchmark pulls
    in xgboost, so skip cleanly where xgboost is unavailable.
    """
    pytest.importorskip('xgboost', reason='trace_driven_benchmark imports xgboost')
    import trace_driven_benchmark as tdb

    cluster = make_cluster(capacity=128, num_nodes=1)
    cluster.nodes = [40]                       # 40 of 128 processors free
    queue = [
        make_job(job_id=0, arrival_time=0, num_gpus=8, runtime=100,
                 est_runtime=200),
        make_job(job_id=1, arrival_time=10, num_gpus=64, runtime=100,
                 est_runtime=50),
        make_job(job_id=2, arrival_time=20, num_gpus=8, runtime=100,
                 est_runtime=999),           # same size as id 0, different est
        make_job(job_id=3, arrival_time=30, num_gpus=3, runtime=100,
                 est_runtime=10),
    ]

    rows = tdb.build_feature_matrix(queue, cluster, running=[], with_est=False)
    # ids 0 and 2 both request 8 processors -> identical rows (size only)
    assert np.array_equal(rows[0], rows[2]), (
        'trace feature builder gave equal-size jobs different rows')

    # The trace vector's constant columns are its four cluster-only features;
    # job_procs, can_fit_now, fit_ratio and queue_pressure are size-derived and
    # vary. Both the trace and (since the train/serve-skew fix) the synthetic
    # builder subtract the scored job's own size from queue_pressure, matching
    # the training rows, which snapshot each job before it joins the queue.
    ptp = rows.max(axis=0) - rows.min(axis=0)
    constant = {n for n, s in zip(tdb.BASE_FEATURES, ptp) if abs(s) <= 1e-12}
    assert constant == {'total_free', 'queue_length', 'running_jobs', 'free_frac'}

    # CONTROL: add the user estimate as a real per-job feature and the identity
    # breaks -- proving the property above is about SIZE, not about the rows
    # being trivially constant. (ids 0 and 2 differ only in est_runtime.)
    rows_est = tdb.build_feature_matrix(queue, cluster, running=[], with_est=True)
    assert not np.array_equal(rows_est[0], rows_est[2])


# ─────────────────────────────────────────────────────────────────────────────
# (e) Guard for the future: a genuine per-job feature WOULD break degeneracy
# ─────────────────────────────────────────────────────────────────────────────

def test_a_true_per_job_feature_would_break_degeneracy(wait_model, make_job,
                                                       make_cluster):
    """INVARIANT (load-bearing guard): the equal-size-identical-score property
    is a consequence of the feature SET, not a tautology. If a feature that is
    NOT a function of size were added -- one distinct value per job -- then two
    equal-size jobs would stop being indistinguishable and any score function of
    the full row would separate them.

    This documents exactly what a NON-degenerate feature set must look like, and
    proves test (a) is not vacuous: it fails today only because no such feature
    exists. If someone adds a genuinely per-job feature (job age, user history,
    a fair-share credit), this is the shape of the change that makes the ML
    scheduler more than a size lookup -- and the degeneracy result would need
    revisiting.
    """
    from conftest import require
    require(_MODEL_ARTEFACT, 'trained wait model')
    import multi_scheduler_benchmark as msb

    cluster = make_cluster(capacity=32, num_nodes=8)
    cluster.nodes = [4, 1, 0, 0, 0, 0, 0, 0]
    # two jobs, SAME size -> identical rows and identical real-model scores
    queue = [make_job(job_id=0, arrival_time=5, num_gpus=4),
             make_job(job_id=1, arrival_time=9, num_gpus=4)]
    x = _feature_matrix(msb, cluster, queue)
    assert np.array_equal(x[0], x[1])
    pred = wait_model['model'].predict(x)
    assert pred[0] == pred[1]           # the degenerate status quo

    # Now simulate augmenting the feature set with a genuine per-job feature:
    # one distinct extra column per job (e.g. a wait-age credit). Use a
    # deterministic scoring function of the FULL row (a fixed linear score with
    # a non-zero weight on the new column) as a stand-in for "any model that
    # reads all its features". The trained model cannot ingest an extra column,
    # so a stand-in scorer is the honest way to demonstrate the mechanism.
    extra = np.array([[0.0], [1.0]])                 # distinct per job
    x_aug = np.hstack([x, extra])
    weights = np.ones(x_aug.shape[1])                # weight on new col != 0
    scores = x_aug @ weights

    assert scores[0] != scores[1], (
        'adding a distinct per-job feature failed to separate equal-size jobs; '
        'the degeneracy test in (a) would then be vacuous')
    # Concretely: the ONLY difference between the two augmented rows is the new
    # column, so the score gap equals that column's contribution -- the new
    # feature, not size, is doing the reordering.
    assert scores[1] - scores[0] == pytest.approx(extra[1, 0] - extra[0, 0])

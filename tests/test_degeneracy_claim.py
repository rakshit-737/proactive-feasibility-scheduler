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

Sections (f)-(j) guard the ARTEFACT rather than the claim. The instant total the
paper quotes is a sum over three settings AT the published protocol (20
synthetic runs, 20 trace windows), so every way of producing a smaller number
under the same column name has to be either impossible or loudly labelled:

  (f) the two tie columns must mean what they say, and the strict one must be
      bounded by the weaker one BY CONSTRUCTION rather than by assumption;
  (g) a missing trace is a hard error; --allow-partial records what was covered;
  (h) --quick, --runs and --windows are equally capable of producing a
      non-published total, so `partial` is derived from all of them;
  (i) the figures a reader actually sees must carry the partial marker, and must
      not describe synthetic instants as "real";
  (j) --figures-only must refuse to re-render publication figures from artefacts
      that are absent, partial, or mutually inconsistent.

Every test here is fast (a handful of jobs, no simulation loop, no benchmark)
and pure (no tracked artefact is written -- the totals-CSV tests write only into
pytest's tmp_path, and the caption tests intercept the renderer so not even a
temporary PNG is produced; artefact READS go through require()).
"""

import numpy as np
import pandas as pd
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


# ─────────────────────────────────────────────────────────────────────────────
# (f) The measured all-ties fraction, and why it is not the arrival-order column
# ─────────────────────────────────────────────────────────────────────────────

# Two columns are enough to exercise the Collector: column 0 is the requested
# size (ranking_degeneracy.SIZE_COL in both real feature vectors) and column 1
# stands in for the cluster-only features, identical for every queued job.
_UNIT_FEATURES = ('job_gpu', 'total_free')


def _observe(col, make_job, sizes, preds, arrivals=None):
    """Drive a Collector exactly the way the RANK_OBSERVER hook does at one
    dispatch instant: (policy, queue, feature matrix, predictions).

    Reuses the SimpleJob fixture the rest of this module builds queues from, so
    the queue carries the real arrival_time/job_id tie-break attributes.
    """
    arrivals = list(range(len(sizes))) if arrivals is None else arrivals
    queue = [make_job(job_id=i, arrival_time=a, num_gpus=int(s))
             for i, (a, s) in enumerate(zip(arrivals, sizes))]
    x = np.column_stack([np.asarray(sizes, dtype=float),
                         np.full(len(sizes), 7.0)])
    col('proactive', queue, x, np.asarray(preds, dtype=float))
    return queue


def test_pct_all_scores_tied_is_bounded_by_order_identical_to_arrival(make_job):
    """INVARIANT: pct_all_scores_tied <= pct_order_identical_to_arrival, by
    construction, and the two are NOT the same measurement.

    The published sentence is "in X% of instants all scores tie, so the policy is
    silently FCFS". All-tied does imply the induced order is arrival order (the
    tie-break is (score, arrival, id)), but the converse fails: distinct scores
    can rank the queue in arrival order too. So the arrival-order column is only
    an UPPER BOUND, and quoting it as the all-ties fraction over-claims. This
    test pins both the bound and the gap.
    """
    import ranking_degeneracy as rd

    col = rd.Collector('unit test', list(_UNIT_FEATURES))

    # instant 1: every queued job scores the same -> all tied, and necessarily
    # ordered by (arrival, id)
    _observe(col, make_job, sizes=[1, 2, 4], preds=[5.0, 5.0, 5.0])
    assert col.all_tied == 1
    assert col.same_as_arrival == 1

    # instant 2: distinct scores that happen to increase with arrival -> the
    # order is STILL arrival order, but nothing is tied. This single instant is
    # the whole gap between the two columns.
    _observe(col, make_job, sizes=[1, 2, 4], preds=[1.0, 2.0, 3.0])
    assert col.all_tied == 1
    assert col.same_as_arrival == 2

    s = col.summary()
    assert s['ranking_instants'] == 2
    assert s['pct_all_scores_tied'] == pytest.approx(50.0)
    assert s['pct_order_identical_to_arrival'] == pytest.approx(100.0)
    assert s['pct_all_scores_tied'] <= s['pct_order_identical_to_arrival'], (
        'all-tied implies order-identical-to-arrival, so the strict measure can '
        'never exceed the bound')

    # The CSV must keep the strict measure next to the bound it strengthens, so
    # a reader cannot pick up one while meaning the other.
    keys = list(s)
    assert keys.index('pct_all_scores_tied') == \
        keys.index('pct_order_identical_to_arrival') + 1


def test_all_tied_and_the_induced_order_read_the_same_quantised_score(make_job):
    """INVARIANT: all_tied <= same_as_arrival holds BY CONSTRUCTION, including
    for scores that differ only below the tie tolerance.

    The bound above is asserted as a mathematical consequence of the tie-break
    being (score, arrival, id). That consequence needs the tie COUNTER and the
    sort KEY to be the same number. They were not: `all_tied` incremented when
    `len(np.unique(np.round(pred, 9))) == 1` while `model_order` sorted on the
    raw `float(pred[i])`. Two predictions 3e-10 apart therefore counted as "all
    scores tied" and still induced an order of their own, so pct_all_scores_tied
    could exceed the column documented as its upper bound -- the comment and the
    test above would have been asserting a convenient fiction.

    The three scores below are the witness: they genuinely differ as floats, in
    an order that is NOT arrival order, and they quantise to a single value.
    With one definition of "tied" the instant is both all-tied and
    arrival-ordered; with two definitions it is all-tied and not.
    """
    import ranking_degeneracy as rd

    col = rd.Collector('unit test', list(_UNIT_FEATURES))

    near = [5.0 + 3e-10, 5.0 + 1e-10, 5.0 + 2e-10]
    assert len(set(near)) == 3, 'the raw scores must genuinely differ as floats'
    assert len(np.unique(rd.tie_keys(near))) == 1, 'and must quantise to one'
    # Sorting on the RAW floats gives 1, 2, 0 -- not the arrival order 0, 1, 2.
    assert sorted(range(3), key=lambda i: near[i]) == [1, 2, 0]

    _observe(col, make_job, sizes=[1, 2, 4], preds=near, arrivals=[0, 1, 2])

    assert col.all_tied == 1
    assert col.same_as_arrival == 1, (
        'the instant was counted as all-tied but its induced order was NOT '
        'arrival order: the tie counter and the sort key disagree, so '
        'pct_all_scores_tied <= pct_order_identical_to_arrival is not a '
        'guarantee')

    s = col.summary()
    assert s['pct_all_scores_tied'] == pytest.approx(100.0)
    assert s['pct_all_scores_tied'] <= s['pct_order_identical_to_arrival']
    # And the distinct-prediction count reads the same quantised score, so a
    # queue reported as all-tied cannot simultaneously report >1 distinct score.
    assert s['mean_distinct_predictions'] == pytest.approx(1.0)


def test_equal_size_violations_use_the_same_tie_tolerance(make_job):
    """INVARIANT: "equal size, different prediction" is decided by the same
    quantised score as everything else in the Collector.

    equal_size_diff_pred_violations is the executable form of "the score is a
    function of requested size alone", and the published value is 0. If the
    violation check used a looser notion of equality than the tie counter, a
    genuine separation could be counted as a tie in one column and a violation
    in another. Both directions are pinned here: scores inside one quantisation
    bin are not a violation, scores plainly outside it are.
    """
    import ranking_degeneracy as rd

    within = rd.Collector('unit test', list(_UNIT_FEATURES))
    # two size-4 jobs whose scores differ by 3e-10 -- one bin, so not a violation
    _observe(within, make_job, sizes=[4, 4, 1], preds=[2.0, 2.0 + 3e-10, 9.0])
    assert within.violations == 0

    beyond = rd.Collector('unit test', list(_UNIT_FEATURES))
    # the same two jobs, now separated well beyond the tolerance
    _observe(beyond, make_job, sizes=[4, 4, 1], preds=[2.0, 2.5, 9.0])
    assert beyond.violations == 1, (
        'two equally-sized jobs received clearly different scores and it was '
        'not counted: the degeneracy claim would report 0 violations while the '
        'score was not a function of size')


# ─────────────────────────────────────────────────────────────────────────────
# (g) A missing trace must be a hard failure, not a smaller published total
# ─────────────────────────────────────────────────────────────────────────────

def _stub_pipeline(rd, monkeypatch, tmp_path, make_job, trace=None):
    """Redirect the script's outputs to tmp_path and replace everything slow.

    run_synthetic is stubbed with a Collector fed one hand-built instant (the
    suite may not run a benchmark, and the behaviour under test is the control
    flow around a missing trace, not the model); the two figure functions are
    stubbed because no test may write a PNG. `trace` is the run_trace stand-in;
    the default one reports every trace as absent.

    Returns the list the figure stubs append their (name, args, kwargs) to, so a
    caller can assert both WHETHER a figure was rendered and WHAT scope it was
    handed.
    """
    monkeypatch.setattr(rd, 'OUT_DIR', str(tmp_path))

    def fake_synthetic(n_runs):
        col = rd.Collector('synthetic (unit test)', list(_UNIT_FEATURES))
        _observe(col, make_job, sizes=[1, 2, 4], preds=[1.0, 2.0, 3.0])
        return col

    def missing(trace_key, *a, **k):
        raise FileNotFoundError(f'02_data/{trace_key}.swf.gz')

    rendered = []
    monkeypatch.setattr(rd, 'run_synthetic', fake_synthetic)
    monkeypatch.setattr(rd, 'run_trace', trace or missing)
    monkeypatch.setattr(rd, 'make_figure',
                        lambda *a, **k: rendered.append(('figure', a, k)))
    monkeypatch.setattr(rd, 'make_size_table_figure',
                        lambda *a, **k: rendered.append(('size_table', a, k)))
    return rendered


def _fake_trace(rd, make_job):
    """A run_trace stand-in that succeeds, contributing one instant per trace."""
    def trace(trace_key, *a, **k):
        col = rd.Collector(f'{trace_key} (unit test)', list(_UNIT_FEATURES))
        _observe(col, make_job, sizes=[1, 2, 4], preds=[1.0, 2.0, 3.0])
        return col
    return trace


def test_missing_trace_is_a_hard_error_without_allow_partial(monkeypatch, tmp_path,
                                                             make_job):
    """INVARIANT: a trace that cannot be loaded aborts the run.

    The published headline is the sum of ranking_instants over three settings.
    The script used to catch FileNotFoundError per trace, print 'skipping ...'
    and write the summary anyway, so 45,432 instants could collapse to the 3,646
    synthetic ones with nothing in the artefact saying so. A missing trace must
    now exit non-zero and write NOTHING.
    """
    import os
    import sys

    import ranking_degeneracy as rd
    _stub_pipeline(rd, monkeypatch, tmp_path, make_job)
    monkeypatch.setattr(sys, 'argv', ['ranking_degeneracy.py', '--quick'])

    with pytest.raises(SystemExit) as exc:
        rd.main()

    msg = str(exc.value)
    assert 'sdsc' in msg                     # names the trace that failed
    assert '--allow-partial' in msg          # says how to proceed deliberately
    assert '02_data' in msg                  # says how to fix it properly
    for name in ('ranking_degeneracy.csv', 'ranking_degeneracy_totals.csv'):
        assert not os.path.exists(os.path.join(str(tmp_path), name)), (
            f'{name} was written despite a missing trace')


def test_allow_partial_records_what_the_run_actually_covered(monkeypatch, tmp_path,
                                                             make_job):
    """INVARIANT: with --allow-partial the run succeeds but the artefact SAYS it
    is partial -- which settings were expected, which are present, which traces
    are missing. That totals row is what a reader (or a later consistency check)
    consults instead of re-summing a table that may be short a setting.
    """
    import sys

    import ranking_degeneracy as rd
    _stub_pipeline(rd, monkeypatch, tmp_path, make_job)
    monkeypatch.setattr(sys, 'argv',
                        ['ranking_degeneracy.py', '--quick', '--allow-partial'])

    rd.main()

    totals = pd.read_csv(tmp_path / 'ranking_degeneracy_totals.csv')
    assert len(totals) == 1
    row = totals.iloc[0]
    assert int(row['settings_expected']) == rd.EXPECTED_SETTINGS == 3
    assert int(row['settings_present']) == 1          # synthetic only
    assert bool(row['partial']) is True
    assert sorted(row['missing_traces'].split(';')) == sorted(rd.TRACE_KEYS)
    assert int(row['total_instants']) == 1            # the one stubbed instant
    assert int(row['total_violations']) == 0

    # The totals live in their OWN file: make_figure draws one bar pair per row
    # of ranking_degeneracy.csv, so a TOTAL row there would plot as a fourth,
    # non-existent setting.
    summary = pd.read_csv(tmp_path / 'ranking_degeneracy.csv')
    assert len(summary) == 1
    assert 'TOTAL' not in set(summary['setting'])


def test_quick_run_is_marked_partial_even_with_every_trace_present(monkeypatch,
                                                                   tmp_path,
                                                                   make_job,
                                                                   capsys):
    """INVARIANT: --quick reduces runs and windows, so its instant total is not
    the published number even when no trace is missing. The totals row must say
    so, and the printed line must report settings covered over settings expected.
    """
    import sys

    import ranking_degeneracy as rd

    _stub_pipeline(rd, monkeypatch, tmp_path, make_job,
                   trace=_fake_trace(rd, make_job))
    monkeypatch.setattr(sys, 'argv', ['ranking_degeneracy.py', '--quick'])

    rd.main()

    row = pd.read_csv(tmp_path / 'ranking_degeneracy_totals.csv').iloc[0]
    assert int(row['settings_present']) == 3
    # Nothing missing is written as the explicit sentinel 'none', never as an
    # empty field: an empty CSV cell round-trips through pd.read_csv as NaN, so
    # a consumer written as `row['missing_traces'] == ''` would never match and
    # every read would have to special-case a float. 'none' is itself on the way
    # back in, which is why this asserts equality rather than "NaN or empty".
    assert row['missing_traces'] == rd.NO_MISSING_TRACES == 'none'
    assert not pd.isna(row['missing_traces'])
    assert bool(row['partial']) is True                # because of --quick alone
    assert int(row['total_instants']) == 3

    out = capsys.readouterr().out
    assert 'Total ranking instants across 3/3 settings: 3' in out
    assert 'partial=True' in out


# ─────────────────────────────────────────────────────────────────────────────
# (h) --runs / --windows can under-report the total exactly as --quick can
# ─────────────────────────────────────────────────────────────────────────────

def test_the_published_protocol_is_the_argparse_default_and_is_not_partial(
        monkeypatch, tmp_path, make_job):
    """INVARIANT: the BARE invocation reproduces the published protocol and is
    the only invocation NOT marked partial.

    This is the non-vacuity half of the pair below. If `partial` were simply
    always True, every test asserting partial=True would still pass while the
    column stopped carrying any information. So the default run must land on
    PUBLISHED_RUNS / PUBLISHED_WINDOWS, must record them in the totals row, and
    must be the case that is not flagged.
    """
    import sys

    import ranking_degeneracy as rd

    _stub_pipeline(rd, monkeypatch, tmp_path, make_job,
                   trace=_fake_trace(rd, make_job))
    monkeypatch.setattr(sys, 'argv', ['ranking_degeneracy.py'])

    rd.main()

    row = pd.read_csv(tmp_path / 'ranking_degeneracy_totals.csv').iloc[0]
    assert int(row['runs']) == rd.PUBLISHED_RUNS == 20
    assert int(row['windows']) == rd.PUBLISHED_WINDOWS == 20
    assert int(row['published_runs']) == rd.PUBLISHED_RUNS
    assert int(row['published_windows']) == rd.PUBLISHED_WINDOWS
    assert bool(row['partial']) is False
    assert row['missing_traces'] == 'none'


@pytest.mark.parametrize('flag, value, runs, windows', [
    ('--windows', '2', 20, 2),
    ('--runs', '5', 5, 20),
])
def test_reduced_runs_or_windows_is_partial_with_every_trace_present(
        flag, value, runs, windows, monkeypatch, tmp_path, make_job):
    """INVARIANT: ANY departure from the published protocol marks the run
    partial -- not only --quick, and not only a missing trace.

    --runs and --windows are user-settable and their defaults ARE the published
    protocol, so `ranking_degeneracy.py --windows 2` produces a total that is
    not 45,432 while every trace is present and nothing is quick. `partial` was
    computed as `bool(missing) or bool(args.quick)`, so that run wrote
    partial=False beside a number a reader would take for the published one.
    The totals row must also record the runs/windows actually used, so the file
    says what produced it without anyone having to recall the command line.
    """
    import sys

    import ranking_degeneracy as rd

    _stub_pipeline(rd, monkeypatch, tmp_path, make_job,
                   trace=_fake_trace(rd, make_job))
    monkeypatch.setattr(sys, 'argv', ['ranking_degeneracy.py', flag, value])

    rd.main()

    row = pd.read_csv(tmp_path / 'ranking_degeneracy_totals.csv').iloc[0]
    assert int(row['settings_present']) == 3        # nothing is missing
    assert row['missing_traces'] == 'none'
    assert int(row['runs']) == runs
    assert int(row['windows']) == windows
    assert bool(row['partial']) is True, (
        f'{flag} {value} departs from the published protocol '
        f'({rd.PUBLISHED_RUNS} runs, {rd.PUBLISHED_WINDOWS} windows) yet the '
        f'totals row claims these are the published numbers')


def test_published_windows_matches_the_trace_benchmarks_own_protocol():
    """INVARIANT: this script's window default is the trace benchmark's N_WINDOWS.

    The degeneracy instants are collected by driving `trace_driven_benchmark`,
    so "the published protocol" only means something if the two agree about how
    many windows that is. If N_WINDOWS moves, PUBLISHED_WINDOWS has to move with
    it, or the default invocation stops reproducing the published total while
    still reporting partial=False.
    """
    pytest.importorskip('xgboost',
                        reason='trace_driven_benchmark imports xgboost')
    import ranking_degeneracy as rd
    import trace_driven_benchmark as tdb

    assert rd.PUBLISHED_WINDOWS == tdb.N_WINDOWS


def test_run_scope_partial_predicate_covers_every_reduction():
    """INVARIANT: RunScope is the single place `partial` is decided, and it
    answers True for each independent way of shrinking the total.

    Unit-level companion to the end-to-end tests above: it pins the predicate
    itself, so a future caller that builds a RunScope by hand cannot get a
    different answer from the one `main()` writes into the CSV.
    """
    import ranking_degeneracy as rd

    full = rd.RunScope(runs=rd.PUBLISHED_RUNS, windows=rd.PUBLISHED_WINDOWS)
    assert full.partial is False
    assert full.reasons() == []
    assert full.missing_field() == 'none'

    for scope in (
        rd.RunScope(runs=5, windows=rd.PUBLISHED_WINDOWS),
        rd.RunScope(runs=rd.PUBLISHED_RUNS, windows=2),
        rd.RunScope(runs=rd.PUBLISHED_RUNS, windows=rd.PUBLISHED_WINDOWS,
                    missing=['sdsc']),
        rd.RunScope(runs=rd.PUBLISHED_RUNS, windows=rd.PUBLISHED_WINDOWS,
                    quick=True),
    ):
        assert scope.partial is True
        assert scope.reasons(), 'a partial run must be able to say why'

    assert rd.RunScope(runs=20, windows=20,
                       missing=['sdsc', 'lanl']).missing_field() == 'sdsc;lanl'


# ─────────────────────────────────────────────────────────────────────────────
# (i) The FIGURE caption: the only part of the artefact most readers ever see
# ─────────────────────────────────────────────────────────────────────────────

def _figure_frames():
    """Minimal summary/feature/curve tables shaped like the real CSVs."""
    summary = pd.DataFrame([
        {'setting': 'synthetic (12 features)', 'ranking_instants': 3646,
         'equal_size_diff_pred_violations': 0,
         'pct_order_identical_to_size': 80.3,
         'pct_order_identical_to_arrival': 19.7,
         'pct_size_table_monotone': 63.1},
        {'setting': 'SDSC SP2 (1998) (8 features)', 'ranking_instants': 11843,
         'equal_size_diff_pred_violations': 0,
         'pct_order_identical_to_size': 71.5,
         'pct_order_identical_to_arrival': 18.1,
         'pct_size_table_monotone': 57.1},
    ])
    feats = pd.DataFrame([
        {'setting': 'synthetic (12 features)', 'feature': 'job_gpu',
         'pct_instants_varying_across_queue': 100.0},
        {'setting': 'synthetic (12 features)', 'feature': 'total_free',
         'pct_instants_varying_across_queue': 0.0},
    ])
    curves = pd.DataFrame([
        {'setting': 'synthetic (12 features)', 'size': 1.0, 'norm_score': 0.0,
         'n': 10},
        {'setting': 'synthetic (12 features)', 'size': 8.0, 'norm_score': 1.0,
         'n': 10},
        {'setting': 'SDSC SP2 (1998) (8 features)', 'size': 2.0,
         'norm_score': 0.2, 'n': 5},
        {'setting': 'SDSC SP2 (1998) (8 features)', 'size': 64.0,
         'norm_score': 0.9, 'n': 5},
    ])
    return summary, feats, curves


def _captions(rd, monkeypatch, scope):
    """Render both figures with the renderer intercepted, returning subtitles.

    `finish` is where every caption string lands and `save_both` is the only
    call that touches the filesystem, so replacing the two means the real
    caption code runs while no PNG is written anywhere -- not even a temporary
    one.
    """
    summary, feats, curves = _figure_frames()
    subtitles = []

    def fake_finish(fig, mode='light', title=None, subtitle=None, **kw):
        subtitles.append(subtitle)
        return fig

    monkeypatch.setattr(rd, 'finish', fake_finish)
    monkeypatch.setattr(rd, 'save_both',
                        lambda fig, stem, mode, **kw: rd.plt.close(fig))

    rd.make_figure(summary, feats, curves, 'unused-stem', scope)
    rd.make_size_table_figure(curves, summary, 'unused-stem', scope)
    return subtitles


def test_full_run_caption_splits_synthetic_from_trace_instants(monkeypatch):
    """INVARIANT: the caption never calls the whole total "real" instants.

    3,646 of the published 45,432 instants come from the synthetic simulator,
    which is not a real machine. The caption read "Across 45,432 real dispatch
    instants", which upgrades simulator output to field evidence for free. It
    must report the split instead.
    """
    import ranking_degeneracy as rd

    scope = rd.RunScope(runs=rd.PUBLISHED_RUNS, windows=rd.PUBLISHED_WINDOWS)
    subtitles = _captions(rd, monkeypatch, scope)
    assert subtitles, 'no caption was rendered'

    main_caption = subtitles[0]
    assert 'real dispatch instants' not in main_caption, (
        'the caption describes synthetic simulator instants as real')
    # 3646 + 11843 = 15489 in this fixture: the split must be stated, not just
    # the bare total.
    assert '15,489 dispatch instants' in main_caption
    assert '3,646 synthetic' in main_caption
    assert '11,843' in main_caption

    for caption in subtitles:
        assert 'PARTIAL' not in caption, (
            'a complete run must not be labelled partial')


def test_partial_run_caption_says_so_and_names_the_settings_present(monkeypatch):
    """INVARIANT: a partial run's FIGURES say they are partial.

    The hard failure on a missing trace exists so that a smaller total can never
    be presented as the published one. --allow-partial reopened exactly that
    hole at the figure: the caption printed "Across {total} real dispatch
    instants" from the same sum with no marker at all, so a partial PNG was
    indistinguishable from the published one. Both figures must carry the
    marker, name the settings actually present, and say why the run is partial.
    """
    import ranking_degeneracy as rd

    scope = rd.RunScope(runs=rd.PUBLISHED_RUNS, windows=2, missing=['lanl'])
    subtitles = _captions(rd, monkeypatch, scope)
    assert len(subtitles) >= 2, 'both figures must render a caption'

    for caption in subtitles:
        assert caption.startswith('PARTIAL RUN:'), (
            'a partial run rendered a caption indistinguishable from the '
            'published figure')
        # the settings that ARE present, named
        assert 'synthetic + SDSC SP2' in caption
        # and why the run is partial, in the caption itself
        assert 'lanl' in caption
        assert '--windows 2' in caption
        assert 'NOT the published totals' in caption


# ─────────────────────────────────────────────────────────────────────────────
# (j) --figures-only must not re-render from artefacts it cannot vouch for
# ─────────────────────────────────────────────────────────────────────────────

def _complete_artefacts(rd, monkeypatch, tmp_path, make_job):
    """Run the stubbed pipeline at the published protocol, leaving CSVs behind.

    Returns the figure-call log, cleared, with argv already switched to
    --figures-only: every test below then calls rd.main() once and inspects
    whether anything was rendered.
    """
    import sys

    rendered = _stub_pipeline(rd, monkeypatch, tmp_path, make_job,
                              trace=_fake_trace(rd, make_job))
    monkeypatch.setattr(sys, 'argv', ['ranking_degeneracy.py'])
    rd.main()
    seed = pd.read_csv(tmp_path / 'ranking_degeneracy_totals.csv').iloc[0]
    assert bool(seed['partial']) is False, 'the seeded artefacts must be complete'
    rendered.clear()
    monkeypatch.setattr(sys, 'argv', ['ranking_degeneracy.py', '--figures-only'])
    return rendered


def test_figures_only_re_renders_when_the_totals_row_agrees(monkeypatch, tmp_path,
                                                            make_job):
    """INVARIANT (non-vacuity): a consistent, complete artefact set re-renders.

    Without this, the four refusal tests below would all pass on a
    --figures-only branch that refused unconditionally.
    """
    import ranking_degeneracy as rd

    rendered = _complete_artefacts(rd, monkeypatch, tmp_path, make_job)
    rd.main()

    assert [name for name, _, _ in rendered] == ['figure', 'size_table']
    # the scope handed to the renderer describes a complete run
    scope = rendered[0][1][-1]
    assert scope.partial is False


def test_figures_only_refuses_when_the_totals_file_is_absent(monkeypatch, tmp_path,
                                                             make_job):
    """INVARIANT: no totals file => refuse, and say which file.

    --figures-only recomputes nothing, so the totals row is the only record that
    the CSVs it is about to caption came from a complete run. Proceeding without
    it re-publishes a figure whose quoted total nothing vouches for.
    """
    import ranking_degeneracy as rd

    rendered = _complete_artefacts(rd, monkeypatch, tmp_path, make_job)
    (tmp_path / 'ranking_degeneracy_totals.csv').unlink()

    with pytest.raises(SystemExit) as exc:
        rd.main()

    assert 'ranking_degeneracy_totals.csv' in str(exc.value)
    assert not rendered, 'a figure was rendered despite the refusal'


def test_figures_only_refuses_to_re_render_a_partial_run(monkeypatch, tmp_path,
                                                         make_job):
    """INVARIANT: partial=True in the totals row => refuse.

    Re-rendering the publication figures from a partial run is precisely the
    mistake the hard failure on a missing trace was added to prevent: the
    caption would present a reduced total in the published figure's own words.
    """
    import ranking_degeneracy as rd

    rendered = _complete_artefacts(rd, monkeypatch, tmp_path, make_job)
    totals_path = tmp_path / 'ranking_degeneracy_totals.csv'
    totals = pd.read_csv(totals_path)
    totals.loc[0, 'partial'] = True
    totals.loc[0, 'missing_traces'] = 'lanl'
    totals.to_csv(totals_path, index=False)

    with pytest.raises(SystemExit) as exc:
        rd.main()

    assert 'partial=True' in str(exc.value)
    assert not rendered, 'a partial run was re-rendered as a publication figure'


def test_figures_only_refuses_when_the_instant_total_is_stale(monkeypatch,
                                                              tmp_path, make_job):
    """INVARIANT: the totals row must add up to the table beside it.

    --figures-only re-renders from ranking_degeneracy.csv while leaving the
    totals row untouched, so the two drift apart the moment anything regenerates
    one and not the other -- and nothing flagged the mismatch, so the figure
    quoted the sum of whichever table it happened to read.
    """
    import ranking_degeneracy as rd

    rendered = _complete_artefacts(rd, monkeypatch, tmp_path, make_job)
    totals_path = tmp_path / 'ranking_degeneracy_totals.csv'
    totals = pd.read_csv(totals_path)
    # a plausible-looking larger total, the shape of a stale row left behind by
    # an earlier, smaller run
    totals.loc[0, 'total_instants'] = int(totals.loc[0, 'total_instants']) + 41786
    totals.to_csv(totals_path, index=False)

    with pytest.raises(SystemExit) as exc:
        rd.main()

    assert 'disagrees with ranking_degeneracy.csv' in str(exc.value)
    assert not rendered, 'stale totals were re-rendered anyway'


def test_figures_only_refuses_when_only_the_violation_total_is_stale(
        monkeypatch, tmp_path, make_job):
    """INVARIANT: the violation total is checked too, not only the instants.

    equal_size_diff_pred_violations == 0 is a headline claim in its own right, so
    a totals row that agrees about instants and disagrees about violations is
    exactly as stale as one that disagrees about both. Checking a single column
    would leave half the file unguarded.
    """
    import ranking_degeneracy as rd

    rendered = _complete_artefacts(rd, monkeypatch, tmp_path, make_job)
    totals_path = tmp_path / 'ranking_degeneracy_totals.csv'
    totals = pd.read_csv(totals_path)
    totals.loc[0, 'total_violations'] = int(totals.loc[0, 'total_violations']) + 7
    totals.to_csv(totals_path, index=False)

    with pytest.raises(SystemExit) as exc:
        rd.main()

    assert 'disagrees with ranking_degeneracy.csv' in str(exc.value)
    assert not rendered

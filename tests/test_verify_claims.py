"""Mutation tests for tools/verify_claims.py.

The claims verifier exists to fail when an artefact stops supporting a statement
the repository makes. A verifier that passes on a broken tree is worse than none
at all, because it launders the break as a check. So every claim check here is
tested the only way that means anything: reintroduce the defect it is supposed to
catch, prove it goes red, and prove it is green on the untouched tree.

The trees are copies of the committed artefacts, mutated in `tmp_path`. Nothing
here writes to the repository.
"""

import importlib.util
import os
import shutil
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    'verify_claims', os.path.join(ROOT, 'tools', 'verify_claims.py'))
verify_claims = importlib.util.module_from_spec(_spec)
sys.modules['verify_claims'] = verify_claims
_spec.loader.exec_module(verify_claims)


def run(root):
    """Run every check against `root`; return {claim: status}."""
    c = verify_claims.Checker(str(root))
    for claim, fn in verify_claims.CHECKS:
        c.check(claim, lambda fn=fn: fn(c))
    return {claim: status for status, claim, _ in c.rows}


@pytest.fixture(scope='module')
def pristine(tmp_path_factory):
    """A copy of the committed 05_results tree -- the only directory read."""
    dst = tmp_path_factory.mktemp('pristine')
    shutil.copytree(os.path.join(ROOT, '05_results'), dst / '05_results')
    return dst


@pytest.fixture
def tree(tmp_path, pristine):
    dst = tmp_path / 'tree'
    shutil.copytree(pristine, dst)
    return dst


def edit(tree, rel, fn):
    """Apply `fn` to the CSV at `rel` and write it back."""
    path = tree / '05_results' / rel
    df = pd.read_csv(path, float_precision='round_trip')
    fn(df)
    df.to_csv(path, index=False)


def test_every_claim_holds_on_the_committed_artefacts(pristine):
    """The baseline. If this fails, the repository does not support its own claims."""
    statuses = run(pristine)
    failed = [c for c, s in statuses.items() if s != verify_claims.PASS]
    assert not failed, f'claims failing on the committed tree: {failed}'


# --- each mutation below must turn exactly the claim it targets red ----------

def test_a_single_violation_fails_the_central_claim(tree):
    edit(tree, 'degeneracy/ranking_degeneracy.csv',
         lambda d: d.__setitem__('equal_size_diff_pred_violations',
                                 [1] + [0] * (len(d) - 1)))
    assert run(tree)['no two equally-sized co-queued jobs ever scored differently'] \
        == verify_claims.FAIL


def test_a_partial_run_fails_the_protocol_claim(tree):
    edit(tree, 'degeneracy/ranking_degeneracy_totals.csv',
         lambda d: d.__setitem__('partial', True))
    assert run(tree)['the diagnostic ran the full protocol and totals agree'] \
        == verify_claims.FAIL


def test_an_instant_count_far_from_the_reference_fails(tree):
    """Platform noise moved the count 0.36%. A 10% move is not platform noise."""
    edit(tree, 'degeneracy/ranking_degeneracy_totals.csv',
         lambda d: d.__setitem__('total_instants', int(verify_claims.REFERENCE_INSTANTS * 0.9)))
    assert run(tree)['the diagnostic ran the full protocol and totals agree'] \
        == verify_claims.FAIL


def test_a_small_instant_drift_still_passes(tree):
    """The Linux runner's 45,268 must NOT fail: that is the whole point."""
    edit(tree, 'degeneracy/ranking_degeneracy_totals.csv',
         lambda d: d.__setitem__('total_instants', 45268))
    assert run(tree)['the diagnostic ran the full protocol and totals agree'] \
        == verify_claims.PASS


def test_queue_pressure_drifting_from_job_gpu_fails(tree):
    def bump(d):
        m = d['feature'] == 'queue_pressure'
        d.loc[m, 'pct_instants_varying_across_queue'] = \
            d.loc[m, 'pct_instants_varying_across_queue'] + 0.01
    edit(tree, 'degeneracy/feature_variation.csv', bump)
    assert run(tree)['7 of 12 synthetic features cannot separate two queued jobs'] \
        == verify_claims.FAIL


def test_nn_diverging_from_the_size_sort_fails(tree):
    def nudge(d):
        i = d.index[d['scheduler'] == 'NN'][0]
        d.loc[i, 'mean_wait'] = d.loc[i, 'mean_wait'] + 1e-9
    edit(tree, 'schedulers/multi_scheduler_runs.csv', nudge)
    assert run(tree)['the neural net IS the size sort, not merely equivalent to it'] \
        == verify_claims.FAIL


def test_losing_equivalence_fails(tree):
    edit(tree, 'schedulers/multi_scheduler_equivalence.csv',
         lambda d: d.__setitem__('equivalent', False))
    assert run(tree)['PROACTIVE is TOST-equivalent to SMALLEST (synthetic)'] \
        == verify_claims.FAIL


def test_an_augmented_arm_that_stops_breaking_it_fails(tree):
    def zero_one(d):
        i = d.index[d['feature_set'] != 'baseline'][0]
        d.loc[i, 'equal_size_diff_pred_violations'] = 0
    edit(tree, 'degeneracy/non_degeneracy_sweep.csv', zero_one)
    assert run(tree)['every genuine per-job feature breaks the degeneracy'] \
        == verify_claims.FAIL


def test_a_variant_beating_the_heuristic_fails(tree):
    """THE finding. If a variant ever wins, this must go red so the claim is
    restated deliberately rather than drifting."""
    def win(d):
        d.loc[0, 'beats_sjf_userest'] = True
    edit(tree, 'degeneracy/non_degeneracy_utility.csv', win)
    assert run(tree)['breaking it beats nothing: 0 of 12 beat SJF on user estimates'] \
        == verify_claims.FAIL


def test_an_attack_that_should_fail_succeeding_fails(tree):
    def break_a1(d):
        i = d.index[d['attack_id'] == 'A1'][0]
        d.loc[i, 'violations'] = 7
    edit(tree, 'degeneracy/robustness_attacks.csv', break_a1)
    assert run(tree)['three attacks fail, only enqueue-time caching succeeds'] \
        == verify_claims.FAIL


def test_the_monotone_attack_diverging_fails(tree):
    def diverge(d):
        i = d.index[d['attack_id'] == 'A2'][0]
        d.loc[i, 'ranking_instants'] = int(d.loc[i, 'ranking_instants']) + 1
    edit(tree, 'degeneracy/robustness_attacks.csv', diverge)
    assert run(tree)['three attacks fail, only enqueue-time caching succeeds'] \
        == verify_claims.FAIL


def test_utilisation_moving_fails_the_reordering_claim(tree):
    def nudge(d):
        i = d.index[d['scheduler'] == 'PROACTIVE'][0]
        d.loc[i, 'gpu_util'] = d.loc[i, 'gpu_util'] + 0.5
    edit(tree, 'schedulers/multi_scheduler_runs.csv', nudge)
    assert run(tree)['the improvement is reordering: utilisation is unchanged'] \
        == verify_claims.FAIL


def test_a_confidence_interval_spanning_zero_fails(tree):
    def span(d):
        d.loc[d['metric'] == 'improvement_ci95_low', 'value'] = -1.0
    edit(tree, 'benchmark_statistical_summary.csv', span)
    assert run(tree)['the wait improvement is positive and significant'] \
        == verify_claims.FAIL


def test_the_split_ordering_reversing_fails(tree):
    def swap(d):
        d.loc[d['split'] == 'chronological', 'r2_mean'] = 0.99
    edit(tree, 'models/evaluation_splits.csv', swap)
    assert run(tree)['random-split accuracy is optimistic; chronological is worst'] \
        == verify_claims.FAIL


def test_a_missing_artefact_fails_rather_than_skips(tree):
    """A truncated pipeline must not pass by silently having nothing to check."""
    os.remove(tree / '05_results' / 'degeneracy' / 'ranking_degeneracy.csv')
    assert run(tree)['no two equally-sized co-queued jobs ever scored differently'] \
        == verify_claims.FAIL

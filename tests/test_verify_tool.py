"""Unit tests for the artefact verifier's comparison logic.

`tools/verify_artifacts.py` is what turns "no number exists unless a committed
script regenerates it" from a claim into a check, so its comparison functions
are themselves load-bearing: a verifier that silently calls everything equal is
worse than no verifier, because it looks like evidence.

These tests exercise the PURE functions only -- no subprocess, no git, no
pipeline run. They pin the three properties that decide whether a report can be
trusted:

  1. the numeric tolerance actually discriminates (1e-12 passes, 1e-6 fails),
  2. the wall-clock exemption is scoped to the named columns of the named
     files, and does not leak into anything else,
  3. line endings are never reported as a difference, and scripts and
     hand-written prose are never treated as artefacts.
"""

import importlib.util
import os
import sys

import pytest

from conftest import PROJECT_ROOT

# The tools directory is not on sys.path (conftest adds the numbered dirs only),
# and `verify_artifacts` is a script rather than a package, so load it from its
# path the way a user runs it.
TOOL_PATH = os.path.join(PROJECT_ROOT, 'tools', 'verify_artifacts.py')


def _load_tool():
    spec = importlib.util.spec_from_file_location('verify_artifacts', TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


va = _load_tool()

# A real committed value, so the tolerance test operates at the magnitude the
# tool actually sees rather than at a toy scale.
BASE = 4.611225553394174

# The timing exemption is asserted against the module's own table: if a future
# edit drops the entry, these tests fail instead of quietly passing.
TIMING_REL = '05_results/model_comparison_table1.csv'
TIMING_COL = 'training_time_sec'


def _write_csv(path, header, rows):
    """Write a CSV with round-trippable floats (repr, not str formatting)."""
    lines = [','.join(header)]
    for row in rows:
        lines.append(','.join(repr(v) if isinstance(v, float) else str(v)
                              for v in row))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def _pair(tmp_path, header, ref_rows, new_rows):
    ref = _write_csv(tmp_path / 'ref.csv', header, ref_rows)
    new = _write_csv(tmp_path / 'new.csv', header, new_rows)
    return ref, new


# ---------------------------------------------------------------------------
# Numeric tolerance
# ---------------------------------------------------------------------------

def test_timing_table_entry_exists():
    """Guard the fixture the exemption tests below depend on."""
    assert TIMING_COL in va.NONDETERMINISTIC_COLUMNS[TIMING_REL]


def test_perturbation_below_tolerance_compares_ok(tmp_path):
    ref, new = _pair(tmp_path, ['model', 'mae'],
                     [['LightGBM', BASE]],
                     [['LightGBM', BASE * (1 + 1e-12)]])
    status, detail = va.compare_csv('05_results/anything.csv', ref, new)
    assert status == va.OK, detail


def test_perturbation_above_tolerance_is_mismatch(tmp_path):
    ref, new = _pair(tmp_path, ['model', 'mae'],
                     [['LightGBM', BASE]],
                     [['LightGBM', BASE * (1 + 1e-6)]])
    status, detail = va.compare_csv('05_results/anything.csv', ref, new)
    assert status == va.MISMATCH
    assert 'mae' in detail          # the report must name the offending column


def test_string_columns_are_compared_exactly(tmp_path):
    ref, new = _pair(tmp_path, ['model', 'mae'],
                     [['LightGBM', BASE]],
                     [['LightGBM ', BASE]])
    status, _ = va.compare_csv('05_results/anything.csv', ref, new)
    assert status == va.MISMATCH


def test_row_count_change_is_mismatch(tmp_path):
    ref, new = _pair(tmp_path, ['model', 'mae'],
                     [['LightGBM', BASE], ['CatBoost', BASE]],
                     [['LightGBM', BASE]])
    status, detail = va.compare_csv('05_results/anything.csv', ref, new)
    assert status == va.MISMATCH
    assert 'shape' in detail


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_reordered_columns_are_schema(tmp_path):
    ref = _write_csv(tmp_path / 'ref.csv', ['model', 'mae'], [['LightGBM', BASE]])
    new = _write_csv(tmp_path / 'new.csv', ['mae', 'model'], [[BASE, 'LightGBM']])
    status, detail = va.compare_csv('05_results/anything.csv', ref, new)
    assert status == va.SCHEMA, detail


def test_renamed_column_is_schema(tmp_path):
    ref = _write_csv(tmp_path / 'ref.csv', ['model', 'mae'], [['LightGBM', BASE]])
    new = _write_csv(tmp_path / 'new.csv', ['model', 'MAE'], [['LightGBM', BASE]])
    status, _ = va.compare_csv('05_results/anything.csv', ref, new)
    assert status == va.SCHEMA


# ---------------------------------------------------------------------------
# The wall-clock exemption
# ---------------------------------------------------------------------------

def test_wall_clock_column_difference_is_timing_not_mismatch(tmp_path):
    header = ['model', 'mae', 'r2', TIMING_COL]
    ref, new = _pair(tmp_path, header,
                     [['LightGBM', BASE, 0.8476769173748722, 5.771230599842966]],
                     [['LightGBM', BASE, 0.8476769173748722, 10.6]])
    status, detail = va.compare_csv(TIMING_REL, ref, new)
    assert status == va.TIMING, detail
    # The exemption must be stated in the report, not applied silently.
    assert TIMING_COL in detail
    assert 'NONDETERMINISTIC_COLUMNS' in detail


def test_exemption_does_not_cover_other_columns_of_the_same_file(tmp_path):
    header = ['model', 'mae', 'r2', TIMING_COL]
    ref, new = _pair(tmp_path, header,
                     [['LightGBM', BASE, 0.8476769173748722, 5.771230599842966]],
                     [['LightGBM', BASE * 1.01, 0.8476769173748722, 10.6]])
    status, detail = va.compare_csv(TIMING_REL, ref, new)
    assert status == va.MISMATCH
    assert 'mae' in detail


def test_exemption_is_scoped_to_the_named_file(tmp_path):
    """The same column name in a different CSV is still compared strictly."""
    header = ['model', TIMING_COL]
    ref, new = _pair(tmp_path, header,
                     [['LightGBM', 5.771230599842966]],
                     [['LightGBM', 10.6]])
    status, _ = va.compare_csv('05_results/some_other_table.csv', ref, new)
    assert status == va.MISMATCH


# ---------------------------------------------------------------------------
# Text artefacts
# ---------------------------------------------------------------------------

def test_text_comparison_ignores_crlf_versus_lf(tmp_path):
    ref = tmp_path / 'ref.txt'
    new = tmp_path / 'new.txt'
    # The reference is a git blob (LF); the regenerated file on Windows is CRLF.
    ref.write_bytes(b'PHASE 26\nmean wait 3743.83\nend\n')
    new.write_bytes(b'PHASE 26\r\nmean wait 3743.83\r\nend\r\n')
    status, detail = va.compare_text(ref, new)
    assert status == va.OK, detail


def test_text_comparison_reports_the_first_differing_line(tmp_path):
    ref = tmp_path / 'ref.txt'
    new = tmp_path / 'new.txt'
    ref.write_bytes(b'PHASE 26\nmean wait 3743.83\nend\n')
    new.write_bytes(b'PHASE 26\nmean wait 9999.99\nend\n')
    status, detail = va.compare_text(ref, new)
    assert status == va.MISMATCH
    assert 'line 2' in detail


def test_text_comparison_catches_a_truncated_file(tmp_path):
    ref = tmp_path / 'ref.txt'
    new = tmp_path / 'new.txt'
    ref.write_bytes(b'a\nb\nc\n')
    new.write_bytes(b'a\nb\n')
    status, detail = va.compare_text(ref, new)
    assert status == va.MISMATCH
    assert 'line count' in detail


# ---------------------------------------------------------------------------
# Which files are artefacts at all
# ---------------------------------------------------------------------------

TRACKED_SAMPLE = [
    '05_results/roi_analysis.py',
    '05_results/roi/cost_benefit_analysis.csv',
    '05_results/scaling/scaling_curves.png',
    '03_models/train_improved_model.py',
    '03_models/wait_model_v2.pkl',
    '02_data/improved_wait_dataset.csv',
    '02_data/dataset.csv',
    '02_data/load_real_traces.py',
    '02_data/lanl_trace_sample.csv',
    'phases_22_30/phase_26_scaling/scaling_benchmark.csv',
    'phases_22_30/phase_26_scaling/scaling_measurements.txt',
    'phases_22_30/COMPLETION_SUMMARY.md',
    'phases_22_30/PHASES_ROADMAP.md',
    'phases_22_30/phase_28_manuscript/manuscript.tex',
    'phases_22_30/phase_28_manuscript/manuscript.pdf',
    'README.md',
    'run_all_experiments.sh',
    'conftest.py',
]


def test_artefact_list_is_exactly_the_generated_files():
    assert va.artefact_paths(TRACKED_SAMPLE) == [
        '02_data/dataset.csv',
        '02_data/improved_wait_dataset.csv',
        '03_models/wait_model_v2.pkl',
        '05_results/roi/cost_benefit_analysis.csv',
        '05_results/scaling/scaling_curves.png',
        'phases_22_30/phase_26_scaling/scaling_benchmark.csv',
        'phases_22_30/phase_26_scaling/scaling_measurements.txt',
    ]


def test_artefact_list_excludes_every_script():
    assert not [p for p in va.artefact_paths(TRACKED_SAMPLE) if p.endswith('.py')]


@pytest.mark.parametrize('prose', [
    'phases_22_30/COMPLETION_SUMMARY.md',
    'phases_22_30/PHASES_ROADMAP.md',
    'phases_22_30/phase_28_manuscript/manuscript.tex',
    'phases_22_30/phase_28_manuscript/manuscript.pdf',
])
def test_handwritten_prose_is_never_an_artefact(prose):
    """No script produces these, so a diff would flag an author's edit."""
    assert not va.is_artefact(prose)


def test_windows_separators_are_normalised():
    assert va.is_artefact('05_results\\scaling\\scaling_analysis.csv')
    assert not va.is_artefact('phases_22_30\\PHASES_ROADMAP.md')

#!/usr/bin/env python3
"""Re-run the pipeline in a scratch copy and diff every tracked artefact.

The claim this tool makes checkable is the one the repository rests on: no
number exists unless a committed script regenerates it. It copies the tree into
a scratch directory, runs `run_all_experiments.sh` there, and diffs every
regenerated artefact against the committed blob.

    python tools/verify_artifacts.py            # copy tree, run, diff (~9 min)
    python tools/verify_artifacts.py --quick    # diff working tree vs HEAD
    python tools/verify_artifacts.py --smoke    # SMOKE=1 run; existence + headers

What is compared, and what deliberately is not:

  * .csv  cell by cell. Numeric columns with np.isclose(--rtol, --atol), every
          other column as an exact string. Every number this project publishes
          lives in a CSV, so this is the comparison that carries the claim.
  * .txt  exact, line by line, after rstrip -- so CRLF vs LF is never a finding.
  * .png  existence and non-zero size ONLY. Raster bytes depend on the
          matplotlib / freetype / zlib build that drew them, so two PNGs whose
          bytes differ routinely plot identical numbers.
  * .pkl  existence, non-zero size, and that pickle.load succeeds. A pickle
          embeds library version strings, so an identically-fitted model
          serialises to different bytes on a different machine.

  Neither exemption hides a number: a figure is drawn from a CSV and a model's
  measured quality is written to a CSV, and the CSV IS compared.

Exit status is 1 if any artefact is MISMATCH, MISSING or SCHEMA. TIMING (see
NONDETERMINISTIC_COLUMNS) and STALE alone exit 0; --strict-stale promotes STALE.
"""

from __future__ import annotations

import argparse
import io
import os
import pathlib
import re
import pickle
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import Counter

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# What counts as an artefact
# --------------------------------------------------------------------------

# Directories whose generated content the pipeline owns.
ARTEFACT_ROOTS = ('05_results/', 'phases_22_30/', '03_models/')

ARTEFACT_SUFFIXES = ('.csv', '.txt', '.png', '.pkl')

# Two generated inputs that live outside those roots: the training dataset and
# the phase-01 simulation dump. Everything else in 02_data is either raw trace
# data or a script.
EXTRA_ARTEFACTS = frozenset({
    '02_data/improved_wait_dataset.csv',
    '02_data/dataset.csv',
})

# Hand-written prose that lives under an artefact root but that no script
# produces. Diffing these would report an author's edit as a reproducibility
# failure.
EXCLUDED_ARTEFACTS = frozenset({
    'phases_22_30/COMPLETION_SUMMARY.md',
    'phases_22_30/PHASES_ROADMAP.md',
    'phases_22_30/phase_28_manuscript/manuscript.tex',
    'phases_22_30/phase_28_manuscript/manuscript.pdf',
})

# Wall-clock measurements. These columns time the machine, not the algorithm: a
# fresh-clone run of this repository reproduced every other column bit for bit
# while these drifted by up to 84%. Differences here are reported as TIMING and
# can never fail the run -- and the report names the exemption, so it is visible
# rather than hidden. Every OTHER column of these same files is compared
# strictly.
NONDETERMINISTIC_COLUMNS = {
    '05_results/model_comparison_table1.csv': frozenset({
        'training_time_sec',
    }),
    '05_results/scaling/scaling_analysis.csv': frozenset({
        'scheduler_overhead_sec',
        'inference_time_sec',
        'proactive_total_sim_time_sec',
    }),
    'phases_22_30/phase_26_scaling/scaling_benchmark.csv': frozenset({
        'inference_latency_ms',
        'throughput_overhead_pct',
    }),
}

# The same exemption, for TEXT reports that RENDER those columns. A wall-clock
# number does not stop being wall-clock because a script printed it into a .txt:
# phases_22_30/phase_26_scaling/scaling_measurements.txt tabulates the very
# inference_latency_ms and throughput_overhead_pct that are exempt in the CSV
# above, so comparing that file byte-for-byte failed for exactly the reason the
# column exemption exists. The mask is deliberately NARROW -- it blanks the two
# timing fields of each table row and the two summary lines that quote them, and
# leaves every other character on those lines, and every other line in the file,
# compared exactly. Wait and throughput sit in the same table and stay checked.
#
# Each entry maps a repo-relative path to (regex, description). A line matching
# the regex is compared with capture group 1 replaced by a placeholder.
NONDETERMINISTIC_TEXT = {
    'phases_22_30/phase_26_scaling/scaling_measurements.txt': [
        (re.compile(r'^(?:Small|Medium|Large|XLarge)\s+[\d.]+\s+[\d.]+(\s+[\d.]+\s+[\d.]+%)\s*$'),
         'latency and overhead columns of the metrics table'),
        (re.compile(r'^(Range: [\d.]+ - [\d.]+ ms .*)$'),
         'measured latency range'),
        (re.compile(r'^(Peak scheduling overhead: [\d.]+% of throughput)$'),
         'peak overhead'),
        (re.compile(r'^(Worst inference latency: [\d.]+ ms .*)$'),
         'worst latency'),
    ],
}

DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 1e-12
MAX_REPORTED_CELLS = 10
DETAIL_CLIP = 120

OK = 'OK'
MISMATCH = 'MISMATCH'
MISSING = 'MISSING'
SCHEMA = 'SCHEMA'
TIMING = 'TIMING'
STALE = 'STALE'
NOREF = 'NOREF'          # tracked but not in HEAD, so there is nothing to diff

FAILING = frozenset({MISMATCH, MISSING, SCHEMA})
STATUS_ORDER = (MISMATCH, SCHEMA, MISSING, STALE, TIMING, NOREF, OK)


def is_artefact(rel):
    """True if a repo-relative tracked path is an artefact this tool diffs."""
    rel = rel.replace('\\', '/')
    if rel in EXCLUDED_ARTEFACTS:
        return False
    if not rel.endswith(ARTEFACT_SUFFIXES):
        return False
    if rel in EXTRA_ARTEFACTS:
        return True
    return rel.startswith(ARTEFACT_ROOTS)


def artefact_paths(tracked):
    """Filter a list of tracked paths down to the artefacts, sorted."""
    return sorted({p.replace('\\', '/') for p in tracked if is_artefact(p)})


# --------------------------------------------------------------------------
# Comparison primitives (pure: paths in, (status, detail) out)
# --------------------------------------------------------------------------

def _clip(value):
    text = repr(value)
    return text if len(text) <= DETAIL_CLIP else text[:DETAIL_CLIP] + '...'


def _differing_mask(ref_col, new_col, rtol, atol):
    """Boolean mask of the cells that differ, dispatched on column dtype."""
    if pd.api.types.is_numeric_dtype(ref_col) and pd.api.types.is_numeric_dtype(new_col):
        a = ref_col.to_numpy(dtype=float)
        b = new_col.to_numpy(dtype=float)
        return ~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    # Anything non-numeric is compared as an exact string; fillna('') keeps a
    # missing cell from comparing unequal to itself.
    a = ref_col.fillna('').astype(str).to_numpy()
    b = new_col.fillna('').astype(str).to_numpy()
    return a != b


def compare_csv(rel, ref_path, new_path, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
    """Diff two CSVs cell by cell. Returns (status, detail)."""
    try:
        # round_trip parsing: the default float parser is faster but lossy, and
        # a lossy re-read would mask a real last-digit regression.
        ref = pd.read_csv(ref_path, float_precision='round_trip')
        new = pd.read_csv(new_path, float_precision='round_trip')
    except Exception as exc:                    # report it, do not abort the run
        return MISMATCH, f'unreadable as CSV: {exc}'

    if list(ref.columns) != list(new.columns):
        return SCHEMA, (f'columns differ: expected {list(ref.columns)}, '
                        f'got {list(new.columns)}')
    if ref.shape != new.shape:
        return MISMATCH, f'shape {new.shape} != expected {ref.shape}'

    exempt = NONDETERMINISTIC_COLUMNS.get(rel.replace('\\', '/'), frozenset())
    drifted_timing = []
    cells = []
    for col in ref.columns:
        bad = _differing_mask(ref[col], new[col], rtol, atol)
        count = int(np.count_nonzero(bad))
        if count == 0:
            continue
        if col in exempt:
            drifted_timing.append(f'{col} ({count} cells)')
            continue
        for pos in np.flatnonzero(bad):
            cells.append(f'row {int(pos)} col {col}: expected '
                         f'{_clip(ref[col].iloc[pos])}, got {_clip(new[col].iloc[pos])}')

    if cells:
        detail = '; '.join(cells[:MAX_REPORTED_CELLS])
        if len(cells) > MAX_REPORTED_CELLS:
            detail += f' (+{len(cells) - MAX_REPORTED_CELLS} more cells)'
        return MISMATCH, detail
    if drifted_timing:
        return TIMING, ('wall-clock columns only, exempt by NONDETERMINISTIC_COLUMNS: '
                        + ', '.join(drifted_timing))
    return OK, f'{len(ref)} rows x {len(ref.columns)} cols identical'


def compare_csv_header(ref_path, new_path):
    """Column-list check only, for --smoke: a smoke run writes reduced values."""
    try:
        ref = list(pd.read_csv(ref_path, nrows=0).columns)
        new = list(pd.read_csv(new_path, nrows=0).columns)
    except Exception as exc:
        return MISMATCH, f'unreadable as CSV: {exc}'
    if ref != new:
        return SCHEMA, f'columns differ: expected {ref}, got {new}'
    return OK, f'{len(ref)} columns match (smoke run: values not compared)'


def _mask_timing_lines(rel, lines):
    """Blank the wall-clock fields of `lines`, returning (masked, n_masked).

    Returns the lines untouched when the artefact declares no text exemption.
    """
    patterns = NONDETERMINISTIC_TEXT.get(rel)
    if not patterns:
        return lines, 0
    masked, count = [], 0
    for line in lines:
        for pattern, _why in patterns:
            match = pattern.match(line)
            if match:
                start, end = match.span(1)
                line = line[:start] + '<wall-clock>' + line[end:]
                count += 1
                break
        masked.append(line)
    return masked, count


def compare_text(rel, ref_path, new_path):
    """Diff two text artefacts line by line, ignoring line endings.

    Lines that render a wall-clock measurement are masked first; see
    NONDETERMINISTIC_TEXT for why, and which.
    """
    try:
        ref = pathlib.Path(ref_path).read_bytes().decode('utf-8')
        new = pathlib.Path(new_path).read_bytes().decode('utf-8')
    except UnicodeDecodeError as exc:
        return MISMATCH, f'not valid UTF-8: {exc}'
    # rstrip per line: the reference is a git blob (LF) while the regenerated
    # file may be CRLF, which is not a difference in content.
    ref_lines = [line.rstrip() for line in ref.splitlines()]
    new_lines = [line.rstrip() for line in new.splitlines()]
    if ref_lines == new_lines:
        return OK, f'{len(ref_lines)} lines identical'
    ref_lines, n_masked = _mask_timing_lines(rel, ref_lines)
    new_lines, _ = _mask_timing_lines(rel, new_lines)
    if ref_lines == new_lines:
        return TIMING, (f'identical apart from {n_masked} wall-clock line(s), '
                        f'exempt by NONDETERMINISTIC_TEXT')
    for lineno, (a, b) in enumerate(zip(ref_lines, new_lines), start=1):
        if a != b:
            return MISMATCH, (f'first difference at line {lineno}: '
                              f'expected {_clip(a)}, got {_clip(b)}')
    return MISMATCH, (f'line count differs: expected {len(ref_lines)}, '
                      f'got {len(new_lines)} (common prefix identical)')


def compare_binary(rel, new_path):
    """Existence / size check for .png and .pkl; see the module docstring."""
    size = pathlib.Path(new_path).stat().st_size
    if size == 0:
        return MISMATCH, 'empty file'
    if pathlib.PurePosixPath(rel.replace('\\', '/')).suffix == '.pkl':
        try:
            # Not untrusted input: this is the model the pipeline wrote minutes
            # ago in our own scratch tree. Loading it is the only way to tell a
            # real model from a truncated file that merely has a plausible size.
            with open(new_path, 'rb') as handle:
                pickle.load(handle)
        except Exception as exc:
            return MISMATCH, f'pickle.load failed: {exc}'
        return OK, f'{size} bytes, unpickles (bytes not compared)'
    return OK, f'{size} bytes (bytes not compared)'


def compare_artefact(rel, ref_path, new_path, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL,
                     header_only=False):
    """Dispatch on suffix. Returns (status, detail)."""
    ref_path = pathlib.Path(ref_path)
    new_path = pathlib.Path(new_path)
    if not new_path.exists():
        return MISSING, 'not produced by the pipeline'
    if not ref_path.exists():
        return NOREF, 'tracked but not in HEAD; nothing to compare against'
    suffix = pathlib.PurePosixPath(rel.replace('\\', '/')).suffix
    if suffix == '.csv':
        if header_only:
            return compare_csv_header(ref_path, new_path)
        return compare_csv(rel, ref_path, new_path, rtol=rtol, atol=atol)
    if suffix == '.txt':
        if header_only:
            size = new_path.stat().st_size
            if size == 0:
                return MISMATCH, 'empty file'
            return OK, f'{size} bytes (smoke run: content not compared)'
        return compare_text(rel, ref_path, new_path)
    return compare_binary(rel, new_path)


# --------------------------------------------------------------------------
# git plumbing
# --------------------------------------------------------------------------

def _git(root, *args):
    return subprocess.run(['git', *args], cwd=str(root), stdout=subprocess.PIPE,
                          check=True).stdout


def list_tracked(root):
    """Every tracked path, NUL-separated so no name needs quoting."""
    raw = _git(root, 'ls-files', '-z').decode('utf-8')
    return [p.replace('\\', '/') for p in raw.split('\0') if p]


def list_head(root):
    """Every path in HEAD. A staged-only file is tracked but is not in HEAD."""
    raw = _git(root, 'ls-tree', '-r', '--name-only', '-z', 'HEAD').decode('utf-8')
    return {p.replace('\\', '/') for p in raw.split('\0') if p}


def _chunked(paths, budget=12000):
    """Split a pathspec list so no single command line grows unbounded."""
    chunk, size = [], 0
    for path in paths:
        if chunk and size + len(path) + 1 > budget:
            yield chunk
            chunk, size = [], 0
        chunk.append(path)
        size += len(path) + 1
    if chunk:
        yield chunk


def extract_head(root, paths, dest):
    """Extract HEAD blobs into `dest` via tar.

    `git archive` emits the raw blob, so the reference copy has LF endings
    whatever core.autocrlf did to the working tree. A checkout-based reference
    would report every text artefact as different on a Windows clone.
    """
    dest = pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for chunk in _chunked(list(paths)):
        blob = _git(root, 'archive', '--format=tar', 'HEAD', '--', *chunk)
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            tar.extractall(dest, filter='data')
    return dest


def populate_tree(root, tracked, dest):
    """Copy every tracked file into the scratch tree, uncommitted edits included."""
    root = pathlib.Path(root)
    dest = pathlib.Path(dest)
    copied = 0
    for rel in tracked:
        src = root / rel
        if not src.is_file():
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)        # copy2 keeps mtime: the staleness baseline
        copied += 1
    return copied


# --------------------------------------------------------------------------
# Running the pipeline
# --------------------------------------------------------------------------

def find_bash(override=None):
    """Locate a bash that can run the pipeline.

    On Windows the `bash` first on PATH is usually the WSL stub
    (C:/Windows/System32/bash.exe). WSL sees a Linux filesystem, so it cannot
    exec the Windows interpreter PY points at and the pipeline dies on step 1.
    Git for Windows ships a bash that can, so derive that one from git itself:
    walk up from `git --exec-path` looking for bin/bash.exe.
    """
    if override:
        return override
    if os.name == 'nt':
        try:
            exec_path = subprocess.run(['git', '--exec-path'], capture_output=True,
                                       text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            exec_path = ''
        if exec_path:
            here = pathlib.Path(exec_path)
            while True:
                candidate = here / 'bin' / 'bash.exe'
                if candidate.is_file():
                    return str(candidate)
                if here.parent == here:
                    break
                here = here.parent
    found = shutil.which('bash')
    if not found:
        raise SystemExit('no bash found; pass --bash PATH')
    return found


def pipeline_env(smoke=False):
    """Environment for the scratch run: fixed interpreter, encoding, backend."""
    env = dict(os.environ)
    # PY: the pipeline honours it, so the scratch run uses THIS interpreter and
    # not whatever `python3` resolves to (on Windows, the Store shim).
    env['PY'] = sys.executable
    env['PYTHONUTF8'] = '1'
    env['PYTHONHASHSEED'] = '0'
    env['MPLBACKEND'] = 'Agg'            # no display in a scratch run or in CI
    if smoke:
        env['SMOKE'] = '1'
    else:
        env.pop('SMOKE', None)
    return env


def run_pipeline(tree, smoke=False, bash=None):
    """Run run_all_experiments.sh inside the scratch tree. Returns its exit code."""
    shell = find_bash(bash)
    print(f'running: {shell} run_all_experiments.sh  (cwd={tree}, '
          f'SMOKE={"1" if smoke else "0"})', flush=True)
    proc = subprocess.run([shell, 'run_all_experiments.sh'], cwd=str(tree),
                          env=pipeline_env(smoke))
    return proc.returncode


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _mtime(path):
    try:
        return pathlib.Path(path).stat().st_mtime
    except OSError:
        return None


def render_report(results, elapsed, mode, tree, pipeline_rc=None, strict_stale=False):
    """Markdown report: one row per artefact, then the exemptions and a count."""
    counts = Counter(status for status, _, _ in results)
    escaped_pipe = '\\|'
    lines = ['# Artefact verification', '',
             f'mode: {mode}', f'tree: {tree}', '']
    if pipeline_rc is not None:
        lines += [f'pipeline exit code: {pipeline_rc}', '']
    lines += ['| status | artefact | detail |', '| --- | --- | --- |']
    ordered = sorted(results, key=lambda row: (STATUS_ORDER.index(row[0]), row[1]))
    for status, rel, detail in ordered:
        lines.append(f'| {status} | `{rel}` | {detail.replace("|", escaped_pipe)} |')
    lines += ['', '## Timing exemptions', '',
              'These columns measure wall-clock time -- the machine, not the algorithm',
              '-- and cannot reproduce across machines. A difference in one of them is',
              'reported as TIMING and never fails the run. Every OTHER column of these',
              'same files is compared strictly:', '']
    for path, cols in sorted(NONDETERMINISTIC_COLUMNS.items()):
        lines.append(f'* `{path}`: {", ".join(sorted(cols))}')
    if NONDETERMINISTIC_TEXT:
        lines += ['',
                  'The same exemption applies to the text reports that RENDER those',
                  'columns -- a wall-clock number does not stop being wall-clock because',
                  'a script printed it into a .txt. Only the named fields are masked;',
                  'every other field on those lines, and every other line, is compared',
                  'exactly:', '']
        for path, patterns in sorted(NONDETERMINISTIC_TEXT.items()):
            for _pattern, why in patterns:
                lines.append(f'* `{path}`: {why}')
    summary = ', '.join(f'{counts[status]} {status}'
                        for status in STATUS_ORDER if counts[status])
    lines += ['', f'{len(results)} artefacts: {summary or "none"} '
                  f'-- {elapsed:.1f} s elapsed']
    if counts[STALE] and not strict_stale:
        lines.append('STALE means no script rewrote the file; rerun with '
                     '--strict-stale to fail on it.')
    return '\n'.join(lines) + '\n'


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description='Regenerate every artefact in a scratch copy and diff it '
                    'against the committed blob.')
    parser.add_argument('--quick', action='store_true',
                        help='diff the working tree against HEAD; no pipeline run')
    parser.add_argument('--smoke', action='store_true',
                        help='SMOKE=1 run; existence and CSV headers only')
    parser.add_argument('--strict-stale', action='store_true',
                        help='treat STALE (artefact not rewritten) as a failure')
    parser.add_argument('--tree', default=None,
                        help='tree that --quick diffs (default: this repository)')
    parser.add_argument('--workdir', default=None,
                        help='scratch directory (default: a fresh temp dir, removed)')
    parser.add_argument('--keep', action='store_true',
                        help='keep the scratch directory')
    parser.add_argument('--source', choices=('worktree', 'HEAD'), default='worktree',
                        help='populate the scratch tree from the worktree '
                             '(default; includes uncommitted edits) or from HEAD')
    parser.add_argument('--bash', default=None,
                        help='bash to run the pipeline with')
    parser.add_argument('--report', default=None,
                        help='also write the markdown report to this path')
    parser.add_argument('--rtol', type=float, default=DEFAULT_RTOL,
                        help=f'relative tolerance for numeric cells (default {DEFAULT_RTOL})')
    parser.add_argument('--atol', type=float, default=DEFAULT_ATOL,
                        help=f'absolute tolerance for numeric cells (default {DEFAULT_ATOL})')
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    started = time.time()
    root = pathlib.Path(__file__).resolve().parent.parent

    tracked = list_tracked(root)
    artefacts = artefact_paths(tracked)
    in_head = list_head(root)

    # An explicit --workdir is the caller's directory: never delete it.
    keep = args.keep or args.workdir is not None
    workdir = (pathlib.Path(args.workdir) if args.workdir
               else pathlib.Path(tempfile.mkdtemp(prefix='verify_artifacts_')))
    workdir.mkdir(parents=True, exist_ok=True)

    pipeline_rc = None
    baseline = None
    results = []
    try:
        ref_dir = extract_head(root, [p for p in artefacts if p in in_head],
                               workdir / 'ref')
        if args.quick:
            tree = pathlib.Path(args.tree).resolve() if args.tree else root
            mode = 'quick (working tree vs HEAD, no pipeline run)'
        else:
            tree = workdir / 'tree'
            if args.source == 'HEAD':
                extract_head(root, [p for p in tracked if p in in_head], tree)
            else:
                populate_tree(root, tracked, tree)
            # Baseline taken BEFORE the run: an artefact whose mtime has not
            # advanced afterwards was not rewritten by any script.
            baseline = {rel: _mtime(tree / rel) for rel in artefacts}
            pipeline_rc = run_pipeline(tree, smoke=args.smoke, bash=args.bash)
            mode = 'smoke (SMOKE=1 run)' if args.smoke else f'full ({args.source})'

        for rel in artefacts:
            status, detail = compare_artefact(rel, ref_dir / rel, tree / rel,
                                              rtol=args.rtol, atol=args.atol,
                                              header_only=args.smoke)
            # --quick has no baseline, so it never reports STALE.
            if baseline is not None and status in (OK, TIMING):
                before, after = baseline.get(rel), _mtime(tree / rel)
                if before is not None and after is not None and after <= before:
                    status, detail = STALE, 'not rewritten by the pipeline'
            results.append((status, rel, detail))

        report = render_report(results, time.time() - started, mode, tree,
                               pipeline_rc=pipeline_rc, strict_stale=args.strict_stale)
        print(report)
        if args.report:
            pathlib.Path(args.report).write_text(report, encoding='utf-8')
            print(f'report written to {args.report}')
        if keep:
            print(f'scratch kept at {workdir}')
    finally:
        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)

    counts = Counter(status for status, _, _ in results)
    failed = sum(counts[status] for status in FAILING)
    if args.strict_stale:
        failed += counts[STALE]
    if pipeline_rc:
        failed += 1
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())

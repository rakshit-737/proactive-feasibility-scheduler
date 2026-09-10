"""Guards for the three ways this repository used to degrade quietly.

Every test here protects the same principle from a different angle: a component
that meets an input it does not recognise must either handle it correctly or say
so out loud. It must never produce output that is indistinguishable from success.

  1. READER UNIFICATION -- `02_data/load_real_traces.py` kept a private
     `open(path, 'r', errors='ignore')` after the rest of the project moved to
     the one gzip-aware reader in `02_data/swf_io.py`. Only the `.swf.gz` copies
     of the Parallel Workloads Archive traces are committed, so that reader
     handed a compressed trace decoded the gzip bytes as text and parsed ZERO
     rows -- an unreadable trace that looks exactly like an empty one.

  2. TABLE MISMATCH -- `phases_22_30/phase_24_extended_schedulers/
     scheduler_comparison.py` filled unknown scheduler keys with
     `{'type': 'unknown', 'reference': 'n/a'}`, so a renamed scheduler reached
     the published comparison table as a row of real numbers with fabricated
     provenance.

  3. LABEL FALLBACK -- `vizstyle.label_of` returned the raw key for anything it
     could not name, so a not-yet-regenerated CSV rendered a raw 'PRIORITY' tick
     into a figure with nothing reporting it. `color_of` keeps its silent
     fallback on purpose; the asymmetry is asserted below so a later "tidy-up"
     cannot collapse the two.

Nothing here writes a tracked artefact: the SWF fixtures are built in `tmp_path`,
and the one Phase 24 function under test that writes (`write_novelty_claim`) is
called only through the `redirected_claim` fixture, which points `OUTPUT_CLAIM`
into `tmp_path` and then asserts the tracked file was not touched.
"""

import ast
import gzip
import importlib.util
import os
import sys
import warnings

import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE24_PATH = os.path.join(PROJECT_ROOT, 'phases_22_30',
                            'phase_24_extended_schedulers',
                            'scheduler_comparison.py')

# Two SWF data lines in the real 18-column Standard Workload Format order.
# `load_real_traces.parse_lanl_swf` reads indices 1 (submit), 2 (wait),
# 3 (runtime), 4 (allocated procs) and 7 (requested procs); everything else is
# the archive's "not recorded" sentinel -1.
SWF_TEXT = (
    '; MaxProcs: 1024\n'
    '; Note: a header line with a stray byte is skipped by the parser\n'
    '1 100 5 60 4 -1 -1 4 120 -1 1 -1 -1 -1 -1 -1 -1 -1\n'
    '2 200 0 30 2 -1 -1 -1 60 -1 1 -1 -1 -1 -1 -1 -1 -1\n'
)
# What SWF_TEXT must parse to. Row 2 exercises the '-1' requested-processors
# sentinel falling back to allocated processors.
EXPECTED_ROWS = [
    {'arrival_time': 100, 'wait_time': 5, 'runtime': 60, 'num_gpus': 4},
    {'arrival_time': 200, 'wait_time': 0, 'runtime': 30, 'num_gpus': 2},
]


@pytest.fixture(scope='module')
def lrt():
    """`02_data/load_real_traces.py` (conftest puts 02_data on sys.path)."""
    import load_real_traces
    return load_real_traces


@pytest.fixture(scope='module')
def phase24():
    """Phase 24's comparison script, loaded by path.

    `phases_22_30/` holds no importable packages, so the module is loaded from
    its file. Importing it has no side effects: everything that writes lives
    behind `main()`.
    """
    pytest.importorskip('scipy', reason='scheduler_comparison imports scipy.stats')
    spec = importlib.util.spec_from_file_location(
        'phase24_scheduler_comparison_under_test', PHASE24_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_gz(path, text=SWF_TEXT):
    with gzip.open(path, 'wb') as fh:
        fh.write(text.encode('utf-8'))
    return str(path)


def _assert_parsed(df):
    assert list(df.columns) == ['arrival_time', 'wait_time', 'runtime', 'num_gpus']
    assert df.to_dict('records') == EXPECTED_ROWS


# ─────────────────────────────────────────────────────────────────────────────
# 1. One gzip-aware SWF reader, used by every SWF reader in the project
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_lanl_swf_falls_back_to_the_committed_gz_copy(lrt, tmp_path):
    """INVARIANT: a plain `.swf` path is satisfied by the `.swf.gz` beside it.

    This is the only form that exists in a fresh clone -- `.gitignore` excludes
    `*.swf` -- so a reader without the fallback cannot open a committed trace at
    all. With the old bare `open()` this raises FileNotFoundError.
    """
    _write_gz(tmp_path / 'trace.swf.gz')
    assert not os.path.exists(tmp_path / 'trace.swf')

    _assert_parsed(lrt.parse_lanl_swf(str(tmp_path / 'trace.swf')))


def test_parse_lanl_swf_decompresses_a_gz_path_given_directly(lrt, tmp_path):
    """INVARIANT: `--input .../LANL-CM5-1994-4.1-cln.swf.gz` parses the trace.

    A user pointing `--input` at a committed `.swf.gz` is the exact scenario the
    private `open(..., errors='ignore')` mishandled: gzip bytes decoded as text
    yield no line with >= 9 whitespace-separated fields, so the parser returned
    an EMPTY DataFrame and the run reported '0 rows' as though the trace were
    empty rather than unreadable.
    """
    gz_path = _write_gz(tmp_path / 'trace.swf.gz')

    df = lrt.parse_lanl_swf(gz_path)
    assert len(df) == len(EXPECTED_ROWS), (
        'a .swf.gz handed in directly parsed to '
        f'{len(df)} rows -- it is being read as text, not decompressed')
    _assert_parsed(df)


def test_parse_lanl_swf_still_reads_an_uncompressed_trace(lrt, tmp_path):
    """The gzip support must not cost the plain path: both forms parse alike."""
    plain = tmp_path / 'trace.swf'
    plain.write_text(SWF_TEXT, encoding='utf-8')

    _assert_parsed(lrt.parse_lanl_swf(str(plain)))


def test_parse_lanl_swf_goes_through_swf_io_rather_than_its_own_open(lrt, tmp_path,
                                                                    monkeypatch):
    """INVARIANT: the reader is SHARED, not reimplemented.

    Two separate assertions, because two different mistakes are possible: the
    module could stop importing `swf_io.open_swf` (identity check), or it could
    import it and still open the file some other way (call check). A second
    local implementation of the gzip fallback is how the readers drifted apart
    in the first place.
    """
    import swf_io

    assert lrt.open_swf is swf_io.open_swf, (
        'load_real_traces must use 02_data/swf_io.open_swf itself, not a copy')

    gz_path = _write_gz(tmp_path / 'trace.swf.gz')
    calls = []

    def recording_open_swf(path):
        calls.append(path)
        return swf_io.open_swf(path)

    monkeypatch.setattr(lrt, 'open_swf', recording_open_swf)
    lrt.parse_lanl_swf(gz_path)

    assert calls == [gz_path], (
        f'parse_lanl_swf opened the trace without the shared reader: {calls}')


def test_main_converts_a_gzipped_input_to_the_four_column_schema(lrt, tmp_path,
                                                                 monkeypatch):
    """End-to-end on the CLI path: `--input <trace>.swf.gz` writes real rows."""
    gz_path = _write_gz(tmp_path / 'trace.swf.gz')
    out_csv = tmp_path / 'out' / 'converted.csv'
    monkeypatch.setattr(sys, 'argv',
                        ['load_real_traces.py', '--input', gz_path,
                         '--output', str(out_csv)])

    lrt.main()

    _assert_parsed(pd.read_csv(out_csv))


def test_main_reports_a_missing_input_by_naming_both_candidate_paths(lrt, tmp_path,
                                                                    monkeypatch):
    """A missing `--input` exits cleanly and says which files were looked for.

    The old code pre-checked `os.path.exists(args.input)`, which is a SECOND
    definition of "present" that disagrees with the reader's: it rejected a
    plain `.swf` path whose `.swf.gz` was sitting right there. The check now
    lives only in `open_swf`, so the message names both candidates.
    """
    missing = str(tmp_path / 'absent.swf')
    monkeypatch.setattr(sys, 'argv',
                        ['load_real_traces.py', '--input', missing,
                         '--output', str(tmp_path / 'unused.csv')])

    with pytest.raises(SystemExit) as excinfo:
        lrt.main()

    message = str(excinfo.value)
    assert 'absent.swf' in message and 'absent.swf.gz' in message, message
    assert not os.path.exists(tmp_path / 'unused.csv')


# ─────────────────────────────────────────────────────────────────────────────
# 2. Phase 24: an unrecognised scheduler key is a hard error
# ─────────────────────────────────────────────────────────────────────────────

def _summary_row(scheduler, mean_wait=10.0):
    return {
        'scheduler': scheduler,
        'mean_wait': mean_wait,
        'max_wait': 100.0,
        'fairness_gini': 0.5,
        'throughput': 0.35,
        'gpu_util': 0.6,
        'pred_wait_mae': float('nan'),
    }


def _improvement_row(wait_pct):
    """One entry of the `improvements` mapping `write_novelty_claim` consumes."""
    return {
        'wait_improvement_pct': wait_pct,
        'throughput_improvement_pct': 1.0,
        'util_improvement_pct': 1.0,
        'fairness_improvement_pct': 1.0,
    }


@pytest.fixture
def redirected_claim(phase24, tmp_path, monkeypatch):
    """Send `novelty_claim.txt` to `tmp_path`, and prove the tracked one is untouched.

    `write_novelty_claim` is the only function exercised here that writes, and
    what it writes is a committed artefact. The fixture yields the temporary
    destination and, on teardown, fails if the real file changed -- so a future
    edit that drops the redirection is caught by the suite rather than by
    `git status`.
    """
    tracked = phase24.OUTPUT_CLAIM
    before = os.stat(tracked).st_mtime_ns if os.path.exists(tracked) else None
    target = tmp_path / 'novelty_claim.txt'
    monkeypatch.setattr(phase24, 'OUTPUT_CLAIM', str(target))

    yield target

    after = os.stat(tracked).st_mtime_ns if os.path.exists(tracked) else None
    assert after == before, f'the test wrote the tracked artefact {tracked}'


def test_build_comparison_dataframe_rejects_an_unknown_scheduler_key(phase24):
    """INVARIANT: a key the SCHEDULERS table does not know stops the phase.

    The old default produced a row with measured metrics, the type 'unknown'
    and the reference 'n/a'. Published beside vetted rows in
    `baseline_comparison.csv`, that row is indistinguishable from a scheduler
    somebody actually documented.
    """
    summary = pd.DataFrame([_summary_row('FIFO'), _summary_row('NOT_A_SCHEDULER'),
                            _summary_row('PROACTIVE')])

    with pytest.raises(SystemExit) as excinfo:
        phase24.build_comparison_dataframe(summary)

    message = str(excinfo.value)
    # The error has to be actionable on its own: which key, which CSV it came
    # from, and which table it is missing from.
    assert 'NOT_A_SCHEDULER' in message, message
    assert 'multi_scheduler_benchmark.csv' in message, message
    assert 'SCHEDULERS' in message and 'scheduler_comparison.py' in message, message


def test_the_renamed_priority_key_is_rejected_rather_than_labelled_unknown(phase24):
    """The concrete rename this guard exists for: PRIORITY -> STATIC_PRIORITY.

    A results CSV written before the rename still carries 'PRIORITY'. Phase 24
    must refuse it instead of publishing it as an 'unknown' scheduler.
    """
    assert 'PRIORITY' not in phase24.SCHEDULERS, (
        'the retired key must not be re-added as an alias; regenerate the CSV')

    summary = pd.DataFrame([_summary_row('FIFO'), _summary_row('PRIORITY'),
                            _summary_row('PROACTIVE')])
    with pytest.raises(SystemExit, match='PRIORITY'):
        phase24.build_comparison_dataframe(summary)


def test_build_comparison_dataframe_keeps_documented_metadata_for_known_keys(phase24):
    """The strict lookup must not break the normal path.

    Every row still carries the name/type/reference from the SCHEDULERS table,
    and no row may carry the placeholder metadata the old fallback invented.
    """
    keys = ['FIFO', 'STATIC_PRIORITY', 'PROACTIVE']
    summary = pd.DataFrame([_summary_row(k, mean_wait=10.0 + i)
                            for i, k in enumerate(keys)])

    out = phase24.build_comparison_dataframe(summary)

    assert list(out['scheduler']) == keys
    assert list(out['scheduler_name']) == [phase24.SCHEDULERS[k]['name'] for k in keys]
    assert 'unknown' not in set(out['scheduler_type'])
    assert 'n/a' not in set(out['reference'])


def test_scheduler_info_refuses_an_unknown_key_however_the_lookup_is_written(phase24):
    """INVARIANT: the one lookup helper raises on a key the table does not have.

    Every scheduler lookup in the script goes through `_scheduler_info`, so this
    is the narrowest place the guarantee can be pinned -- and it is pinned as
    BEHAVIOUR (what the function does with an unknown key) rather than as the
    shape of the code. A silent fallback has unlimited spellings --
    `SCHEDULERS.get(k, default)`, the same call through a local alias,
    `try: SCHEDULERS[k] except KeyError:`, `k if k in SCHEDULERS else ...`, a
    `dict(SCHEDULERS)` copy -- and every one of them fails this assertion, while
    a check on the source can only ever enumerate the spellings somebody
    happened to think of.
    """
    with pytest.raises(SystemExit) as excinfo:
        phase24._scheduler_info('NOT_A_SCHEDULER')

    message = str(excinfo.value)
    assert 'NOT_A_SCHEDULER' in message, message
    assert 'SCHEDULERS' in message and 'scheduler_comparison.py' in message, message

    # ...and the strictness costs the normal path nothing: a known key still
    # returns the table's own entry, not a copy or a reconstruction of it.
    assert phase24._scheduler_info('FIFO') is phase24.SCHEDULERS['FIFO']


def test_write_novelty_claim_also_refuses_an_unknown_scheduler_key(phase24,
                                                                   redirected_claim):
    """INVARIANT: the OTHER consumer of the table refuses the key too.

    `build_comparison_dataframe` was never the only lookup: `write_novelty_claim`
    labels the ranked rows and the head-to-head section the same way, and it is
    the function that writes `novelty_claim.txt` -- the artefact the manuscript
    quotes. Fixing the lookup in one function and leaving it defaulted in the
    other is how a fix looks complete while the defect survives in the file
    nobody diffs, so this asserts the second function's behaviour directly
    instead of inferring it from the source.

    Both of its lookups feed off the same `improvements` mapping, so the ranking
    loop is the one an unknown key reaches first; the syntactic guard below is
    what watches the head-to-head site separately.
    """
    comparison = phase24.build_comparison_dataframe(
        pd.DataFrame([_summary_row('FIFO'), _summary_row('PROACTIVE')]))
    improvements = {
        'PROACTIVE': _improvement_row(12.0),
        'NOT_A_SCHEDULER': _improvement_row(5.0),
        'FIFO': _improvement_row(0.0),
    }

    with pytest.raises(SystemExit) as excinfo:
        phase24.write_novelty_claim(comparison, improvements, None)

    assert 'NOT_A_SCHEDULER' in str(excinfo.value), str(excinfo.value)

    # This second assertion is not decoration: the function has two lookups, so
    # degrading only the first still ends in SystemExit from the second -- and a
    # test that stopped at `raises` would pass while the ranked list had already
    # been written with a fabricated label. What the artefact CONTAINS is the
    # thing that matters, so it is what is asserted. A half-written claim file
    # is obviously broken; a plausible one is not.
    partial = (redirected_claim.read_text(encoding='utf-8')
               if redirected_claim.exists() else '')
    assert 'NOT_A_SCHEDULER' not in partial, partial


def _scheduler_table_aliases(tree):
    """Every name bound to the SCHEDULERS dict: the table itself plus its aliases.

    Iterated to a fixed point so a chain (`t = SCHEDULERS; u = t`) is followed,
    and applied to plain assignments, annotated assignments, walrus bindings and
    parameter defaults (`def f(table=SCHEDULERS)`) -- the cheap ways to give the
    table a second name and slip a defaulted lookup past a check that only knows
    the literal spelling.
    """
    aliases = {'SCHEDULERS'}
    while True:
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                positional = a.posonlyargs + a.args
                bound = list(zip(positional[len(positional) - len(a.defaults):],
                                 a.defaults))
                bound += [(arg, d) for arg, d in zip(a.kwonlyargs, a.kw_defaults) if d]
                found |= {arg.arg for arg, d in bound
                          if isinstance(d, ast.Name) and d.id in aliases}
                continue
            if isinstance(node, ast.Assign):
                value, targets = node.value, node.targets
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
                value, targets = node.value, [node.target]
            else:
                continue
            if isinstance(value, ast.Name) and value.id in aliases:
                found |= {t.id for t in targets if isinstance(t, ast.Name)}
        if found <= aliases:
            return aliases
        aliases |= found


def test_no_defaulted_scheduler_table_lookup_survives_anywhere_in_phase_24():
    """SECONDARY guard. The functional tests above are the real one.

    This reads the source, so it can only ever catch spellings it was taught:
    a silent lookup written as `try/except KeyError`, as a membership test, or
    against a copy of the table walks straight past it. That is not a defect to
    be patched by adding one more pattern -- it is why the two tests above feed
    the module an unrecognised key and assert on what it DOES, which no
    rewriting of the lookup can evade. This one is kept only because it is the
    only check that sees each call site individually, including the head-to-head
    site in `write_novelty_claim` that an unknown key never reaches (the ranking
    loop raises first), and because it names the offending line.

    Broadened from the original, which matched only a Call whose func was
    literally the attribute `SCHEDULERS.get`: the reviewer's evasion was to bind
    the table to a local name first. Any `.get` on the table under ANY of its
    names now fails, with or without a default -- inside `_scheduler_info` the
    sanctioned read is the subscript `SCHEDULERS[key]`, so no `.get` on this
    table has a legitimate caller.
    """
    tree = ast.parse(open(PHASE24_PATH, encoding='utf-8').read(), PHASE24_PATH)

    # Parsed, not grepped: the string 'SCHEDULERS.get(' appears in the docstring
    # that explains why the pattern was removed, and a text search would match
    # its own tombstone.
    aliases = _scheduler_table_aliases(tree)
    defaulted = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'get'):
            continue
        receiver = node.func.value
        if not isinstance(receiver, ast.Name):
            continue
        bound_form = receiver.id in aliases
        # The unbound spelling of the same call: dict.get(SCHEDULERS, key, ...).
        unbound_form = (receiver.id == 'dict' and node.args
                        and isinstance(node.args[0], ast.Name)
                        and node.args[0].id in aliases)
        if bound_form or unbound_form:
            defaulted.append(f'line {node.lineno}: {ast.unparse(node)[:70]}')

    assert not defaulted, (
        f'a defaulted lookup on the SCHEDULERS table is back in '
        f'scheduler_comparison.py -- {defaulted}; route it through '
        f'_scheduler_info() so an unknown key fails loudly instead of '
        f'producing a row labelled "unknown". Aliases checked: {sorted(aliases)}')


# ─────────────────────────────────────────────────────────────────────────────
# 3. vizstyle: a label fallback is visible, a colour fallback stays silent
# ─────────────────────────────────────────────────────────────────────────────

def test_label_of_warns_when_it_cannot_name_a_key():
    """INVARIANT: an unnamed key is reported, not quietly drawn as a raw tick.

    The value still comes back so a plotting run that is 40 minutes deep does
    not die at the last draw call -- but the run now says which key it could not
    name, which is the whole difference between a fallback and a silent defect.
    """
    import vizstyle

    with pytest.warns(vizstyle.UnknownPolicyLabel, match='PRIORITY'):
        assert vizstyle.label_of('PRIORITY') == 'PRIORITY'

    # The warning must name the key, not just complain in general.
    with pytest.warns(vizstyle.UnknownPolicyLabel) as record:
        vizstyle.label_of('SOME_NEW_POLICY')
    assert any('SOME_NEW_POLICY' in str(w.message) for w in record)


def test_the_retired_priority_key_has_no_alias_in_vizstyle():
    """DECISION, asserted: no permanent alias for the renamed scheduler.

    'PRIORITY' was renamed to 'STATIC_PRIORITY' because the old name asserts
    aging the scheduler does not do. Adding 'PRIORITY' back as a label alias
    would make a stale CSV render silently and forever -- exactly the class of
    mistake the warning above exists to expose. The artefacts get regenerated
    instead.
    """
    import vizstyle

    assert 'PRIORITY' not in vizstyle.POLICY_LABEL
    assert 'PRIORITY' not in vizstyle.POLICY_ROLE
    assert vizstyle.POLICY_LABEL['STATIC_PRIORITY'] == 'Static priority'


def test_label_of_is_silent_and_correct_for_every_key_it_knows():
    """No warning may fire on the normal path, in either spelling.

    Results CSVs store keys upper-cased and the benchmark loops use lower-case,
    so a case-sensitive membership test would warn on half the real call sites
    and train everyone to ignore the warning.
    """
    import vizstyle

    with warnings.catch_warnings():
        warnings.simplefilter('error')  # any warning here fails the test
        for key, expected in vizstyle.POLICY_LABEL.items():
            assert vizstyle.label_of(key) == expected
            assert vizstyle.label_of(key.lower()) == expected


def test_every_policy_vizstyle_can_colour_is_a_policy_it_can_name():
    """INVARIANT: POLICY_ROLE and POLICY_LABEL share one vocabulary.

    A key with a role but no label is a key that colours correctly and then
    warns when it is drawn -- a trap for the next caller, and the residue a
    rename leaves when only one of the two tables is updated.
    """
    import vizstyle

    unnamed = sorted(k for k in vizstyle.POLICY_ROLE if k not in vizstyle.POLICY_LABEL)
    assert not unnamed, (
        f'POLICY_ROLE keys with no POLICY_LABEL entry: {unnamed} -- label_of '
        f'will warn and draw the raw key for each of them')


def test_color_of_keeps_its_silent_fallback_on_purpose():
    """INVARIANT: the asymmetry between the two fallbacks is intentional.

    `color_of` must NOT warn. Its fallback is correct rather than merely
    tolerable -- baseline grey is the recessive ink, so an unrecognised policy
    renders de-emphasised and asserts nothing false -- and callers depend on it:
    `phase_27_fairness/fairness_sla_analysis.py` asks for
    `color_of(POLICY_KEY.get(k, k))` while supplying its own name for the tick,
    and `04_scheduler/estimate_sensitivity.py` does the same for its modal
    variants. Making colour strict would break those runs; making it warn would
    make the label warning noise.
    """
    import vizstyle

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for mode in ('light', 'dark'):
            fallback = vizstyle.color_of('NOT_A_POLICY', mode)
            assert fallback == vizstyle.PALETTE[mode]['muted']
            assert fallback == vizstyle.color_of('FIFO', mode)

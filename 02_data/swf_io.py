"""One gzip-aware reader for the Parallel Workloads Archive SWF traces.

WHY THIS EXISTS
---------------
Only the compressed traces are committed -- `.gitignore` excludes `*.swf`, so a
fresh clone has `LANL-CM5-1994-4.1-cln.swf.gz` and `SDSC-SP2-1998-4.2-cln.swf.gz`
and neither expanded file. Every reader in the project therefore has to accept
either form, and every reader has to accept it the SAME way: when the gzip
fallback lived in one script and not another, that second script could not run
on a fresh clone at all while continuing to work on a machine where the
uncompressed file happened to be lying around. This module is the single
implementation both `02_data/build_real_trace_datasets.py` and
`04_scheduler/trace_driven_benchmark.py` call.
"""

import gzip
import io
import os


def open_swf(path):
    """Open an SWF trace, transparently falling back to the gzipped copy.

    Callers pass the plain `.swf` path; if it is absent the `.swf.gz` beside it
    is read instead. Decoding is `utf-8` with `errors='replace'` because the
    archive headers carry occasional non-UTF-8 bytes in site/contact lines, and
    a trace must not fail to load over a comment the parser skips anyway.

    Pass the PLAIN name. An existing path is opened as text as given, so a
    `.swf.gz` argument that exists on disk would be read undecompressed; the
    `.gz` suffix is handled here only as the fallback for a missing `.swf`.
    """
    if os.path.exists(path):
        return open(path, 'r', encoding='utf-8', errors='replace')
    gz = path if path.endswith('.gz') else path + '.gz'
    if os.path.exists(gz):
        return io.TextIOWrapper(gzip.open(gz, 'rb'), encoding='utf-8',
                                errors='replace')
    raise FileNotFoundError(
        f'Neither {path} nor {gz} exists. The Parallel Workloads Archive '
        f'traces ship with the repository as .swf.gz.')

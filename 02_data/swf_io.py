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


def _open_gzip_text(path):
    """Decode a gzip member as text with the archive's tolerant decoding."""
    return io.TextIOWrapper(gzip.open(path, 'rb'), encoding='utf-8',
                            errors='replace')


def open_swf(path):
    """Open an SWF trace, transparently handling the gzipped copy.

    Two accepted forms, and the SUFFIX decides which -- never the order the
    checks happen to be written in:

      * a `.gz` path is always decompressed, whether or not the plain file
        exists beside it;
      * a plain path is read as text when it exists, and otherwise the
        `<path>.gz` beside it is decompressed.

    The suffix test comes FIRST on purpose. When existence was tested first, an
    existing `.swf.gz` handed in directly fell into the plain-text branch and
    was read as UTF-8 with `errors='replace'`, so the caller got a stream of
    replacement characters instead of a trace. The parser skips anything that
    does not split into >= 11 numeric-looking fields, so that garbage did not
    raise: it parsed to ZERO jobs, which reads downstream like an empty trace
    rather than like a bug.

    Decoding is `utf-8` with `errors='replace'` because the archive headers
    carry occasional non-UTF-8 bytes in site/contact lines, and a trace must not
    fail to load over a comment the parser skips anyway.
    """
    if path.endswith('.gz'):
        if os.path.exists(path):
            return _open_gzip_text(path)
        raise FileNotFoundError(
            f'{path} does not exist. The Parallel Workloads Archive traces '
            f'ship with the repository as .swf.gz.')
    if os.path.exists(path):
        return open(path, 'r', encoding='utf-8', errors='replace')
    gz = path + '.gz'
    if os.path.exists(gz):
        return _open_gzip_text(gz)
    raise FileNotFoundError(
        f'Neither {path} nor {gz} exists. The Parallel Workloads Archive '
        f'traces ship with the repository as .swf.gz.')

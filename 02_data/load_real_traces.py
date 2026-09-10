import os
import argparse
import pandas as pd

# The one gzip-aware SWF reader. This module lives in 02_data beside swf_io, and
# every script that touches an SWF trace must go through it: when a reader kept
# its own `open()`, the two disagreed about which forms of the file they could
# read, and the disagreement only surfaced on a machine where the uncompressed
# trace happened to exist.
from swf_io import open_swf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# This script builds a SYNTHETIC LANL-SCHEMA PROXY, not a real trace.
#
# It used to look for '02_data/lanl_trace_sample.swf', a filename that has never
# existed in this repository, so the fallback below fired on every run on every
# machine while the pipeline step was labelled "Real trace loading". The two
# real traces the project evaluates ship as .swf.gz and are handled by
# 02_data/build_real_trace_datasets.py, not here.
#
# --input is kept so a caller who has an SWF log can convert it to this 4-column
# schema, but there is no default input any more: the honest default is to
# generate the proxy and say so.
DEFAULT_INPUT = None
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, '02_data',
                              'synthetic_proxy_lanl_schema_trace.csv')


def parse_lanl_swf(path):
    """Parse a Standard Workload Format (SWF) trace.

    SWF fields (0-indexed after whitespace split):
      parts[0] = job number, parts[1] = submit time, parts[2] = wait time,
      parts[3] = run time, parts[4] = allocated processors,
      parts[7] = requested number of processors, parts[8] = requested time.
    Requested processors (parts[7]) is preferred; the '-1' sentinel falls
    back to allocated processors (parts[4]).

    The file is opened through `swf_io.open_swf`, so BOTH committed forms work:
    a `.swf.gz` path is decompressed, and a plain `.swf` path falls back to the
    `.swf.gz` beside it. The previous bare `open(..., errors='ignore')` here
    read a gzip member as text and produced an empty DataFrame rather than an
    error -- an unreadable trace that looked like an empty one. Decoding is
    `errors='replace'` (swf_io's choice, not 'ignore'): the archive's header
    comments carry occasional non-UTF-8 bytes, and every reader in this
    repository must mangle them the same way.
    """
    rows = []
    with open_swf(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            submit_time = int(float(parts[1]))
            wait_time = int(float(parts[2]))
            runtime = max(1, int(float(parts[3])))
            req_procs = max(1, int(float(parts[7])) if parts[7] != '-1' else int(float(parts[4])))
            rows.append({
                'arrival_time': submit_time,
                'wait_time': wait_time,
                'runtime': runtime,
                'num_gpus': min(8, max(1, req_procs)),
            })
    return pd.DataFrame(rows)


def build_fallback_trace():
    """Build a SYNTHETIC proxy trace with a LANL-like schema, derived from
    improved_wait_dataset.csv.

    This is NOT real LANL data and carries no real-world provenance. Downstream
    evaluations must label it 'synthetic_proxy_trace'. It exists only to give
    the out-of-distribution check in synthetic_vs_real_comparison.py a workload
    whose distribution differs from the training set.
    """
    data = pd.read_csv(os.path.join(PROJECT_ROOT, '02_data', 'improved_wait_dataset.csv'))
    out = pd.DataFrame({
        'arrival_time': (data.index.values % 150) * 2,
        'wait_time': (data['wait_time'] * 1.15).round().astype(int),
        'runtime': (6 + data['job_gpu'] * 1.8 + data['queue_length'] * 0.3).round().clip(1, 80).astype(int),
        'num_gpus': data['job_gpu'].clip(1, 8).astype(int),
    })
    return out.sample(n=min(2000, len(out)), random_state=42).sort_values('arrival_time').reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description='Generate the synthetic LANL-schema proxy trace, or convert an SWF log '
                    'into the same 4-column schema.')
    parser.add_argument('--input', default=DEFAULT_INPUT,
                        help='optional SWF log to convert instead of generating the proxy')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='Output CSV path')
    args = parser.parse_args()

    if args.input:
        # No `os.path.exists` pre-check: open_swf decides what "present" means
        # (a plain `.swf` is satisfied by the `.swf.gz` beside it), and a second
        # existence rule here would reject an input the reader can actually
        # open. Its FileNotFoundError names both candidate paths; it is turned
        # into a clean exit rather than a traceback.
        try:
            df = parse_lanl_swf(args.input)
        except FileNotFoundError as exc:
            raise SystemExit(f'--input {args.input}: {exc}') from None
        source = f'SWF log {os.path.basename(args.input)}'
    else:
        df = build_fallback_trace()
        source = 'synthetic_proxy_trace (generated; NOT real trace data)'

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f'Loaded {len(df)} rows from {source}. Saved to {args.output}')


if __name__ == '__main__':
    main()

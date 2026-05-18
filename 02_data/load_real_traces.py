import os
import argparse
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(PROJECT_ROOT, '02_data', 'lanl_trace_sample.swf')
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, '02_data', 'lanl_trace_sample.csv')


def parse_lanl_swf(path):
    rows = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
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
            req_procs = max(1, int(float(parts[8])) if parts[8] != '-1' else int(float(parts[4])))
            rows.append({
                'arrival_time': submit_time,
                'wait_time': wait_time,
                'runtime': runtime,
                'num_gpus': min(8, max(1, req_procs)),
            })
    return pd.DataFrame(rows)


def build_fallback_trace():
    data = pd.read_csv(os.path.join(PROJECT_ROOT, '02_data', 'improved_wait_dataset.csv'))
    out = pd.DataFrame({
        'arrival_time': (data.index.values % 150) * 2,
        'wait_time': (data['wait_time'] * 1.15).round().astype(int),
        'runtime': (6 + data['job_gpu'] * 1.8 + data['queue_length'] * 0.3).round().clip(1, 80).astype(int),
        'num_gpus': data['job_gpu'].clip(1, 8).astype(int),
    })
    return out.sample(n=min(2000, len(out)), random_state=42).sort_values('arrival_time').reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description='Parse LANL SWF trace into project CSV schema.')
    parser.add_argument('--input', default=DEFAULT_INPUT, help='Path to LANL SWF file')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='Output CSV path')
    args = parser.parse_args()

    if os.path.exists(args.input):
        df = parse_lanl_swf(args.input)
        source = 'LANL SWF'
    else:
        df = build_fallback_trace()
        source = 'fallback synthetic-real proxy'

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f'Loaded {len(df)} rows from {source}. Saved to {args.output}')


if __name__ == '__main__':
    main()

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_PATH = os.path.join(PROJECT_ROOT, '05_results', 'benchmark_statistical_summary.csv')
OUT_DIR = os.path.join(PROJECT_ROOT, '05_results', 'roi')
os.makedirs(OUT_DIR, exist_ok=True)

HOURS_PER_JOB = 2.5
GPU_COST_PER_HOUR = 2.2
ENERGY_KWH_PER_GPU_HOUR = 0.35
ENERGY_COST_PER_KWH = 0.14
ANNUAL_JOBS = 180_000


def main():
    improvement_pct = 7.5
    if os.path.exists(BENCH_PATH):
        summary = pd.read_csv(BENCH_PATH)
        if {'metric', 'value'}.issubset(summary.columns):
            # benchmark_statistical.py writes a long-format (metric,value) CSV
            match = summary.loc[summary['metric'] == 'mean_improvement_pct', 'value']
            if not match.empty:
                improvement_pct = float(match.iloc[0])
        elif 'mean_improvement_pct' in summary.columns:
            improvement_pct = float(summary.loc[0, 'mean_improvement_pct'])

    baseline_gpu_hours = ANNUAL_JOBS * HOURS_PER_JOB
    saved_gpu_hours = baseline_gpu_hours * (improvement_pct / 100.0)
    cost_reduction = saved_gpu_hours * GPU_COST_PER_HOUR
    energy_saved_kwh = saved_gpu_hours * ENERGY_KWH_PER_GPU_HOUR
    energy_cost_reduction = energy_saved_kwh * ENERGY_COST_PER_KWH
    total_savings = cost_reduction + energy_cost_reduction
    deployment_cost = 42_000
    roi_pct = (total_savings - deployment_cost) / deployment_cost * 100

    out = pd.DataFrame([{
        'annual_jobs': ANNUAL_JOBS,
        'mean_wait_improvement_pct': improvement_pct,
        'baseline_gpu_hours': baseline_gpu_hours,
        'saved_gpu_hours': saved_gpu_hours,
        'cloud_cost_reduction_usd': cost_reduction,
        'energy_saved_kwh': energy_saved_kwh,
        'energy_cost_reduction_usd': energy_cost_reduction,
        'total_annual_savings_usd': total_savings,
        'estimated_deployment_cost_usd': deployment_cost,
        'roi_pct': roi_pct,
    }])

    out_csv = os.path.join(OUT_DIR, 'cost_benefit_analysis.csv')
    out.to_csv(out_csv, index=False)

    plt.figure(figsize=(7.5, 4.8))
    labels = ['Cloud Cost Savings', 'Energy Savings', 'Deployment Cost']
    values = [cost_reduction, energy_cost_reduction, deployment_cost]
    colors = ['#34d399', '#38bdf8', '#fb923c']
    plt.bar(labels, values, color=colors)
    plt.title('ROI summary for proactive scheduler deployment')
    plt.ylabel('USD')
    plt.tight_layout()
    out_plot = os.path.join(OUT_DIR, 'roi_summary.png')
    plt.savefig(out_plot, dpi=160)

    print(out.to_string(index=False, formatters={'roi_pct': '{:.2f}'.format}))
    print('Saved:', out_csv)
    print('Saved:', out_plot)

if __name__ == '__main__':
    main()

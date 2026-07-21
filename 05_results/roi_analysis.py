import os
import sys

import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from vizstyle import (figure, finish, save_both, PALETTE, color_of, label_of,
                      bar_ends, legend_roles)

BENCH_PATH = os.path.join(PROJECT_ROOT, '05_results', 'benchmark_statistical_summary.csv')
OUT_DIR = os.path.join(PROJECT_ROOT, '05_results', 'roi')
os.makedirs(OUT_DIR, exist_ok=True)

HOURS_PER_JOB = 2.5
GPU_COST_PER_HOUR = 2.2
ENERGY_KWH_PER_GPU_HOUR = 0.35
ENERGY_COST_PER_KWH = 0.14
ANNUAL_JOBS = 180_000

SOURCE = ('Source: 05_results/roi/cost_benefit_analysis.csv | wait-time improvement from '
          '05_results/benchmark_statistical_summary.csv')


def _usd(value, _pos=None):
    """Axis tick formatter: whole thousands of dollars."""
    return f'${value / 1000:,.0f}k'


def plot_roi(cost_reduction, energy_cost_reduction, deployment_cost,
             total_savings, roi_pct, improvement_pct, stem):
    """Annual savings vs one-off deployment cost, in both light and dark modes.

    Two hues only: blue = recurring saving, orange = the one-off cost it is
    weighed against. No third hue, no colour ramp over the three categories.
    """
    labels = ['Cloud cost saved\n(GPU hours)',
              'Energy cost saved',
              'Deployment cost\n(one-off)']
    values = [cost_reduction, energy_cost_reduction, deployment_cost]
    is_cost = [False, False, True]

    for mode in ('light', 'dark'):
        p = PALETTE[mode]
        fig, ax = figure(mode, figsize=(8.6, 4.4))

        colors = [p['series_2'] if c else p['series_1'] for c in is_cost]
        y = range(len(labels))
        ax.barh(list(y), values, color=colors, height=0.62)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel('US dollars')
        ax.xaxis.set_major_formatter(FuncFormatter(_usd))
        ax.set_xlim(0, max(values) * 1.18)
        bar_ends(ax, 'h')

        # Direct-label the dominant term only; the rest is read off the axis.
        biggest = max(range(len(values)), key=lambda i: values[i])
        ax.annotate(f'\\${values[biggest]:,.0f}', xy=(values[biggest], biggest),
                    xytext=(8, 0), textcoords='offset points',
                    va='center', ha='left', fontsize=10, fontweight='semibold',
                    color=p['ink'])

        ax.legend(handles=[Patch(facecolor=p['series_1'], label='Recurring annual saving'),
                           Patch(facecolor=p['series_2'], label='One-off deployment cost')],
                  loc='lower right')

        finish(fig, mode,
               title='Projected annual savings outweigh the one-off deployment cost',
               subtitle=(f'{improvement_pct:.1f}% mean wait-time reduction applied to '
                         f'{ANNUAL_JOBS:,} jobs/yr at {HOURS_PER_JOB} GPU-h/job\n'
                         f'\\${total_savings:,.0f} saved per year against a '
                         f'\\${deployment_cost:,.0f} deployment cost - first-year ROI {roi_pct:.0f}%'),
               source=SOURCE)
        fig.subplots_adjust(top=0.76, bottom=0.18, left=0.20, right=0.98)
        save_both(fig, stem, mode)


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

    out_plot = os.path.join(OUT_DIR, 'roi_summary.png')
    plot_roi(cost_reduction, energy_cost_reduction, deployment_cost,
             total_savings, roi_pct, improvement_pct,
             os.path.join(OUT_DIR, 'roi_summary'))

    print(out.to_string(index=False, formatters={'roi_pct': '{:.2f}'.format}))
    print('Saved:', out_csv)
    print('Saved:', out_plot)

if __name__ == '__main__':
    main()

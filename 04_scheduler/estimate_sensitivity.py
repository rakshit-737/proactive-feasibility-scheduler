# estimate_sensitivity.py
#
# How good must user runtime estimates be before estimate-driven classical
# schedulers (SJF, EASY backfill) beat the estimate-free proactive scheduler?
#
# Sweeps the f-model over-estimation factor C (est = runtime * f, f ~ U(1,C);
# Mu'alem & Feitelson 2001) over C in {1, 2, 3, 5, 10} for SJF-EST and
# EASY-BF-EST, against C-independent references (FIFO, SJF oracle, EASY-BF
# oracle, PROACTIVE). C=1 is perfect information; C=10 approximates the
# order-of-magnitude inflation seen in real user estimates.
#
# Pairing: same 20 out-of-training workloads as multi_scheduler_benchmark.py
# (seeds 1000+run); the estimate noise u ~ U(0,1) per job is drawn from a
# fresh random.Random(20000+run) for every C, so the SAME u is rescaled by
# (C-1) across the sweep — estimate QUALITY varies, the noise realisation
# does not.
#
# MODAL variants (SJF_MODAL, BACKFILL_MODAL): the f-model is kind to SJF
# because multiplicative noise largely preserves job RANKING. Real user
# estimates are modal — most users pick the same few round values (Tsafrir &
# Feitelson 2005) — which destroys rank information. Modelled here as
# deterministic menu rounding: est = runtime rounded UP to the next multiple
# of 10, i.e. only {10, 20} on this 5–20-tick workload (two classes; ties
# fall back to arrival order). This is the rank-poor regime the f-model
# cannot reach.
#
# Outputs (05_results/schedulers/):
#   estimate_sensitivity.csv        per-(run, C, scheduler) rows
#   estimate_sensitivity_summary.csv
#   estimate_sensitivity.png        mean wait vs C

import os
import random
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from multi_scheduler_benchmark import (
    OUT_DIR, NUM_RUNS, EST_SEED_BASE,
    generate_jobs, assign_estimates, run_once,
)

C_VALUES = [1.0, 2.0, 3.0, 5.0, 10.0]
SWEPT = ['sjf_est', 'backfill_est']            # estimate-driven classical
REFERENCES = ['fifo', 'sjf', 'backfill', 'proactive']   # C-independent
MODAL_MENU_STEP = 10                            # round est up to next multiple


def assign_modal_estimates(jobs, step=MODAL_MENU_STEP):
    for job in jobs:
        job.est_runtime = float(-(-job.runtime // step) * step)


def main():
    rows = []
    for run in range(NUM_RUNS):
        random.seed(1000 + run)
        np.random.seed(1000 + run)
        jobs = generate_jobs()

        for sch in REFERENCES:
            out = run_once(jobs, sch, None)     # nn_model unused by these
            out.update(run=run + 1, scheduler=sch.upper(), over_factor=np.nan)
            rows.append(out)

        # rank-poor modal estimates (deterministic, C-independent)
        assign_modal_estimates(jobs)
        for sch, label in (('sjf_est', 'SJF_MODAL'), ('backfill_est', 'BACKFILL_MODAL')):
            out = run_once(jobs, sch, None)
            out.update(run=run + 1, scheduler=label, over_factor=np.nan)
            rows.append(out)

        for C in C_VALUES:
            # fresh RNG with the same seed per C -> identical u, scaled by C-1
            assign_estimates(jobs, random.Random(EST_SEED_BASE + run), over_factor=C)
            for sch in SWEPT:
                out = run_once(jobs, sch, None)
                out.update(run=run + 1, scheduler=sch.upper(), over_factor=C)
                rows.append(out)
        print(f'Run {run + 1:02d}/{NUM_RUNS} done')

    df = pd.DataFrame(rows)
    runs_path = os.path.join(OUT_DIR, 'estimate_sensitivity.csv')
    df.to_csv(runs_path, index=False)

    summary = (df.groupby(['scheduler', 'over_factor'], dropna=False, as_index=False)
                 .agg(mean_wait=('mean_wait', 'mean'),
                      max_wait=('max_wait', 'mean'),
                      mean_bounded_slowdown=('mean_bounded_slowdown', 'mean'),
                      fairness_gini=('fairness_gini', 'mean')))

    # paired uncertainty vs PROACTIVE per (scheduler, C): 95% CI on the
    # per-run wait difference and a paired t-test — the crossing evidence
    pro = (df[df['scheduler'] == 'PROACTIVE'].sort_values('run')
           ['mean_wait'].to_numpy())
    ci_lo, ci_hi, pvals = [], [], []
    for _, row in summary.iterrows():
        sel = df['scheduler'] == row['scheduler']
        sel &= df['over_factor'].isna() if pd.isna(row['over_factor']) \
            else df['over_factor'] == row['over_factor']
        w = df[sel].sort_values('run')['mean_wait'].to_numpy()
        if row['scheduler'] == 'PROACTIVE' or len(w) != len(pro):
            ci_lo.append(np.nan); ci_hi.append(np.nan); pvals.append(np.nan)
            continue
        diff = w - pro
        sem = diff.std(ddof=1) / np.sqrt(len(diff))
        tcrit = stats.t.ppf(0.975, len(diff) - 1)
        ci_lo.append(diff.mean() - tcrit * sem)
        ci_hi.append(diff.mean() + tcrit * sem)
        pvals.append(float(stats.ttest_rel(w, pro).pvalue))
    summary['wait_diff_vs_proactive_ci95_low'] = ci_lo
    summary['wait_diff_vs_proactive_ci95_high'] = ci_hi
    summary['ttest_p_vs_proactive'] = pvals

    summary_path = os.path.join(OUT_DIR, 'estimate_sensitivity_summary.csv')
    summary.to_csv(summary_path, index=False)

    plt.figure(figsize=(9, 5.5))
    styles = {'SJF_EST': ('#38bdf8', 'o'), 'BACKFILL_EST': ('#f472b6', 's')}
    for sch, (color, marker) in styles.items():
        sub = summary[summary['scheduler'] == sch].sort_values('over_factor')
        plt.plot(sub['over_factor'], sub['mean_wait'], marker=marker,
                 color=color, label=sch.replace('_', '-'))
    ref_styles = {'PROACTIVE': ('#34d399', '-'), 'FIFO': ('#94a3b8', '--'),
                  'SJF': ('#0ea5e9', ':'), 'BACKFILL': ('#fb923c', ':'),
                  'SJF_MODAL': ('#a78bfa', '-.'), 'BACKFILL_MODAL': ('#f59e0b', '-.')}
    for sch, (color, ls) in ref_styles.items():
        val = summary.loc[summary['scheduler'] == sch, 'mean_wait'].iloc[0]
        note = ('oracle' if sch in ('SJF', 'BACKFILL')
                else 'modal est.' if sch.endswith('_MODAL') else 'C-independent')
        plt.axhline(val, color=color, linestyle=ls, linewidth=1.2,
                    label=f'{sch.replace("_", "-")} ({note})')
    plt.xlabel('Over-estimation factor C  (est = runtime × f,  f ~ U(1, C))')
    plt.ylabel('Mean wait (ts)')
    plt.title('Estimate quality vs scheduling performance '
              f'({NUM_RUNS} paired runs)')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'estimate_sensitivity.png')
    plt.savefig(plot_path, dpi=160)

    print('\n=== Estimate-sensitivity summary (mean wait) ===')
    pivot = summary.pivot_table(index='over_factor', columns='scheduler',
                                values='mean_wait', dropna=False)
    print(pivot.to_string(float_format='{:.3f}'.format))
    print('Saved:', runs_path)
    print('Saved:', summary_path)
    print('Saved:', plot_path)


if __name__ == '__main__':
    main()

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
#   estimate_sensitivity.csv             per-(run, C, scheduler) rows
#   estimate_sensitivity_summary.csv
#   estimate_sensitivity.png             sweep + C-independent references
#   estimate_sensitivity-dark.png        same figure, dark mode

import os
import sys
import random
import numpy as np
import pandas as pd
from scipy import stats

# ── Paths (PROJECT_ROOT pattern used across the repo) ────────────────────────
# Works whether the script is launched from 04_scheduler/ or from the repo root:
# __file__ anchors PROJECT_ROOT, and Python already puts the script's own
# directory on sys.path for the multi_scheduler_benchmark import below.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from vizstyle import (figure, finish, save_both, PALETTE, color_of, label_of,
                      bar_ends, legend_roles)

from multi_scheduler_benchmark import (
    OUT_DIR, NUM_RUNS, EST_SEED_BASE,
    generate_jobs, assign_estimates, run_once,
)

C_VALUES = [1.0, 2.0, 3.0, 5.0, 10.0]
SWEPT = ['sjf_est', 'backfill_est']            # estimate-driven classical
REFERENCES = ['fifo', 'sjf', 'backfill', 'proactive']   # C-independent
MODAL_MENU_STEP = 10                            # round est up to next multiple

# Tick labels for the two modal variants, which vizstyle does not name (it maps
# them to the 'baseline' ROLE, which is what drives their colour).
EXTRA_LABELS = {
    'SJF_MODAL': 'SJF (modal estimates)',
    'BACKFILL_MODAL': 'EASY backfill (modal estimates)',
}

# Marker shape (NOT hue) separates the two swept classical policies: both are
# classical baselines, so both wear the baseline grey. Filled vs hollow is the
# secondary channel.
SWEPT_MARKER = {'SJF_EST': ('o', True), 'BACKFILL_EST': ('s', False)}

# C-independent policies drawn as the reference panel.
REF_POLICIES = ['SJF', 'SJF_MODAL', 'PROACTIVE', 'FIFO', 'BACKFILL',
                'BACKFILL_MODAL']


def name_of(policy):
    """Human-readable policy name; vizstyle's map first, local fallback after."""
    return EXTRA_LABELS.get(policy, label_of(policy))


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

    # ── Figure ──────────────────────────────────────────────────────────────
    # Both panels measure the SAME quantity (mean wait), so there is one scale
    # and never two y-axes on one plot. The old chart drew eight policies in
    # eight hues, six of them as horizontal rules that read as gridlines; here
    # colour follows the ENTITY -- PROACTIVE is the ML method and is blue in
    # both panels, every classical policy is baseline grey in both panels --
    # and the C-independent references move out of the sweep into their own
    # ranked panel instead of being stacked on top of it.
    def ref_wait(sch):
        return float(summary.loc[summary['scheduler'] == sch, 'mean_wait'].iloc[0])

    ref_vals = sorted(((s, ref_wait(s)) for s in REF_POLICIES),
                      key=lambda kv: kv[1])
    plot_stem = os.path.join(OUT_DIR, 'estimate_sensitivity')
    written = {}

    for mode in ('light', 'dark'):
        p = PALETTE[mode]
        fig, (ax1, ax2) = figure(mode, figsize=(13, 5.6), nrows=1, ncols=2,
                                 gridspec_kw={'width_ratios': [1.25, 1]})

        # Panel 1 -- the sweep. Two classical policies (same role, same grey,
        # separated by marker shape) against the estimate-free ML reference.
        for sch, (marker, filled) in SWEPT_MARKER.items():
            sub = summary[summary['scheduler'] == sch].sort_values('over_factor')
            c = color_of(sch, mode)
            ax1.plot(sub['over_factor'], sub['mean_wait'], marker=marker,
                     color=c, label=name_of(sch), zorder=3,
                     markerfacecolor=c if filled else p['surface'],
                     markeredgecolor=c, markeredgewidth=1.6)

        ax1.axhline(ref_wait('PROACTIVE'), color=color_of('PROACTIVE', mode),
                    linewidth=1.8, zorder=2,
                    label=f'{name_of("PROACTIVE")} — needs no estimates')
        ax1.set_xticks(C_VALUES)
        ax1.set_xticklabels([f'{c:g}' for c in C_VALUES])
        ax1.set_xlim(0.4, 10.6)
        ax1.set_ylim(9.8, 20.4)
        ax1.set_xlabel('Over-estimation factor C   '
                       '(est = runtime × f,  f ~ U(1, C);  C = 1 is perfect information)')
        ax1.set_ylabel('Mean wait (ticks)')
        ax1.set_title('Estimate quality sweep')
        ax1.legend(loc='lower right')
        ax1.set_axisbelow(True)

        # Panel 2 -- the C-independent references, ranked. Ranking is layout
        # only; the colour of every bar still comes from color_of(policy).
        ypos = np.arange(len(ref_vals))
        ax2.barh(ypos, [v for _, v in ref_vals], height=0.62, zorder=3,
                 color=[color_of(s, mode) for s, _ in ref_vals])
        ax2.set_yticks(ypos)
        ax2.set_yticklabels([name_of(s) for s, _ in ref_vals])
        ax2.set_ylim(-1.15, len(ref_vals) - 0.45)
        ax2.set_xlim(0, max(v for _, v in ref_vals) * 1.16)
        ax2.set_xlabel('Mean wait (ticks)')
        ax2.set_title('C-independent references')
        bar_ends(ax2, 'h')
        # one direct label, on the policy the sweep is measured against --
        # not a value on every bar
        i_pro = [s for s, _ in ref_vals].index('PROACTIVE')
        ax2.annotate(f'{ref_vals[i_pro][1]:.1f}', xy=(ref_vals[i_pro][1], i_pro),
                     xytext=(7, 0), textcoords='offset points', va='center',
                     fontsize=9, fontweight='semibold', color=p['ink'])
        legend_roles(ax2, mode, roles=('ml', 'baseline'), loc='lower right')

        fig.tight_layout(rect=(0, 0.03, 1, 0.86))
        finish(fig, mode,
               title='How good must user runtime estimates be to beat the estimate-free policy?',
               subtitle=(f'Mean wait over {NUM_RUNS} paired workloads. SJF with user estimates '
                         'stays ahead of the proactive policy at every C tested; '
                         'EASY backfill never does.'),
               source='05_results/schedulers/estimate_sensitivity_summary.csv')
        written[mode] = save_both(fig, plot_stem, mode)

    plot_path = written['light']

    print('\n=== Estimate-sensitivity summary (mean wait) ===')
    pivot = summary.pivot_table(index='over_factor', columns='scheduler',
                                values='mean_wait', dropna=False)
    print(pivot.to_string(float_format='{:.3f}'.format))
    print('Saved:', runs_path)
    print('Saved:', summary_path)
    print('Saved:', plot_path)
    print('Saved:', written['dark'])


if __name__ == '__main__':
    main()

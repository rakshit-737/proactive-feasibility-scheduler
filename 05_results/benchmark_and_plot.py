"""
benchmark_and_plot.py
─────────────────────────────────────────────────────────────────────────────
10-run benchmark: FIFO Baseline  vs  Proactive Scheduler (wait_model_v2)
Produces:
  • console table          — per-run and aggregate stats
  • benchmark_results.csv — full per-run data
  • wait_comparison.png   — per-run wait times (line) + distribution (hist)
  • util_comparison.png   — per-run GPU utilisation
  • scatter_pred_vs_actual.png — model quality check
  Each PNG is written as a light/dark pair via vizstyle.save_both, so the
  light-mode filename above is unchanged and '<stem>-dark.png' is added.

Results (10 runs, seeds fixed for reproducibility):
  Baseline  — Avg Wait: 20.64 ts | Avg Util: 65.6%
  Proactive — Avg Wait: 18.08 ts | Avg Util: 65.6%
  Wait reduction: +10.4%  (std 11.5%)
  Model quality — MAE: 2.87 ts | R²: 0.9369

Key fix applied (vs original proactive_Schedule_v2.py):
  The original scheduler broke after dispatching one job per tick, which
  artificially throttled throughput vs the baseline that dispatches all
  fitting jobs per tick. Fixed: removed the `break` so both schedulers
  have identical dispatching capacity — only the queue ORDER differs.
─────────────────────────────────────────────────────────────────────────────
Run from: anywhere (all paths are resolved relative to this file).
"""

import sys, os, random, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import mean_absolute_error, r2_score

RESULTS_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(RESULTS_DIR)
MODEL_PATH   = os.path.join(PROJECT_ROOT, "03_models", "wait_model_v2.pkl")
DATA_PATH    = os.path.join(PROJECT_ROOT, "02_data",   "improved_wait_dataset.csv")

# Shared figure style (one palette, colour follows the ENTITY not the panel).
# PROJECT_ROOT is on sys.path so this import works whether the script is run
# from 05_results/ or from the repository root.
sys.path.insert(0, PROJECT_ROOT)
from vizstyle import (figure, finish, save_both, PALETTE, color_of, label_of,
                      bar_ends, legend_roles)  # noqa: F401  (shared helper set)

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)
model    = bundle["model"]
FEATURES = bundle["features"]
print(f"Model loaded: wait_model_v2 | {len(FEATURES)} features\n")

class Job:
    def __init__(self, job_id, arrival_time, num_gpus, runtime):
        self.job_id=job_id; self.arrival_time=arrival_time
        self.num_gpus=num_gpus; self.runtime=runtime
        self.start_time=None; self.end_time=None; self.allocated_nodes=[]

class Cluster:
    def __init__(self, num_nodes, gpus_per_node):
        self.num_nodes=num_nodes; self.gpus_per_node=gpus_per_node
        self.nodes=[gpus_per_node]*num_nodes
    def total_free_gpus(self): return sum(self.nodes)
    def allocate(self, job, t):
        req,alloc=job.num_gpus,[]
        for i in range(self.num_nodes):
            if req<=0: break
            av=self.nodes[i]
            if av>0:
                u=min(av,req); self.nodes[i]-=u; alloc.append((i,u)); req-=u
        if req==0:
            job.start_time=t; job.end_time=t+job.runtime; job.allocated_nodes=alloc; return True
        for nid,u in alloc: self.nodes[nid]+=u
        return False
    def release(self, job):
        for nid,u in job.allocated_nodes: self.nodes[nid]+=u

def generate_jobs(n=110, max_t=300):
    jobs=[Job(i,random.randint(0,max_t//2),random.randint(1,8),random.randint(5,20)) for i in range(n)]
    return sorted(jobs, key=lambda x: x.arrival_time)

def get_features(job, cluster, queue, running):
    """Matches generate_improved_dataset.py exactly. `queue` includes the scored
    job at dispatch time; training rows exclude it (snapshot at arrival, before
    enqueue), so queue_length/queue_pressure subtract the job's own share."""
    tf=cluster.total_free_gpus(); mf=max(cluster.nodes); vf=np.var(cluster.nodes)
    return [job.num_gpus, tf, len(queue)-1, len(running), mf, vf,
            int(tf>=job.num_gpus),
            min(tf/(job.num_gpus+1e-6), 1.0),
            float(np.std(cluster.nodes)),
            (sum(q.num_gpus for q in queue)-job.num_gpus)/(tf+1),
            sum(1 for n in cluster.nodes if n>=job.num_gpus)/cluster.num_nodes,
            tf/cluster.num_nodes]

SIM_TIME=300; NN=8; GPN=4; CAP=NN*GPN

def run_baseline(jobs_in):
    cluster=Cluster(NN,GPN)
    jobs=[Job(j.job_id,j.arrival_time,j.num_gpus,j.runtime) for j in jobs_in]
    q,run,comp,gpu=[],[],[],[]
    for t in range(SIM_TIME):
        for j in run[:]:
            if j.end_time==t: cluster.release(j);run.remove(j);comp.append(j)
        for j in jobs:
            if j.arrival_time==t: q.append(j)
        for j in q[:]:
            if cluster.total_free_gpus()>=j.num_gpus:
                if cluster.allocate(j,t): run.append(j);q.remove(j)
        gpu.append((CAP-cluster.total_free_gpus())/CAP)
    waits=[j.start_time-j.arrival_time for j in comp if j.start_time is not None]
    return np.mean(waits),np.mean(gpu),len(comp),waits

def run_proactive(jobs_in):
    cluster=Cluster(NN,GPN)
    jobs=[Job(j.job_id,j.arrival_time,j.num_gpus,j.runtime) for j in jobs_in]
    q,run,comp,gpu=[],[],[],[]
    for t in range(SIM_TIME):
        for j in run[:]:
            if j.end_time==t: cluster.release(j);run.remove(j);comp.append(j)
        for j in jobs:
            if j.arrival_time==t: q.append(j)
        # Re-score queue each tick — sort ascending by predicted wait time
        if q:
            scored=[(model.predict([get_features(j,cluster,q,run)])[0],j) for j in q]
            scored.sort(key=lambda x:x[0]); q=[j for _,j in scored]
        # Dispatch ALL fitting jobs (same throughput as FIFO — only order differs)
        for j in q[:]:
            if cluster.total_free_gpus()>=j.num_gpus:
                if cluster.allocate(j,t): run.append(j);q.remove(j)
        gpu.append((CAP-cluster.total_free_gpus())/CAP)
    waits=[j.start_time-j.arrival_time for j in comp if j.start_time is not None]
    return np.mean(waits),np.mean(gpu),len(comp),waits

# ── Benchmark ──────────────────────────────────────────────────────────────
NUM_RUNS=10
results=[]; all_bw=[]; all_pw=[]
print(f"{'Run':>4}  {'B-Wait':>7} {'B-Util':>7}  {'P-Wait':>7} {'P-Util':>7}  {'Δ Wait%':>8}")
print("─"*56)
for i in range(NUM_RUNS):
    random.seed(i*42+7)
    jobs=generate_jobs()
    bw,bu,bj,bws=run_baseline(jobs)
    pw,pu,pj,pws=run_proactive(jobs)
    imp=(bw-pw)/bw*100
    results.append(dict(run=i+1,b_wait=bw,b_util=bu,b_jobs=bj,
                         p_wait=pw,p_util=pu,p_jobs=pj,improvement=imp))
    all_bw.extend(bws); all_pw.extend(pws)
    print(f"{i+1:>4}  {bw:>7.2f} {bu:>7.3f}  {pw:>7.2f} {pu:>7.3f}  {imp:>+7.1f}%")

df_r=pd.DataFrame(results)
b_mean=df_r.b_wait.mean(); p_mean=df_r.p_wait.mean()
b_util=df_r.b_util.mean(); p_util=df_r.p_util.mean()
imp_mean=df_r.improvement.mean(); imp_std=df_r.improvement.std()

print("\n"+"═"*56)
print(f"  Baseline  — Avg Wait: {b_mean:.2f} ts | Avg Util: {b_util:.3f}")
print(f"  Proactive — Avg Wait: {p_mean:.2f} ts | Avg Util: {p_util:.3f}")
print(f"  Wait reduction: {imp_mean:+.1f}%  (std {imp_std:.1f}%)")
print("═"*56)
df_r.to_csv(os.path.join(RESULTS_DIR,"benchmark_results.csv"),index=False)
print("Saved: benchmark_results.csv")

# ── Model quality ───────────────────────────────────────────────────────────
df_data=pd.read_csv(DATA_PATH)
X_all=df_data[FEATURES].values; y_true=df_data['wait_time'].values
y_pred=model.predict(X_all)
mae_all=mean_absolute_error(y_true,y_pred); r2_all=r2_score(y_true,y_pred)
print(f"Model quality — MAE: {mae_all:.2f} ts | R²: {r2_all:.4f}\n")

# ── Figure styling constants (presentation only) ────────────────────────────
# Colour follows the ENTITY: FCFS is the classical baseline (neutral) and the
# proactive policy is the ML method (blue) in EVERY panel of EVERY figure here.
runs = df_r.run.values
POL_BASE, POL_ML = 'FIFO', 'PROACTIVE'
# Shared bin edges so the two pooled histograms are actually comparable
# (rendering choice only — no statistic is recomputed).
WAIT_BINS = np.histogram_bin_edges(np.concatenate([all_bw, all_pw]), bins=35)

# ── Plot 1: per-run wait + distribution ────────────────────────────────────
for mode in ('light', 'dark'):
    p = PALETTE[mode]
    c_base, c_ml = color_of(POL_BASE, mode), color_of(POL_ML, mode)

    fig, (ax1, ax2) = figure(mode, figsize=(13.5, 5.4), ncols=2,
                             gridspec_kw={'width_ratios': [1.4, 1]})

    # Panel A — per-run average wait
    ax1.fill_between(runs, df_r.b_wait, df_r.p_wait, where=df_r.p_wait < df_r.b_wait,
                     color=c_ml, alpha=0.14, linewidth=0,
                     label='Runs where proactive waits less')
    ax1.plot(runs, df_r.b_wait, 'o-', color=c_base, lw=2.0, ms=6,
             label=label_of(POL_BASE))
    ax1.plot(runs, df_r.p_wait, 's-', color=c_ml, lw=2.0, ms=6,
             label=label_of(POL_ML))
    ax1.axhline(b_mean, color=c_base, lw=1.0)
    ax1.axhline(p_mean, color=c_ml, lw=1.0)
    ax1.text(runs[0], b_mean, f'  FCFS mean {b_mean:.1f}', color=p['ink_2'],
             fontsize=8.5, ha='left', va='bottom')
    ax1.text(runs[0], p_mean, f'  Proactive mean {p_mean:.1f}', color=p['ink_2'],
             fontsize=8.5, ha='left', va='top')
    ax1.set_xlabel('Run (fixed seed per run)')
    ax1.set_ylabel('Average wait time (timesteps)')
    ax1.set_title('Per-run average wait time', loc='left')
    ax1.set_xticks(runs)
    ax1.grid(axis='x', visible=False); ax1.set_axisbelow(True)
    ax1.legend(loc='upper right')

    # Panel B — pooled per-job wait distribution (same entities, same colours)
    # Filled baseline + outlined ML series: two overlapping translucent fills
    # would blend into a third apparent hue, which the colour rule forbids.
    ax2.hist(all_bw, bins=WAIT_BINS, density=True, color=c_base, alpha=0.85,
             label=label_of(POL_BASE))
    ax2.hist(all_pw, bins=WAIT_BINS, density=True, histtype='step', color=c_ml,
             lw=1.7, label=label_of(POL_ML))
    ax2.axvline(np.mean(all_bw), color=c_base, lw=1.2)
    ax2.axvline(np.mean(all_pw), color=c_ml, lw=1.2)
    ax2.annotate(f'mean wait\n{np.mean(all_bw):.1f} ts FCFS\n{np.mean(all_pw):.1f} ts proactive',
                 xy=(0.97, 0.74), xycoords='axes fraction', ha='right', va='top',
                 color=p['ink_2'], fontsize=8.5, linespacing=1.5)
    ax2.set_xlabel('Wait time of an individual job (timesteps)')
    ax2.set_ylabel('Density')
    ax2.set_title('Pooled job waits, all 10 runs', loc='left')
    ax2.grid(axis='x', visible=False); ax2.set_axisbelow(True)
    ax2.legend(loc='upper right')

    fig.tight_layout(rect=(0, 0.02, 1, 0.86))
    finish(fig, mode,
           title='Ordering the queue by predicted wait shortens the average wait',
           subtitle=(f'10 seeded runs, 110 jobs on an 8-node x 4-GPU cluster  ·  '
                     f'mean wait {b_mean:.2f} to {p_mean:.2f} timesteps  ·  '
                     f'mean per-run reduction {imp_mean:+.1f}% (sd {imp_std:.1f}%)'),
           source='05_results/benchmark_results.csv  ·  05_results/benchmark_and_plot.py')
    print(f"Saved: {os.path.basename(save_both(fig, os.path.join(RESULTS_DIR, 'wait_comparison'), mode))}")

# ── Plot 2: GPU utilisation ─────────────────────────────────────────────────
for mode in ('light', 'dark'):
    p = PALETTE[mode]
    c_base, c_ml = color_of(POL_BASE, mode), color_of(POL_ML, mode)

    fig, ax = figure(mode, figsize=(9.5, 4.8))
    # The two curves coincide exactly (both policies dispatch every fitting job
    # each tick), so the baseline is drawn as a wide halo under a thin ML line
    # rather than being hidden beneath it.
    ax.plot(runs, df_r.b_util, 'o-', color=c_base, lw=4.0, ms=9,
            label=label_of(POL_BASE))
    ax.plot(runs, df_r.p_util, 's-', color=c_ml, lw=1.4, ms=5,
            markerfacecolor='none', markeredgewidth=1.5, label=label_of(POL_ML))
    ax.set_xlabel('Run (fixed seed per run)')
    ax.set_ylabel('Average GPU utilisation')
    ax.set_xticks(runs); ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(axis='x', visible=False); ax.set_axisbelow(True)
    ax.legend(loc='lower right')

    fig.tight_layout(rect=(0, 0.03, 1, 0.84))
    finish(fig, mode,
           title='Queue order changes who waits, not how busy the cluster is',
           subtitle=(f'Both policies dispatch every fitting job each tick  ·  '
                     f'mean utilisation {b_util:.1%} (FCFS) vs {p_util:.1%} (proactive)'),
           source='05_results/benchmark_results.csv')
    print(f"Saved: {os.path.basename(save_both(fig, os.path.join(RESULTS_DIR, 'util_comparison'), mode))}")

# ── Plot 3: predicted vs actual ─────────────────────────────────────────────
for mode in ('light', 'dark'):
    p = PALETTE[mode]

    fig, ax = figure(mode, figsize=(7.2, 6.4))
    # Not a scheduler chart: exactly ONE series colour, reference line in ink.
    ax.scatter(y_true, y_pred, s=11, alpha=0.18, color=p['series_1'],
               edgecolors='none', rasterized=True)
    mn, mx = 0, max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], color=p['muted'], lw=1.4)
    ax.set_xlabel('Actual wait time (timesteps)')
    ax.set_ylabel('Predicted wait time (timesteps)')
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Line2D([], [], marker='o', linestyle='none', color=p['series_1'],
               markersize=5.5, label='One dataset row'),
        Line2D([], [], color=p['muted'], lw=1.4, label='Perfect prediction'),
    ], loc='upper left')

    fig.tight_layout(rect=(0, 0.02, 1, 0.87))
    finish(fig, mode,
           title='Wait-time model: predicted vs actual',
           subtitle=(f'MAE {mae_all:.2f} timesteps  ·  R² {r2_all:.4f}  ·  scored on every '
                     f'dataset row, which includes the training split'),
           source='02_data/improved_wait_dataset.csv  ·  model 03_models/wait_model_v2.pkl')
    print(f"Saved: {os.path.basename(save_both(fig, os.path.join(RESULTS_DIR, 'scatter_pred_vs_actual'), mode))}")

print("\nAll done.")

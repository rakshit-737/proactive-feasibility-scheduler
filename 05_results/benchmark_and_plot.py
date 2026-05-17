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
Run from: E:\\Research Project\\Proactive\\05_results\\
"""

import sys, os, random, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_absolute_error, r2_score

RESULTS_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(RESULTS_DIR)
MODEL_PATH   = os.path.join(PROJECT_ROOT, "03_models", "wait_model_v2.pkl")
DATA_PATH    = os.path.join(PROJECT_ROOT, "02_data",   "improved_wait_dataset.csv")

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
    """Matches generate_improved_dataset.py exactly."""
    tf=cluster.total_free_gpus(); mf=max(cluster.nodes); vf=np.var(cluster.nodes)
    return [job.num_gpus, tf, len(queue), len(running), mf, vf,
            int(tf>=job.num_gpus),
            min(tf/(job.num_gpus+1e-6), 1.0),
            float(np.std(cluster.nodes)),
            sum(q.num_gpus for q in queue)/(tf+1),
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

# ── Plot 1: per-run wait + distribution ────────────────────────────────────
runs=df_r.run.values
fig=plt.figure(figsize=(14,5))
gs=gridspec.GridSpec(1,2,figure=fig,width_ratios=[1.4,1])
ax1=fig.add_subplot(gs[0])
ax1.plot(runs,df_r.b_wait,'o-',color='#94a3b8',lw=2.2,ms=8,label='FIFO Baseline')
ax1.plot(runs,df_r.p_wait,'s-',color='#38bdf8',lw=2.2,ms=8,label='Proactive (wait_model_v2)')
ax1.fill_between(runs,df_r.b_wait,df_r.p_wait,
                 where=df_r.p_wait<df_r.b_wait,alpha=0.13,color='#38bdf8',label='Improvement region')
ax1.axhline(b_mean,color='#94a3b8',ls='--',lw=1.2,alpha=0.7,label=f'B avg={b_mean:.1f}')
ax1.axhline(p_mean,color='#38bdf8',ls='--',lw=1.2,alpha=0.7,label=f'P avg={p_mean:.1f}')
ax1.set_xlabel('Run',fontsize=11); ax1.set_ylabel('Average Wait Time (timesteps)',fontsize=11)
ax1.set_title('Per-Run Average Wait Time',fontsize=12,fontweight='bold')
ax1.legend(fontsize=9); ax1.set_xticks(runs); ax1.grid(axis='y',alpha=0.25)
ax2=fig.add_subplot(gs[1])
ax2.hist(all_bw,bins=35,color='#94a3b8',alpha=0.6,label='Baseline',density=True)
ax2.hist(all_pw,bins=35,color='#38bdf8',alpha=0.65,label='Proactive',density=True)
ax2.axvline(np.mean(all_bw),color='#64748b',ls='--',lw=1.5,label=f'B mean={np.mean(all_bw):.1f}')
ax2.axvline(np.mean(all_pw),color='#0ea5e9',ls='--',lw=1.5,label=f'P mean={np.mean(all_pw):.1f}')
ax2.set_xlabel('Wait Time (timesteps)',fontsize=11); ax2.set_ylabel('Density',fontsize=11)
ax2.set_title('Wait Time Distribution (pooled, 10 runs)',fontsize=12,fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(axis='y',alpha=0.25)
plt.suptitle(f'FIFO Baseline vs Proactive Scheduler — 10-Run Benchmark  ({imp_mean:+.1f}% avg wait reduction)',
             fontsize=12,fontweight='bold',y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,'wait_comparison.png'),dpi=150,bbox_inches='tight'); plt.close()
print("Saved: wait_comparison.png")

# ── Plot 2: GPU utilisation ─────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(9,4))
ax.plot(runs,df_r.b_util,'o-',color='#94a3b8',lw=2.2,ms=8,label='FIFO Baseline')
ax.plot(runs,df_r.p_util,'s-',color='#34d399',lw=2.2,ms=8,label='Proactive (wait_model_v2)')
ax.axhline(b_util,color='#94a3b8',ls='--',lw=1.2,alpha=0.7,label=f'B avg={b_util:.1%}')
ax.axhline(p_util,color='#34d399',ls='--',lw=1.2,alpha=0.7,label=f'P avg={p_util:.1%}')
ax.set_xlabel('Run',fontsize=11); ax.set_ylabel('Average GPU Utilisation',fontsize=11)
ax.set_title('GPU Utilisation — FIFO vs Proactive Scheduler',fontsize=12,fontweight='bold')
ax.legend(fontsize=10); ax.set_xticks(runs); ax.set_ylim(0,1.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_:f'{y:.0%}'))
ax.grid(axis='y',alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,'util_comparison.png'),dpi=150,bbox_inches='tight'); plt.close()
print("Saved: util_comparison.png")

# ── Plot 3: predicted vs actual ─────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(7,6))
ax.scatter(y_true,y_pred,alpha=0.2,s=12,color='#818cf8',rasterized=True)
mn,mx=0,max(y_true.max(),y_pred.max())
ax.plot([mn,mx],[mn,mx],'r--',lw=1.5,label='Perfect prediction')
ax.set_xlabel('Actual Wait Time (timesteps)',fontsize=11)
ax.set_ylabel('Predicted Wait Time (timesteps)',fontsize=11)
ax.set_title(f'Model Quality: Predicted vs Actual\nMAE = {mae_all:.2f} ts  |  R² = {r2_all:.4f}',
             fontsize=12,fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,'scatter_pred_vs_actual.png'),dpi=150,bbox_inches='tight'); plt.close()
print("Saved: scatter_pred_vs_actual.png")
print("\nAll done.")

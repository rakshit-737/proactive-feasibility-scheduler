# Methodology (Phases 01–21 + v3.3/v3.4 baselines; current as of v3.6 + Phase C)

## Core system
- Discrete-time cluster simulation with heterogeneous queue states (synthetic study).
- **Event-driven, second-exact trace replay** for the real-workload study (v3.4):
  state changes only at arrivals and completions, so there is no tick-quantisation
  bias. Capacity-only allocator — SWF records no per-node placement, and inventing
  node-level fragmentation would fabricate data.
- Wait-time regression model trained on engineered cluster-state features.
- Proactive scheduler reorders queue using predicted wait-time.

## v3.4: ranking-degeneracy analysis
The central v3.4 experiment tests whether the wait-time model can influence queue
*ordering* at all. At a fixed dispatch instant all queued jobs share the cluster
state, so only per-job features can differ — and in this feature set each of those
(`can_fit_now`, `gpu_fit_ratio`, `node_availability`, `queue_pressure`) is a
deterministic function of the requested size given the state. The predicted score is
therefore `g_S(size)`, so the ranking is a function of size alone: it can order
size classes but cannot distinguish two jobs inside one. It is NOT necessarily the
ascending-size order — see Proposition 1 and the note below it.

`04_scheduler/ranking_degeneracy.py` instruments real dispatch decisions through a
`RANK_OBSERVER` hook in both benchmarks (no reimplementation of the simulators) and
measures, per instant: which features actually vary across the queue; whether two
equally-sized jobs ever receive different scores; Kendall τ against the size order;
the fraction of instants whose order is identical to smallest-first and to arrival
order; the fraction in which *every* queued job receives an identical score
(`pct_all_scores_tied`, measured directly as of v3.6); and the recovered
size→priority table with its monotonicity. The arrival-order fraction and the
all-tied fraction are distinct quantities and must be reported as such: an
all-tied instant is necessarily order-identical to arrival, but not the
converse, so `pct_order_identical_to_arrival` only
*upper-bounds* the all-tied fraction. Before v3.6 the two were conflated and the
arrival-order figure was reported as if it were the all-tied one. The matching
control policy is `04_scheduler/size_scheduler.py` (`SMALLEST` / `SMALLEST_FIRST`):
sort by requested size, no model.

## The degeneracy, stated formally

The argument above is correct but informal, and one of its informal phrasings is
slightly wrong. This section states it precisely, because the imprecision matters for
what the paper may claim.

**Setup.** Fix a dispatch instant *t*. Let *S* denote the cluster state at *t* —
free GPUs per node, the running set, the queue contents, and any quantity derived
from them. Let the queue be *Q* = {*j*₁ … *j*ₙ}, and write *g*(*j*) for the
requested size of job *j*. A feature map assigns to each queued job a vector

&nbsp;&nbsp;&nbsp;&nbsp;φ(*j*, *S*) = ( ψ(*j*, *S*), χ(*S*) ),

where χ(*S*) collects the features that depend on the cluster alone and ψ the ones
that depend on the job. A score function *f* induces the score *s*(*j*) =
*f*(φ(*j*, *S*)), and the dispatcher ranks *Q* by *s* with a fixed tie-break.

**Proposition 1 (ranking degeneracy).** Suppose every per-job feature factors
through the requested size given the state — that is, there is a map ψ̃ with
ψ(*j*, *S*) = ψ̃(*g*(*j*), *S*) for every *j* ∈ *Q*. Then there exists
*h*_*S* : sizes → ℝ with *s*(*j*) = *h*_*S*(*g*(*j*)) for every *j* ∈ *Q*.

*Proof.* χ(*S*) does not depend on *j*, so at the fixed instant *t* it is a
constant. Define *h*_*S*(*u*) := *f*(ψ̃(*u*, *S*), χ(*S*)). Then for any *j* ∈ *Q*,
*s*(*j*) = *f*(ψ̃(*g*(*j*), *S*), χ(*S*)) = *h*_*S*(*g*(*j*)). ∎

**Corollary 1.1 (no cluster-only feature can reorder anything).** Every feature in
χ enters *h*_*S* as a constant argument at a fixed instant. It therefore cannot
separate two queued jobs, whatever *f* is. This needs no additivity or monotonicity
assumption on *f* — the feature is simply held fixed across the comparison. In the
12-feature synthetic set, seven features are of this kind, and they are measured to
vary across a queue in 0.0% of instants.

**Corollary 1.2 (equal size implies equal score).** If *g*(*i*) = *g*(*j*) then
*s*(*i*) = *s*(*j*). This is the falsifiable form of the proposition, and it is what
`04_scheduler/ranking_degeneracy.py` counts: **zero counterexamples over 45,432
dispatch instants**.

**Corollary 1.3 (the ranking's resolution is bounded by the size alphabet).** The
number of distinct scores at an instant is at most |{ *g*(*j*) : *j* ∈ *Q* }|. The
induced order is therefore *measurable with respect to the partition of the queue
into size classes*: the policy can order size classes, and can do nothing whatsoever
within one. Measured: a queue of ~8.5–10.2 jobs receives 2.28–3.09 distinct priority
levels, and **every** queued job receives an identical score in 14.4–20.7% of
instants, in which the policy is exactly FCFS.

### What the proposition does *not* say

The paper has said in places that the ranking is "a permutation of the size order".
That is too strong, and the repository's own measurements contradict it: the
recovered size→priority table is monotone in only 57–63% of instants. Proposition 1
says the score is a *function of size*, not that it is an *increasing* function of
size. *h*_*S* may order the size classes in any way at all; it simply may not
distinguish jobs within one. The honest statement is the one in Corollary 1.3.

This distinction is why the equivalence to a size sort is established
*statistically*, by TOST against `SMALLEST_FIRST`, rather than deduced. Degeneracy
is a structural fact; that the resulting policy performs like ascending-size order
is a separate empirical claim.

### The converse, and its limits

**Proposition 2 (non-degeneracy condition).** If some per-job feature does *not*
factor through (*g*, *S*) — if ψ(*i*, *S*) ≠ ψ(*j*, *S*) is possible for two jobs
with *g*(*i*) = *g*(*j*) — then no such *h*_*S* need exist, and Corollary 1.2 can
fail. A feature set can produce a ranking finer than the size partition only if it
contains such an attribute.

Three limits on that converse, each of which the project either measures or must
state as a caveat:

1. **Necessary, not sufficient.** Breaking the degeneracy lets the model express a
   finer ranking; it does not make that ranking *good*. `PROACTIVE_EST` carries the
   user's runtime estimate, is non-degenerate by Proposition 2, and still loses to
   plain SJF on both traces. This limit is no longer an argument from one policy:
   the Phase D non-degeneracy sweep (below) breaks the degeneracy twelve separate
   ways, and **0 of the 12 augmented variants beat `SJF_USEREST`** on mean wait
   (`05_results/degeneracy/non_degeneracy_utility.csv`).
2. **The attribute must be informative, not merely distinct.** Any per-job nonce
   would break Corollary 1.2 while carrying no information about wait.
3. **The shared-state premise is a modelling choice, not a law.** Proposition 1
   assumes every queued job is scored against the *same* *S*. A policy that scores a
   job when it *enters* the queue and caches the result violates that premise by
   construction, and its ranking need not be a function of size. Whether such a
   policy schedules *better* is an empirical question, and staleness is a cost as
   well as a source of variation. This limit is now measured rather than asserted:
   Phase D attack **A4** builds exactly that policy and records 20,482 (SDSC) /
   37,741 (LANL) counterexamples -- the only one of the four attacks that breaks the
   degeneracy -- while scheduling no better
   (`05_results/degeneracy/robustness_attacks.csv`, `robustness_attack_utility.csv`).

## Phase D: testing the converse, and attacking the claim

Phase D adds two trace-only experiments that sit either side of Proposition 2: one
asks what it takes to *break* the degeneracy on purpose, the other asks whether a
reviewer can break it by accident. Both follow the published trace protocol
(`trace_driven_benchmark`'s window count, warm-up, measured span and chronological
train split, imported rather than restated) on the SDSC and LANL traces. **Neither
experiment touches the synthetic study**, which is not attacked and is unchanged.

### D1: the non-degeneracy sweep (`04_scheduler/non_degeneracy_sweep.py`, pipeline step 12)

Artefacts: `05_results/degeneracy/non_degeneracy_sweep.csv` and
`non_degeneracy_utility.csv`.

**What is added.** Five per-job attributes are appended to the 8-feature trace
vector, one at a time and then all together, giving a baseline arm plus six variants
per trace:

- `est_runtime` -- the user's requested time, SWF field 9.
- `user_hist_wait` -- causal per-user mean wait.
- `user_hist_runtime` -- causal per-user mean runtime.
- `queue_id` -- queue identity, SWF field 15 (6 distinct queues on SDSC, 17 on LANL).
- `user_id` -- user identity, SWF field 12 (437 distinct users on SDSC, 213 on LANL).

SWF field 16 (partition) is **deliberately excluded, and the exclusion is stated
rather than left silent**: it is constant at -1 on both committed traces, so it
varies across no pair of co-queued jobs and could not break anything.

**The causality guarantee, on which the experiment lives or dies.** A per-user
history computed over the full trace would leak the future into every row and
*manufacture* a fake non-degeneracy -- the feature would carry information no
scheduler could ever have had. The implementation therefore sweeps submit order with
a min-heap of pending completions and absorbs a job into its user's running means
only once `submit + wait + runtime <= t`: **completed** earlier, not merely
*submitted* earlier, because a wait is unobservable before the job starts and a
runtime before it ends. Completion is taken from the recorded schedule, which is
exactly what the real machine knew at *t*, and `tests/test_non_degeneracy.py` pins
this down with a leakage test that goes red under a full-trace groupby. Cold start --
a job whose user has no completed prior job -- is left as **NaN**, which XGBoost
handles with a learned default branch direction, so that "unseen user" stays
distinguishable from "user who happens to average the global mean"; the cold-start
rate is 1.2% on SDSC and 0.2% on LANL.

**A conservative encoding.** `user_id` and `queue_id` enter as raw SWF integer
codes, not one-hot. This is the conservative choice and is reported as such: an
integer code is an arbitrary but *consistent* partition of the categories, and a
one-hot expansion could only separate co-queued jobs *further*. It would strengthen
the non-degeneracy half of the result, never weaken it, so the violation counts
below are a floor rather than a ceiling.

**Instrumentation is imported, not re-implemented.** The published `Collector` is
imported from `ranking_degeneracy.py`, so the sweep and the published measurement
agree by construction on what a violation is, what counts as a tie, and how the
induced order is formed. `build_feature_matrix` is **monkey-patched inside a context
manager** that calls the original for columns 0-7 and appends only the extra
columns, so the size column and the eight base features stay bit-identical and the
published module is neither edited nor left altered. The baseline arm therefore
reproduces the published diagnostic field for field, and that agreement is what
validates every other row of the artefact. The utility comparison pairs by window
**label** through `04_scheduler/simstats.py`.

**Summary of what it measures** (full tables in
`phases_22_30/COMPLETION_SUMMARY.md`): every one of the five attributes breaks
Corollary 1.2, taking violations from 0 to 6,194-10,718 on SDSC and from 0 to
13,801-28,497 on LANL, while **0 of the 12 augmented variants beat `SJF_USEREST`**.
Two caveats travel with that. SDSC's TOST equivalence to `SMALLEST_FIRST` breaks in
every augmented arm, as the converse predicts. On LANL the baseline was **never**
equivalent to `SMALLEST_FIRST` (p_tost 0.687), so there the degeneracy breaks but no
equivalence is lost -- the LANL arms must not be read as "the sweep destroyed an
equivalence".

### D4: four adversarial attacks (`04_scheduler/robustness_attacks.py`, pipeline step 13)

Artefacts: `05_results/degeneracy/robustness_attacks.csv` and
`robustness_attack_utility.csv`. Same imported `Collector`, same published window
protocol, both traces. Models trained in this file use a new seed
**`ROBUST_SEED = 90210`**, chosen to fall outside every protected seed family in the
repository (`42+i`, `1000+run`, `20000+run`, `800+run`, `4000+id+nodes`, `5000+i`,
`7000+run`, `SEED=42` for the traces, and `POWER_SEED=31337`, which the TOST power
study had already taken), so no existing experiment can collide with it. A reference
row `A0` reproduces the published measurement under this run's window count, so
every attack is read against a like-for-like control. The verdict criterion is the
counterexample counter `equal_size_diff_pred_violations`, not the supporting
statistics: an attack can weaken the claim without falsifying it, so Kendall tau
against size, the all-tied fraction and the mean number of distinct score levels are
reported alongside.

- **A1 -- a different learning objective.** `XGBRanker` with `rank:pairwise` and with
  `rank:ndcg` over the identical 8 features. The argument is that degeneracy is a
  property of the *inputs*, not of the loss. Two modelling choices have to be made,
  because the wait data carries no native query groups and no native relevance
  grades, and the artefact records both: query groups are **1-hour (3600 s)
  submit-time buckets**, and relevance is **5 global quantile grades of log1p(wait)**
  with the shortest wait most relevant. Scores are negated so that ascending order
  remains dispatch order -- a monotone map, so it moves no counter but the sign of
  tau. A different grouping would change tau; it cannot produce a violation.
- **A2 -- monotone transforms** (`log1p`, `sigmoid`) applied to the *published*
  model's own predictions, with nothing retrained. This fails by construction, since
  a strictly monotone map is order-preserving and can neither reorder a queue nor
  split a tie; it is run anyway because a non-zero count here would mean the
  instrumentation is broken, not that the claim is.
- **A3 -- a history window** of 5 lagged cluster states alongside the current one.
  Lagged cluster state is still *shared* by every co-queued job at the dispatch
  instant, so the argument predicts it lands in the same "identical for every queued
  job" bucket as the contemporaneous block. Two approximations are documented rather
  than hidden: a "tick" is one **training row** back when fitting and one **dispatch
  instant** back when simulating, and the lag block repeats the oldest available
  state until 5 instants have been seen.
- **A4 -- score computed at enqueue time and cached.** If a job is scored when it
  *enters* the queue and the score is then cached, two co-queued jobs were scored
  against *different* cluster states, and the shared-state premise of Proposition 1
  fails by construction. This is a **different policy**, not a different
  implementation of the published one, so the utility question is asked separately
  and written to its own CSV: staleness is a cost as well as a source of variation.

A1, A2 and A3 record **zero** counterexamples on both traces; A4 breaks the
degeneracy and schedules no better. The tables are in
`phases_22_30/COMPLETION_SUMMARY.md`.

## Phase C: evaluation split protocol
"How accurate is the model?" has no answer until the split is named, so the
protocol is now stated explicitly and all four splits are reported side by side
in `05_results/models/evaluation_splits.csv` (`03_models/evaluate_splits.py`).
This matters here more than it usually would: the dataset is 2,200 rows that are
20 simulation runs of 110 jobs, and rows from one run share a cluster
trajectory, so a random *row* split puts near-duplicates on both sides.

- **random** — `train_test_split(test_size=0.2, random_state=42)` over rows.
  R² 0.836840, MAE 4.6935. This is the published protocol, and it is optimistic.
- **run-wise — THE HEADLINE** — `GroupKFold(n_splits=5)` on `run_id`: 16 training
  runs / 4 held-out runs per fold, no run on both sides. **R² 0.810910 ±
  0.020672, MAE 4.8970 ± 0.4492.** This is the number to quote for "does the
  model work", because it is the only split that asks whether the model
  transfers to a cluster trajectory it has not seen. It is **lower** than the
  0.837 this repository previously advertised as its headline, and the reason is
  the split, not any change to the model.
- **chronological** — within each run, train on arrivals before that run's own
  0.8 arrival-time quantile, test on arrivals at or after it. R² 0.725147,
  MAE 7.2403. This is what deployment means: predicting the future of a queue
  from its past. Any *deployment* claim must quote MAE 7.24 — roughly 54% more
  error than the 4.69 the repository used to advertise.
- **leave-one-run-out** — 20 folds, kept as a sensitivity check on the fold
  count: R² 0.793486 ± 0.054176, MAE 4.8207 ± 1.0156.

**Fold-count rationale.** Five folds are used for the headline so the training
set stays at 1,760 rows — exactly the size of the random split's training set —
which forecloses the obvious objection that the gap between 0.8368 and 0.8109 is
merely less training data. Leave-one-run-out trains on more (2,090 rows) and
still scores lower, but its per-fold R² spans 0.6899 to 0.8744; that spread is
also why a single grouped hold-out would have been unquotable, since the answer
would have depended on which run happened to be held out.

**Scale.** Constant-predictor baselines are recomputed inside every fold of every
split in the same file, so the reader can see what "R² 0.81" is worth. On the
run-wise folds a mean-constant predictor scores R² −0.019958 and a
median-constant −0.088624, with MAE around 13 (13.4502 and 13.0214). The learned
model's 4.90 is about a 64% reduction in error against the mean-constant
predictor. It is a real but ordinary regressor rather than a dressed-up mean,
and the reader is owed both halves of that sentence.

**This correction does not touch the ranking-degeneracy result.** Degeneracy is a
statement about the *functional form* of the score — at a fixed dispatch instant
every queued job sees the same cluster state, so the score is a function of
requested size alone — and it holds whatever the accuracy is. A more accurate
model would produce the same size-ordered permutation. The accuracy figure and
the degeneracy finding are independent claims; neither weakens the other.

## Enhancements
1. **Ablation (intervals as of Phase C)**: remove each of 12 features and measure
   the R² drop. The drop is no longer one number from one fixed row split: each
   ablation is re-fit over **20 leave-one-run-out folds on `run_id`**, the drop is
   taken **paired within fold**, and a Student-t 95% interval is formed over the
   20 paired differences (`05_results/models/ablation_study_results.csv`, which
   is duplicated byte-identically at `05_results/ablation_study_results.csv` —
   one artefact under two names, not two studies). The baseline falls with the
   protocol: R² 0.836840 (single fixed row split) becomes 0.793486
   [0.767472, 0.819500] leave-one-run-out.
   **Only three of the twelve drops are distinguishable from zero**: `job_gpu`
   0.2051 [0.1565, 0.2538], `queue_length` 0.0217 [0.0081, 0.0353], and
   `queue_pressure` 0.0158 [0.0051, 0.0264]. The other nine intervals span zero.
   Two readings must be kept apart there: an interval spanning zero means the
   drop is **not distinguishable from zero at this sample size**, *not* that the
   feature contributes nothing. Accepting a null because a test failed to reject
   it is precisely the error this repository criticises elsewhere — it is why
   equivalence claims here use TOST rather than a large p-value.
   Two structural limits bound what a single-feature ablation can say at all,
   both measured in `05_results/models/feature_collinearity.csv`: `total_free`
   and `avg_free_per_node` correlate at 1.000000 with **infinite VIF** — in the
   synthetic generator one is the other divided by a constant node count, so they
   are the same variable twice — and `fragmentation` and `variance_free`
   correlate at 0.950720 (VIF 66.8 and 34.1). Dropping one member of such a pair
   leaves the information intact in the other, so a near-zero drop there is
   arithmetic, not evidence about the feature.
   Finally, `queue_pressure` being one of only three real contributors is
   *consistent* with the degeneracy result rather than in tension with it: v3.5
   established that `queue_pressure` is itself a deterministic function of
   requested size given the cluster state.
2. **Fairness**: evaluate max wait, Gini index, completion by size, starvation count.
3. **Scheduler baselines (14 as of v3.4)**: FCFS/first-fit (historical 'FIFO' key), strict head-blocking FIFO, SJF with true runtimes (oracle), SJF with f-model user estimates (est = runtime·f, f~U(1,C), C=5; Mu'alem & Feitelson 2001) and modal estimates (menu rounding, Tsafrir & Feitelson 2005), STATIC_PRIORITY (relabelled in v3.6 — the old "Priority + aging" name was false: the key expands to `(priority_score + 0.03·arrival_time) − 0.03·current_time`, and the `current_time` term is a common additive shift at any one instant, so it cancels pairwise, the induced order is time-invariant, and a waiting job can never overtake; its value as a baseline is precisely that it does not age, and no anti-starvation property may be claimed for it), HRRN (this repository's genuinely aging baseline), **SMALLEST (sort by requested size — the ML-free control implied by the degeneracy analysis)**, canonical two-condition EASY backfill (oracle and estimated runtimes), conservative backfill (per-job reservations on a capacity profile), preemptive SRPT (1-tick checkpoint penalty per preemption), Proactive (XGBoost), NN (MLP), predicted-wait EASY hybrid. Unified wait definition: wait = turnaround − true runtime (identical to start − arrival for non-preemptive policies; charges preemptive requeue time and checkpoint overhead as waiting). A runtime-estimate-quality sweep (C ∈ {1,2,3,5,10} + modal) isolates how much of classical schedulers' advantage survives realistic estimate error.
4. **SHAP (held out as of Phase C)**: summary, dependence, and force plots,
   computed on **held-out rows only** — 400 rows drawn from the 440-row test
   split, with the background drawn from that same split. They previously
   explained 400 rows sampled from the *full* dataset against a full-dataset
   background, roughly 320 of which the model had been fitted on, which made the
   attributions partly in-sample. `05_results/shap/shap_provenance.csv` records
   `rows_from_training = 0` and `background_rows_from_training = 0`, and verifies
   it with a sha256 of the actual explained row indices rather than by
   self-report. The conclusion survives the fix: the attribution ordering is
   essentially unchanged, `job_gpu` still dominates, and the only rank movement
   is a 7/8 swap between `running_jobs` and `variance_free`, whose mean |SHAP|
   values differ by 0.0003 — noise. That the fix changed essentially nothing is
   the point of reporting it: the explanation-based half of the degeneracy
   argument no longer rests on in-sample attributions.
5. **Real traces (validated as of v3.2; used for scheduling as of v3.4)**: two cleaned Parallel Workloads Archive traces are committed — LANL CM-5 1994 (1024 procs, 122,055 kept jobs) and SDSC SP2 1998 (128 procs, 43,117 kept jobs). `02_data/build_real_trace_datasets.py` reconstructs each job's submit-instant cluster state by replaying the recorded schedule; `02_data/real_trace_validation.py` evaluates prediction quality. **v3.4** adds `04_scheduler/trace_driven_benchmark.py`, which replays the traces through all 12 policies. Key point: SWF field 9 records the user's *requested time*, so the study uses the **real runtime estimates the traces contain** rather than the simulated f-model — real error is both larger and differently shaped (SDSC median 6.9× over-estimate; LANL 36.3% under-estimates, which the over-estimate-only f-model cannot generate). Jobs with missing estimates fall back to the trace median, deliberately *not* the true runtime, so estimate-driven policies get no free oracle. Protocol: chronological 60% train split, 20 evenly spaced windows (3-day warm-up not measured + 7 measured days), windows spaced evenly rather than selected by load.
6. **Scaling**: 4/8/16/32 nodes (8/32/128/256 GPUs), overhead and inference-latency
   measurement. Latency and overhead are wall-clock and machine-dependent, so no
   complexity class is fitted from them; the phase-26 "O(1) / latency is CONSTANT"
   verdict and every projection past the largest measured cluster were withdrawn in
   v3.6 (the fit had used four non-monotone timing points, and its classifier's
   one-sided test labelled a negative exponent constant by fall-through).
7. **Online learning**: incremental updates on streaming data.
8. **Concept drift**: rolling MAE trigger for adaptive retraining.
9. **ROI — withdrawn in full (Phase C)**: this item previously read "GPU-hour
   savings, energy savings, and annual cost-benefit metrics". It is retracted
   rather than caveated, because the defect is a category error and not an
   uncertain assumption. The quantity being monetised was GPU-hours saved, and
   this repository's own 40-run benchmark records `baseline_util ==
   proactive_util` in 40 of 40 runs (mean 0.641177, identical to six decimal
   places) with the same 110 jobs completing under both policies. The cluster
   performs the same compute either way: reordering a queue changes *when* jobs
   start, not how many GPU-hours they consume. The quantity was measured at zero,
   and widening the error bars around a number whose true value is zero still
   reports a saving — which is why no version of the study is retained.
   `05_results/roi_analysis.py`, `05_results/roi/`, the pipeline step and the
   dashboard panel are deleted. The measured wait-time reduction itself
   (7.897862% against FIFO) is unaffected and remains a real result; what is
   withdrawn is only the claim that it converts into money.
10. **Reproducibility and dashboard**: one-command pipeline plus interactive explorers.

## Statistical treatment
- Seeded paired runs for scheduler comparisons (40-run FIFO-vs-proactive benchmark; 20-run 14-scheduler benchmark with out-of-training seeds).
- Paired t-test, Wilcoxon signed-rank, and **Student-t** 95% confidence intervals; Benjamini–Hochberg correction across metrics; zero-variance comparisons reported as "n/a" rather than as significance.
- **CI provenance (corrected in v3.6)**: the headline 40-run interval [4.8824%, 10.9133%] comes from `04_scheduler/benchmark_statistical.py`, which uses `stats.t.ppf(0.975, df=n−1)` — it is a Student-t interval and was mislabelled a bootstrap. The one genuine percentile bootstrap in the repository is `phases_22_30/phase_22_stats/stats_bootstrap.py` (10,000 resamples of the mean, fixed seed); over the same 40 runs it gives [4.9102%, 10.6714%]. Cite whichever is meant by name and file — the two are not interchangeable labels for one number.
- v3.3: every scheduler is compared pairwise against both PROACTIVE and FCFS with Holm step-down correction applied separately to the t-test and Wilcoxon families (`05_results/schedulers/multi_scheduler_significance.csv`), plus Cohen's dz effect sizes; per-C paired CIs in the estimate sweep.
- **v3.4 equivalence testing**: claims that two policies perform *the same* use paired TOST (two one-sided tests, `simstats.tost_equivalence`) with an equivalence margin of 10% of the reference mean, reporting both one-sided p-values, p_TOST, and the 90% CI of the paired difference. A large p from a difference test is **not** evidence of sameness; without TOST the size-sort-vs-XGBoost comparison (Holm-adjusted p = 0.17) would read as "no significant difference" and be discarded rather than recognised as the result. Trace comparisons are paired by window; synthetic ones by run.
- **v3.4 metrics**: mean bounded slowdown `max(turnaround/max(runtime, τ), 1)` (τ = 60 s on traces, 1 tick synthetic) is reported alongside mean wait as a first-class metric — it is the standard batch-scheduling measure and mean wait alone hides the effect on short jobs.
- Mean and max wait reporting.
- Fairness measured via per-job Gini coefficient, run-level Jain index, starvation counts, and SLA compliance (Phase 27). One definition of starvation holds repository-wide: a job is starved when its wait exceeds **3× its own runtime**. That is what `04_scheduler/fairness_analysis.py` has always computed and what phase 27's SLA-2 uses; `04_scheduler/fairness_budget_sweep.py` used a distribution-relative "wait > 3× the run's mean wait" rule until v3.6 and now uses the per-job one (column `starved_jobs_wait_gt_3x_own_runtime`).
- Cross-dataset and OOD diagnostics via MAE and R² on freshly simulated shifted workloads (Phase 23). The cross-dataset check runs against a **synthetic proxy written in the LANL SWF schema** (`02_data/synthetic_proxy_lanl_schema_trace.csv`, results in `05_results/traces/synthetic_proxy_validation_results.csv`) — renamed in v3.6 because the old `lanl_trace_sample` / `lanl_validation_results` names invited it to be read as real LANL data. It is not; the real-trace study is item 5 above.
- **Phase C — power for the equivalence tests.** An equivalence margin without a
  power figure cannot separate "these two policies perform the same" from "this
  test could never have said otherwise", so `04_scheduler/tost_power.py` now
  computes achieved power and the n needed for 80% power for every equivalence
  claim, at the same margin `simstats.tost_equivalence` uses (10% of the
  reference mean), by Monte-Carlo over 20,000 replicates with seed 31337 —
  outside every protected seed family — into
  `05_results/trace_schedulers/tost_power.csv`. It produces one correction. On
  **SDSC**, SMALLEST_FIRST vs PROACTIVE on mean wait has achieved power 1.000 and
  would have needed only 3 of the 29 available windows; the equivalence claim
  rests on SDSC, and there it rests solidly. On **LANL** the same pair shows an
  observed paired difference of 320.02 s against a margin of 222.93 s and
  achieved power of **0.47%** (0.00465; 5.8% on the bootstrap variant). A test
  with a 0.47% chance of certifying equivalence returning "not equivalent" is not
  evidence, so the LANL row is reported **INCONCLUSIVE** and must never be
  reported as "different". Nor is "more windows would settle it" available: the
  observed difference *exceeds* the margin, so no sample size can certify
  equivalence there at 10%, and establishing a *difference* instead would need 29
  windows — 50 after Holm over the family of 11 — where LANL supplies 28 disjoint
  7-day windows. It is not settleable on that trace with disjoint windows at all.
  Two limits travel with the figure and are part of the claim: power computed
  from an observed effect is post-hoc and is an estimate rather than a design
  guarantee, and the window supply is a property of the trace, not of the
  experiment.
- **Phase C — censoring and the started set.** A policy comparison that averages
  wait over the jobs that started can be biased if the policies start different
  jobs, so `04_scheduler/censoring_analysis.py` decomposes every pair into jobs
  started under both, under A only, and under B only, and recomputes the
  improvement on the common set
  (`05_results/uncertainty/censoring_analysis.csv`). The selection effect is
  real, but it is confined to one of five scenarios and it runs in the direction
  that *penalised* the learned policies — so this is a check that passed, not a
  concern that was refuted. Four of the five scenarios start every job under
  every policy: 24 of the 30 pair-rows have a selection gap of exactly 0.0 and
  there is nothing to correct, and all censoring sits in `arr2.0_nodes4`. There
  all six non-zero gaps are **negative**, from −0.38 to −3.25 percentage points,
  meaning the common-set improvement is *larger* than the published one and the
  published statistic understated the learned policies by up to 3.3 pp. Nothing
  disappears on the common set. What the common set cannot do is adjudicate the
  exchange: for FIFO vs PROACTIVE in that scenario the means over 10 runs are
  91.9 jobs common, 21.8 FIFO-only and 39.4 PROACTIVE-only — a two-way exchange,
  not a one-way rescue — so the common-set figure decides only the ~92 shared
  jobs, and the ~61 exchanged jobs are described but not adjudicated.
- **Phase C — the OOD failure taxonomy was a constant and is now a ranking.** Any
  earlier sentence describing "the OOD failure taxonomy" was describing a column
  that said the same thing 72 times: all 72 scenarios in
  `phases_22_30/phase_23_sensitivity/ood_failure_modes.csv` were labelled
  `DISTRIBUTION_MISMATCH`, zero entropy. The cause was a classifier that tested
  `mape >= 35.0` ahead of most branches while the minimum MAPE over the 72
  scenarios is 54.01, so that gate fired 72 times out of 72 and seven of the
  eight categories were unreachable dead code; `risk_level` was correspondingly
  65 MEDIUM / 7 HIGH / 0 LOW. The replacement is a continuous severity score
  standardised across the grid, four data-derived quantile bands of 18 scenarios
  each, and a dominant-axis label: `failure_mode` now spans five values
  (COMPLETION_DOMINATED 22, POLICY_DOMINATED 21, FIT_DOMINATED 14,
  NO_DOMINANT_AXIS 8, CALIBRATION_DOMINATED 7) and `risk_level` three
  (MEDIUM_RISK 37, HIGH_RISK 24, LOW_RISK 11). The simulation, the scenarios and
  the seeds are unchanged; only the classification of the results is. One reading
  rule travels with the new labels: severity is standardised **within this grid**,
  so `LOW_RISK` means "least severe among 72 shifted regimes" and never "safe" —
  none of these regimes is good, and the mean R² across them is negative
  (−0.3104).

## Reproducibility
- The entire pipeline is seeded: `bash run_all_experiments.sh` (**22 steps** as of Phase C, which added `03_models/evaluate_splits.py`, `04_scheduler/tost_power.py` and `04_scheduler/censoring_analysis.py` and removed the deleted ROI step) regenerates the dataset, model, and every result deterministically on a fresh checkout. This became true only in v3.6 — until then seven studies sat outside every pipeline script, so the claim was false as written; `phases_22_30/run_all_experiments_v2.sh` is now a forwarding shim rather than a second entry point. A regenerated tree can be checked with `python tools/verify_artifacts.py` (`--quick` / `--smoke`).
- One honest exception to "deterministically": three artefacts carry wall-clock columns that cannot reproduce on any machine — `05_results/model_comparison_table1.csv` (`training_time_sec`), `05_results/scaling/scaling_analysis.csv` (its three `*_sec` columns), and `phases_22_30/phase_26_scaling/scaling_benchmark.csv` (`inference_latency_ms`, `throughput_overhead_pct`). Between two runs on the same machine here these drifted by up to 84%. Every other column reproduced identically, and no claim in this repository should rest on a timing column.
- `PYTHONUTF8=1` is exported by the pipeline scripts so Unicode console output works on Windows (cp1252) as well as Linux/macOS.

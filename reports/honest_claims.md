# Honest claims

What this repository may say, what it may say only with a caveat, and what it must never
say. Every entry is traceable to a committed artifact; every caveat has exact wording you
can paste.

The test of a claim here is not whether it is true but whether **a stranger on a fresh
clone can check it**. `python tools/verify_artifacts.py` re-runs all 24 pipeline steps in
a scratch copy and diffs every artifact; `python -m pytest` pins every headline number to
its committed file.

---

## A. Fully evidenced — say these plainly

**The degeneracy.** Over 45,432 dispatch instants — 41,786 replayed from two Parallel
Workloads Archive traces and 3,646 from a synthetic generator — two equally-sized
co-queued jobs never received different scores. Zero counterexamples.
*`05_results/degeneracy/ranking_degeneracy.csv`, step 9.*

**Why.** Seven of twelve synthetic features (four of eight on each trace) vary across a
queue in 0.0% of instants. At a fixed instant they are constants, so they cannot separate
two jobs whatever the score function is. `queue_pressure` varies exactly as often as
`job_gpu` — it is a function of requested size given the state.
*`05_results/degeneracy/feature_variation.csv`.*

**The consequence.** A queue of 8.5–10.2 jobs receives 2.28–3.09 distinct priority levels,
and every queued job receives an identical score in 14.41% (SDSC), 20.73% (LANL) and
17.11% (synthetic) of instants, where the policy is exactly FCFS.

**Equivalence to a size sort, on the synthetic benchmark and on SDSC.** Paired TOST gives
+0.7952% with a 90% CI of [+0.025, +0.228] ts against a ±1.59477 margin, p = 2.6e-16;
SDSC −0.051%, p = 1.76e-12. *`multi_scheduler_equivalence.csv`,
`trace_scheduler_equivalence.csv`.*

**The MLP is not merely equivalent to the size sort — it is identical to it.** Bit-identical
on nine metrics across all twenty runs, both at 16.074545 ts.
*`multi_scheduler_runs.csv`.*

**Non-degeneracy is necessary and not sufficient.** Five per-job attributes that do not
factor through requested size each break the degeneracy on both traces: violations rise
from 0 to 6,194–10,718 (SDSC) and 0 to 13,801–28,497 (LANL), Kendall τ falls, distinct
levels rise, ties fall. And **0 of 12** augmented variants beat shortest-job-first on the
users' own estimates; the best is +1.48% slower and certified *equivalent* to it.
*`non_degeneracy_sweep.csv`, `non_degeneracy_utility.csv`, steps 12–13.*

**The claim survives three of four attacks.** A pairwise or listwise learning-to-rank
objective, monotone transforms, and a five-tick history window all give zero
counterexamples. *`robustness_attacks.csv`.*

**Real user estimates are not the f-model.** LANL under-estimates 36.27% of the time,
which `runtime × U(1,C)` cannot produce at all; SDSC's median estimate ratio is 6.907 and
LANL's 1.514. *`trace_estimate_quality.csv`.*

**Utilisation and completions are unchanged.** Identical in all 40 paired runs to six
decimal places. This is why the ROI study was deleted rather than caveated.

---

## B. True, but never without the caveat — exact wording provided

**The 7.9% wait reduction.**
> 7.9% mean wait reduction versus FIFO on the synthetic benchmark (40 seeded paired runs,
> paired t-test p = 2.0e-06, **Student-t** 95% CI [4.9%, 10.9%]). The interval is a
> Student-t interval, not a bootstrap; the repository's one genuine percentile bootstrap
> over the same 40 runs gives [4.9%, 10.7%]. Utilisation and completions are identical in
> all 40 runs, so this is a reordering effect, not a throughput gain.

**Model accuracy.**
> R² 0.811 ± 0.021, MAE 4.90 (run-wise `GroupKFold` on `run_id`). The random-row-split
> figure of 0.837 that earlier versions quoted is optimistic: the 2200 rows are 20
> simulation runs, so a random split puts rows of the same run on both sides. Under a
> within-run chronological split — which is what deployment means — R² is 0.725 and MAE
> 7.24. **The degeneracy result does not depend on this number**: it is a statement about
> the functional form of the score, not its accuracy.

**The LANL equivalence row.**
> On LANL the equivalence test is **inconclusive**, not negative. Achieved power was 0.47%,
> and because the observed difference (320.02 s) exceeds the margin (222.93 s), no sample
> size can certify equivalence there at a 10% margin. Establishing a *difference* would
> need 29 windows, or 50 after Holm over the family of 11; the trace supplies 28 disjoint
> 7-day windows. The equivalence claim rests on SDSC, where achieved power is 1.000.

**The scope of the degeneracy.**
> The claim holds for a score computed **at the dispatch instant** over cluster-state
> features. A policy that scores a job at enqueue time and caches the result scores
> different jobs against different cluster states; that variant is measurably
> non-degenerate (20,482 and 37,741 counterexamples). It is also worse: on LANL +11.7%
> mean wait and 27% worse bounded slowdown, and on both traces behind plain SJF.

**The trace-driven comparison.**
> Replayed on a capacity-only simulator that models occupancy, not the real machine's
> scheduler. The fidelity gap is measured and is large on SDSC: simulated FCFS mean wait
> 174.6 min against 630.9 min recorded (LANL 38.8 against 33.2). Relative comparisons
> between policies under the same simulator are meaningful; absolute waits are not
> comparable to the machine's own logs.

**The ablation.**
> Only three of twelve feature drops are distinguishable from zero (`job_gpu` 0.205,
> `queue_length` 0.022, `queue_pressure` 0.016; 20 leave-one-run-out folds). An interval
> spanning zero means the drop is **not distinguishable from zero at this sample size** —
> it does not establish that the feature adds nothing. Two features are the same variable
> twice: `total_free` and `avg_free_per_node` correlate at 1.000000 with infinite VIF.

**The OOD severity ranking.**
> Severity is standardised *within* a grid of 72 shifted regimes, so the least-severe band
> means "least severe among these", never "safe". Mean R² across the grid is negative.

**The dispatch-instant count.**
> 45,432 dispatch instants on the reference platform; 45,268 on a Linux runner;
> **zero violations on both**. The count is platform-dependent and the result is not.
> XGBoost's histogram build reduces in parallel, so the fitted model depends on thread
> count and library build, and the model drives dispatch decisions — so a different
> machine visits slightly different instants. Never write the count in a form that
> implies it is machine-independent. See `reports/cross_platform_reproduction.md`.

**External validity.**
> Established on two real machines, both from the 1990s, plus a synthetic generator. The
> argument is structural and predicts the same outcome on modern GPU clusters, but that
> prediction is **untested** — see `reports/d3_modern_trace_gap.md`.

**EASY's cost of real estimates.**
> +6.2% mean wait on SDSC and +74% on LANL. On significance: the raw paired t p is 0.025
> but 0.23 after Holm; the Wilcoxon signed-rank test gives Holm p = 2.1e-05. The effect
> survives correction under the rank test and not under the t-test — the wait
> distribution's tail is what carries it.

---

## C. Never say these

**"The ranking is a permutation of the size order."** It is a *function* of size, not an
*increasing* one. The recovered size-to-priority table is monotone in only 57–63% of
instants. Say: the ranking is measurable with respect to the partition of the queue into
size classes — it may order size classes arbitrarily and may only not distinguish jobs
within one. (Proposition 1, Corollary 1.3.)

**"All scores tie in 18–27% of instants."** That range is the fraction where the induced
*order* equals arrival order. All-tied implies that; the converse does not. The measured
all-ties fraction is 14–21%. Both are true; quoting one for the other is not.

**"45,432 real dispatch instants."** 3,646 are synthetic. Say 41,786 real + 3,646
synthetic.

**"Bootstrap 95% CI [4.9%, 10.9%]."** That is a Student-t interval. The bootstrap gives
[4.9%, 10.7%].

**"ML scheduling does not work."** The result is about one feature family used for ranking
at a dispatch instant. Phase D demonstrates five ways to break the degeneracy.

**"Priority + aging."** The baseline's aging term is a common additive shift at any instant
and cancels pairwise; the order is time-invariant and no job overtakes by waiting. It is
`STATIC_PRIORITY`.

**Any ROI, dollar figure or percentage return.** The study is deleted. It monetised
GPU-hours saved while the benchmark records identical utilisation and identical completions
in all 40 runs — the quantity was measured at zero. A category error, not an uncertain
assumption.

**"O(1) inference latency" / "constant regardless of cluster size" / any projection to
512, 1024 or 4096 GPUs.** Withdrawn. The fitted exponent was −0.234 over four non-monotone
wall-clock points, from a one-sided classifier that labelled a negative exponent constant
by fall-through.

**"Utilisation stays above 99.7%."** The minimum is 99.694. Say "above 99.69%".

**"Inference costs 10–48 ms per decision."** Not the range in any committed artifact, and
wall-clock besides.

**"The predicted-wait backfill hybrid ties plain EASY (n.s.)."** No paired test between
those two exists anywhere. State the two means without the significance claim.

**"Proactive ≥ FIFO on fairness."** Withdrawn. Per-job Gini rises from 0.526 to 0.794 and
max wait from 57.85 to 122.65 ts. Run-level Jain and the composite SLA score do slightly
favour it, because they aggregate per run rather than per job — which is why the honest
framing is a trade-off, not a reversal.

**"The OOD failure taxonomy shows…"** Before v3.6 that column held one constant value
across all 72 scenarios. Any claim resting on the old labels is void.

**"run_all_experiments.sh regenerates every result" — for any version before v3.6.** It
was false when written; seven studies sat outside every pipeline script. It is true now.

**Any number from `docs/project_report.html` or `docs/research_progress.html`.** Both were
deleted as frozen v3.2 snapshots that never mentioned the degeneracy result.

---

## D. Three things that are easy to get subtly wrong

1. **The accuracy correction is not a weakening of the contribution.** Lower R² and the
   degeneracy result are independent; the latter is about functional form. Never present
   them as connected.
2. **The censoring audit did not refute the censoring concern.** Selection exists in one
   scenario of five, and there it *penalised* the learned policies by up to 3.3 points. The
   common set adjudicates ~92 shared jobs; ~61 exchanged jobs are described, not
   adjudicated.
3. **The digits are platform-scoped; the claims are not.** Every committed number was
   generated on one machine, and a second platform reproduces the claims but not the
   low decimals. `tools/verify_artifacts.py` checks the digits on the reference
   platform; `tools/verify_claims.py` checks the claims anywhere. Quoting a number
   without that scope is the subtlest available overstatement.
4. **Six columns in three CSVs are wall-clock and cannot reproduce anywhere.** They are
   reported as `TIMING` exemptions by the verifier and must never be quoted as results.

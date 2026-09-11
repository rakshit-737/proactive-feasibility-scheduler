# Phase D — pushing the science forward

Companion documents: `reports/phase_A_report.md`, `reports/phase_B_report.md`,
`reports/phase_C_report.md`, `reports/d3_modern_trace_gap.md`, `CHANGELOG.md`.

Phases A–C made the repository's claims true and its methods honest. Phase D asks what
the negative result is *worth*: it turns a stated condition into a measured one, writes
the argument down as a proposition, and then tries to break the whole thing.

---

## 1. The four deliverables

| item | outcome |
|---|---|
| **D1** demonstrate the non-degeneracy condition | done — five attributes, both traces, and the utility half that makes it a finding |
| **D2** formalise the degeneracy as a theorem | done — Proposition 1 with three corollaries, Proposition 2 for the converse |
| **D3** break the 1990s trace dependency | **not done** — documented as a gap rather than faked |
| **D4** adversarial self-attack | done — three attacks fail, one succeeds and forces a scope limitation |

---

## 2. D1 — the condition, demonstrated

The paper stated that a feature set can rank more finely than the size partition only
if it carries a per-job attribute that is not a function of requested size given the
cluster state. It never showed it. Now it does.

Five such attributes were added to the 8-feature trace vector, one at a time and all
together: the user's requested time, causal per-user mean wait, causal per-user mean
runtime, queue identity, and user identity. Partition is constant at −1 on both traces
and is excluded, which the module says explicitly rather than leaving it quietly out.

**The sweep validates itself before it claims anything.** Its baseline arm reproduces
the published diagnostic field for field — SDSC 11,843 instants, τ 0.751883, 14.41%
all-tied; LANL 29,943 instants, τ 0.620662, 20.73% all-tied — and reproduces the
published mean waits and `PROACTIVE_EST`. The parallel experiment and the published one
agree, which is what makes the other twelve rows readable.

| quantity | SDSC baseline → augmented | LANL baseline → augmented |
|---|---|---|
| violations | 0 → 6,194–10,718 | 0 → 13,801–28,497 |
| Kendall τ vs size | 0.752 → 0.593–0.750 | 0.621 → 0.420–0.597 |
| distinct score levels | 3.09 → 3.58–4.55 | 2.28 → 2.86–4.54 |
| all-tied fraction | 14.41% → 6.31–9.48% | 20.73% → 6.01–13.48% |
| order = size order | 71.5% → 33.1–50.6% | 77.6% → 32.0–56.0% |

Every predicted direction holds, for all five attributes, on both machines.

**The caveat, carried rather than hidden.** The TOST-equivalence column only *breaks*
on SDSC. On LANL the published baseline was never TOST-equivalent to `SMALLEST_FIRST`
in the first place — and Phase C established why that row is **inconclusive** rather
than "different": the test had 0.47% power, and because the observed difference exceeds
the margin, no sample size can certify equivalence there. So on LANL what Phase D
breaks is the degeneracy itself, not an equivalence that was never certified.

### The half that makes it a finding

Non-degeneracy is *necessary* for the model to rank at all. It is not *sufficient* for
the model to be useful. Paired by window label against shortest-job-first on the users'
own runtime estimates:

**0 of 12 augmented variants beat the heuristic.** Every one is slower in the paired
mean. The best case — LANL with all five attributes — is +1.48% slower and is certified
**TOST-equivalent** to it (p = 0.0014). A full non-degenerate gradient-boosted pipeline
over five extra per-job attributes buys a statistical tie with sorting on the walltime
the user already typed into the submission script.

### Causality, which is the whole game

A per-user history computed over the full trace leaks the future and would manufacture
a fake non-degeneracy — the experiment would then prove nothing at all. The
implementation sweeps submit order with a min-heap of pending completions and absorbs a
job into its user's running means only once `submit + wait + runtime ≤ t`: **completed**
earlier, not merely submitted earlier, because a wait is unobservable before the job
starts and a runtime before it ends. Cold start is `NaN` rather than a global mean, so
"unseen user" stays distinguishable from "user who averages the global mean"; the
cold-start rate is 1.2% on SDSC and 0.2% on LANL. Three leakage tests go red under a
full-trace groupby.

`user_id` and `queue_id` enter as raw SWF integer codes rather than one-hot. That is the
conservative choice: an integer code is an arbitrary but consistent partition, and
one-hot could only separate co-queued jobs *further*, strengthening the non-degeneracy
half rather than weakening it.

---

## 3. D2 — the proposition, and what writing it down exposed

`METHODOLOGY.md` and the manuscript now carry the argument formally: Proposition 1 with
a three-line proof, Corollaries 1.1–1.3, and Proposition 2 for the converse with three
limits.

**Writing it down caught an overstatement in the project's own prose.** The methodology
said the ranking is "a permutation of the size order". That is too strong, and the
repository's own measurement contradicts it: the recovered size-to-priority table is
monotone in only 57–63% of instants. Proposition 1 says the score is a *function* of
size, not an *increasing* function of it. The correct statement is Corollary 1.3 — the
ranking is measurable with respect to the partition of the queue into size classes: it
may order size classes arbitrarily, and may only not distinguish two jobs within one.

That distinction is load-bearing. It is precisely why equivalence to a size sort has to
be established *statistically*, by TOST against `SMALLEST_FIRST`, rather than deduced.
Degeneracy is a structural fact; that the resulting policy *performs* like ascending-size
order is a separate empirical claim. Conflating them would have handed a reviewer an
easy objection.

---

## 4. D4 — four attacks, three failures, one scope limitation

Full 20-window protocol, both traces, with a control row that again reproduces the
published diagnostic exactly, so each attack is read against a like-for-like reference.

| attack | violations | verdict |
|---|---|---|
| A1 learning-to-rank, `rank:pairwise` and `rank:ndcg` | 0 | survives |
| A2 monotone transforms, log1p and sigmoid | 0 | survives |
| A3 history window, 5 lagged cluster states | 0 | survives |
| **A4 score at enqueue time, cached** | **20,482 / 37,741** | **breaks it** |

The failures are informative rather than empty. A ranking objective changes *which*
size-to-priority table is learned — τ moves between 0.47 and 0.82 — but never produces a
single counterexample, because the degeneracy is a property of the **inputs**, not the
loss: if every per-job input is a function of size given the state, so is any function
of them. A lagged cluster state is still **shared** by every co-queued job at the
dispatch instant, so it lands in the same bucket as the contemporaneous columns. And A2
fails by construction, visibly: its rows are numerically *identical* to the baseline in
every order-derived column.

### The scope limitation

Caching a score computed at enqueue time scores different jobs against *different*
cluster states, so Proposition 1's shared-state premise fails by construction. The
all-tied fraction collapses from 14–21% to 0.06–0.21%, distinct levels rise from 2–3 to
8–9, and the induced order stops being the size order.

**The paper must state the claim conditionally: it holds for a score computed at the
dispatch instant.** Claiming the degeneracy for "wait-time-model schedulers" in general
would be wrong.

### But it does not rescue the approach

| trace | policy | mean wait | bounded slowdown |
|---|---|---|---|
| LANL | Proactive | 2229.3 s | 6.13 |
| LANL | enqueue-cached | **2491.1 s (+11.7%)** | **7.78 (+27%)** |
| LANL | SJF on user estimates | 1889.2 s | 4.51 |
| SDSC | Proactive | 8701.7 s | 18.49 |
| SDSC | enqueue-cached | 8559.0 s (−1.6%) | **21.56 (worse)** |
| SDSC | SJF on user estimates | 6946.3 s | 14.70 |

On LANL the non-degenerate variant is worse on both metrics. On SDSC its 1.6% mean-wait
edge is contradicted by median wait (76.0 → 105.4 s) and bounded slowdown. Staleness is
a real cost and it eats the variation it buys. Both traces put it well behind an ML-free
one-line heuristic.

**The negative result is narrowed in scope, not weakened in force.** The only
non-degenerate variant these attacks found is a different, and largely worse, policy.

---

## 5. What Phase D adds up to

Before: "a learned wait-time score cannot order a queue by anything except requested
size, and sorting by size does as well."

After, with a table behind each clause: the degeneracy follows from a proposition about
the feature map; it survives a different learning objective, a monotone
reparameterisation, and a history window; it is broken by five different per-job
attributes and by moving the scoring point, exactly as the proposition predicts; and in
all thirteen non-degenerate variants measured, a one-line heuristic still wins.

That is the difference between reporting that something did not work and identifying the
property that decides whether it can.

---

## 6. `[GAP]` — what Phase D did not settle

- **No modern GPU trace.** The external-validity story still rests on two machines from
  the 1990s. `reports/d3_modern_trace_gap.md` records what a candidate must supply, the
  specific obstacle per candidate, the procedure for adding one, and the prediction
  (zero violations, because the proposition is about the feature map and not the
  machine). This is the largest remaining weakness and it is stated as such.
- **The attacks are trace-only.** The synthetic setting was not attacked. It is untouched
  and uncontradicted, but it is also untested against A1–A4.
- **A1 needed modelling choices the data does not supply.** Learning-to-rank requires
  query groups and graded relevance; 1-hour submit buckets and five quantile grades of
  log-wait were used. A different grouping changes τ but cannot produce a violation,
  which is why the conclusion is robust to the choice — but the choice is recorded in
  the artifact rather than buried.
- **A3's lag is an approximation.** One training row back when fitting, one dispatch
  instant back when simulating, with the first five instants repeating the oldest
  available state. A sharper definition needs a lagged replay the feature builder does
  not expose.
- **The utility comparison is mean wait only.** Bounded slowdown is available in the same
  rows and would be a cheap extension.
- **One-hot encoding untested.** Categorical attributes enter as integer codes. The
  argument that this is conservative is sound, but unverified.

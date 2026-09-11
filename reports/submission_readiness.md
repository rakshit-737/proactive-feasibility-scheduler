# Submission readiness

A blunt go/no-go. Companion: `reports/honest_claims.md` (what may be said and how),
`reports/phase_D_report.md` (what the science now covers),
`reports/cross_platform_reproduction.md` (what a second platform reproduced, and what
it did not), `CHANGELOG.md` (every retraction, old value to new value, with the reason).

---

## Verdict

**GO for a workshop or a negative-results / reproducibility track. NOT READY for a
top-tier main conference track.**

The result is correct, it is reproducible end to end from a fresh clone, and it is now
stated at exactly the strength the evidence supports. What it is missing is not rigour —
it is *reach*. Two machines from the 1990s and a synthetic generator is a thin empirical
base for a claim about ML scheduling, and a strong main-track reviewer will say so. That
objection cannot be answered by more analysis of the data on hand; it needs a trace the
repository does not have.

Submitting this to a main track now would most likely produce a reject whose review reads
"convincing on the traces used, but the traces are not representative of the systems the
claim is about". Submitting it to a venue that exists for exactly this kind of result
should produce an accept.

---

## What is genuinely strong

- **The claim is structural, not empirical.** Proposition 1 is three lines and its premise
  is checkable by reading a feature builder. The 45,432 instants confirm it; they are not
  what establishes it. A reviewer who disputes the traces still has to dispute the
  proposition.
- **It survives attack.** Three of four adversarial attacks fail (learning-to-rank
  objective, monotone reparameterisation, history window) and the one that succeeds is
  reported, scoped, and shown to schedule worse.
- **The second half is the interesting half.** Five attributes break the degeneracy, and
  0 of 12 augmented variants beat sorting by the user's own walltime estimate. "Necessary
  but not sufficient" is a more useful finding than the negative result alone.
- **Reproducibility is unusually good for this kind of submission.** One entry point, 24
  steps; `tools/verify_artifacts.py` re-runs the whole pipeline in a scratch copy and
  diffs every tracked artifact; 399 tests, 31 of them pinning headline numbers to their
  committed files. Six wall-clock columns in three CSVs are declared non-deterministic and
  exempted by name rather than quietly tolerated.
- **The retraction record is an asset.** A CHANGELOG that lists withdrawn claims with old
  value, new value and cause is evidence of method. Do not hide it; cite it.

---

## Blockers, in order of how much they will cost

### 1. No modern trace — the only real blocker

Both real traces predate 2000. The claim is about GPU-cluster schedulers. The proposition
predicts the outcome is machine-independent, and that prediction is untested.
`reports/d3_modern_trace_gap.md` records the candidates, the specific obstacle for each,
and the exact procedure for adding one. **Fix:** one modern trace with per-job size,
submit, wait and runtime, run through the existing degeneracy harness. The harness is
already written; this is data access, not engineering. Expected effort: days, if a trace
can be obtained at all. **Until then, external validity is a stated limitation.**

### 2. Simulator fidelity

The capacity-only simulator reproduces SDSC's recorded FCFS mean wait at 174.6 min against
630.9 min. Relative policy comparisons under one simulator are fine; the gap is measured
and stated, but a reviewer will press on it. **Fix (cheap, worth doing):** state the gap
in Threats to Validity with the number, and argue explicitly why the degeneracy result is
invariant to it — the counterexample count is computed from scores at dispatch instants,
not from simulated waits.

### 3. The model's own accuracy is modest and split-sensitive

R-squared 0.811 run-wise, 0.725 chronological. A reviewer may argue the degeneracy is an
artefact of a weak model. It is not — the proposition holds for any score over that
feature family, and the learning-to-rank attack demonstrates it empirically — but the
paper must make that argument explicitly rather than leaving it to be inferred.

### 4. LANL equivalence is inconclusive and must be labelled so

0.47% power; the observed difference exceeds the margin, so no sample size certifies
equivalence at a 10% margin. The equivalence claim rests on SDSC and the synthetic
benchmark. Stating this plainly is strictly better than being caught on it.

### 5. Related Work must engage, not enumerate

A negative result lives or dies on whether the reader believes the thing being negated was
worth trying. Backfilling and its variants (Mu'alem and Feitelson; Tsafrir et al. on
estimate quality), the learning-to-rank literature, and reported ML-for-systems gains all
need engaging on their own terms. **A fabricated citation would destroy the paper's one
asset — its credibility about its own evidence.** Every reference must be verified against
a real record before submission.

### Not blockers (already handled)

Reproducibility, statistical machinery, claim/artifact traceability, the withdrawn ROI and
scaling-law claims, the fairness fork, the PRIORITY mislabel.

---

## Venues

Deadlines below are **cycle patterns, not verified dates** — confirm every one against the
venue's own call for papers before planning around it.

### 1. A reproducibility / negative-results venue (first choice)

Examples: the ACM REP conference, an MLSys or SC workshop on reproducibility, or a
"negative results in ML" track.

This is the paper's natural home and the only venue where the artifact is a first-class
contribution rather than an appendix. The submission is three things at once: a negative
result, a methodological rule ("check whether your per-job features can separate two
co-queued jobs before you train anything"), and a repository where every number in the
paper regenerates from one command and a verifier proves it. The 1990s traces are a much
weaker objection here, because the claim on offer is structural and the venue's readers
are primed to evaluate whether the *evidence* supports the *statement* rather than whether
the benchmark is fashionable. **Still missing:** nothing that blocks submission. The
modern-trace gap is stated and the paper is honest about it, which at this venue reads as
method rather than weakness.

### 2. JSSPP (Job Scheduling Strategies for Parallel Processing), co-located with IPDPS

The exact audience: the people who built the Parallel Workloads Archive, wrote the
backfilling literature, and have spent thirty years on the question of whether user runtime
estimates are usable. They will find the LANL/SDSC choice normal rather than dated, and the
finding that no non-degenerate variant beats SJF on user estimates is a direct contribution
to a conversation they are already having. It is the venue most likely to read the result
correctly. **Still missing:** a sharper positioning against backfilling — the paper compares
against EASY but should say precisely where the degeneracy sits relative to a backfilling
scheduler's own decision structure. Roughly a section's worth of work, not an experiment.

### 3. An MLSys or EuroSys workshop

Broader ML-for-systems audience, which is where the methodological rule has the most
leverage: the feature-separability check costs nothing and would have prevented this
project's own first two years. **Still missing:** this is the venue where the 1990s traces
hurt most. Without a modern trace, expect the review to be "plausible but unproven for the
hardware we care about". If a modern trace can be obtained, this becomes the highest-value
target; without one, it is the weakest of the three.

**Recommendation:** submit to (1) or (2) now with the limitations stated; pursue a modern
trace in parallel and hold (3) for the version that has one.

---

## Pre-submission checklist

- [ ] `python tools/verify_artifacts.py` clean on a fresh clone (reference platform)
- [ ] `python tools/verify_claims.py` clean — this is the one that must pass anywhere
- [ ] `python -m pytest` green; `ruff check .` clean
- [ ] `make paper-check` — zero undefined references or citations
- [ ] **Every citation verified against a real record.** No exceptions.
- [ ] Every number in the manuscript checked against `reports/honest_claims.md` section C
- [ ] Every table and figure caption names its artifact path and the command that produces it
- [ ] Anonymise the repository link if the venue is double-blind

---

## `[GAP]`

- **Venue deadlines are unverified.** Stated as cycle patterns only.
- **Cross-platform reproduction is disproven, and now handled.** The first Linux run
  reproduced 45,268 dispatch instants against the reference platform's 45,432, with **zero
  violations on both**. XGBoost's parallel histogram build makes the fitted model a
  function of thread count and library build, and the model drives dispatch decisions, so
  one perturbation reaches every downstream count. The claims are platform-independent; the
  digits are not. `tools/verify_claims.py` adjudicates the claims on any platform and CI
  runs it on every PR; `reports/cross_platform_reproduction.md` holds the evidence. Every
  quoted count must carry the reference-platform scope.
- **Page limit not checked** against any specific venue's format.
- **No external reviewer has read the paper.** Everything in this file is self-assessment,
  which is exactly the kind of evidence the rest of this repository refuses to accept.

# Phase B — integrity and reproducibility repairs

Companion documents: `reports/phase_A_report.md` (ground truth), `reports/claim_inventory.md`
(910 claims), `CHANGELOG.md` v3.6 (every retraction, old → new).

---

## 1. What the central result did

**It survived, unchanged, and is now defended by tests.**

A `git clone` into a clean virtualenv on Python 3.14 reproduces 45,432 dispatch instants with zero
equal-size/different-score violations, byte-identically. So do the TOST equivalence, the 7.9%
improvement, the twelve-policy trace table, the seven-of-twelve constant features, and the exact
bit-identity of the neural network and the size sort at 16.074545 ts.

Nothing in this pass weakened the degeneracy result. One supporting sentence was overstated and is
now corrected; the result itself needed no repair.

---

## 2. The retraction

README, RESULTS.md, the manuscript and the changelog all stated that *in 18–27% of instants all
scores tie, so the policy silently is FCFS*.

The column behind that sentence is `pct_order_identical_to_arrival`, which counts instants where the
induced **order** equals plain arrival order under the tie-breaks `(score, arrival, id)`. All-tied
implies order-equals-arrival; the converse does not hold. The column therefore **upper-bounds** the
claim rather than measuring it, and the diagnostic already computed the distinct-prediction count
per instant, so the true quantity was one line away.

| setting | order = arrival order | all scores tied |
|---|---|---|
| synthetic | 19.72% | 17.11% |
| SDSC SP2 | 18.11% | 14.41% |
| LANL CM-5 | 27.47% | 20.73% |

The honest pair of statements: the learned order coincides with plain arrival order in 18–27% of
instants, and every queued job receives an identical score in 14–21%. Both support the degeneracy
argument. A golden test now pins the two separately, and asserts they differ by more than a point,
so quoting one for the other is a test failure rather than a rounding choice.

---

## 3. Every number that moved

| quantity | old | new | cause |
|---|---|---|---|
| ROI annual savings | $78,073.65 | $79,930.32 | committed table computed from the superseded 7.7144% |
| ROI first-year return | 85.89% | 90.31% | same |
| estimate-sweep PROACTIVE mean wait | 16.1036 | 15.9477 | never regenerated after the v3.5 train/serve fix |
| estimate-sweep PROACTIVE max wait | 127.25 | 128.75 | same |
| estimate-sweep PROACTIVE bounded slowdown | 2.5670 | 2.5482 | same |
| estimate-sweep PROACTIVE Gini | 0.7915 | 0.7934 | same |
| proactive max wait (fairness) | 125 / 124.8 | 122.65 | phase-27 copy was stale |
| anti-starvation max wait | 87 / 87.15 | 88.15 | same |
| proactive per-job Gini | 0.80 | 0.79363 | same |
| phase-27 fairness table rows | 6 | 15 | same |
| contended-cluster advantage | 14.4% | 14.5% (14.5219) | rounding carried across four documents |
| utilisation floor | ">99.7%" | above 99.69% (min 99.694) | bound was false as written |
| inference latency range | "10–48 ms" | 10.24–15.49 ms this run | wall-clock; not the artefact's range |
| proxy transfer R² | 0.015 | 0.0321 | proxy was a v3.1 artefact the pipeline could not overwrite |
| proxy transfer MAE | 14.98 | 13.53 | same |
| phase-24 header | "15-run" | "20-run" | producer's literal contradicted its own numbers |
| budget-sweep starvation column | `starvation` | `starved_jobs_wait_gt_3x_own_runtime` | definition unified; values change and the trend inverts |
| phase-01 dataset | — | regenerated | never matched its own deterministic generator |

Row order moved in three CSVs without any value changing: `STATIC_PRIORITY` sorts after `SRPT`, and
NN and SMALLEST swap because they are exactly tied and the sort is stable. The tie itself is the
evidence.

---

## 4. What was repaired

### Fabrication removed

`phase_25_real_traces/trace_preprocessing.py` fell from 679 to 464 lines. Two mappers, `_map_lanl`
and `_map_alibaba`, built the wait-time **target** from a hard-coded linear formula while their
trace candidates were tagged `source_type="real"`. `estimate_cross_trace_mae` then scored a
heuristic whose coefficients nearly matched that generator's, so the reported error was circular,
and its output row carried `mape_pct` pinned at the 200.0 saturation value. No file those mappers
read has ever existed in this repository.

`phase_27_fairness/fairness_formal_analysis.md` was deleted. All 26 values in its two example tables
were tested against every numeric cell of every CSV in the repository; **exactly one appears as the
quantity claimed**. Its headers name columns that do not exist, its "~3.7% Gini improvement" matches
nothing under any Gini definition, and its SLA-2 threshold was never implemented.

### Unsupported verdict withdrawn

The scaling law reported `O(1)` and "latency is CONSTANT regardless of cluster size" from a fitted
exponent of −0.234, then extrapolated to 4,096 GPUs. The classifier's test was one-sided, so a
negative exponent was labelled constant by fall-through; the four points are non-monotone with a 5×
spread; and the fitted column is wall-clock, measured drifting up to 84% between runs. Replaced by a
function that reports the measured range and infers nothing.

### Reproducibility

A missing trace is now a hard failure, so the headline instant count cannot silently collapse. One
gzip-aware SWF reader replaces three readers, one of which could not run on a fresh clone at all.
The unvalidated cached-feature branch is gone. The interpreter is overridable, because a bare
`python3` on Windows resolves to the Store shim rather than an activated virtualenv. The pipeline is
one 20-step command that genuinely regenerates every result, which is what makes
`METHODOLOGY.md`'s long-standing claim true for the first time.

### Statistics

Paired statistics now align by run or window label instead of by row position, and raise on
differing label sets, duplicate labels, a missing unit column, or an absent reference. **No
published number moved**, which is provable: for equal, duplicate-free label sets the sorted label
sequence is identical for both schedulers, so reindexing reproduces the old float order exactly.
Regenerating all four affected CSVs confirmed it.

Starvation has one definition. The priority baseline is relabelled static priority with its formula
untouched, because its aging term cancels pairwise and cannot rescue a waiting job. An unrecognised
scheduler key is a hard error in phase 24 rather than a row labelled `unknown` flowing into a
published table.

### Tests

135 → 268 tests. `tests/test_golden_numbers.py` pins every headline number to its committed
artefact and fails rather than skips when an input is missing. Two tests that pinned known defects
as intended behaviour now pin the corrected contract.

---

## 5. How the work was checked

Every repair was written by one agent and then adversarially reviewed by another whose instruction
was to find what it got wrong. That found **48 substantive issues** in the first round, of which the
most valuable were four new tests that **passed equally well against the old broken code**. They
looked like evidence and guarded nothing.

The second round therefore required mutation testing: reintroduce the defect, prove the test goes
red, restore. Nine mutations were applied and all nine were caught. A third round closed the
remaining behaviours that survived mutation — the tie tolerance, the quantised-score consumers, the
CSV boolean round-trip, and two refusal paths — and replaced a syntactic guard that could be evaded
by rewriting the same defect in a different form.

The verify tool was itself checked this way: changing one instant count in a committed CSV turns two
golden tests red, and restoring it turns them green.

---

## 6. `[GAP]` — what this phase did not settle

- **The seed collision stands.** `scaling_analysis.py` derives its seed as
  `4000 + run_id + nodes`, a sum, so run 8 with 4 nodes collides with run 4 with 8 nodes. Seed
  families were explicitly out of scope; this is reported, not changed.
- **Cross-platform bit-reproduction is unproven.** Everything here was verified on Windows with
  CPython 3.14.3. Scheduling is discrete, so a last-ulp difference in one prediction can flip a
  queue order and move a published mean by percent. The weekly CI job is what will establish it, and
  until it has run once the full verification is deliberately not a per-PR gate.
- **Six wall-clock columns cannot reproduce anywhere.** They are reported as `TIMING` and never fail
  a run. That exemption is printed in every verification report rather than hidden, because an
  invisible exemption is indistinguishable from a silent pass.
- **The ROI study's premise is unresolved.** Its figures are regenerated and internally consistent,
  but it monetises a wait-time percentage while the project's own benchmark shows utilisation and
  completions are identical in all 40 runs. Whether the study should exist is a question for the
  methodology pass, not an integrity repair.
- **Two unreproducible numbers were preserved before deletion.** The phase-03 classifier ROC-AUCs
  and a phase-04 leaky-model MAE existed only in a frozen documentation page with no producer. They
  are recorded as `[UNVERIFIED]` in the Phase A report so the values survive the file.
- **`docs/explanation.html` embeds a distilled browser model** with no producer in the repository.
  It powers a working interactive demo, so it was annotated rather than deleted: the page now states
  that its numbers are a browser-side approximation and gives the measured model quality alongside.
- **The claim inventory's line-number citations are now stale** wherever this pass edited the file
  they point into. The verdicts remain accurate; the line numbers do not.

# Phase C — methodological hardening

Companion documents: `reports/phase_A_report.md` (ground truth), `reports/phase_B_report.md`
(integrity repairs), `CHANGELOG.md`.

Phase B made the repository's numbers traceable. Phase C asks a harder question about the same
numbers: are the *methods* behind them honest? Six studies, five of which lower a published figure.

---

## 1. What changed, in one table

| item | was | is | direction |
|---|---|---|---|
| model accuracy (headline) | R² 0.8368, MAE 4.69 | **R² 0.8109 ± 0.0207, MAE 4.90** run-wise | weaker |
| model accuracy (deployment reading) | 4.69 MAE implied | **7.24 MAE** chronological | weaker |
| ablation baseline | R² 0.8368 single split | **0.7935 [0.7675, 0.8195]** leave-one-run-out | weaker |
| ablation drops | 12 point estimates | **3 of 12 distinguishable from zero** | weaker |
| SHAP | 400 rows, ~320 in the fit | **400 held-out rows, 0 from training** | sound |
| LANL equivalence | "not equivalent" | **inconclusive, and unsettleable on that trace** | weaker |
| OOD taxonomy | one constant label | **5 categories, 4 bands** | informative |
| censoring | unexamined | **confined to 1 of 5 scenarios, and it penalised the model** | sound |
| ROI | ~$80k/yr, ~90% return | **deleted** | withdrawn |

**None of this touches the central result.** Ranking degeneracy is a statement about the functional
form of the score: at a fixed dispatch instant every queued job sees the same cluster state, so the
score is a function of requested size alone. That holds whatever the score's accuracy is. A lower
honest R² costs the paper nothing.

---

## 2. C1 — the evaluation was optimistic

The published R² came from a uniformly random 80/20 split of 2200 rows that are 20 simulation runs
of 110 jobs. Rows from one run share a cluster trajectory; adjacent rows share almost the same
cluster state. A random *row* split therefore puts near-duplicates on both sides.

`03_models/evaluate_splits.py` holds the model configuration fixed and evaluates four ways:

| split | folds | R² | MAE |
|---|---|---|---|
| random (anchor) | 1 | 0.836840 | 4.6935 |
| run-wise | 5 | 0.810910 ± 0.020672 | 4.8970 ± 0.4492 |
| leave-one-run-out | 20 | 0.793486 ± 0.054176 | 4.8207 ± 1.0156 |
| chronological | 1 | 0.725147 | 7.2403 |

Constant-predictor baselines, so the reader can scale it: under the random and run-wise folds a
constant scores R² between −0.006 and −0.134 at MAE about 13, so the model beats a constant by
roughly 64% MAE. Under the chronological split the constants collapse to −0.70 and −1.18, because
the target drifts within a run, and the model still beats them comfortably.

**The run-wise number should be the headline** — it is the only one that answers whether the model
works on a cluster trajectory it has not seen. The leakage penalty is modest at 0.026 R², and I will
not overstate it: the model does generalise across held-out runs.

**The number that costs something is the chronological one.** Fitting on the past and predicting the
future within a run is what deployment means, and there the error is 54% higher than the repository
advertised. A deployment claim should quote 7.24.

Five folds were chosen so the training set stays at exactly 1760 rows, the same as the random split,
so the gap cannot be blamed on less data. Leave-one-run-out is reported alongside because its
per-fold R² spans 0.6899 to 0.8744 — a single grouped hold-out could have produced anything in that
range, so quoting one would have been indefensible.

**Incidental finding:** the published anchor is row-*order* dependent. Sorting the same 1760 training
indices moves it to 0.8382 / 4.6773, because XGBoost's `subsample=0.8` draws rows in presentation
order.

---

## 3. C2 — SHAP was explaining memorised rows

It sampled 400 rows from the full dataset with the full dataset as background; roughly 320 were in
the fit. It now explains 400 rows from the 440-row held-out test split with a held-out background,
and the training split is reconstructed and *asserted* against the published test MAE of 4.693508
before anything runs — a silently wrong reconstruction would be worse than the defect.

**The conclusion survives**, which is the reportable outcome. `job_gpu` still dominates at 35.4% of
attribution mass and 76.2% of the mass carried by the four job-dependent features. The only rank
movement is a 7/8 swap between two features separated by 0.0003.

The first version of this fix wrote its provenance from the script's own variables, so an adversarial
reviewer reintroduced the defect and the guard tests stayed green — the artifact was asserting its
own correctness. It now records a sha256 of the actual explained row indices.

---

## 4. C3 — the LANL row is not "not equivalent", and cannot be settled there

| | LANL | SDSC |
|---|---|---|
| observed paired difference | 320.02 s | −4.44 s |
| equivalence margin | 222.93 s | 870.17 s |
| achieved power | **0.47%** | 100% |
| n for 80% power (equivalence) | **not achievable** | 3 |
| n for 80% power (difference) | 29, or 50 after Holm | — |
| disjoint 7-day windows available | **28** | 29 |

Two conclusions, the second sharper than the brief anticipated.

A test with a 0.47% chance of certifying equivalence saying "not equivalent" is not evidence. The
LANL row must read **inconclusive**.

And because the observed difference *exceeds* the margin, no sample size can certify equivalence
there at a 10% margin. So the honest sentence is not "more windows would settle it". Establishing a
*difference* would need 29 windows, or 50 after Holm over the family of 11, and the trace supplies
28 — it is not settleable with disjoint windows on LANL at all. The equivalence claim rests on SDSC,
where it rests solidly.

Both limits are recorded: power computed from an observed effect is post-hoc, and the window supply
is a property of the trace rather than a budget decision.

---

## 5. C4 — the censoring bias runs the other way

Four of five scenarios start every job under every policy: 24 of 30 pair-rows have a selection gap of
exactly zero. All censoring lives in one scenario, and there all six non-zero gaps are **negative**
(−0.38 to −3.25 percentage points), meaning the common-set improvement is *larger* than the published
one. The published statistic understated the learned policies by up to 3.3 points.

The mechanism, decomposed rather than asserted: for FIFO against the proactive policy the started
sets are 91.9 jobs in common, 21.8 FIFO-only and 39.4 proactive-only. That is a two-way exchange,
not a one-way rescue.

**This does not show the concern was unfounded**, and the study no longer says "refuted". The common
set adjudicates only the shared jobs; the roughly 61 exchanged jobs are described, not adjudicated.

---

## 6. C5 — the OOD taxonomy was a constant

All 72 scenarios carried the label `DISTRIBUTION_MISMATCH`. The classifier tested `mape >= 35.0`
ahead of most branches and the minimum MAPE across the grid is 54.01, so that gate fired 72 times out
of 72. Seven of eight categories were unreachable dead code, and `risk_level` was 65 MEDIUM, 7 HIGH,
0 LOW.

Replaced with a continuous severity score, four data-derived quantile bands and a dominant-axis
label. `failure_mode` now spans five values and `risk_level` three. The cut-points are derived from
the observed grid rather than hard-coded, which is exactly how 35.0 went stale unnoticed.

Severity is standardised *within* this grid, so a LOW_RISK band means "least severe among 72 shifted
regimes", never "safe". None of these regimes is good; the mean R² across them is negative.

---

## 7. C6 — two features are the same variable twice

| pair | correlation | VIF |
|---|---|---|
| `total_free`, `avg_free_per_node` | **1.000000** | infinite |
| `fragmentation`, `variance_free` | 0.950720 | 66.8 / 34.1 |

The first pair is *perfectly* collinear: in the synthetic generator one is the other divided by a
constant node count. A single-feature ablation cannot say anything about either member — dropping one
leaves the information intact in the other, so a near-zero drop there is arithmetic, not evidence.

With 20 leave-one-run-out folds and paired within-fold drops, **only three of twelve drops are
distinguishable from zero**: `job_gpu` 0.2051 [0.1565, 0.2538], `queue_length` 0.0217 [0.0081,
0.0353], `queue_pressure` 0.0158 [0.0051, 0.0264]. The other nine intervals span zero.

An interval crossing zero means the drop is not distinguishable from zero at this sample size. It does
**not** establish that the feature adds nothing. The figure previously asserted the stronger reading;
accepting a null from a failure to reject it is precisely the error this project criticises elsewhere,
which is why it uses TOST for equivalence rather than a large p-value.

`queue_pressure` appearing among the three real contributors is consistent with the degeneracy result
rather than in tension with it: v3.5 established that `queue_pressure` is itself a deterministic
function of requested size given the cluster state.

---

## 8. C7 — the ROI study is deleted

It converted a wait-time percentage into GPU-hours saved and priced them at roughly $80k a year and a
90% first-year return. The project's own 40-run benchmark records `baseline_util == proactive_util` in
40 of 40 runs — identical to six decimal places — and the same 110 jobs completing under both
policies. The cluster performs the same compute either way; reordering a queue changes *when* jobs
start, not how many GPU-hours they consume.

The quantity being monetised was measured at zero. That is a category error, not an uncertain
assumption, and no sensitivity band rescues it: widening the error bars on a number whose true value
is zero still reports a saving.

The wait-time reduction itself remains a real measured result. What is withdrawn is the claim that it
converts into money.

---

## 9. How the work was checked

Every study was written by one agent and adversarially reviewed by another instructed to find what it
got wrong. That found real problems in four of six, and the pattern repeated from Phase B: **guard
tests that did not guard.**

- The SHAP provenance asserted its own correctness, so reintroducing the defect left the tests green.
- The TOST decision rule was documented as "the single thing this module's tests guard", but loosening
  it to 1.75× alpha left all 24 tests passing — and a loosened rule inflates every power number.
- The ablation tests never referenced `run_id`, so the grouped cross-validation that is the entire
  point of the repair was unpinned.
- The censoring study stated its own finding with the **sign backwards** in the delivered artifact,
  claiming improvement "disappears on the common set" when every non-zero gap showed the opposite.

Round two closed all of them with mandatory mutation testing: reintroduce the defect, prove the test
goes red, restore.

---

## 10. `[GAP]` — what Phase C did not settle

- **The LANL difference is unresolvable on that trace.** Not a defect to fix; a property of the data,
  now quantified.
- **Post-hoc power is an estimate.** Computed from the observed effect size, it is not a guarantee.
- **The chronological split is a single partition.** Its reported standard deviation is 0 by
  construction, not a spread estimate. A fold-to-fold chronological variant was not built.
- **The collinear pair was not removed.** Measuring it was in scope; deciding whether the feature set
  should carry the same variable twice is a modelling change that would move published numbers, and
  belongs with the manuscript revision.
- **Censoring was analysed only in the uncertainty benchmark.** The phase-23 OOD grid reports a
  failure rate up to 79% and was not given the same common-set treatment.
- **The duplicate ablation artifact remains.** One table is written to two tracked paths; both now
  carry the interval columns and the verifier compares both, so they cannot drift, but the
  duplication predates this pass.

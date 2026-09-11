# Cross-platform reproduction: what holds, what does not, and why

Phases A–E closed with one `[GAP]` that could not be closed from inside the
repository: every artefact had been generated on one Windows machine, and no
second platform had ever run the pipeline. `reports/submission_readiness.md`
recorded the risk as *unproven*.

It is no longer unproven. It is **disproven**, and this document is the evidence.
The finding is favourable, but only because the distinction it forces — between a
number and a claim — is one the repository had not been making carefully enough.

---

## What happened

The first Linux run of the pipeline (GitHub Actions, `ubuntu-latest`, 2 vCPU,
Python 3.14, pinned requirements) failed. It failed at step 15, on a guard this
project wrote in Phase C:

```
SPLIT RECONSTRUCTION FAILED.
  model MAE on the reconstructed test split : 4.731409
  published hold-out MAE of wait_model_v2   : 4.693500
  tolerance                                 : 0.0005
```

The obvious reading — a broken train/test split, inconsistent preprocessing, a
feature-order mismatch — is wrong, and acting on it would have broken the
reproduction that already worked. Three facts rule it out:

1. **Same machine, same dataset, same split code, varying only the thread count:**

   | threads | hold-out MAE |
   |---|---|
   | 1 | 4.692659 |
   | 2 | 4.669915 |
   | 4 | 4.637280 |
   | 16 (reference platform) | **4.693508** |
   | 2 (Linux runner) | **4.731409** |

   The split was byte-identical across all five. A split defect cannot produce
   that table.

2. **The same reconstruction passes exactly on the reference platform.** A full
   `tools/verify_artifacts.py` run there reports 131 OK, 4 TIMING, 0 MISMATCH.
   An inconsistent split would fail everywhere.

3. **Linux diverged in 26 artefacts spanning the whole pipeline** — trace
   benchmarks, fairness metrics, estimate sensitivity, degeneracy diagnostics.
   The SHAP script's split code cannot move `trace_scheduler_windows.csv`.

## The actual cause

XGBoost's histogram build reduces floating point in parallel. The reduction order
depends on the thread count and on the library build, so the same data under the
same seed fits a slightly different model on a different machine. That model then
drives dispatch decisions; different decisions produce different queue
trajectories; different trajectories produce different dispatch instants. One
perturbation at the root reaches every downstream count, which is why a single
cause presented as twenty-six symptoms.

Note what this means for a pinned thread count: `trace_driven_benchmark.py`,
`non_degeneracy_sweep.py` and `robustness_attacks.py` all pin `n_jobs=4`, and
their outputs still moved on Linux. Thread count is one mechanism; the platform's
library build is another. Pinning threads addresses only the first, which is why
this document does not recommend pinning threads as the fix.

---

## What moved, and what did not

The decisive comparison is `05_results/degeneracy/ranking_degeneracy_totals.csv`.
Exactly one column differed:

```
row 0 col total_instants: expected 45432, got 45268
```

`total_violations` and `partial` are absent from the diff — they matched.

| | reference platform | Linux runner |
|---|---|---|
| dispatch instants | 45,432 | 45,268 (−0.36%) |
| equal-size / different-score violations | **0** | **0** |
| protocol complete | yes | yes |

**The claim reproduced. The count did not.**

That is the right way round, and it is what Proposition 1 predicts. The
degeneracy is a property of the feature map, not of the arithmetic: if every
per-job input is a function of requested size given the cluster state, then so is
any function of them — on any machine, in any float order. A perturbation nobody
designed has now confirmed the structural claim is invariant to it.

Other numbers moved in their low decimals: mean waits in the third significant
figure, the synthetic TOST `pct_diff` from +0.795% to +0.221%, and a permutation
of the ablation's feature ordering.

### The complete run

With the guard corrected, the full 24-step pipeline ran to completion on Linux and
every claim was checked against the artefacts it regenerated there:

```
135 artefacts: 42 DRIFT, 1 TIMING, 92 OK -- 1590.2 s elapsed
14 of 14 claims hold.
```

**92 of 135 artefacts reproduced bit-identically on a different operating system.**
The drift is confined to the 42 that depend on a fitted model, exactly as the
mechanism predicts; everything that does not train a model — the simulation
outputs, the trace inventories, the derived tables — is byte-for-byte identical
across platforms. That is a much sharper result than "it does not reproduce".

Every headline claim held, with its number alongside:

| claim | reference platform | Linux |
|---|---|---|
| dispatch instants | 45,432 | 45,268 (-0.36%) |
| violations | 0 | 0 |
| wait improvement vs FIFO | 7.90% [4.88, 10.91] | 7.79% [4.90, 10.69] |
| paired t p-value | 2.00e-06 | 1.94e-06 |
| synthetic TOST vs size sort | equivalent, +0.795% | equivalent, +0.221% |
| SDSC TOST vs size sort | equivalent, −0.051% | equivalent, −0.025% |
| NN identical to the size sort | yes, 8 metrics x 20 runs | yes, 8 metrics x 20 runs |
| augmented arms breaking the degeneracy | 12 of 12 | 12 of 12 |
| augmented arms beating SJF | 0 of 12 | 0 of 12 |
| utilisation FIFO vs PROACTIVE | identical | identical |
| split ordering | 0.837 > 0.811 > 0.725 | 0.833 > 0.808 > 0.727 |

One detail worth stating rather than smoothing over: *which* augmented variant comes
closest to the heuristic is platform-dependent — LANL `+all` at +1.48% here, SDSC
`+user_hist_runtime` at +1.76% on Linux. The claim is "0 of 12 beat SJF", and that
held on both; the identity of the runner-up is not a claim and should never be
written as one.

---

## What was changed in response

**Nothing was loosened.** In particular the 5e-4 tolerance was not widened, and
the split was not reordered to make the number come out.

1. **The guard was checking the wrong thing** (`03_models/explainability_shap.py`).
   It compared the reconstruction against a literal `4.6935` copied into its
   source, which conflates two questions: *did I rebuild the same split?* and *is
   this the same model the number was published from?* The bundle now records the
   split it was fitted under and the score it earned there, and the guard checks
   against that. A genuine split mismatch still fails — the model would be scoring
   rows it was fitted on and the MAE would collapse far outside the tolerance —
   while an honestly refitted model does not. The tolerance is unchanged, and no
   published number moved: retrained on the reference platform the model
   reproduces MAE 4.693508148193359 and R² 0.836840033531189 exactly.

2. **A second verifier was added** (`tools/verify_claims.py`) that checks the 14
   statements `reports/honest_claims.md` permits, rather than the digits that
   happen to express them. Directional claims assert direction and report
   magnitude; identities assert exactness, because there the identity is the
   claim. 16 mutation tests prove each check goes red against a tree carrying the
   defect it exists to catch — and prove that 45,268 does **not** fail, which is
   the point.

3. **CI now asks each platform the question it can answer.** Every PR runs the
   claim checks against the committed tree. The scheduled full run uses
   `--expect claims`: digit differences are reported as `DRIFT`, and the verdict
   comes from the claim checks. Digit-exactness remains verified on the reference
   platform by `make verify` and pinned by `tests/test_golden_numbers.py`.

The two instruments are complementary and neither replaces the other:

| tool | question | scope |
|---|---|---|
| `verify_artifacts.py` | does this tree regenerate itself, digit for digit? | reference platform |
| `verify_claims.py` | do these artefacts support what the paper says? | any platform |

---

## What this costs the paper, honestly

`45,432` must be read as a reference-platform figure. The honest form is: **45,432
dispatch instants on the reference platform, 45,268 on Linux, zero violations on
both.** That is a weaker statement about the count and a *stronger* statement
about the result, since the result now has two independent platforms behind it
instead of one.

`reports/honest_claims.md` carries the caveat wording. Any restatement that
implies the count itself is machine-independent is wrong.

## `[GAP]`

- **Only two platforms.** Windows/16-thread and Linux/2-vCPU. The mechanism
  predicts every distinct (platform, thread count, library build) triple gives its
  own digits; that is untested beyond these two.
- **The first Linux run compared only the first 14 steps.** It aborted at step 15 on
  the old guard, leaving 74 artefacts unregenerated. The re-run with the corrected
  guard completed all 24 steps and is the run reported above, so this gap is closed:
  the 7.9% figure has now been checked on a second platform and holds at 7.79%
  with a confidence interval that still excludes zero.
- **The drift bands in `verify_claims.py` are judgement, not theory.** The 2%
  instant tolerance was chosen to sit well outside the observed 0.36% and well
  inside anything that would indicate a changed protocol. There is no principled
  derivation of that number, and it is stated here rather than buried.

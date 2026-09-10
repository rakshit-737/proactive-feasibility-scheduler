# Phase A — ground truth

Read-only reconnaissance. Nothing in the repository was modified while this was compiled.
Companion document: `reports/claim_inventory.md` (910 rows).

Baseline: branch `manuscript-submission-prep`, HEAD `888b5ed`, clean working tree.

---

## 1. Environment

| item | value |
|---|---|
| host | Windows 11, Git Bash (`C:\Program Files\Git\usr\bin\bash.exe`) |
| interpreter under test | CPython 3.14.3 in a venv created for this run |
| dependencies | `pip install -r requirements.txt` into an empty venv, exit 0, every pin resolved |
| import check | numpy, pandas, scipy, sklearn, xgboost, catboost, lightgbm, shap, matplotlib, seaborn, joblib, plotly, streamlit all import |
| LaTeX | `pdflatex` and `latexmk` present (MiKTeX) |
| absent | `make`, `docker` |

The pinned dependency set installs and imports cleanly on 3.14, which is what `requirements.txt`
claims. That part of the reproducibility story holds.

### A fresh-clone hazard the brief did not list

`run_all_experiments.sh:8` resolves its interpreter as:

```bash
PY="$(command -v python3 || command -v python)"
```

On this machine `command -v python3` resolves to the Windows Store shim at
`~/AppData/Local/Microsoft/WindowsApps/python3`, **not** to an activated virtual environment.
Activating a venv and running the pipeline therefore silently executes it against a different
interpreter and a different set of installed packages. The Phase A run had to shim `python3`
explicitly to test the venv at all. Any caller who believes they are testing a pinned environment
may not be. This is the defect the planned `PY="${PY:-...}"` override fixes.

---

## 2. Fresh-clone pipeline run

Method: `git clone` of the branch into a scratch directory, fresh venv, `pip install -r
requirements.txt`, then `bash run_all_experiments.sh` with `PYTHONUTF8=1` and every line
timestamped.

**Result: exit 0, no traceback, no step failure. Total wall clock 534 s (8 min 54 s) on this
machine.** No document in the repository states a runtime; this is the first measurement.

### Per-step wall clock

| step | seconds |
|---|---|
| [0/14] generate dataset + train wait_model_v2 | 11 |
| [1/14] ablation study | 7 |
| [2/14] fairness analysis | 69 |
| [3/14] synthetic scheduler benchmark | 70 |
| [3b/14] estimate-sensitivity sweep | 2 |
| [4/14] ranking-degeneracy diagnostic | 102 |
| [5/14] trace-driven benchmark | 143 |
| [6/14] SHAP explainability | 0 |
| [7/14] "real trace loading" | 4 |
| [8/14] scaling analysis | 37 |
| [9/14] online learning + concept drift | 7 |
| [10/14] baseline statistical benchmark | 66 |
| [11/14] ROI analysis | 0 |
| [12/14] multi-model comparison | 16 |
| **total** | **534** |

### Warnings and silent degradation observed

- No traceback, no non-zero step exit, no `skipping sdsc`/`skipping lanl` line anywhere.
- Both `.swf.gz` traces are present in a fresh clone and the degeneracy diagnostic reads them
  through the gzip fallback, so the three-setting run is intact.
- **Step [7/14] is labelled "Real trace loading and synthetic-vs-real validation" and loads no real
  data.** Its output, verbatim:

  ```
  No SWF found and a real trace CSV already exists; writing fallback proxy to
  .../02_data/lanl_trace_fallback_proxy.csv instead of overwriting it.
  Loaded 2000 rows from synthetic_proxy_trace (fallback; no SWF found).
  ```

  The step looks for `02_data/lanl_trace_sample.swf`, a filename that has never existed in the
  repository (the two shipped traces are `LANL-CM5-1994-4.1-cln.swf.gz` and
  `SDSC-SP2-1998-4.2-cln.swf.gz`), so the fallback fires on every run, on every machine. It then
  **overwrites the tracked file `02_data/lanl_trace_fallback_proxy.csv`**, and writes its output to
  `05_results/traces/lanl_validation_results.csv` — a name that implies LANL validation for a table
  whose own first column honestly reads `synthetic_proxy_trace`. The regenerated content happens to
  be byte-identical, so the overwrite is invisible in `git status`.
- The step numbering is wrong: the script promises `[N/14]`, inserts an unnumbered `[3b/14]`, and
  ends at `[13/14]`, which is a bare `echo`. There is no `[14/14]`.

---

## 3. The headline result reproduces exactly

`05_results/degeneracy/ranking_degeneracy.csv`, regenerated from a fresh clone in a clean venv:

| setting | instants | equal-size different-score violations |
|---|---|---|
| synthetic (12 features) | 3,646 | 0 |
| SDSC SP2 (1998) (8 features) | 11,843 | 0 |
| LANL CM-5 (1994) (8 features) | 29,943 | 0 |
| **total** | **45,432** | **0** |

Byte-identical to the committed file. The project's central claim is reproducible on a fresh
clone, and this pass found nothing that weakens it.

---

## 4. Artifact diff over the whole run

164 tracked artifacts were compared against their committed blobs. Working-tree files carry CRLF
while `git show` returns LF, so every text file differs in bytes; comparison is therefore on parsed
values for CSVs and on EOL-normalised lines for text.

| verdict | count | meaning |
|---|---|---|
| identical | 153 | same values (47 CSVs, 106 PNG/PKL/text byte-identical) |
| genuinely different | 5 | see below |
| PNG re-rendered | 6 | raster bytes differ, all belong to the 5 changed CSVs |

**Only five CSVs in the entire repository differ after a full fresh-clone run**, and every one is
explained:

| artefact | cells | cause |
|---|---|---|
| `05_results/model_comparison_table1.csv` | 7 | **wall clock only** — `training_time_sec`. Every R² and MAE is bit-identical, so the headline R² = 0.837 reproduces exactly |
| `05_results/scaling/scaling_analysis.csv` | 12 | **wall clock only** — the three `*_sec` columns, differing by up to **84% relative**. `wait_advantage_pct` = 14.5219 is bit-identical |
| `05_results/roi/cost_benefit_analysis.csv` | 7 | **stale input** — the committed file was computed from the pre-v3.5 7.7144%; the script reads the current 7.897862% at run time |
| `05_results/schedulers/estimate_sensitivity.csv` | 101 | **stale artifact** — not regenerated since v3.5 |
| `05_results/schedulers/estimate_sensitivity_summary.csv` | 49 | same |

Everything else — the dataset, the model pickle, the ablation table, the fairness table, the whole
14-scheduler benchmark with its significance and TOST tables, all three degeneracy tables, the
entire trace-driven benchmark, SHAP, drift and online learning — regenerated with identical values.
**The pipeline is genuinely deterministic on a fixed platform**, which is a real strength and was
not previously demonstrated anywhere.

Artifacts whose producers are outside the pipeline (`real_trace_validation.csv`,
`budget_sweep.csv`, `uncertainty_ood_benchmark.csv`, `xgboost_tuning_results.csv`, everything under
`phases_22_30/`) were untouched by the run, which is the ORPHAN finding restated as an observation.

### The two stale scheduler files

Contained and explainable: only the PROACTIVE arm and the comparisons against it move, because the
v3.5 train/serve fix only affected that arm.

| row | field | committed | regenerated |
|---|---|---|---|
| PROACTIVE | mean_wait | 16.1036 | **15.9477** |
| PROACTIVE | max_wait | 127.25 | 128.75 |
| PROACTIVE | mean_bounded_slowdown | 2.56701 | 2.54821 |
| PROACTIVE | fairness_gini | 0.791476 | 0.793438 |

Every other scheduler's own `mean_wait` is unchanged (SJF_MODAL stays 14.0627, BACKFILL_EST at
`over_factor=10` stays 18.9364); only the `wait_diff_vs_proactive_ci95_*` and
`ttest_p_vs_proactive` columns shift, as they must. The regenerated PROACTIVE mean now agrees with
`multi_scheduler_benchmark.csv`, which is the point: the two files share a generator and a seed
family and had disagreed since v3.5.

### The ROI cascade

Re-running step [11/14] moves every figure in the ROI study, because `roi_analysis.py:85-94` reads
`mean_improvement_pct` from `benchmark_statistical_summary.csv` at run time while the committed CSV
still records the superseded input:

| field | committed | regenerated |
|---|---|---|
| mean_wait_improvement_pct | 7.7144 | **7.897862** |
| saved_gpu_hours | 34,714.83 | 35,540.38 |
| cloud_cost_reduction_usd | 76,372.62 | 78,188.84 |
| energy_saved_kwh | 12,150.19 | 12,439.13 |
| energy_cost_reduction_usd | 1,701.03 | 1,741.48 |
| total_annual_savings_usd | **78,073.65** | **79,930.32** |
| roi_pct | **85.89** | **90.31** |

`RESULTS.md:167` and `DEPLOYMENT.md:90` quote the old pair (~$78k, 86%) next to the current 7.9%
headline, so the two are already inconsistent in the prose. Note separately that the quantity being
monetised is a wait-time percentage while the project's own benchmark shows utilisation and
completions are **identical in all 40 runs** — the ROI study's disposition is a Phase C question,
not a Phase B one.

---

## 5. Non-determinism by construction

Three committed CSVs contain wall-clock measurements that cannot reproduce on any platform, and no
document acknowledges it:

| file | columns |
|---|---|
| `05_results/model_comparison_table1.csv` | `training_time_sec` |
| `05_results/scaling/scaling_analysis.csv` | `scheduler_overhead_sec`, `inference_time_sec`, `proactive_total_sim_time_sec` |
| `phases_22_30/phase_26_scaling/scaling_benchmark.csv` | `inference_latency_ms`, `throughput_overhead_pct` |

This makes `README_REPRODUCIBILITY.md:7` ("two runs produce identical datasets and results") false
as written, and it is the direct cause of the scaling defect: `scaling_law_fit.txt` fits a power
law to four wall-clock latency points (48.05 → 9.60 → 28.40 → 19.49 ms) that are **non-monotone
with a 5x spread**, then reports the fitted exponent −0.234 as "Complexity: O(1)" and
"VERDICT: inference latency is CONSTANT regardless of cluster size", and extrapolates it to 4,096
GPUs. The classifier at `scaling_benchmark.py:206` is `"O(1)" if exponent < 0.1`, a one-sided test
that labels any negative exponent constant by fall-through. Simulation `throughput` columns are
jobs-per-timestep and are deterministic; only the six columns above are affected.

Consequence for the planned `make verify`: these six columns must be compared informationally
rather than as pass/fail, and that exemption has to be stated in the tool and in the report it
prints. Nothing else in the tree needs a tolerance beyond CSV round-trip.

---

## 6. Tests and CI

| check | result (fresh clone, clean venv, CPython 3.14.3) |
|---|---|
| `python -m pytest -q` | **135 passed**, 0 failed, 0 skipped |
| `python -m ruff check --select E9,F63,F7,F82 .` | All checks passed |

Coverage of the headline claims by the test suite is the problem, not the pass rate:

| headline claim | pinned by a test? |
|---|---|
| 45,432 instants | **no test references the count** |
| 0 violations | only on a freshly simulated toy queue (`test_degeneracy_claim.py:122`) |
| 7 of 12 features at 0.0% | partially: `test_degeneracy_claim.py:177` reads the committed CSV **only if it exists** and silently no-ops otherwise; it carries no `artefact` marker |
| TOST p and CI for size-sort ≡ XGBoost | only on hand-built synthetic inputs; no published value is pinned |
| NN ≡ SMALLEST bit-identity | **not asserted anywhere**; the suite only checks the two get different plot colours |
| 7.9% improvement | **not asserted anywhere** |
| 12-policy trace table | **not asserted**; only that every scheduler name maps to a defined policy role |

The suite pins invariants and known defects, which is valuable, but not one published number.
Two tests deliberately pin defects as current behaviour: `test_simstats.py:527` asserts that the
statistics helpers pair observations positionally without checking unit labels, and
`test_queue_policies.py:330` asserts that the priority baseline's aging term never reorders.

`.github/workflows/ci.yml` **never runs pytest.** It byte-compiles a subset of directories that
excludes `tests/` and `conftest.py`, runs ruff with four error-only rules, and has a second job
whose only dependency step carries `continue-on-error: true`, so it cannot fail. The `checks` job
installs no dependencies at all, so import-time breakage is invisible to it. Every defect-pinning
test in the repository is unexecuted by CI.

Python version targets disagree across four places: `requirements.txt` says ">= 3.10, verified on
3.14", the `Dockerfile` pins 3.11, CI matrixes 3.11 and 3.12, and `pyproject.toml` has no
`requires-python` at all (deliberately: the numbered directories are not importable packages).
Neither 3.10 nor 3.14 is exercised anywhere.

---

## 7. Claim inventory summary

910 numeric claims checked. **181 are STALE, ORPHAN or UNREPRODUCIBLE** — far past the ~15 the
brief set as the prune-or-repair threshold. Full table in `reports/claim_inventory.md`.

| source | rows | MATCH | STALE | ORPHAN | UNREPRO | HIST | NOT-CLAIMED |
|---|---|---|---|---|---|---|---|
| README.md + CHANGELOG v3.5 | 72 | 58 | 1 | 9 | 0 | 4 | 0 |
| RESULTS.md | 162 | 132 | 8 | 17 | 1 | 1 | 3 |
| METHODOLOGY / DEPLOYMENT / CITATION / README_REPRODUCIBILITY / CONTRIBUTING | 93 | 56 | 16 | 10 | 4 | 2 | 5 |
| manuscript.tex | 219 | 212 | 3 | 2 | 1 | 1 | 0 |
| phases_22_30 prose + txt | 157 | 124 | 13 | 11 | 6 | 1 | 2 |
| docs/*.html | 207 | 116 | 43 | 30 | 6 | 10 | 2 |
| **total** | **910** | **698** | **84** | **79** | **18** | **19** | **12** |

The 181 are not 181 independent errors. They cluster into six causes:

1. **Two frozen documentation pages** (`docs/project_report.html`, `docs/research_progress.html`)
   account for 79 rows, 43 of them STALE. Both are hand-maintained, both are frozen at v3.2, and
   neither contains the strings `45,432`, `7.9%`, `15.95`, `degenerac`, `TOST` or `size sort`. The
   project's entire headline result is absent from them.
2. **Six studies whose producers are in no pipeline script** account for 48 ORPHAN rows. The
   numbers are correct; nothing regenerates them.
3. **One filename holding two different experiments** (`fairness_metrics.csv` exists in both
   `05_results/fairness/` and `phases_22_30/phase_27_fairness/`) explains every 125-vs-123 and
   87-vs-88 split in the prose.
4. **Two artifacts not regenerated since v3.5** (`estimate_sensitivity_summary.csv`, the ROI CSV).
5. **One hand-written document that is almost entirely fabricated**
   (`phase_27_fairness/fairness_formal_analysis.md`).
6. Roughly 28 genuine one-off prose errors.

---

## 8. Findings the brief did not list

Treated with the same seriousness as the listed defects.

**A headline-adjacent claim is stated more strongly than the artifact supports.** README.md:158,
RESULTS.md:42, manuscript.tex:340 and CHANGELOG.md:74 all state that *in 18–27% of instants every
queued job receives the same score, so the policy silently is FCFS*. The supporting column is
`pct_order_identical_to_arrival`, which counts instants where the model's dispatch order equals
arrival order under the tie-breaks `(score, arrival, id)`. All-scores-tied implies that, but not
conversely, so the column is an **upper bound** on the all-ties fraction, not a measurement of it.
`ranking_degeneracy.py` records `n_distinct_pred` per instant and could count the all-ties case
directly, but does not. The honest options are to add the counter and report the measured value, or
to reword all four places to say what the column measures.

**The manuscript quotes one p-value unadjusted while insisting on Holm elsewhere.**
`manuscript.tex:514` reports the cost of real user estimates to EASY on LANL as "+74%, p = 0.025".
That is the raw `ttest_p`; the Holm-adjusted value in the same CSV row is **0.2347**, which is not
significant. The paper requires Holm adjustment at lines 284, 540 and 561.

**Two different 95% intervals over the same 40 runs are both called "bootstrap"**, and a third has
no producer at all. The Student-t interval [4.88%, 10.91%] from `benchmark_statistical.py:141` is
labelled bootstrap in README.md:151, RESULTS.md:154, METHODOLOGY.md:43, DEPLOYMENT.md:53 and
`docs/explanation.html:848`. The only genuine percentile bootstrap in the repository,
[4.91%, 10.67%] from `phase_22_stats/stats_bootstrap.py`, is quoted in exactly one place and by no
HTML page. A third interval, [4.95%, 10.34%], appears five times across the two frozen HTML pages
and in the CHANGELOG's historical v3.1 block, and **no script in the repository produces it**.

**`docs/explanation.html` contradicts itself in three places.** Its scaling chart hard-codes 14.4
at line 1423 while its own caption at line 931 says 14.5; it states the retracted "+45% mean wait"
reservation price at line 1263 while retracting it at line 881; and it says "33-of-40 runs" at line
791 while its own legend at line 847 says 34. It also describes the pipeline as having 12 steps and
omits the ranking-degeneracy, trace-driven and estimate-sensitivity steps from its flow diagram,
and is still stamped v3.4.

**`phase_27_fairness/fairness_formal_analysis.md` is fabricated, not merely stale.** All 26 values
in its two example tables were tested against every numeric cell of every CSV in the repository.
**Exactly one — 18.27 — appears as the quantity claimed.** Its table headers name columns that do
not exist in the artifacts. Its "~3.7% Gini improvement" matches nothing: the per-job Gini is 51%
*worse* under Proactive, and the project's own `novelty_claim.txt:109` says so in plain text. Its
SLA-2 definition ("wait < 200 timesteps for 99% of jobs") was never implemented; `grep 200` in the
producing script returns nothing.

**A retraction was recorded in one file and never propagated.** The claim "proactive ≥ FIFO on
fairness" is withdrawn at `COMPLETION_SUMMARY.md:72` — the only "withdraw" hit in the repository —
but both documents that assert it (`PHASES_ROADMAP.md:17` and `fairness_formal_analysis.md:4,38`,
with a proof sketch) still carry it unchanged.

**`phase_25_real_traces/trace_preprocessing.py` synthesises its target and labels it real** in two
places, not one: `_map_lanl` (lines 302–316) and `_map_alibaba` (lines 363–374) both build the
wait-time target from a hard-coded linear formula while the candidate entries tag
`source_type="real"`. `estimate_cross_trace_mae` (lines 612–640) then scores a heuristic whose
coefficients are near-identical to the generator's, making the reported error circular. The
resulting `cross_trace_mae.csv` holds a single row for `synthetic_proxy_trace` with `mape_pct`
exactly 200.0, a saturated placeholder. The Alibaba trace the roadmap promises has never existed in
the repository; only scaffolding for it does.

**`COMPLETION_SUMMARY.md:41` carries four numbers from v3.1** (Proactive 15.71, FIFO 16.65, Gini
0.518, max wait 53.9) that match no current artifact; `git log -S` traces them to commit 69c733d.

**Two prose bounds are false as written rather than merely rounded.** "Utilisation stays > 99.7% at
all scales" — the minimum is 99.694. "Batched inference 10–48 ms" — the range is 9.60–48.05, so
both endpoints fall outside the stated interval.

**`novelty_claim.txt:16` says "15-run multi-scheduler benchmark"** while every number beneath it
comes from 20 runs. The string is a literal inside the producing script.

**`DEPLOYMENT.md` describes drift detection with numbers that appear nowhere in the code.** It
states a trigger of "> 1.5x training-holdout MAE" and a "rolling window of 50–100 jobs"; the
implementation uses `np.std(y_train) * 0.55` with a rolling window of 80 for the CSV and 30 for the
trigger test.

**`METHODOLOGY.md:42` says "20-run 13-scheduler benchmark"** while the artifact has 14 schedulers,
contradicting `METHODOLOGY.md:32` eleven lines earlier.

**`README.md` still identifies itself as v3.4** in its headline, key-features list, honest-summary
section and BibTeX block, while `CHANGELOG.md` and `CITATION.cff` are at v3.5. Its "14 schedulers"
list at line 86 enumerates 15 policies.

**`README.md:28,97` and `CITATION.cff:25` describe all 45,432 instants as "real".** 3,646 of them
are synthetic; 41,786 are real. The claim should name the split.

**`05_results/schedulers/fairness_metrics.csv` is byte-identical to
`05_results/fairness/fairness_metrics.csv`** — one script writes the same table to two tracked
paths.

---

## 9. [GAP] — things this phase could not settle

- **`04_scheduler/scaling_analysis.py:162` derives its seed as `4000 + run_id + nodes`, a sum, so
  distinct configurations collide** (run 8 with 4 nodes uses the same seed as run 4 with 8 nodes).
  This is inside a protected seed family, so per the brief it is reported and not changed.
- **Cross-platform reproduction is unproven.** Everything in section 4 was verified on Windows with
  CPython 3.14.3. Whether a Linux runner reproduces the same CSVs bit-for-bit is unknown until the
  planned scheduled CI job runs once. Scheduling is discrete, so a last-ulp difference in an
  xgboost prediction can flip a queue order and move a published mean by percent, not by 1e-9.
- **Determinism is proven only for this platform.** 153 of 164 artifacts reproduced identically on
  Windows / CPython 3.14.3, but a second run on the same machine was not performed, so
  same-machine repeat-determinism is inferred from seed hygiene rather than measured. The six
  wall-clock columns are known non-deterministic and were observed drifting by up to 84%.
- **`docs/research_progress.html` holds two claims that exist nowhere else and cannot be
  reproduced**: Phase 03 classifier ROC-AUCs (LogReg 0.87 / RF 0.91 / XGBoost 0.93, line 204) and a
  Phase 04 leaky-model MAE 5.25 / R² 0.833 (line 225). Their producer prints them and writes no
  artifact. Recorded here as `[UNVERIFIED]` so the values survive the file's deletion.
- **`docs/explanation.html:1772` embeds a 40-tree distilled model** (`var MODEL`, holdout MAE 4.64 /
  R² 0.842) that powers the page's interactive demo. No script in the repository produces it. It is
  unreproducible, but deleting it would break a working feature of the documentation rather than
  remove a false claim, so the disposition is a judgement call rather than a mechanical one.

---

## 10. Decision list for Phase B

The brief asks for a prune-or-repair decision when the STALE/ORPHAN count passes about fifteen. It
is 181. The dispositions below follow the policy already agreed for this pass: wire in an orphaned
study when current prose cites it, delete it when nothing does; weaken any claim the evidence does
not carry; never edit a number in prose to match a CSV.

### Regenerate and propagate (numbers move, all verified above)

| artefact | what moves |
|---|---|
| `estimate_sensitivity{,_summary}.csv` | PROACTIVE 16.1036 → 15.9477 and its comparison statistics |
| `roi/cost_benefit_analysis.csv` | $78,073.65 → $79,930.32; ROI 85.89% → 90.31% |

### Wire into the pipeline (prose cites them)

`fairness_budget_sweep.py`, `uncertainty_scheduler_benchmark.py`, `train_quantile_model.py`,
`build_real_trace_datasets.py` + `real_trace_validation.py`, and the six `phases_22_30` phase
scripts, folded into the single entry point so `run_all_experiments.sh` really does regenerate
every result METHODOLOGY.md:52 claims it regenerates.

### Delete (no current prose cites them; nothing is lost that is both unique and true)

`sensitivity_analysis.py`, `tune_xgboost.py`, `generate_load_profiles.py`, `benchmark_and_plot.py`
and their CSVs; the pre-v2 legacy datasets and models; `cross_trace_mae.csv`;
`phase_27_fairness/fairness_formal_analysis.md`; `docs/project_report.html` and
`docs/research_progress.html`. The two unique unverifiable numbers in the last of these are
preserved as `[UNVERIFIED]` in section 9 above before the file goes.

### Weaken or correct in place

The all-ties overstatement (four files), the unadjusted p-value at manuscript.tex:514, the
"bootstrap" CI label (six files), the `>99.7%` and `10–48 ms` bounds, "13-scheduler",
"15-run", the drift-detection numbers in DEPLOYMENT.md, the v3.4 version stamps, the
"45,432 real instants" wording, the `PRIORITY` baseline label, and the two starvation definitions.

### Structural repairs

One name per experiment (rename the phase-27 fairness table), one interpreter (`PY` override),
gzip fallback in every SWF reader, a hard failure when a trace is missing, golden tests for every
headline number, and a CI that actually runs them.

### What Phase B will not touch

The seed families, including the `4000 + run_id + nodes` collision, and the degeneracy result
itself — which reproduced exactly and needs no repair.

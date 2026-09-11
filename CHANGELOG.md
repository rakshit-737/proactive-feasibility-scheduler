# Changelog

## v3.7 — September 2026 · Hostile-review hardening, part two: honest evaluation and the condition made measurable

v3.6 made every number in this repository regenerable. v3.7 asks a harder question of the
same numbers — whether the *method* behind them is sound — and then pushes the science
past the negative result. Full accounts: `reports/phase_C_report.md`,
`reports/phase_D_report.md`, `reports/honest_claims.md`,
`reports/submission_readiness.md`.

**The central result did not move.** 45,432 dispatch instants, zero equal-size /
different-score violations, still byte-identical on a fresh clone. Everything below either
weakens a supporting claim, deletes one, or adds new evidence on either side of the main
one.

### Retraction: the headline accuracy figure was optimistic

- **R² 0.8368, MAE 4.69 → R² 0.8109 ± 0.0207, MAE 4.90 ± 0.45.** *Reason:* the 2200
  training rows are 20 simulation runs, and every trainer used
  `train_test_split(random_state=42)`. A random row split puts rows of the same run on
  both sides of the split, so the model is scored partly on runs it has seen. The new
  figure is `GroupKFold` over `run_id` (5 folds), which is the honest unit.
- **Deployment reading: 4.69 MAE → 7.24 MAE (R² 0.7251).** *Reason:* deployment means
  predicting the later part of a run from its earlier part. A within-run chronological
  split measures that and nothing else did. This is the number a practitioner should plan
  against.
- The random-split figures are retained in `05_results/splits/split_comparison.csv` as a
  labelled anchor, never as the headline.
- **The degeneracy result does not depend on any of these.** It is a statement about the
  functional form of the score, not its accuracy — which the Phase D learning-to-rank
  attack then demonstrates empirically.

### Retraction: the ablation reported twelve effects where the data supports three

- **12 point estimates from one split → 3 of 12 distinguishable from zero**
  (`job_gpu` 0.205, `queue_length` 0.022, `queue_pressure` 0.016; 20 leave-one-run-out
  folds with confidence intervals). Baseline R² restated as **0.7935 [0.7675, 0.8195]**.
  *Reason:* a single-split point estimate cannot separate a real effect from fold noise.
  An interval spanning zero means *not distinguishable at this sample size* — it is not
  evidence the feature adds nothing, and is not written as though it were.

### Retraction: the SHAP analysis was explaining memorised rows

- **400 rows sampled from the full dataset (≈320 of them in the fit) → 400 held-out rows,
  0 from training.** *Reason:* explaining rows the model was fitted on measures
  memorisation, not behaviour.

### Retraction: the LANL equivalence row said "not equivalent"; it is inconclusive

- **"not equivalent" → inconclusive, and unsettleable on that trace.** *Reason:* achieved
  power was **0.47%**, and because the observed paired difference (320.02 s) exceeds the
  equivalence margin (222.93 s), **no sample size certifies equivalence there** at a 10%
  margin. Establishing a *difference* instead would need 29 windows (50 after Holm over
  the family of 11); the trace supplies 28. The equivalence claim rests on SDSC (power
  1.000) and the synthetic benchmark. Reported in
  `05_results/traces/trace_equivalence_power.csv`.

### Retraction: the OOD failure taxonomy was a constant

- **All 72 scenarios labelled `DISTRIBUTION_MISMATCH` → 5 categories across 4 severity
  bands.** *Reason:* the classifier's thresholds were unreachable (minimum observed MAPE
  54.01 against a 35 cut-off), so every row fell through to one label. Any claim resting on
  the old labels is void. Severity is standardised *within* the grid: "least severe" means
  least severe among these 72 shifted regimes, never "safe" — mean R² across the grid is
  negative.

### Correction: two features are the same variable twice

- `total_free` and `avg_free_per_node` correlate at **1.000000** with infinite VIF;
  `fragmentation` and `variance_free` at 0.950720. *Consequence:* per-feature importance
  over this set is not identifiable, and the ablation is read accordingly. The feature set
  is unchanged, because changing it would change the artefacts the claim rests on — the
  collinearity is reported, not silently fixed.

### Deletion: the ROI study

- `05_results/roi_analysis.py`, `05_results/roi/` and every dollar figure (~$80k/yr,
  ~90% return) are **deleted**, not caveated. *Reason:* the study monetised GPU-hours saved
  while the benchmark it reads records **identical utilisation and identical completions in
  all 40 paired runs**. The quantity being monetised was measured at zero. That is a
  category error, not an uncertain assumption.

### Not a retraction: the censoring bias runs the other way

Selection from unfinished jobs exists in **1 of 5 scenarios**, and there it **penalised**
the learned policies by up to 3.3 points rather than flattering them. The common-set
re-analysis adjudicates ~92 shared jobs; ~61 exchanged jobs are described, not adjudicated.

### New: the non-degeneracy condition, measured (Phase D1)

The condition was stated but never shown. Five per-job attributes that do not factor
through requested size given the cluster state — requested time, causal per-user mean wait,
causal per-user mean runtime, queue identity, user identity — each break the degeneracy on
both traces:

| quantity | SDSC baseline → augmented | LANL baseline → augmented |
|---|---|---|
| violations | 0 → 6,194–10,718 | 0 → 13,801–28,497 |
| Kendall τ vs size | 0.752 → 0.593–0.750 | 0.621 → 0.420–0.597 |
| all-tied fraction | 14.41% → 6.31–9.48% | 20.73% → 6.01–13.48% |

**And 0 of 12 augmented variants beat shortest-job-first on the users' own estimates.** The
best is +1.48% slower and is certified TOST-*equivalent* to it. Non-degeneracy is
**necessary and not sufficient** — that is the finding, and
`tests/test_golden_numbers.py` fails if a future change makes one of these variants win, so
the claim is restated deliberately rather than drifting. User history is built causally
(a job enters its user's running means only once `submit + wait + runtime ≤ t`); three
leakage tests go red under a full-trace groupby.

### New: the degeneracy stated as a proposition (Phase D2)

`METHODOLOGY.md` and the manuscript now carry Proposition 1 with a three-line proof,
Corollaries 1.1–1.3, and Proposition 2 for the converse with three limits.

- **Correction found by writing it down:** the methodology said the ranking is "a
  permutation of the size order". Too strong, and contradicted by this repository's own
  measurement — the recovered size-to-priority table is monotone in only 57–63% of
  instants. The score is a **function** of size, not an **increasing** function of it
  (Corollary 1.3). This is precisely why equivalence to a size sort must be established
  statistically by TOST rather than deduced.

### New: the claim attacked four ways (Phase D4)

| attack | violations | verdict |
|---|---|---|
| A1 learning-to-rank (`rank:pairwise`, `rank:ndcg`) | 0 | survives |
| A2 monotone transforms (log1p, sigmoid) | 0 | survives |
| A3 history window (5 lagged cluster states) | 0 | survives |
| **A4 score at enqueue time, cached** | **20,482 / 37,741** | **breaks it** |

**Scope limitation, now stated in every claim:** the degeneracy holds for a score computed
**at the dispatch instant**. Caching a score computed at enqueue time scores different jobs
against different cluster states, so the shared-state premise fails by construction. It
does not rescue the approach — on LANL it is +11.7% mean wait and 27% worse bounded
slowdown, and on both traces it is behind plain SJF on user estimates.

### `[GAP]` — stated, not filled

- **No modern GPU trace.** External validity still rests on two machines from the 1990s.
  `reports/d3_modern_trace_gap.md` records what a candidate must supply, the obstacle per
  candidate, the procedure for adding one, and the prediction. This is the largest
  remaining weakness and `reports/submission_readiness.md` names it as the one real
  blocker. It was documented rather than filled with a substitute.
- **Cross-platform reproduction is unproven.** Every artefact was generated on Windows with
  Python 3.14.3; the weekly Linux full-verify job has not yet reported.

### Tests, pipeline, tooling

- `run_all_experiments.sh` is now **24 steps** and remains the single entry point; the two
  Phase D experiments are wired in as steps 12 and 13.
- **399 tests** (135 at the start of v3.6), including `tests/test_golden_numbers.py`, which
  pins every headline number to its committed artefact and **asserts rather than skips**
  when a committed file is absent. Each round of repairs was mutation-tested: the defect
  was reintroduced and the test had to go red before the fix was accepted. This was not
  ceremony — the first audit round produced four "guard" tests that passed equally well
  against the broken code.
- `tools/verify_artifacts.py` declares six wall-clock columns across three files as
  non-deterministic **by name**, so they are exempted explicitly rather than tolerated
  silently. They must never be quoted as results.
- `04_scheduler/simstats.py` now pairs observations by unit **label** rather than row
  position, and raises on mismatched or duplicated labels. No published number moved: the
  label sets were equal and duplicate-free, so the sorted sequences were already identical.
- `make paper` and `make paper-check` build the manuscript reproducibly; `paper-check`
  fails on any undefined reference or citation.


## v3.6 — September 2026 · Hostile-review hardening: integrity, reproducibility, and one retraction

A full adversarial audit of every numeric claim in the repository. 910 claims were
inventoried against the artefact that should hold them and the script that should
produce it (`reports/claim_inventory.md`); 181 were stale, orphaned or unreproducible.
The central result survived unchanged and is now, for the first time, defended by tests.

### The headline is unchanged and reproduces on a fresh clone

A `git clone` into a clean virtualenv on the pinned interpreter reproduces
**45,432 dispatch instants with zero equal-size/different-score violations**,
byte-identically. So do the TOST equivalence, the 7.9% improvement, the 12-policy
trace table and the 7-of-12 constant features. Of 164 tracked artefacts, 153
reproduced identically; every difference is explained below.

### Retraction: the all-ties claim was overstated

README, RESULTS.md, the manuscript and this changelog all stated that
*"in 18–27% of instants **all scores tie**, so the policy silently is FCFS"*.

That 18–27% is `pct_order_identical_to_arrival` — the fraction of instants where the
induced **order** equals plain arrival order. All-tied implies that; the converse does
not hold, so the column only **upper-bounds** the claim. The quantity is now measured
directly, in a new `pct_all_scores_tied` column of
`05_results/degeneracy/ranking_degeneracy.csv`:

| setting | order = arrival order | all scores tied |
|---|---|---|
| synthetic | 19.72% | **17.11%** |
| SDSC SP2 | 18.11% | **14.41%** |
| LANL CM-5 | 27.47% | **20.73%** |

**old:** "all scores tie in 18–27% of instants" · **new:** "the learned order coincides
with plain arrival order in 18–27% of instants, and every queued job receives an
identical score in 14–21%". Both facts support the degeneracy argument; only the
conflation was wrong. Neither number changes the equivalence result.

### Numbers that moved, and why

| quantity | old | new | cause |
|---|---|---|---|
| ROI annual savings | $78,073.65 | **$79,930.32** | the committed table was computed from the superseded 7.7144% while the prose beside it quoted 7.9%; the producer reads the improvement at run time |
| ROI first-year return | 85.89% | **90.31%** | same |
| `estimate_sensitivity` PROACTIVE mean wait | 16.1036 | **15.9477** | never regenerated after the v3.5 train/serve fix. Every other scheduler's own mean wait is unchanged |
| proactive max wait (fairness) | 125 / 124.8 | **122.65** | the phase-27 copy of the fairness table was stale, not different — see below |
| anti-starvation max wait | 87 / 87.15 | **88.15** | same |
| proactive per-job Gini | 0.80 | **0.79363** | same |
| contended-cluster advantage | 14.4% | **14.5%** (14.5219) | rounding error carried across four documents |
| utilisation at all scales | ">99.7%" | **above 99.69%** (minimum 99.694) | the stated bound was false as written |
| batched inference latency | "10–48 ms" | the committed run records **10.24–15.49 ms** | wall-clock and machine-dependent; the quoted range did not match the artefact even when written (that run recorded 9.60–48.05 ms) |
| proxy-trace transfer R² | 0.015 | **0.0321** (MAE 14.98 → 13.53) | the committed proxy was a v3.1 artefact the pipeline could not overwrite |
| phase-24 header | "15-run benchmark" | **20-run** | the producer's own literal contradicted its numbers; every scheduler has exactly 20 runs |

### Withdrawn claims

- **The scaling law.** `scaling_law_fit.txt` reported `Complexity: O(1)` and
  *"VERDICT: inference latency is CONSTANT regardless of cluster size"* from a fitted
  exponent of **−0.234**, then projected it to 512, 1024 and 4096 GPUs. Three independent
  defects: the classifier's test was `exponent < 0.1`, one-sided, so a strongly
  *negative* exponent was labelled constant by fall-through; the four latencies it was
  fitted to were **non-monotone** (48.05, 9.60, 28.40, 19.49 ms — a 5× spread); and
  `inference_latency_ms` is a wall-clock column. How wall-clock: the *current* committed
  run of the same benchmark records 15.18, 15.49, 12.57, 10.24 ms for those same four
  cluster sizes, and the Phase A fresh-clone run measured the related timing columns in
  `scaling_analysis.csv` moving by up to **84%** while every non-timing column stayed
  bit-identical (`reports/phase_A_report.md` §5). A "law" whose inputs move that much
  between runs of identical code is fitting the machine, not the algorithm. Four noisy
  points cannot identify a complexity class. The verdict, the complexity label and all
  three projections are withdrawn; the file is renamed `scaling_measurements.txt` and
  reports what was measured. What survives: scheduling overhead stayed under 5% of
  throughput at every measured scale.
- **"All headline claims carry bootstrap CIs."** The interval [4.9%, 10.9%] is a
  **Student-t** interval (`stats.t.ppf` in `benchmark_statistical.py`), not a bootstrap.
  It was called a bootstrap CI in six places. The repository's one genuine percentile
  bootstrap is `phase_22_stats/stats_bootstrap.py`, which gives **[4.9%, 10.7%]** over the
  same 40 runs, and was quoted in exactly one place. Both are now named.
- **"Priority + aging."** `priority_scheduler.py`'s key expands to
  `(priority_score + 0.03·arrival_time) − 0.03·current_time`. The `current_time` term is a
  common additive shift at any instant, so it cancels from every pairwise comparison: the
  order is **time-invariant** and a queued job can never overtake by waiting. The formula
  is unchanged and the mean wait is still 16.5318 — only the false label goes. The
  scheduler key is `STATIC_PRIORITY` everywhere. HRRN is the genuinely aging baseline.
- **"(n.s.)" on the predicted-wait backfill hybrid.** RESULTS.md reported the hybrid as
  tying plain EASY "(19.20 vs 19.25, n.s.)". No paired test between `PROACTIVE_BF` and
  `BACKFILL` exists anywhere: `multi_scheduler_significance.csv` uses only PROACTIVE and
  FIFO as references. The two means stand; the untested significance claim is withdrawn.
- **"45,432 real dispatch instants."** 3,646 of them are synthetic. The split is now
  stated wherever the total is (41,786 real + 3,646 synthetic), including `CITATION.cff`.
- **A "30-phase research pipeline"** (`CITATION.cff`). The directories stop at phase 28.

### Deleted, because nothing regenerated them and no current claim cites them

`phase_27_fairness/fairness_formal_analysis.md` — hand-written, and **fabricated rather
than merely stale**: all 26 values in its two example tables were tested against every
numeric cell of every CSV in the repository and **exactly one (18.27) appears as the
quantity claimed**. Its table headers name columns that do not exist. Its "~3.7% Gini
improvement" matches nothing — the per-job Gini is 51% *worse* under Proactive, which the
project's own `novelty_claim.txt` states in plain text. Its SLA-2 definition ("wait < 200
timesteps for 99% of jobs") was never implemented.

`phase_25_real_traces/cross_trace_mae.csv` and the code path behind it — `_map_lanl` and
`_map_alibaba` **synthesised the wait-time target from a hard-coded linear formula while
their trace candidates were tagged `source_type="real"`**, and `estimate_cross_trace_mae`
then scored a heuristic whose coefficients nearly matched that generator's, so the
reported error was circular. No file those mappers read has ever existed in this
repository. The CSV's only row was the synthetic proxy with `mape_pct` pinned at the 200.0
saturation value. No Alibaba trace has ever shipped here; only scaffolding did.

`docs/project_report.html` and `docs/research_progress.html` — hand-maintained, frozen at
v3.2, 41% and 30% stale respectively, and containing no mention of `45,432`, `7.9%`,
`degenerac`, `TOST` or the equivalence result at all. Their two genuinely unique numbers
(phase-03 classifier ROC-AUCs, and a phase-04 leaky-model MAE) have no producer and no
artefact, and are preserved as `[UNVERIFIED]` in `reports/phase_A_report.md`.

Also deleted: `sensitivity_analysis.py`, `tune_xgboost.py`, `generate_load_profiles.py`,
`benchmark_and_plot.py` and their CSVs, and the pre-v2 legacy datasets, models and
trainers. None was in any pipeline and no current prose cites them.

### Reproducibility

- **One entry point.** `run_all_experiments.sh` is a single 20-step run that now includes
  the six phases 22–27 scripts and the five studies no pipeline regenerated (quantile
  model, fairness budget sweep, real-trace dataset build and validation, uncertainty
  benchmark, phase-01 simulation). `METHODOLOGY.md`'s claim that it "regenerates every
  result" was false when written and is now true. `run_all_experiments_v2.sh` forwards here.
- **`tools/verify_artifacts.py`** re-runs the pipeline in a scratch copy and diffs every
  tracked artefact against its committed blob — CSVs cell by cell, text line by line, PNGs
  and pickles by existence and loadability. Six wall-clock columns in three files are
  reported as `TIMING` and never as failures, and the report prints that exemption rather
  than hiding it. `make verify`, `--quick`, `--smoke`.
- **The interpreter is no longer ambiguous.** `PY` is overridable, because a bare
  `python3` on Windows resolves to the Microsoft Store shim rather than to an activated
  virtualenv — so a caller who believed they were testing a pinned environment may not have
  been. Python **3.14** is named in `requirements.txt`, `requirements-dev.txt`, the
  `Dockerfile` and CI, and `tests/test_python_pin.py` fails if those four drift apart.
- **A missing trace is a hard failure.** `ranking_degeneracy.py` used to catch
  `FileNotFoundError` per trace, print `skipping`, write the summary anyway and exit 0 — so
  the published 45,432 could silently collapse to 3,646. It now exits non-zero unless
  `--allow-partial` is passed, and always writes `ranking_degeneracy_totals.csv` recording
  settings expected and present, missing traces, the runs and windows actually used against
  the published protocol, and whether the run was partial. The figure captions say so too,
  because a finished PNG that looks exactly like the real one is the dangerous output.
- **One gzip-aware SWF reader** (`02_data/swf_io.py`). `build_real_trace_datasets.py` opened
  the plain `.swf` name with a bare `open()` while only `.swf.gz` is committed, so it could
  not run on a fresh clone at all. The unvalidated cached-feature branch in
  `trace_driven_benchmark.py` is deleted: a stale gitignored file could silently change
  published numbers.
- **Honest naming.** Pipeline step 7 was labelled "Real trace loading" and loaded no real
  data on any machine — it looked for `02_data/lanl_trace_sample.swf`, a filename that has
  never existed here, so the synthetic fallback fired on every run and then overwrote a
  tracked file. `lanl_trace_sample.csv` → `synthetic_proxy_lanl_schema_trace.csv`;
  `lanl_validation_results.csv` → `synthetic_proxy_validation_results.csv`.

### Statistics and correctness

- **Paired statistics now pair by label.** `simstats.equivalence_table` and
  `pairwise_significance` sorted each scheduler's rows by the unit column and then paired
  them **by position**, never checking that the two schedulers carried the same run or
  window labels. Two schedulers with equal counts but disjoint labels received a confident
  paired p-value for a comparison that was never paired; a length mismatch made one
  function drop the pair silently and the other die inside numpy. They now align by label
  and raise on differing sets, duplicated labels, a missing unit column or an absent
  reference. **No published number moves**: for equal, duplicate-free label sets the sorted
  label sequence is identical for both schedulers, so reindexing reproduces the old float
  order exactly — verified by regenerating all four affected CSVs.
- **One definition of starvation.** `fairness_budget_sweep.py` counted a job starved when
  its wait exceeded 3× the run's **mean wait**, while `fairness_analysis.py` and phase 27's
  SLA-2 use 3× the job's **own runtime**. Standardised on the per-job rule. Its column is
  renamed `starved_jobs_wait_gt_3x_own_runtime` so the definition travels with the data.
  **The trend inverts** under the corrected rule — a regenerated sweep whose starvation
  count falls as budgets loosen is correct, not a regression. Every other metric in that
  file is computed from the same values as before.
- **The fairness fork was staleness, not disagreement.**
  `phase_27_fairness/fairness_metrics.csv` disagreed with `05_results/fairness/` for
  several releases (124.8 vs 122.65, 87.15 vs 88.15). Its producer has always iterated
  every row of the benchmark; the committed copy predated that benchmark reaching 14
  schedulers and was never regenerated because phase 27 ran only from a script nobody
  invoked. Regenerated, it has 15 rows and **agrees exactly**. A golden test asserts it.
- **An unrecognised scheduler key is a hard error** in phase 24, instead of a row labelled
  `unknown` flowing into the published comparison table. A rename is precisely how that
  happens, and the degraded row is the one artefact that would not reveal it.
- **`02_data/dataset.csv` never matched its own generator.** The generator is
  deterministic; the file simply predated a change and nothing regenerated it. It is now
  step 2 of the pipeline.

### Tests and CI

- **`tests/test_golden_numbers.py`** — 26 tests pinning every headline number to its
  committed artefact. Unlike the rest of the suite these **fail rather than skip** when an
  input is missing, because every file they read is committed: absence means a broken
  checkout, not an un-run pipeline.
- The suite grows from **135 to over 230 tests**. Two tests that pinned known defects as
  intended behaviour (positional pairing in `simstats`, the inert aging term) now pin the
  corrected contract instead.
- Every repair in this release was **adversarially reviewed and mutation-tested**: the
  defect was reintroduced and the guarding test had to go red. That process found four
  tests which passed equally well against the old broken code — they looked like evidence
  and guarded nothing — and they were replaced.
- **CI actually runs the tests.** It previously byte-compiled a subset that excluded
  `tests/` and `conftest.py`, installed no dependencies in that job, and carried
  `continue-on-error: true` on the only step that touched dependencies, so it could not
  fail. It now installs the pinned set on 3.14, lints, runs `pytest`, and runs a smoke
  reproduction of the pipeline; a weekly job runs the full artefact verification.

### Known gaps, recorded rather than fixed

- `scaling_analysis.py` derives its seed as `4000 + run_id + nodes`, a **sum**, so run 8
  with 4 nodes collides with run 4 with 8 nodes. Reported and left alone: seed families are
  out of scope for this pass by design.
- Cross-platform bit-reproduction is **unproven**. Everything above was verified on Windows
  with CPython 3.14.3. Scheduling is discrete, so a last-ulp difference in one prediction
  can flip a queue order and move a published mean by percent, not by 1e-9. The weekly CI
  job is what will establish this.
- The ROI study monetises a wait-time percentage while the project's own benchmark shows
  utilisation and completions are **identical in all 40 runs**. The figures are regenerated
  and internally consistent, but the study's premise is a question for the next pass.

## v3.5 — July 2026 · Train/serve feature-skew fix and full synthetic-result regeneration

### The defect
Training rows snapshot each job's features **at arrival, before it joins the
queue** (`generate_improved_dataset.extract_features`, and likewise
`build_real_trace_datasets.replay_trace_features`: "self NOT yet in the queue").
But every *synthetic dispatch-time* feature builder scored a job while it was
**inside** `queue`, so `queue_length` was off by one and `queue_pressure`
included the job's own GPU demand — a train/serve skew. The trace-side dispatch
builder (`trace_driven_benchmark.build_feature_matrix`) already excluded the
scored job, and the manuscript/RESULTS formula
`queue_pressure = (Σ_Q − g)/(free+1)` already *described* the corrected
semantics; the synthetic serving code simply did not implement them.

### The fix
`queue_length → len(queue) − 1` and `queue_pressure → (Σ_queue − job_gpu)/(free+1)`
at dispatch time in every synthetic builder: `multi_scheduler_benchmark`,
`neural_network_scheduler`, `benchmark_statistical`, `fairness_analysis`,
`fairness_budget_sweep`, `scaling_analysis`, `uncertainty_scheduler_benchmark`,
`proactive_scheduler`, `proactive_Schedule_v2`, `05_results/benchmark_and_plot`,
and the phase-23 dispatch call site (which now passes the queue without the
scored job). Snapshot-style callers were already correct and are unchanged; no
model was retrained (training data never had the skew).

### Consequences for the results (all synthetic numbers regenerated)
- `queue_pressure` now **varies across the queue** (82.9% of synthetic instants,
  exactly as often as `job_gpu` — it is a deterministic function of size given
  the state), so the synthetic feature vector has **7**, not 8, cluster-constant
  features. The degeneracy argument is unchanged and the measured claim still
  holds: **zero equal-size/different-score violations over 45,432 instants**
  (3,646 synthetic + 11,843 SDSC + 29,943 LANL at the trace benchmark's own
  20-window protocol, now also the diagnostic's default).
- 40-run headline: **7.9%** mean-wait reduction vs FIFO (was 7.7%), t p=2.0e-06,
  CI [4.9%, 10.9%]. 14-scheduler table: PROACTIVE 15.95 ts (was 16.10),
  −7.4% vs FCFS/first-fit (Holm p=4.0e-04); NN and SMALLEST remain **bit-identical
  to each other** at 16.07, and both are TOST-equivalent to PROACTIVE
  (+0.80%, CI [+0.03, +0.23] ts inside a ±1.59 margin, p=2.6e-16).
- Every conclusion of v3.3/v3.4 survives: any runtime signal still beats
  PROACTIVE, the ranking is still a size lookup table, the trace-driven results
  are untouched (that code path never had the skew).
- Tests: `tests/` gains a suite that pins the degeneracy claim (equal-size ⇒
  equal score, 7 constant features, all-equal-size queue ⇒ FCFS, trace builder
  parity) plus backfill/queue-policy/statistics/reproducibility checks;
  `pyproject.toml` + `conftest.py` configure it.

## v3.4 — July 2026 · Ranking degeneracy, trace-driven re-evaluation, and a reframed manuscript

> **v3.5 note.** The synthetic numbers quoted below were produced under the
> serving-side feature skew fixed in v3.5 and are superseded by the regenerated
> values in `RESULTS.md`; the qualitative findings are unchanged.

v3.3 established that any runtime signal beats the proactive scheduler on mean
wait, leaving it a claimed niche in the *zero-runtime-information* regime. v3.4
tests that niche and removes it, and moves the scheduler evaluation off the
synthetic generator onto real workloads.

### The headline: the ML score cannot distinguish co-queued jobs
At a fixed dispatch instant every queued job observes the same cluster, so only
per-job features can differ — and in this feature set each of those
(`can_fit_now`, `gpu_fit_ratio`, `node_availability`, `queue_pressure`) is a
deterministic function of the requested size given the state. The predicted
score is therefore `g_S(size)` and the ranking is a permutation of the size
order; the remaining **eight** features describe only the cluster, so they shift
all scores equally and **cannot reorder anything**.

- **NEW `04_scheduler/ranking_degeneracy.py`** — instruments real dispatch
  decisions via a `RANK_OBSERVER` hook added to both benchmarks (no simulator
  duplication). Over **25,306 instants in three settings: zero counterexamples**
  — two equally-sized co-queued jobs never receive different scores. 8 of the 12
  synthetic features vary across the queue in **0.0%** of instants. A ~9-job
  queue receives only 2.3–3.1 distinct priority levels, and in 18–26% of instants
  every score ties, so the policy silently *is* FCFS. Recovers and plots the
  size→priority lookup table (monotone in 52–67% of instants).
- **NEW `04_scheduler/size_scheduler.py`** — `SMALLEST` / `SMALLEST_FIRST`: sort
  by requested size, no model. The ML-free control the feature set implies.
- **Result**: statistically **equivalent** to the XGBoost pipeline — synthetic
  −0.18%, paired TOST p=2e-16, diff CI [−0.14,+0.08] ts vs a ±1.61 margin; SDSC
  SP2 −0.05%, TOST p=1.8e-12. The **MLP baseline is bit-identical to the size
  sort on every metric of all 20 runs**. On LANL the verdict is **inconclusive**,
  not "different": the size sort is 14.4% worse on mean wait, but that does not
  survive Holm correction on the t-test (p=0.25; Wilcoxon disagrees at p=0.015),
  shrinks to +7.3% at p=0.60 on bounded slowdown, and TOST cannot certify
  equivalence either (p=0.77). On that machine the ML scheduler also fails to
  beat plain FCFS (p=0.48).
- **Retraction within this release.** A draft of this entry explained the LANL
  result by its learned size table being less monotone. That is wrong: the
  per-instant monotone fraction is 51.9% on SDSC — where equivalence *does* hold
  — against 61.5% on LANL, so monotonicity does not track where the equivalence
  fires. LANL is reported as underpowered, not mechanistically explained. The
  jagged SDSC curve in `size_priority_table.png` is likewise mostly sampling
  noise (51 distinct sizes vs 6 on LANL, 8 synthetic), not ranking instability.

### Trace-driven scheduler benchmark on real workloads
- **NEW `04_scheduler/trace_driven_benchmark.py`** — event-driven, second-exact
  replay of LANL CM-5 and SDSC SP2 through **12 policies**, 20 paired windows per
  trace (3-day warm-up not measured + 7 measured days), offered load ≈0.70,
  windows spaced evenly rather than selected by load. Capacity-only allocator
  (SWF records no per-node placement). The ML scheduler is given its best case:
  a model **retrained on the same machine's earlier data** (chronological 60%
  split, no leakage).
- Uses the **real user runtime estimates the traces contain** (SWF field 9)
  instead of the simulated f-model. Missing estimates fall back to the trace
  median, deliberately *not* the true runtime, so estimate-driven policies get no
  free oracle.
- **The synthetic gain does not replicate**: Proactive vs FCFS is −20.4% on SDSC
  (p=0.042) but **−4.5%, p=0.48 on LANL**. SJF on real user estimates beats
  Proactive by 20.2% (SDSC, Holm p=0.009) and 15.3% (LANL). `PROACTIVE_EST` —
  the model retrained *with* the estimate as a feature — still loses to plain SJF
  on both machines (+7.9% / +17.5%): the pipeline's ceiling is the sort it is
  trying to learn.

### Real estimate error vs the f-model
- Measured, not simulated: SDSC median 6.91× over-estimate with 0.1%
  under-estimates; LANL median 1.51× but **36.3% under-estimates** — which the
  over-estimate-only f-model (`est = runtime·U(1,C)`) **cannot produce at all**.
- v3.3 concluded from the f-model that EASY is "near-insensitive to estimate
  quality". On the traces real estimate error costs EASY **+6.2% on SDSC but
  +74% on LANL** (p=0.025) versus perfect estimates: under-estimates break the
  reservation guarantee, and over-estimate-only noise cannot reveal it. The
  f-model should not be the sole estimate-error model in backfill studies.

### Methodology hardening
- **Equivalence testing** (`simstats.tost_equivalence`, `equivalence_table`):
  claims of *sameness* now use paired TOST with a 10%-of-reference margin,
  reporting both one-sided p-values and the 90% CI of the difference. Without
  it, the p=0.65 size-sort comparison reads as "no significant difference" and
  gets discarded instead of recognised as the result.
- **Mean bounded slowdown** `max(turnaround/max(runtime,τ),1)` promoted to a
  first-class reported metric everywhere (it was computed but never surfaced).
- **NEW `04_scheduler/simstats.py`** — shared Gini / Holm / paired-significance /
  TOST helpers, so the synthetic and trace studies apply identical statistical
  machinery; removes the duplicated copies from `multi_scheduler_benchmark.py`.
- `hrrn_scheduler.py` gains `order_queue_estimated` (deployable HRRN on user
  estimates), mirroring `sjf_scheduler.py`.
- Synthetic benchmark now batches PROACTIVE's per-queue predictions into one
  call — **verified bit-identical** to the previous per-job loop (max change in
  `mean_wait` across all pre-existing schedulers: 0.0).
- `run_all_experiments.sh` extended to 14 steps (adds the degeneracy diagnostic
  and the trace-driven benchmark); phase 24 positioning statement regenerated
  with the SMALLEST control.

### Manuscript reframed
`phases_22_30/phase_28_manuscript/manuscript.tex` rewritten from "we propose an
ML scheduler that beats FIFO" to what the evidence supports: **"Ranking
Degeneracy: A Learned Wait-Time Model Schedules No Better Than Sorting by
Requested Size"** — an evaluation study and a negative result, with a stated
non-degeneracy condition for feature sets and two methodological
recommendations. Also fixes two long-standing manuscript defects: a stale
in-sample `R² ≈ 0.937` presented as holdout quality, and a claim that runtimes
were "sampled from real distributions (LANL, Alibaba traces)" when the synthetic
generator uses `randint(5,20)`.

### Known limitation (stated, not hidden)
Simulator fidelity against recorded waits is good on LANL (38.8 vs 33.2 min mean;
the simulator is slightly pessimistic) but permissive on SDSC (174.6 vs 630.9 min
mean, recorded **median** 19.3 min) — SDSC's mean is dominated by a tail our
capacity-only model cannot reproduce, since SWF records no placement or site
policy. SDSC results read as "a correctly-loaded 128-processor machine driven by
real distributions", not a replay of SDSC's queue. The degeneracy result does not
depend on this: it is a property of the feature vector and holds in all three
substrates.

## v3.3 — July 2026 · Classical-baseline landscape: estimated SJF, canonical EASY, conservative BF, HRRN, preemptive SRPT

Response to faculty review: "add EASY/reservation backfill, SJF with perfect or
estimated lengths, and preemption/starvation-aware policies — do they explain
the improvement?" The 20-run paired benchmark now covers **13 schedulers**
with Holm-adjusted pairwise significance (t + Wilcoxon) against both
PROACTIVE and FCFS (`05_results/schedulers/multi_scheduler_{runs,benchmark,significance}.csv`).

### New classical baselines
- **SJF-EST** — SJF on user estimates (f-model, Mu'alem & Feitelson 2001:
  est = runtime·f, f~U(1,C), C=5) and **SJF-MODAL** (estimates rounded up to a
  {10, 20}-tick menu — the rank-destroying regime of Tsafrir & Feitelson 2005).
- **Canonical EASY** — added the missing second backfill condition (a candidate
  fitting in the shadow-time surplus may start even if it outlives the shadow);
  `04_scheduler/backfill_scheduler.py` now takes `est_runtime_of` so EASY also
  runs on estimates (**EASY-EST**). Verified: 300-trial property test that a
  reserved head is never delayed.
- **Conservative backfill (CONS-BF)** — a reservation for EVERY queued job via
  a future free-capacity profile (helper property-tested against a brute-force
  oracle on 3000 random instances).
- **HRRN** — highest response ratio next (starvation-aware aging built into the
  dispatch ratio).
- **Preemptive SRPT** — shortest-remaining-processing-time with a 1-tick
  checkpoint penalty per preemption (runtime oracle; ~41.5 preemptions/run).
- **Strict FIFO** — textbook head-blocking FIFO, exposing that the historical
  'FIFO' baseline is FCFS + unrestricted first-fit dispatch (label kept, now
  documented; strict FIFO is the honest EASY reference).
- **Estimate-quality sweep** — `04_scheduler/estimate_sensitivity.py` sweeps
  f-model C ∈ {1,2,3,5,10} + modal estimates with coupled noise and per-C
  paired CIs vs PROACTIVE (`05_results/schedulers/estimate_sensitivity*.csv/png`).

### Headline findings (honest)
- **Any runtime signal beats the wait-prediction scheduler on mean wait**:
  SJF-oracle 12.34 < SJF-EST 13.32 < SRPT 14.02 < HRRN 15.55 < PROACTIVE 16.10
  ts; even modal SJF (two estimate classes) reaches 14.06. Runtime-ordered
  policies pay in tails (SJF max 146 ts / Gini 0.79; SRPT 151 / 0.81); HRRN is
  the classical all-rounder (15.55 / max 65 / Gini 0.54, beating PROACTIVE's
  tail by 2×). PROACTIVE's defensible niche: **zero-runtime-information**
  (−6.5% vs FCFS, Holm p=0.0012, no estimates required).
- **Canonical EASY re-prices the reservation guarantee**: +11.8% mean wait vs
  FCFS/first-fit (v3.2's non-canonical variant overstated it as ~45%), −26% vs
  strict FIFO, and near-insensitive to estimate quality (19.0–19.8 ts across
  perfect/f-model/modal). Hybrid predicted-wait backfill still ties plain EASY.

### Metric & statistics fixes (from a 13-agent adversarial review; 7 confirmed)
- **Unified wait = turnaround − runtime** for all schedulers: bit-identical for
  the non-preemptive ones, but closes SRPT's time-to-first-dispatch artifact
  that hid ~29% of its queueing delay (SRPT mean wait 10.66 → honest 14.02,
  which flips its ranking vs SJF).
- New metrics for every scheduler: p95 wait, mean turnaround, mean bounded
  slowdown, preemption count.
- Holm correction no longer converts NaN p-values into finite ones; Wilcoxon
  p-values now get their own Holm column instead of being reported raw next to
  adjusted t-tests; HRRN docstring no longer overclaims "cannot starve".
- Pre-existing schedulers (FIFO/FCFS, SJF, Priority, Proactive, NN) verified
  **bit-identical** to v3.2 outputs; estimate RNG uses a dedicated stream
  (seed 20000+run) so workload pairing is untouched.

---

## v3.2 — July 2026 · Research extensions: real traces, backfill, fairness budget, uncertainty

The four highest-leverage items from the v3.1 future-scope roadmap, implemented,
run at full scale, and documented. Every result below is seeded and regenerable.

### ① Real-trace validation (Phase 25 closed)
- Downloaded and committed two genuine Parallel Workloads Archive traces
  (`02_data/LANL-CM5-1994-4.1-cln.swf.gz`, 1,024 procs, 122k jobs;
  `02_data/SDSC-SP2-1998-4.2-cln.swf.gz`, 128 nodes, 54k jobs).
- New pipeline: `02_data/build_real_trace_datasets.py` (chronological replay of the
  recorded schedule reconstructs 8 honest cluster-state features per arrival) →
  `02_data/real_trace_validation.py` (transfer + retrain + baselines).
- **Zero-shot transfer of the synthetic model ≈ 0 on both traces** (best-case
  protocol with rescaled features and affine calibration).
- **Retrained on the trace: SDSC SP2 R²(log) = 0.494** (vs −0.694 median baseline)
  — half the log-wait variance of a real batch supercomputer explained; LANL CM-5
  only 0.101 (interactive machine, median wait ~4 s) — machine-dependent signal,
  reported honestly.

### ② EASY-backfill baseline (+ hybrid)
- `04_scheduler/backfill_scheduler.py` (EASY with head reservation, perfect runtime
  estimates = strongest baseline) + integration into the now-7-scheduler benchmark.
- Finding: this simulator's "FIFO" already backfills without reservations, so
  EASY's guarantee **costs ~45% mean wait but is the fairest policy in the study**
  (Gini 0.365, max wait ~55 ts) — the reservation price, measured.
- The predicted-wait backfill hybrid ties plain EASY — honest null result.

### ③ Bounded-fairness wait budget
- `04_scheduler/fairness_budget_sweep.py`: hard escalation budget B swept over
  9 values × 20 paired runs → a smooth mean-wait ↔ tail-latency Pareto frontier
  (`05_results/fairness/budget_sweep.csv` + `budget_pareto.png`).
- **B=60 keeps +7.1% of the +13.2% unbounded gain while capping max wait at 81 ts
  (vs 136) and Gini at 0.69.** B≤30 is slightly worse than FIFO (churn) — reported.

### ④ Uncertainty-aware scheduling (honest negative)
- `03_models/train_quantile_model.py` (q10/q50/q90 XGBoost; holdout q50 MAE 4.61;
  **interval coverage 68% vs 80% nominal — under-dispersed, reported as-is**) +
  `04_scheduler/uncertainty_scheduler_benchmark.py` (UCB + spread-guarded FIFO
  fallback across 5 scenarios).
- Parity with the point model everywhere; the guard fails to fire (0.2% of ticks)
  in the one regime where reordering hurts. Interval width is **not** a reliable
  OOD alarm here; the rolling-MAE drift trigger remains the deployment mechanism.

### Fixes & hygiene
- Eliminated train/eval seed leakage in three benchmarks (evaluation now uses
  out-of-training seed ranges 1000+/5000+/7000+); PROACTIVE's edge over FIFO on
  fresh seeds: 6.5% (vs 7.7% near-training).
- Fixed a unit-scale bug in the real-trace transfer mapping (raw processor counts
  were fed to a model trained on a 32-GPU world).
- All three HTML docs, README, RESULTS, DEPLOYMENT, roadmap/status docs, and the
  manuscript updated with the v3.2 results (including the negatives).

---

## v3.1 — July 2026 · Full audit, repair & regeneration

This release is the result of a complete project audit: every file was read, every
claim re-verified against the code and data, and every script executed. A pre-repair
snapshot of the project was kept locally (outside the repo).

### 🔬 Scientific-integrity fixes
- **Trace relabelled honestly** — `02_data/lanl_trace_sample.csv` was proven to be the
  loader's synthetic fallback (row-for-row reproduction), not a real LANL trace. All
  code, results, and docs now label it `synthetic_proxy_trace`; the loader falls back
  loudly and documents how to obtain real traces. Real-trace validation is tracked as
  the open Phase 25.
- **Split-before-fit** — `synthetic_vs_real_comparison.py` fit the model on all data
  *before* splitting, inflating the "holdout" to R² ≈ 0.96. Fixed; honest holdout is
  R² 0.837 / MAE 4.69 ts.
- **Fabricated results removed** — Phases 23 (OOD), 24 (scheduler landscape),
  26 (scaling), and 27 (fairness/SLA) previously reported hardcoded constants or a
  silent heuristic stand-in for the model. All four now compute from real data and
  real simulations; baselines without real implementations (SLURM/K8s/Yarn rows) were
  dropped rather than invented.
- **Manuscript corrected** — `manuscript.tex` now compiles (inputenc fix), all
  citations resolve, and every number matches the regenerated artifacts, including
  the fairness/tail-latency trade-off and the unvalidated sim-to-real transfer.
- **v1 scheduler dispatch bug** — a stray `break` limited the v1 proactive experiment
  to one job per tick; fixed (v2 semantics: dispatch everything that fits).
- **Degenerate profile fixed** — the low-contention dataset produced all-zero waits
  (trivial R² = 1.0); it now has real variance, plus a zero-variance guard.

### 🔧 Engineering fixes
- **Determinism** — dataset generators and benchmarks are seeded (global 42,
  per-run 42+i); two runs produce byte-identical outputs.
- **Windows support** — pipelines export `PYTHONUTF8=1`; all text writes specify
  UTF-8 (previously three scripts crashed on cp1252 consoles).
- **Path independence** — eight scripts that only worked from a specific working
  directory now resolve paths from their own location.
- **Pipeline order** — the 40-run statistical benchmark now runs *before* the ROI
  analysis that consumes it; the ROI parser reads the file's actual (long) format.
- **SWF parser** — off-by-one in the Standard Workload Format field mapping fixed
  (requested processors was read from the requested-time column).
- **Dependencies** — `requirements.txt` re-pinned to versions verified working on
  Python 3.14 (adds previously-missing seaborn/joblib; shap/lightgbm/catboost/
  streamlit/plotly now installable and used by the pipeline).
- **Archive hygiene** — `07_archive/` scripts carry ARCHIVED headers naming their
  successors and fail fast with clear messages instead of cryptic errors.

### 📊 Regenerated results (canonical)
- Model: holdout R² 0.837 / MAE 4.69 ts (5-fold CV 4.74 ± 0.42; tuned 4.57 / 0.850).
- Scheduler: 7.71% mean-wait reduction vs FIFO (40 seeded paired runs; bootstrap
  95% CI [4.95%, 10.34%]; paired t-test p = 1.4e-06; utilisation unchanged ≈64%).
- Trade-offs: max wait ~58 → ~125 ts; per-job Gini 0.526 → 0.796 (anti-starvation
  variant: 0.688 / 87 ts).
- OOD: mean R² −0.31 across 72 shifted scenarios → retrain per regime + FIFO fallback.
- Proxy-trace transfer: R² ≈ 0.015 (real-trace validation open).

### 📚 New & reorganised documentation
- **`docs/index.html`** (since renamed to **`docs/explanation.html`**) — complete
  interactive project documentation: every concept
  explained from scratch, 9 data charts, architecture diagrams, a **live in-browser
  simulator** running a distilled 40-tree export of the real XGBoost model
  (distilled holdout R² 0.842), per-file repository guide, audit changelog, future
  roadmap, glossary. Light/dark themes, mobile navigation, accessible table views.
- `docs/project_report.html` and `docs/research_progress.html` — moved into `docs/`
  and updated to the regenerated numbers (July 2026 banners link to the main docs).
- `DEPLOYMENT.md` — new operations runbook (rollout modes, monitoring thresholds,
  drift response, rollback).
- README / RESULTS / METHODOLOGY / phases_22_30 summaries — synced to regenerated
  artifacts; stale paths and version references fixed.
- Removed from the tree: stray `Microsoft.VisualStudio.Services.VSIXPackage`,
  `__pycache__/`, `catboost_info/` (also gitignored).

### How to publish this update to GitHub
From your existing clone of the repository:

```bash
# 1. copy the contents of this folder over your clone (or use it directly)
# 2. review what changed
git status
git diff --stat

# 3. commit and push
git add -A
git commit -m "v3.1: full audit & repair — honest regenerated results, seeded pipeline, interactive docs"
git push
```

If this folder itself should become the repo (fresh start):

```bash
git init
git add -A
git commit -m "v3.1: full audit & repair — honest regenerated results, seeded pipeline, interactive docs"
git remote add origin https://github.com/<you>/proactive-feasibility-scheduler.git
git push -u origin main --force   # only if you intend to replace history
```

Tip: enable **GitHub Pages → deploy from branch → /docs** to serve the HTML
documentation as the project site. (The entry page was `docs/index.html` at the
time of this release; it is now `docs/explanation.html`.)

---

## v3.0 — May 2026
Phases 01–21 research pipeline (simulation, features, model, schedulers,
benchmarks, explainability, ROI) plus phases 22–30 scaffolding.

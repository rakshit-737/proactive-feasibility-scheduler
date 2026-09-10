# Claim inventory

Every numeric claim that appears in prose anywhere in this repository, checked against the
artifact that should contain it and against the script that should produce that artifact.
Built in Phase A of the hostile-review hardening pass. Read-only: nothing in the repository was
modified while this was compiled.

Baseline: branch `manuscript-submission-prep`, HEAD `888b5ed`, clean working tree.

## Verdicts

| verdict | meaning |
|---|---|
| MATCH | value equals the artifact value at the stated precision, and a committed script regenerates that artifact |
| STALE | the artifact exists but holds a different number, or the prose contradicts the artifact at the stated precision |
| ORPHAN | the value matches an artifact, but no committed pipeline script produces that artifact |
| UNREPRODUCIBLE | no artifact anywhere contains the value (console output or prose only) |
| HISTORICAL | presented in place as a superseded earlier-version number; correctly labelled |
| NOT-CLAIMED | a definition or configuration constant, not asserted as a research result |

## Totals

| source | rows | MATCH | STALE | ORPHAN | UNREPRO | HIST | NOT-CLAIMED |
|---|---|---|---|---|---|---|---|
| README.md + CHANGELOG.md (v3.5) | 72 | 58 | 1 | 9 | 0 | 4 | 0 |
| RESULTS.md | 162 | 132 | 8 | 17 | 1 | 1 | 3 |
| METHODOLOGY / DEPLOYMENT / CITATION / README_REPRODUCIBILITY / CONTRIBUTING | 93 | 56 | 16 | 10 | 4 | 2 | 5 |
| manuscript.tex | 219 | 212 | 3 | 2 | 1 | 1 | 0 |
| phases_22_30 prose + txt | 157 | 124 | 13 | 11 | 6 | 1 | 2 |
| docs/*.html | 207 | 116 | 43 | 30 | 6 | 10 | 2 |
| **total** | **910** | **698** | **84** | **79** | **18** | **19** | **12** |

**181 rows are STALE, ORPHAN or UNREPRODUCIBLE** — an order of magnitude above the ~15 threshold
at which the brief asks for a prune-or-repair decision before repairs begin.

Concentration matters more than the raw count. Of the 181:

- **79 are in `docs/project_report.html` and `docs/research_progress.html`** (43 of them STALE), two
  hand-maintained pages frozen at v3.2 that never mention the project's headline result at all.
- **48 are ORPHAN rows pointing at six real studies** whose producers sit outside every pipeline
  script (real-trace validation, fairness budget sweep, uncertainty benchmark, quantile model,
  xgboost tuning, load profiles). The numbers are right; the pipeline just cannot regenerate them.
- **26 come from two forks of one file** (`fairness_metrics.csv` exists twice with different values)
  and from artifacts not regenerated since v3.5 (`estimate_sensitivity_summary.csv`, the ROI CSV).
- The remaining ~28 are genuine one-off prose errors spread across DEPLOYMENT.md, COMPLETION_SUMMARY.md
  and the manuscript.

## Claims asserted by the brief that this inventory could not find

| brief's assertion | finding |
|---|---|
| "CITATION.cff says 25,306 instants, version 3.4" | already fixed: the file says 45,432 and version 3.5 |
| "there is no PDF, no .aux, no .log in the tree" | `manuscript.pdf` and `manuscript.log` are both tracked |
| "graphicx is loaded and no figure is included" | two `\includegraphics` are present and both target files exist |
| "none of the four tables cites the artifact file" | all four cite a file; none cites a producing command |
| "Table 1 and §4 give 57.1/56.9; §5 gives 51.9/61.5" | no contradiction: all four locations say 57.1/56.9 and match the CSV. 51.9/61.5 appears only inside the CHANGELOG's own v3.4 retraction |
| "fill the author block" | already filled with a real name and affiliation |
| "add a Threats to Validity section" | already present with four paragraphs |
| "up to 79% of jobs never start" (OOD/uncertainty) | no such phrase in any prose file; the nearest artifact value is a 78.94% maximum `failure_rate_pct` |
| "~320 of 400 SHAP-explained rows were in the fit" | the 400 is real; no prose anywhere states the 320 |

---


---

# README.md + CHANGELOG.md (v3.5 section) — 72 rows: MATCH 58, ORPHAN 9, HISTORICAL 4, STALE 1

| claim | value as stated | file:line | artifact | producer | in run_all? | actual | VERDICT |
|---|---|---|---|---|---|---|---|
| Zero equal-size/different-score counterexamples | 0 in 45,432 | README.md:28,97,158,168 | 05_results/degeneracy/ranking_degeneracy.csv | 04_scheduler/ranking_degeneracy.py | yes [4/14] | 0; 3646+11843+29943=45432 | MATCH |
| Cluster-state features never varying across queue | 7 of 12 @ 0.0% | README.md:30,59,98,158 | degeneracy/feature_variation.csv | ranking_degeneracy.py | yes | 7 features at 0.0 of 12 | MATCH |
| Feature-set sizes | 12 synthetic / 8 per trace | README.md:30,42 | feature_variation.csv | ranking_degeneracy.py | yes | 12 / 8 / 8 | MATCH |
| TOST size-sort ≡ XGBoost | p = 2.6e-16 | README.md:34,99,159 | schedulers/multi_scheduler_equivalence.csv | multi_scheduler_benchmark.py | yes [3/14] | 2.62314e-16 | MATCH |
| Difference CI vs margin | [+0.03,+0.23] vs ±1.59 | README.md:99,159 | multi_scheduler_equivalence.csv | same | yes | 0.02537 / 0.22827; 1.59477 | MATCH |
| Size-sort vs proactive gap | +0.80% | README.md:159 | multi_scheduler_equivalence.csv | same | yes | 0.79521 | MATCH |
| ML order identical to size order | 72–80% | README.md:42 | ranking_degeneracy.csv | ranking_degeneracy.py | yes | 71.53 / 77.63 / 80.25 | MATCH |
| ML order identical to arrival order | 18–27% | README.md:42,158 | ranking_degeneracy.csv | ranking_degeneracy.py | yes | 18.11 / 19.72 / 27.47 | MATCH |
| Distinct priority levels, ~9-job queue | 2.3–3.1 | README.md:158 | ranking_degeneracy.csv | ranking_degeneracy.py | yes | 2.277 / 2.805 / 3.088; queue 8.48–10.16 | MATCH |
| Mean-wait reduction vs FIFO | 7.9% ± 9.4% | README.md:35,151 | benchmark_statistical_summary.csv | benchmark_statistical.py | yes [10/14] | 7.89786 / 9.42864 | MATCH |
| Paired benchmark run count | 40 runs | README.md:151 | benchmark_statistical_summary.csv | same | yes | runs=40 | MATCH |
| Paired t-test p | 2.0e-06 | README.md:151 | benchmark_statistical_summary.csv | same | yes | 2.00147e-06 | MATCH |
| "bootstrap" 95% CI on improvement | [4.9%, 10.9%] | README.md:151 | benchmark_statistical_summary.csv | same | yes | 4.88244 / 10.91329 (Student-t, NOT bootstrap — label defect) | MATCH-value / LABEL-DEFECT |
| Trace protocol breadth | 12 policies × 2 traces × 20 windows | README.md:78,108,160 | trace_schedulers/trace_scheduler_{summary,windows}.csv | trace_driven_benchmark.py | yes [5/14] | 12/trace; 480 = 12×2×20 | MATCH |
| Measured window length | 7 days | README.md:108,160 | trace_driven_benchmark.py:139 | same | yes | MEASURE_DAYS=7 | MATCH |
| Offered load | ≈0.70 | README.md:108,160 | trace_scheduler_summary.csv | same | yes | lanl 0.6751 / sdsc 0.7233 | MATCH |
| Synthetic benchmark size | 14 schedulers | README.md:79,86 | multi_scheduler_benchmark.csv | multi_scheduler_benchmark.py | yes | 14 rows (README list at :86 enumerates 15) | MATCH-value / LIST-DEFECT |
| Proactive vs FCFS, SDSC | −20.4%, p=0.042 | README.md:114,160 | trace_scheduler_significance.csv | trace_driven_benchmark.py | yes | 20.407; 0.04203 | MATCH |
| Proactive vs FCFS, LANL | −4.5%, p=0.48 | README.md:115,160 | trace_scheduler_significance.csv | same | yes | 4.4778; 0.47893 | MATCH |
| SJF-userest beats Proactive, SDSC | 20.2%, Holm p=0.009 | README.md:116,160 | trace_scheduler_significance.csv | same | yes | 20.173; 0.008905 | MATCH |
| SJF-userest beats Proactive, LANL | 15.3% | README.md:117,160 | trace_scheduler_significance.csv | same | yes | 15.2575 | MATCH |
| Median est/runtime SDSC | 6.91× | README.md:130,161 | trace_estimate_quality.csv | same | yes | 6.90714 | MATCH |
| Median est/runtime LANL | 1.51× | README.md:130,161 | trace_estimate_quality.csv | same | yes | 1.51430 | MATCH |
| f-model (C=5) median ratio | 3.00× | README.md:130 | trace_estimate_quality.csv | same | yes | 3.0 | MATCH |
| Within 2× SDSC / LANL | 24.0% / 41.7% | README.md:131 | trace_estimate_quality.csv | same | yes | 0.24014 / 0.41677 | MATCH |
| Under-estimates SDSC/LANL/f-model | 0.1% / 36.3% / 0% | README.md:132,161 | trace_estimate_quality.csv | same | yes | 0.00123 / 0.36274 / 0.0 | MATCH |
| EASY cost of real estimates, SDSC | +6.2% | README.md:137,161 | trace_scheduler_summary.csv | same | yes | 11705.17/11024.67 = +6.17% | MATCH |
| EASY cost of real estimates, LANL | +74%, p=0.025 | README.md:137,161 | trace_scheduler_{summary,significance}.csv | same | yes | +74.36%; 0.02484 | MATCH |
| Wait-model quality | R²≈0.84, MAE≈4.69, 20% holdout | README.md:146,150 | model_comparison_table1.csv | compare_multiple_models.py | yes [12/14] | 0.83684 / 4.69351; test_size=0.2 | MATCH |
| 5-fold CV MAE | 4.74 ± 0.42 | README.md:150 | NONE | train_improved_model.py (stdout only) | yes (step 0) | not persisted anywhere | ORPHAN |
| GPU utilisation unchanged | ≈64% | README.md:152 | multi_scheduler_benchmark.csv | multi_scheduler_benchmark.py | yes | 0.63749 for all but SRPT | MATCH |
| Tail max wait FIFO→Proactive | ~58 → 123 ts | README.md:153 | fairness/fairness_metrics.csv | fairness_analysis.py | yes [2/14] | 57.85 → 122.65 (phase_27 fork says 124.8) | MATCH |
| Gini FIFO→Proactive | 0.53 → 0.79 | README.md:154 | fairness/fairness_metrics.csv | same | yes | 0.52619 → 0.79363 | MATCH |
| Anti-starvation Gini recovery | 0.69 | README.md:154 | fairness/fairness_metrics.csv | same | yes | 0.69210 | MATCH |
| Zero-shot transfer R² | ≈ 0 both traces | README.md:155 | traces/real_trace_validation.csv | real_trace_validation.py | NO | −0.00071 / 0.07218 | ORPHAN |
| Retrained R²(log) SDSC | 0.49 | README.md:156 | traces/real_trace_validation.csv | same | NO | 0.49426 | ORPHAN |
| Median-baseline R² SDSC | −0.69 | README.md:156 | traces/real_trace_validation.csv | same | NO | −0.69356 | ORPHAN |
| Retrained R²(log) LANL | 0.10 | README.md:156 | traces/real_trace_validation.csv | same | NO | 0.10112 | ORPHAN |
| SJF-oracle mean wait | 12.34 ts | README.md:157 | multi_scheduler_benchmark.csv | multi_scheduler_benchmark.py | yes | 12.34 | MATCH |
| SJF-est mean wait | 13.32 ts | README.md:157 | multi_scheduler_benchmark.csv | same | yes | 13.31773 | MATCH |
| SJF-modal mean wait | 14.06 ts | README.md:157 | schedulers/estimate_sensitivity_summary.csv | estimate_sensitivity.py | yes [3b/14] | 14.06273 in a pre-v3.5, non-regenerated file | STALE |
| SRPT mean wait | 14.02 ts | README.md:157 | multi_scheduler_benchmark.csv | same | yes | 14.02318 | MATCH |
| HRRN mean wait | 15.55 ts | README.md:157 | multi_scheduler_benchmark.csv | same | yes | 15.55364 | MATCH |
| Proactive mean wait | 15.95 ts | README.md:157 | multi_scheduler_benchmark.csv | same | yes | 15.94773 | MATCH |
| SJF tails | max 146, Gini 0.79 | README.md:157 | multi_scheduler_benchmark.csv | same | yes | 145.7 / 0.78537 | MATCH |
| HRRN tails | max 65, Gini 0.54 | README.md:157 | multi_scheduler_benchmark.csv | same | yes | 65.2 / 0.53772 | MATCH |
| SDSC trace TOST, size-sort | −0.05%, p=1.8e-12 | README.md:159 | trace_scheduler_equivalence.csv | trace_driven_benchmark.py | yes | −0.051075; 1.7601e-12 | MATCH |
| MLP reproduces size sort bit-identically | all 20 runs | README.md:159 | multi_scheduler_runs.csv + equivalence.csv | multi_scheduler_benchmark.py | yes | 20 runs, NN==SMALLEST elementwise | MATCH |
| EASY vs FCFS/first-fit | +11.8% | README.md:162 | multi_scheduler_significance.csv | same | yes | 11.80198 | MATCH |
| EASY vs strict FIFO | −26% | README.md:162 | multi_scheduler_benchmark.csv | same | yes | 19.2477 vs 26.1745 = −26.5% | MATCH |
| v3.2 reservation price overstated | ~45% | README.md:162 | none | none | n/a | labelled superseded in place | HISTORICAL |
| EASY Gini / max wait | 0.45 / 52 ts | README.md:162 | multi_scheduler_benchmark.csv | same | yes | 0.45148 / 51.55 | MATCH |
| Fairness budget B=60 | +7.4%, max 81 ts | README.md:163 | fairness/budget_sweep.csv | fairness_budget_sweep.py | NO | 7.37961 / 81.3 | ORPHAN |
| Unbounded proactive | +13.1%, max 138 | README.md:163 | fairness/budget_sweep.csv | same | NO | 13.14364 / 138.1 | ORPHAN |
| OOD robustness | mean R² < 0, 72 scenarios | README.md:164 | phase_23_sensitivity/ood_failure_modes.csv | sensitivity_ood_analysis.py | NO (v2 only) | 72 rows; mean −0.31042 | ORPHAN |
| Interval-guard coverage | 68% | README.md:164 | 03_models/wait_model_quantile.pkl (binary only) | train_quantile_model.py | NO | 0.679545 | ORPHAN |
| Holm-adjusted size-sort difference test | p = 0.17 | README.md:170 | multi_scheduler_significance.csv | multi_scheduler_benchmark.py | yes | 0.174496 | MATCH |
| queue_pressure now varies across queue | 82.9% of synthetic instants | CHANGELOG.md:28 | feature_variation.csv | ranking_degeneracy.py | yes | 82.88535 | MATCH |
| queue_pressure varies exactly as often as job_gpu | equal | CHANGELOG.md:28-29 | feature_variation.csv | same | yes | both 82.88535 | MATCH |
| Cluster-constant feature count after fix | 7 | CHANGELOG.md:30 | feature_variation.csv | same | yes | 7 at 0.0% | MATCH |
| Cluster-constant count before fix | 8 | CHANGELOG.md:30 | — | — | n/a | superseded in place | HISTORICAL |
| Zero violations after regeneration | 0 / 45,432 | CHANGELOG.md:32-33 | ranking_degeneracy.csv | same | yes | 0 | MATCH |
| Instant split | 3,646 + 11,843 + 29,943 | CHANGELOG.md:33-34 | ranking_degeneracy.csv | same | yes | exact | MATCH |
| Diagnostic default window count | 20 | CHANGELOG.md:34 | ranking_degeneracy.py:374 | same | yes | default=20 | MATCH |
| 40-run headline | 7.9% | CHANGELOG.md:35 | benchmark_statistical_summary.csv | benchmark_statistical.py | yes | 7.89786 | MATCH |
| Pre-fix headline | was 7.7% | CHANGELOG.md:35 | — | — | n/a | superseded in place | HISTORICAL |
| Headline t p and CI | 2.0e-06; [4.9%,10.9%] | CHANGELOG.md:35-36 | benchmark_statistical_summary.csv | same | yes | 2.00147e-06; 4.88244/10.91329 | MATCH |
| PROACTIVE in 14-scheduler table | 15.95 ts | CHANGELOG.md:36-37 | multi_scheduler_benchmark.csv | multi_scheduler_benchmark.py | yes | 15.94773 | MATCH |
| Pre-fix PROACTIVE | was 16.10 | CHANGELOG.md:36-37 | estimate_sensitivity_summary.csv STILL shows 16.10364 | estimate_sensitivity.py | yes | superseded in place | HISTORICAL |
| PROACTIVE vs FCFS/first-fit | −7.4%, Holm p=4.0e-04 | CHANGELOG.md:37-38 | multi_scheduler_significance.csv | same | yes | −7.36634; 3.9758e-04 | MATCH |
| NN and SMALLEST bit-identical | 16.07 ts | CHANGELOG.md:38-39 | multi_scheduler_benchmark.csv + runs.csv | same | yes | both 16.07455, identical on 20/20 | MATCH |
| TOST equivalence to PROACTIVE | +0.80%, [+0.03,+0.23], ±1.59, p=2.6e-16 | CHANGELOG.md:39 | multi_scheduler_equivalence.csv | same | yes | 0.79521; 0.02537/0.22827; 1.59477; 2.62314e-16 | MATCH |

## Non-numeric defects found in the same sweep

- README headline, key-features list, honest-summary and the BibTeX block still say **v3.4** while CHANGELOG and CITATION.cff are at v3.5. STALE metadata.
- README.md:28,97 and CITATION.cff:25 describe all **45,432 instants as "real"**, but 3,646 of them are synthetic (11,843 SDSC + 29,943 LANL are real). Claim-strength defect on the headline sentence.
- README.md:86 calls the list "14 schedulers" and then enumerates **15** policies (SJF-modal is listed but is not one of the 14 benchmark rows).


---

# Claim inventory — RESULTS.md (whole file)

Repo: D:\Academics\Research Project\proactive-feasibility-scheduler-main
Target: RESULTS.md (199 lines)

## Verdict convention
- MATCH — value equals the artifact value at the stated precision AND the artifact is regenerated by `run_all_experiments.sh` or `phases_22_30/run_all_experiments_v2.sh` (RESULTS.md:3-4 names both as "the seeded pipeline").
- STALE — artifact exists but holds a different / superseded number, or the quoted number contradicts the artifact at the stated precision.
- ORPHAN — value matches an artifact, but no committed .sh pipeline produces that artifact.
- UNREPRODUCIBLE — no artifact anywhere contains the value (stdout / prose only).
- HISTORICAL — presented in RESULTS.md itself as a superseded earlier-version number.
- NOT-CLAIMED — a definitional/config constant, not asserted as a research result.

`run_all` column: `yes` = root run_all_experiments.sh; `v2` = phases_22_30/run_all_experiments_v2.sh only; `NO` = neither.

---

## A. Ranking degeneracy (lines 6-77)

| claim (short text) | value as stated | file:line | artifact file | producer script | in run_all? | value actually in artifact | VERDICT |
|---|---|---|---|---|---|---|---|
| Cluster-only features cannot reorder | "remaining seven features" (7) | RESULTS.md:16-17 | 05_results/degeneracy/feature_variation.csv | 04_scheduler/ranking_degeneracy.py | yes [4/14] | exactly 7 synthetic features at 0.0% variation | MATCH |
| Ranking instants (>=2 queued) | 3,646 / 11,843 / 29,943 | RESULTS.md:27 | 05_results/degeneracy/ranking_degeneracy.csv | ranking_degeneracy.py | yes | 3646 / 11843 / 29943 | MATCH |
| Features total | 12 / 8 / 8 | RESULTS.md:28 | ranking_degeneracy.csv | ranking_degeneracy.py | yes | 12 / 8 / 8 | MATCH |
| Features varying across queue (mean) | 2.62 / 2.87 / 2.69 | RESULTS.md:29 | ranking_degeneracy.csv | ranking_degeneracy.py | yes | 2.62205 / 2.86819 / 2.68557 | MATCH |
| Equal size -> different score violations | 0 / 0 / 0 | RESULTS.md:30 | ranking_degeneracy.csv | ranking_degeneracy.py | yes | 0 / 0 / 0 | MATCH |
| Distinct scores per queue | 2.81 / 3.09 / 2.28 | RESULTS.md:31 | ranking_degeneracy.csv | ranking_degeneracy.py | yes | 2.80499 / 3.08832 / 2.27726 | MATCH |
| Kendall tau vs size order | 0.71 / 0.75 / 0.62 | RESULTS.md:32 | ranking_degeneracy.csv | ranking_degeneracy.py | yes | 0.71315 / 0.75188 / 0.62066 | MATCH |
| Order identical to smallest-first | 80.3% / 71.5% / 77.6% | RESULTS.md:33 | ranking_degeneracy.csv | ranking_degeneracy.py | yes | 80.2523 / 71.5275 / 77.6308 | MATCH |
| Order identical to arrival (FCFS) | 19.7% / 18.1% / 27.5% | RESULTS.md:34 | ranking_degeneracy.csv | ranking_degeneracy.py | yes | 19.7202 / 18.1120 / 27.4655 | MATCH |
| Size->priority table monotone | 63.1% / 57.1% / 56.9% | RESULTS.md:35 | ranking_degeneracy.csv | ranking_degeneracy.py | yes | 63.1377 / 57.1139 / 56.9014 | MATCH |
| Zero counterexamples in N instants | 45,432 | RESULTS.md:37 | ranking_degeneracy.csv (sum) | ranking_degeneracy.py | yes | 3646+11843+29943 = 45432 | MATCH |
| Synthetic cluster-state features at 0.0% | 7 | RESULTS.md:37-38 | feature_variation.csv | ranking_degeneracy.py | yes | 7 rows at 0.0 | MATCH |
| Trace features at 0.0% | 4 of 8 | RESULTS.md:38 | feature_variation.csv | ranking_degeneracy.py | yes | 4 of 8 on each trace | MATCH |
| Variation of those features | 0.0% | RESULTS.md:38-39 | feature_variation.csv | ranking_degeneracy.py | yes | 0.0 exactly | MATCH |
| Queue size | ~9-job queue | RESULTS.md:41 | ranking_degeneracy.csv (mean_queue_len) | ranking_degeneracy.py | yes | 10.164 / 8.476 / 9.098 | MATCH |
| Priority classes per queue | 2.3-3.1 | RESULTS.md:41-42 | ranking_degeneracy.csv | ranking_degeneracy.py | yes | 2.277 min, 3.088 max | MATCH |
| Instants where all scores tie | 18-27% | RESULTS.md:42 | ranking_degeneracy.csv | ranking_degeneracy.py | yes | 18.11 / 19.72 / 27.47 | MATCH |
| TOST margin fraction | 10% of reference | RESULTS.md:45 | multi_scheduler_equivalence.csv, trace_scheduler_equivalence.csv | multi_scheduler_benchmark.py, trace_driven_benchmark.py | yes [3],[5] | margin_frac = 0.1 every row | MATCH |
| Synthetic mean wait TOST | +0.80%, p_TOST 2.6e-16 | RESULTS.md:50 | 05_results/schedulers/multi_scheduler_equivalence.csv | 04_scheduler/multi_scheduler_benchmark.py | yes | 0.795212%, 2.6231e-16 | MATCH |
| Synthetic bsld TOST | +0.69%, 3.8e-18 | RESULTS.md:51 | multi_scheduler_equivalence.csv | multi_scheduler_benchmark.py | yes | 0.689658%, 3.7508e-18 | MATCH |
| SDSC mean wait TOST | -0.05%, 1.8e-12 | RESULTS.md:52 | 05_results/trace_schedulers/trace_scheduler_equivalence.csv | 04_scheduler/trace_driven_benchmark.py | yes | -0.051075%, 1.7601e-12 | MATCH |
| SDSC bsld TOST | +2.7%, 7.4e-04 | RESULTS.md:53 | trace_scheduler_equivalence.csv | trace_driven_benchmark.py | yes | 2.71711%, 7.4463e-04 | MATCH |
| LANL mean wait TOST | +14.4%, 0.77 | RESULTS.md:54 | trace_scheduler_equivalence.csv | trace_driven_benchmark.py | yes | 14.3549%, 0.765481 | MATCH |
| LANL bsld TOST | +7.3%, 0.30 | RESULTS.md:55 | trace_scheduler_equivalence.csv | trace_driven_benchmark.py | yes | 7.25262%, 0.300120 | MATCH |
| Synthetic size-sort vs ML difference | +0.127 ts | RESULTS.md:57 | multi_scheduler_equivalence.csv (mean_diff) | multi_scheduler_benchmark.py | yes | 0.1268182 | MATCH |
| 90% CI on that difference | [+0.025, +0.228] | RESULTS.md:57 | multi_scheduler_equivalence.csv (ci_low/ci_high) | multi_scheduler_benchmark.py | yes | [0.0253704, 0.2282659] | MATCH |
| TOST margin, synthetic mean wait | +/-1.59 ts | RESULTS.md:57 | multi_scheduler_equivalence.csv (margin) | multi_scheduler_benchmark.py | yes | 1.5947727 | MATCH |
| MLP bit-identical to size sort, all runs | 20 runs | RESULTS.md:59-61 | multi_scheduler_equivalence.csv (SMALLEST vs NN) | multi_scheduler_benchmark.py | yes | n=20; mean_diff 0.0, CI [0,0], both metrics | MATCH |
| LANL size sort worse on mean wait | 14.4% | RESULTS.md:64-65 | trace_scheduler_significance.csv | trace_driven_benchmark.py | yes | 14.3549% | MATCH |
| LANL paired t after Holm | p = 0.25 | RESULTS.md:65 | trace_scheduler_significance.csv (ttest_p_holm) | trace_driven_benchmark.py | yes | 0.2489870 | MATCH |
| LANL Wilcoxon disagrees | p = 0.015 | RESULTS.md:65 | trace_scheduler_significance.csv (wilcoxon_p_holm) | trace_driven_benchmark.py | yes | 0.0145105 | MATCH |
| LANL bsld gap and p | +7.3% at p = 0.60 | RESULTS.md:66 | trace_scheduler_significance.csv | trace_driven_benchmark.py | yes | 7.25262%, ttest_p_holm 0.6038358 | MATCH |
| LANL TOST cannot certify equivalence | p = 0.77 | RESULTS.md:66-67 | trace_scheduler_equivalence.csv (p_tost) | trace_driven_benchmark.py | yes | 0.765481 | MATCH |
| LANL evaluation windows | 20 windows | RESULTS.md:67 | trace_scheduler_equivalence.csv (n) | trace_driven_benchmark.py | yes | n = 20 | MATCH |
| Monotone fraction SDSC (retraction) | 57.1% | RESULTS.md:71 | ranking_degeneracy.csv | ranking_degeneracy.py | yes | 57.1139 | MATCH |
| Monotone fraction LANL (retraction) | 56.9% | RESULTS.md:71-72 | ranking_degeneracy.csv | ranking_degeneracy.py | yes | 56.9014 | MATCH |
| Distinct job sizes LANL / synthetic | 6 / 8 (SDSC "dozens") | RESULTS.md:75 | 05_results/degeneracy/size_priority_table.csv | ranking_degeneracy.py | yes | LANL 6, synthetic 8, SDSC 60 | MATCH |
| LANL ML fails to beat plain FCFS | p = 0.48 | RESULTS.md:77 | trace_scheduler_significance.csv (lanl PROACTIVE/FCFS ttest_p) | trace_driven_benchmark.py | yes | 0.4789341 (raw t; Holm = 0.9579) | MATCH |

## B. Trace-driven re-evaluation (lines 79-113)

All 12 policy rows below: prose is in MINUTES, artifact `trace_scheduler_summary.csv` is in SECONDS (divide by 60). All 48 numbers re-derived and confirmed.

| claim (short text) | value as stated | file:line | artifact file | producer script | in run_all? | value actually in artifact | VERDICT |
|---|---|---|---|---|---|---|---|
| Paired windows per trace | 20 | RESULTS.md:81 | trace_scheduler_windows.csv / N_WINDOWS=20 (trace_driven_benchmark.py:137) | trace_driven_benchmark.py | yes [5/14] | 20 per trace | MATCH |
| Measured window length | 7-day | RESULTS.md:81 | trace_driven_benchmark.py:139 MEASURE_DAYS = 7 | trace_driven_benchmark.py | yes | 7 | MATCH |
| Warm-up, not measured | 3-day | RESULTS.md:81 | trace_driven_benchmark.py:138 WARMUP_DAYS = 3 | trace_driven_benchmark.py | yes | 3 | MATCH |
| Offered load | ~0.70 | RESULTS.md:81 | trace_scheduler_summary.csv / trace_fidelity.csv (offered_load) | trace_driven_benchmark.py | yes | SDSC 0.72326, LANL 0.67507 | MATCH |
| SRPT (oracle, preemptive) | 60.1 / 1.1 / 14.6 / 1.1 | RESULTS.md:88 | trace_scheduler_summary.csv | trace_driven_benchmark.py | yes | 3607.02s=60.12, 1.1066; 878.37s=14.64, 1.0718 | MATCH |
| SJF (oracle) | 110.8 / 13.2 / 23.0 / 2.8 | RESULTS.md:89 | trace_scheduler_summary.csv | trace_driven_benchmark.py | yes | 6650.46s=110.84, 13.1791; 1380.43s=23.01, 2.8050 | MATCH |
| SJF (real user estimates) | 115.8 / 14.7 / 31.5 / 4.5 | RESULTS.md:90 | trace_scheduler_summary.csv | trace_driven_benchmark.py | yes | 6946.29s=115.77, 14.7027; 1889.18s=31.49, 4.5121 | MATCH |
| HRRN (real estimates) | 126.3 / 16.8 / 29.5 / 5.0 | RESULTS.md:91 | trace_scheduler_summary.csv | trace_driven_benchmark.py | yes | 7578.28s=126.30, 16.7889; 1771.81s=29.53, 4.9846 | MATCH |
| Proactive + estimate feature | 125.0 / 14.7 / 37.0 / 5.8 | RESULTS.md:92 | trace_scheduler_summary.csv | trace_driven_benchmark.py | yes | 7497.49s=124.96, 14.7067; 2219.44s=36.99, 5.8383 | MATCH |
| Smallest-first (no ML) | 145.0 / 19.0 / 42.5 / 6.6 | RESULTS.md:93 | trace_scheduler_summary.csv | trace_driven_benchmark.py | yes | 8697.25s=144.95, 18.9906; 2549.34s=42.49, 6.5709 | MATCH |
| Proactive (XGBoost) | 145.0 / 18.5 / 37.2 / 6.1 | RESULTS.md:94 | trace_scheduler_summary.csv | trace_driven_benchmark.py | yes | 8701.70s=145.03, 18.4882; 2229.32s=37.16, 6.1266 | MATCH |
| FCFS + first-fit | 174.6 / 34.7 / 38.8 / 8.2 | RESULTS.md:95 | trace_scheduler_summary.csv | trace_driven_benchmark.py | yes | 10477.45s=174.62, 34.6873; 2329.15s=38.82, 8.2128 | MATCH |
| EASY backfill (oracle) | 183.7 / 35.7 / 46.0 / 8.5 | RESULTS.md:96 | trace_scheduler_summary.csv | trace_driven_benchmark.py | yes | 11024.67s=183.74, 35.6763; 2758.85s=45.98, 8.4682 | MATCH |
| EASY backfill (real estimates) | 195.1 / 42.5 / 80.2 / 20.6 | RESULTS.md:97 | trace_scheduler_summary.csv | trace_driven_benchmark.py | yes | 11705.17s=195.09, 42.4655; 4810.49s=80.17, 20.5466 | MATCH |
| Conservative backfill | 207.5 / 40.5 / 113.6 / 28.4 | RESULTS.md:98 | trace_scheduler_summary.csv | trace_driven_benchmark.py | yes | 12449.33s=207.49, 40.5402; 6815.03s=113.58, 28.4256 | MATCH |
| Strict FIFO | 755.1 / 281.5 / 205.7 / 71.4 | RESULTS.md:99 | trace_scheduler_summary.csv | trace_driven_benchmark.py | yes | 45308.48s=755.14, 281.530; 12343.72s=205.73, 71.3754 | MATCH |
| Proactive beats FCFS on SDSC | 20.4% | RESULTS.md:101 | trace_scheduler_significance.csv (pct_vs_ref) | trace_driven_benchmark.py | yes | 20.40702% | MATCH |
| SDSC p and Holm p | p=0.042, Holm 0.084 | RESULTS.md:102 | trace_scheduler_significance.csv | trace_driven_benchmark.py | yes | 0.0420251, 0.0840503 | MATCH |
| Proactive vs FCFS on LANL | 4.5%, p=0.48 | RESULTS.md:102-104 | trace_scheduler_significance.csv | trace_driven_benchmark.py | yes | 4.47778%, ttest_p 0.4789341 | MATCH |
| SJF-user-estimate beats Proactive SDSC | 20.2%, Holm p=0.009 | RESULTS.md:106 | trace_scheduler_significance.csv | trace_driven_benchmark.py | yes | -20.17315%, ttest_p_holm 0.0089045 | MATCH |
| SJF-user-estimate beats Proactive LANL | 15.3% | RESULTS.md:106 | trace_scheduler_significance.csv | trace_driven_benchmark.py | yes | -15.25752% | MATCH |
| PROACTIVE_EST still loses to SJF, SDSC | +7.9% | RESULTS.md:110 | trace_scheduler_equivalence.csv (pct_diff) | trace_driven_benchmark.py | yes | 7.935214% | MATCH |
| PROACTIVE_EST still loses to SJF, LANL | +17.5% | RESULTS.md:110 | trace_scheduler_equivalence.csv (pct_diff) | trace_driven_benchmark.py | yes | 17.481507% | MATCH |

## C. Real user-estimate error vs the f-model (lines 115-137)

| claim (short text) | value as stated | file:line | artifact file | producer script | in run_all? | value actually in artifact | VERDICT |
|---|---|---|---|---|---|---|---|
| Estimate present | 99.9% / 90.8% | RESULTS.md:121 | 05_results/trace_schedulers/trace_estimate_quality.csv | 04_scheduler/trace_driven_benchmark.py | yes [5/14] | 99.93042 / 90.75335 | MATCH |
| Median est/runtime | 6.91x / 1.51x / 3.00x | RESULTS.md:122 | trace_estimate_quality.csv (ratio_median, fmodel_C5_ratio_median) | trace_driven_benchmark.py | yes | 6.90714 / 1.51430 / 3.0 | MATCH |
| Mean est/runtime | 61.4x / 23.8x / 3.00x | RESULTS.md:123 | trace_estimate_quality.csv (ratio_mean) | trace_driven_benchmark.py | yes | 61.42012 / 23.82619; f-model 3.00 has no MEAN column (artifact has fmodel_C5_ratio_median=3.0; E[U(1,5)]=3 so the value is correct but re-used from the median column) | MATCH |
| Within 2x | 24.0% / 41.7% | RESULTS.md:124 | trace_estimate_quality.csv (accurate_within_2x_frac) | trace_driven_benchmark.py | yes | 0.2401420 / 0.4167682 | MATCH |
| Under-estimates | 0.1% / 36.3% / 0% | RESULTS.md:126 | trace_estimate_quality.csv (under_estimate_frac, fmodel_under_estimate_frac) | trace_driven_benchmark.py | yes | 0.00123007 (=0.12%) / 0.3627369 / 0.0 | MATCH |
| f-model constant C | C=5 (est = runtime x U(1,C)) | RESULTS.md:120,129 | trace_estimate_quality.csv column name fmodel_C5_* | trace_driven_benchmark.py | yes | C=5 | NOT-CLAIMED |
| v3.3: EASY nearly insensitive to estimate quality | 19.0-19.8 ts | RESULTS.md:132 (repeated at :163) | 05_results/schedulers/estimate_sensitivity_summary.csv | 04_scheduler/estimate_sensitivity.py | yes [3b/14] | true range across BACKFILL/BACKFILL_EST(C=1..10)/BACKFILL_MODAL = 18.936-19.831 | STALE |
| Real estimate error costs EASY, SDSC | +6.2% mean wait | RESULTS.md:132 | trace_scheduler_significance.csv (EASY_USEREST vs EASY_ORACLE) | trace_driven_benchmark.py | yes | 680.504/11024.67 = +6.173% | MATCH |
| Real estimate error costs EASY, LANL | +74% mean wait | RESULTS.md:132 | trace_scheduler_significance.csv | trace_driven_benchmark.py | yes | 4810.49/2758.85 = +74.36% | MATCH |
| p for the LANL EASY estimate effect | p = 0.025 | RESULTS.md:133 | trace_scheduler_significance.csv (ttest_p) | trace_driven_benchmark.py | yes | 0.0248403 raw (Holm = 0.2347) | MATCH |

## D. Simulator fidelity (lines 139-150)

| claim (short text) | value as stated | file:line | artifact file | producer script | in run_all? | value actually in artifact | VERDICT |
|---|---|---|---|---|---|---|---|
| LANL simulated FCFS mean wait | 38.8 min | RESULTS.md:141 | 05_results/trace_schedulers/trace_fidelity.csv (sim_fcfs_mean_wait_min) | trace_driven_benchmark.py | yes | 38.81914 (mean of 20 windows) | MATCH |
| LANL recorded mean wait | 33.2 min | RESULTS.md:141-142 | trace_fidelity.csv (recorded_mean_wait_min) | trace_driven_benchmark.py | yes | 33.24814 | MATCH |
| SDSC simulated FCFS mean wait | 174.6 min | RESULTS.md:143 | trace_fidelity.csv | trace_driven_benchmark.py | yes | 174.62422 | MATCH |
| SDSC recorded mean wait | 630.9 min | RESULTS.md:143 | trace_fidelity.csv | trace_driven_benchmark.py | yes | 630.91956 | MATCH |
| SDSC recorded median wait | 19.3 min | RESULTS.md:143 | trace_fidelity.csv (recorded_median_wait_min) | trace_driven_benchmark.py | yes | mean of per-window medians = 19.31583 (median of those medians = 7.23) | MATCH |
| SDSC machine size | 128-processor | RESULTS.md:147 | 02_data/SDSC-SP2-1998-4.2-cln.swf.gz header MaxProcs | trace_driven_benchmark.py (parse_swf) | yes | MaxProcs: 128 | MATCH |

## E. High-level findings (lines 152-169)

| claim (short text) | value as stated | file:line | artifact file | producer script | in run_all? | value actually in artifact | VERDICT |
|---|---|---|---|---|---|---|---|
| Model quality R2 | ~0.84 | RESULTS.md:153 | 05_results/model_comparison_table1.csv (XGBoost) / traces/lanl_validation_results.csv | 03_models/compare_multiple_models.py; 03_models/train_improved_model.py | yes [12/14] | 0.83684 | MATCH |
| Model quality MAE | ~4.69 ts | RESULTS.md:153 | model_comparison_table1.csv / lanl_validation_results.csv | compare_multiple_models.py | yes | 4.693508 | MATCH |
| Holdout fraction | 20% | RESULTS.md:153 | 03_models/train_improved_model.py:40 test_size=0.2 | train_improved_model.py | yes | 0.2 | MATCH |
| 5-fold CV MAE | 4.74 +/- 0.42 | RESULTS.md:153 | NO CSV — stdout only (train_improved_model.py:64-66) | train_improved_model.py | yes | re-ran: 4.7354 +/- 0.4186 | MATCH |
| Proactive mean wait-time reduction vs FIFO | 7.9% | RESULTS.md:154 | 05_results/benchmark_statistical_summary.csv | 04_scheduler/benchmark_statistical.py | yes [10/14] | 7.897862356147857 | MATCH |
| Benchmark run count | 40-run seeded paired | RESULTS.md:154 | benchmark_statistical_summary.csv (runs) / benchmark_statistical_results.csv | benchmark_statistical.py | yes | 40 | MATCH |
| Paired t-test p | 2.0e-06 | RESULTS.md:154 | benchmark_statistical_summary.csv (ttest_pvalue) | benchmark_statistical.py | yes | 2.001470727467938e-06 | MATCH |
| Wilcoxon p | 2.7e-06 | RESULTS.md:154 | benchmark_statistical_summary.csv (wilcoxon_pvalue) | benchmark_statistical.py | yes | 2.6737125153886154e-06 | MATCH |
| "bootstrap 95% CI on improvement" | [4.9%, 10.9%] | RESULTS.md:154 | benchmark_statistical_summary.csv (improvement_ci95_*) vs phases_22_30/phase_22_stats/stats_summary.csv | benchmark_statistical.py:141 (Student-t) / stats_bootstrap.py (percentile bootstrap) | yes / v2 | t-interval [4.88244, 10.91329]; ACTUAL bootstrap [4.91022, 10.67139] -> would round to [4.9%, 10.7%] | STALE |
| GPU utilisation, identical both arms | ~64% | RESULTS.md:154 | 05_results/benchmark_statistical_results.csv (baseline_util, proactive_util) | benchmark_statistical.py | yes | 0.6411771 both arms, identical row-by-row | MATCH |
| Mean max-wait FIFO -> Proactive | ~58 -> ~123 ts | RESULTS.md:155 | 05_results/fairness/fairness_metrics.csv | 04_scheduler/fairness_analysis.py | yes [2/14] | 57.85 -> 122.65 | MATCH |
| Per-job Gini FIFO -> Proactive | 0.53 -> 0.79 | RESULTS.md:155 | 05_results/fairness/fairness_metrics.csv | fairness_analysis.py | yes | 0.5261905 -> 0.7936274 | MATCH |
| Anti-starvation Gini | 0.69 | RESULTS.md:155 | 05_results/fairness/fairness_metrics.csv | fairness_analysis.py | yes | 0.6921033 (phase_27 duplicate says 0.68777) | MATCH |
| Anti-starvation max wait | ~88 ts | RESULTS.md:155 | 05_results/fairness/fairness_metrics.csv | fairness_analysis.py | yes | 88.15 (phase_27 duplicate says 87.15) | MATCH |
| Committed PWA traces | 2 (.swf.gz) | RESULTS.md:156 | 02_data/SDSC-SP2-1998-4.2-cln.swf.gz, 02_data/LANL-CM5-1994-4.1-cln.swf.gz | 02_data/load_real_traces.py | yes [7/14] | both present | MATCH |
| Zero-shot transfer R2 SDSC | 0.072 | RESULTS.md:156 | 05_results/traces/real_trace_validation.csv | 02_data/build_real_trace_datasets.py + real_trace_validation.py | NO | 0.07218296 | ORPHAN |
| Zero-shot transfer R2 LANL | -0.001 | RESULTS.md:156 | real_trace_validation.csv | build_real_trace_datasets.py + real_trace_validation.py | NO | -0.00070661 | ORPHAN |
| Chronological split | 80/20 | RESULTS.md:156 | real_trace_validation.csv (n_train/n_test) | real_trace_validation.py | NO | LANL 97644/24411 = 80.0%; SDSC 34493/8624 = 80.0% | ORPHAN |
| SDSC retrained R2(log) | 0.494 | RESULTS.md:156 | real_trace_validation.csv | real_trace_validation.py | NO | 0.4942594 | ORPHAN |
| SDSC retrained MAE | 622 min | RESULTS.md:156 | real_trace_validation.csv | real_trace_validation.py | NO | 622.5331 | ORPHAN |
| SDSC median baseline R2 | -0.694 | RESULTS.md:156 | real_trace_validation.csv | real_trace_validation.py | NO | -0.6935626 | ORPHAN |
| SDSC median baseline MAE | 684 min | RESULTS.md:156 | real_trace_validation.csv | real_trace_validation.py | NO | 684.2447 | ORPHAN |
| LANL retrained R2(log) | 0.101 | RESULTS.md:156 | real_trace_validation.csv | real_trace_validation.py | NO | 0.1011160 | ORPHAN |
| LANL median wait | ~4 s | RESULTS.md:156 | 02_data/real_trace_dataset_lanl.csv / real_trace_validation.csv note | build_real_trace_datasets.py | NO | full-trace median 2.0 s; train-split median 3.0 s (CSV note: "0.05 min") | STALE |
| OOD shifted scenarios | 72 | RESULTS.md:157 | phases_22_30/phase_23_sensitivity/ood_failure_modes.csv | sensitivity_ood_analysis.py | v2 | 72 rows | MATCH |
| OOD mean R2 | ~-0.31 | RESULTS.md:157 | ood_failure_modes.csv (r2_score) | sensitivity_ood_analysis.py | v2 | -0.3104164 | MATCH |
| OOD R2 range | -2.3 to +0.85 | RESULTS.md:157 | ood_failure_modes.csv | sensitivity_ood_analysis.py | v2 | min -2.3236643, max 0.8535501 | MATCH |
| OOD improvement swing | -10% to +53% | RESULTS.md:157 | ood_failure_modes.csv (improvement_pct) | sensitivity_ood_analysis.py | v2 | min -9.65517, max 52.57319 | MATCH |
| OOD mean failure rate | ~32% | RESULTS.md:157 | ood_failure_modes.csv (failure_rate_pct) | sensitivity_ood_analysis.py | v2 | 31.52788 | MATCH |
| LightGBM MAE/R2 | 4.61 / 0.848 | RESULTS.md:158 | 05_results/model_comparison_table1.csv | 03_models/compare_multiple_models.py | yes [12/14] | 4.6112256 / 0.8476769 | MATCH |
| CatBoost MAE/R2 | 4.62 / 0.845 | RESULTS.md:158 | model_comparison_table1.csv | compare_multiple_models.py | yes | 4.6183197 / 0.8449712 | MATCH |
| XGBoost MAE/R2 | 4.69 / 0.837 | RESULTS.md:158 | model_comparison_table1.csv | compare_multiple_models.py | yes | 4.6935081 / 0.8368400 | MATCH |
| Linear regression MAE/R2 | 6.32 / 0.761 | RESULTS.md:158 | model_comparison_table1.csv | compare_multiple_models.py | yes | 6.3182596 / 0.7610821 | MATCH |
| Ablation: job_gpu R2 drop | 0.201 | RESULTS.md:159 | 05_results/ablation_study_results.csv | 03_models/ablation_study.py | yes [1/14] | 0.2012289 | MATCH |
| Ablation: queue_pressure R2 drop | 0.027 | RESULTS.md:159 | ablation_study_results.csv | ablation_study.py | yes | 0.0267010 | MATCH |
| Ablation: queue_length R2 drop | 0.021 | RESULTS.md:159 | ablation_study_results.csv | ablation_study.py | yes | 0.0208933 | MATCH |
| SLA compliance FIFO | 0.933 | RESULTS.md:160 | phases_22_30/phase_27_fairness/sla_compliance.csv | fairness_sla_analysis.py | v2 | 0.9325758 | MATCH |
| SLA compliance Proactive | 0.945 | RESULTS.md:160 | sla_compliance.csv | fairness_sla_analysis.py | v2 | 0.9448485 | MATCH |
| SLA compliance anti-starvation | 0.863 | RESULTS.md:160 | sla_compliance.csv | fairness_sla_analysis.py | v2 | 0.8627273 | MATCH |
| Scheduler count in benchmark | 14 schedulers | RESULTS.md:161 | 05_results/schedulers/multi_scheduler_benchmark.csv | 04_scheduler/multi_scheduler_benchmark.py | yes [3/14] | 14 rows | MATCH |
| Paired runs | 20 | RESULTS.md:161 | multi_scheduler_runs.csv | multi_scheduler_benchmark.py | yes | 280 rows = 14 x 20 unique runs | MATCH |
| Mean wait ladder (13 values) | SJF 12.34 < SJF-EST 13.32 < SRPT 14.02 < HRRN 15.55 < Proactive 15.95 < NN 16.07 = SMALLEST 16.07 < Priority 16.53 < FCFS 17.22 < Hybrid-BF 19.20 ~ EASY-EST 19.23 ~ EASY 19.25 < Cons-BF 21.84 < FIFO 26.17 ts | RESULTS.md:161 | multi_scheduler_benchmark.csv (mean_wait) | multi_scheduler_benchmark.py | yes | 12.34, 13.31773, 14.02318, 15.55364, 15.94773, 16.07455, 16.07455, 16.53182, 17.21591, 19.20136, 19.22773, 19.24773, 21.84364, 26.17455 | MATCH |
| HRRN gap vs Proactive | -2.5%, t p=0.17 Holm | RESULTS.md:161 | 05_results/schedulers/multi_scheduler_significance.csv | multi_scheduler_benchmark.py | yes | -2.471142%, ttest_p_holm 0.1744957 | MATCH |
| NN/SMALLEST gap vs Proactive | +0.8%, Holm n.s., TOST-equivalent | RESULTS.md:161 | multi_scheduler_significance.csv + multi_scheduler_equivalence.csv | multi_scheduler_benchmark.py | yes | +0.795212%, ttest_p_holm 0.1744957, equivalent=True | MATCH |
| Priority gap Holm n.s. | (implied n.s.) | RESULTS.md:161 | multi_scheduler_significance.csv | multi_scheduler_benchmark.py | yes | +3.662534%, ttest_p_holm 0.1744957 | MATCH |
| NN and SMALLEST tie exactly | exact tie | RESULTS.md:161 | multi_scheduler_benchmark.csv / multi_scheduler_equivalence.csv | multi_scheduler_benchmark.py | yes | both 16.074545..., mean_diff 0.0 | MATCH |
| SJF on modal estimates | 14.06 | RESULTS.md:162 | 05_results/schedulers/estimate_sensitivity_summary.csv (SJF_MODAL) | 04_scheduler/estimate_sensitivity.py | yes [3b/14] | 14.0627273 | MATCH |
| Proactive mean wait (comparison anchor) | 16.10 | RESULTS.md:162 | estimate_sensitivity_summary.csv (PROACTIVE) | estimate_sensitivity.py | yes | 16.1036364 in that (NOT-REGENERATED) file; current value in multi_scheduler_benchmark.csv is 15.9477 (quoted 3 lines earlier at :161) | STALE |
| Estimate-quality sweep degrades SJF-EST | 12.34 -> 13.49 (C=1->10) | RESULTS.md:162 | estimate_sensitivity_summary.csv / estimate_sensitivity.csv | estimate_sensitivity.py | yes | 12.34 (C=1) -> 13.4886 (C=10) | MATCH |
| SJF max wait / Gini | 146 ts, 0.79 | RESULTS.md:162 | multi_scheduler_benchmark.csv | multi_scheduler_benchmark.py | yes | 145.7, 0.7853698 | MATCH |
| SRPT max wait / Gini / preemptions | 151, 0.81, 41.5/run | RESULTS.md:162 | multi_scheduler_benchmark.csv | multi_scheduler_benchmark.py | yes | 150.95, 0.8102613, 41.45 | MATCH |
| SRPT checkpoint cost | 1-tick | RESULTS.md:162 | 04_scheduler/multi_scheduler_benchmark.py (config) | multi_scheduler_benchmark.py | yes | config constant | NOT-CLAIMED |
| HRRN mean / max / Gini | 15.55, 65, 0.54 | RESULTS.md:162 | multi_scheduler_benchmark.csv | multi_scheduler_benchmark.py | yes | 15.5536364, 65.2, 0.5377217 | MATCH |
| Proactive vs FCFS/first-fit | -7.4%, Holm p=4.0e-04 | RESULTS.md:162 | multi_scheduler_significance.csv | multi_scheduler_benchmark.py | yes | -7.366337%, ttest_p_holm 3.975794e-04 | MATCH |
| v3.2 overstated reservation price | ~45% | RESULTS.md:163 | none (v3.2 artifact overwritten) | n/a | n/a | superseded implementation; no current artifact | HISTORICAL |
| Canonical EASY cost vs FCFS/first-fit | +11.8% (19.25 vs 17.22) | RESULTS.md:163 | multi_scheduler_significance.csv / multi_scheduler_benchmark.csv | multi_scheduler_benchmark.py | yes | 11.801980%; 19.24773 vs 17.21591 | MATCH |
| EASY beats strict FIFO by | 26% | RESULTS.md:163 | multi_scheduler_benchmark.csv | multi_scheduler_benchmark.py | yes | (26.17455-19.24773)/26.17455 = 26.46% | MATCH |
| EASY insensitivity range | 19.0-19.8 across perfect/f-model/modal | RESULTS.md:163 | estimate_sensitivity_summary.csv | estimate_sensitivity.py | yes | 18.9364 (BACKFILL_EST C=10) to 19.8309 (BACKFILL_MODAL) | STALE |
| Conservative BF extra mean wait | +13% | RESULTS.md:163 | multi_scheduler_benchmark.csv | multi_scheduler_benchmark.py | yes | (21.84364-19.24773)/19.24773 = +13.49% | MATCH |
| Conservative BF best tail | max 50 ts | RESULTS.md:163 | multi_scheduler_benchmark.csv (CONS_BF max_wait) | multi_scheduler_benchmark.py | yes | 50.25 (study min) | MATCH |
| Hybrid ties plain EASY | 19.20 vs 19.25, n.s. | RESULTS.md:163 | multi_scheduler_benchmark.csv (values); NO pairwise test row exists for PROACTIVE_BF vs BACKFILL | multi_scheduler_benchmark.py | yes | 19.20136 vs 19.24773; significance CSV only has PROACTIVE and FIFO as references, so "n.s." is untested | UNREPRODUCIBLE |
| Budget B=60 mean-wait gain | +7.4% | RESULTS.md:164 | 05_results/fairness/budget_sweep.csv | 04_scheduler/fairness_budget_sweep.py | NO | 7.379608 | ORPHAN |
| Unbounded gain | +13.1% | RESULTS.md:164 | budget_sweep.csv (budget=None) | fairness_budget_sweep.py | NO | 13.143644 | ORPHAN |
| B=60 max wait (vs unbounded) | 81 ts (vs 138) | RESULTS.md:164 | budget_sweep.csv (max_wait) | fairness_budget_sweep.py | NO | 81.3 vs 138.1 | ORPHAN |
| B=60 Gini cap | 0.68 | RESULTS.md:164 | budget_sweep.csv (gini) | fairness_budget_sweep.py | NO | 0.6846367 | ORPHAN |
| B<=30 slightly worse than FIFO | negative improvement | RESULTS.md:164 | budget_sweep.csv | fairness_budget_sweep.py | NO | B=0 -0.852, B=10 -0.640, B=20 -1.417, B=30 -0.109 (all negative) | ORPHAN |
| Quantile levels | q10/q50/q90 | RESULTS.md:165 | 03_models/wait_model_quantile.pkl (quantiles) | 03_models/train_quantile_model.py | NO | q10/q50/q90 | NOT-CLAIMED |
| Holdout q50 MAE | 4.61 | RESULTS.md:165 | 03_models/wait_model_quantile.pkl (holdout_q50_mae) | train_quantile_model.py | NO | 4.6062965 | ORPHAN |
| Interval coverage vs nominal | 68% vs 80% nominal | RESULTS.md:165 | wait_model_quantile.pkl (holdout_coverage) | train_quantile_model.py | NO | 0.6795455 = 67.95%; nominal 80% by q10-q90 construction | ORPHAN |
| 0.5x arrivals / 4 nodes vs FIFO | -2...-5% across all three variants | RESULTS.md:165 | 05_results/uncertainty/uncertainty_ood_benchmark.csv | 04_scheduler/uncertainty_scheduler_benchmark.py | NO | PROACTIVE -2.013, UCB -2.904, GUARDED -4.826 | ORPHAN |
| Guard fired on ticks | 0.5% | RESULTS.md:165 | uncertainty_ood_benchmark.csv (fallback_tick_pct) | uncertainty_scheduler_benchmark.py | NO | 0.4860872 | ORPHAN |
| SHAP dependence plots | 12 | RESULTS.md:166 | 05_results/shap/shap_dependence_*.png | 03_models/explainability_shap.py | yes [6/14] | 12 distinct features (x2 light/dark) | MATCH |
| SHAP force plots | 3 | RESULTS.md:166 | 05_results/shap/shap_force_{0,1,2}.png | explainability_shap.py | yes | 3 | MATCH |
| SHAP global summary | 1 | RESULTS.md:166 | 05_results/shap/shap_summary.png | explainability_shap.py | yes | 1 | MATCH |
| Scaling advantage, small contended cluster | 14.5% | RESULTS.md:167 | 05_results/scaling/scaling_analysis.csv (wait_advantage_pct) | 04_scheduler/scaling_analysis.py | yes [8/14] | 14.521943 (4 nodes / 8 GPUs) | MATCH |
| Scaling advantage at capacity | 0% | RESULTS.md:167 | scaling_analysis.csv | scaling_analysis.py | yes | 0.0 at 16 nodes/128 GPUs and 32/256 | MATCH |
| Online learning MAE before/after | 6.98 -> 6.88 | RESULTS.md:167 | 05_results/models/online_learning_results.csv | 03_models/online_learning.py | yes [9/14] | 6.9763122 -> 6.8765076 | MATCH |
| ROI annual savings | ~$78k/yr | RESULTS.md:167 | 05_results/roi/cost_benefit_analysis.csv | 05_results/roi_analysis.py | yes [11/14] | 78073.65 — but computed from mean_wait_improvement_pct = 7.7144 (superseded); its own input benchmark_statistical_summary.csv now says 7.8979, which regenerates to ~$79.9k | STALE |
| ROI first-year | 86% | RESULTS.md:167 | cost_benefit_analysis.csv (roi_pct) | roi_analysis.py | yes | 85.8896 — same stale 7.7144 input; regenerates to ~90% | STALE |
| Phase 22 CIs are bootstrap + BH-corrected | (qualitative + CI/p columns) | RESULTS.md:168 | phases_22_30/phase_22_stats/stats_summary.csv | phase_22_stats/stats_bootstrap.py | v2 | ci_lower/ci_upper + p_value_bh columns present; wait_improvement_pct CI [4.9102, 10.6714], p_bh 2.0015e-06 | MATCH |
| Degenerate comparisons flagged | "n/a (zero variance)" x2 | RESULTS.md:168 | stats_summary.csv | stats_bootstrap.py | v2 | 2 rows: paired_ttest_gpu_utilization, paired_ttest_completed_jobs | MATCH |
| Stress-test cluster range | 32-256 GPUs | RESULTS.md:169 | phases_22_30/phase_26_scaling/scaling_benchmark.csv | phase_26_scaling/scaling_benchmark.py | v2 | 32, 64, 128, 256 | MATCH |
| Utilisation under saturation | >99.7% | RESULTS.md:169 | scaling_benchmark.csv (gpu_utilization_pct) | scaling_benchmark.py | v2 | min 99.69416 — strictly BELOW 99.7 | STALE |
| Batched inference per decision | 10-48 ms | RESULTS.md:169 | scaling_benchmark.csv (inference_latency_ms) | scaling_benchmark.py | v2 | 9.6018 / 19.4892 / 28.4027 / 48.0514 (true range 9.60-48.05) | MATCH |
| Scheduling overhead | <5% of throughput | RESULTS.md:169 | scaling_benchmark.csv (throughput_overhead_pct) | scaling_benchmark.py | v2 | max 4.8051 | MATCH |

---

## Summary

**Total rows: 162**

| VERDICT | count |
|---|---|
| MATCH | 132 |
| ORPHAN | 17 |
| STALE | 8 |
| NOT-CLAIMED | 3 |
| HISTORICAL | 1 |
| UNREPRODUCIBLE | 1 |

### STALE (8)
1. RESULTS.md:154 — "bootstrap 95% CI on improvement [4.9%, 10.9%]". The quoted numbers are the **Student-t** interval from `benchmark_statistical.py:141` ([4.88244, 10.91329]); the genuine percentile bootstrap over the same 40 runs (`phase_22_stats/stats_summary.csv`) is [4.91022, 10.67139] -> [4.9%, **10.7%**]. Either the label "bootstrap" or the upper bound is wrong.
2. RESULTS.md:162 — "Proactive's **16.10**". Superseded; the current value is 15.9477 (`multi_scheduler_benchmark.csv`), quoted correctly one line earlier at :161. 16.10 survives only in the un-regenerated `estimate_sensitivity_summary.csv`.
3. RESULTS.md:132 — "EASY 19.0-19.8 ts across perfect/f-model/modal". True artifact range is **18.936-19.831** (`BACKFILL_EST` at C=10 is 18.9364).
4. RESULTS.md:163 — same "19.0-19.8" range restated. Same defect.
5. RESULTS.md:156 — LANL "median wait ~4 s". Artifact median is **2.0 s** full-trace / 3.0 s on the train split (`real_trace_dataset_lanl.csv`; the CSV note says "0.05 min" = 3 s).
6. RESULTS.md:167 — ROI "~$78k/yr". `cost_benefit_analysis.csv` holds 78073.65 but its `mean_wait_improvement_pct` field is **7.7144** (pre-v3.5); its own input `benchmark_statistical_summary.csv` now reads 7.8979, which regenerates to ~$79.9k. The artifact was not re-run after the benchmark.
7. RESULTS.md:167 — ROI "86% first-year". `roi_pct` = 85.8896, same stale 7.7144 input; regenerates to ~90%.
8. RESULTS.md:169 — "utilisation stays **>99.7%**". `scaling_benchmark.csv` minimum is **99.69416**, strictly below the stated bound (XLarge, 256 GPUs).

### ORPHAN (17) — value verified, but no committed .sh runs the producer
- RESULTS.md:156 x8 — the entire real-trace validation paragraph (zero-shot R2 SDSC 0.072, LANL -0.001; 80/20 chronological split; SDSC retrained 0.494 / 622 min; median baseline -0.694 / 684 min; LANL retrained 0.101) reads `05_results/traces/real_trace_validation.csv`, produced by `02_data/build_real_trace_datasets.py` + `real_trace_validation.py`, neither of which appears in `run_all_experiments.sh` or `run_all_experiments_v2.sh`.
- RESULTS.md:164 x5 — whole wait-budget Pareto bullet (B=60 +7.4%, unbounded +13.1%, max 81 vs 138, Gini 0.68, B<=30 negative) reads `05_results/fairness/budget_sweep.csv`; `04_scheduler/fairness_budget_sweep.py` is in no pipeline.
- RESULTS.md:165 x4 — uncertainty bullet (q50 MAE 4.61, 68% coverage vs 80% nominal, -2...-5% at 0.5x/4-node, guard 0.5% of ticks) reads `03_models/wait_model_quantile.pkl` and `05_results/uncertainty/uncertainty_ood_benchmark.csv`; `train_quantile_model.py` and `uncertainty_scheduler_benchmark.py` are in no pipeline.

### UNREPRODUCIBLE (1)
- RESULTS.md:163 — "the predicted-wait backfill hybrid still ties plain EASY (19.20 vs 19.25, **n.s.**)". Both means are in `multi_scheduler_benchmark.csv`, but `multi_scheduler_significance.csv` only carries PROACTIVE and FIFO as references, so no PROACTIVE_BF-vs-BACKFILL test exists anywhere. The "n.s." is asserted, never computed.

### HISTORICAL (1)
- RESULTS.md:163 — "the v3.2 EASY implementation ... overstated the reservation price at ~45%". Explicitly framed as the superseded v3.2 implementation; no current artifact holds 45%.

### NOT-CLAIMED (3)
- RESULTS.md:120,129 — f-model C=5 (config constant of `est = runtime x U(1,C)`).
- RESULTS.md:162 — SRPT "1-tick checkpoint cost" (simulator config).
- RESULTS.md:165 — "q10/50/90" quantile levels (model config).

### Cross-artifact inconsistencies worth flagging (not counted as separate rows)
- RESULTS.md:155 fairness numbers match `05_results/fairness/fairness_metrics.csv` (max 122.65, Gini 0.79363; anti-starvation 88.15 / 0.69210) but the duplicate `phases_22_30/phase_27_fairness/fairness_metrics.csv` disagrees (124.8 / 0.79592; 87.15 / 0.68777). Two artifacts, one claim.
- RESULTS.md:169 "10-48 ms" is a rounding of the true 9.60-48.05 ms range; scored MATCH but both endpoints are outside the stated interval.
- RESULTS.md:88-99 trace table: prose is in minutes, artifact in seconds. All 48 values re-derived; no discrepancy.


---

# METHODOLOGY.md + DEPLOYMENT.md + CITATION.cff + README_REPRODUCIBILITY.md + CONTRIBUTING.md
93 rows: MATCH 56, STALE 16, ORPHAN 10, NOT-CLAIMED 5, UNREPRODUCIBLE 4, HISTORICAL 2.
Per file: DEPLOYMENT.md 40 · METHODOLOGY.md 36 · CITATION.cff 6 · README_REPRODUCIBILITY.md 6 · CONTRIBUTING.md 5.

## STALE (16)

| claim | stated | file:line | artifact | actual | note |
|---|---|---|---|---|---|
| contended-cluster wait advantage | 14.4% | DEPLOYMENT.md:15 | 05_results/scaling/scaling_analysis.csv (nodes=4,gpus=8) | 14.521943154401422 → 14.5% | producer in pipeline [8/14] |
| proactive max wait | ~125 ts | DEPLOYMENT.md:17 | 05_results/fairness/fairness_metrics.csv | **122.65**; 125 only reachable from the phase-27 fork (124.8) | fork is not in the pipeline |
| per-job Gini proactive | 0.80 | DEPLOYMENT.md:17 | 05_results/fairness/fairness_metrics.csv | **0.7936274161073568** → 0.79; fork says 0.7959 → 0.80 | claim taken from fork, path cited is root |
| anti-starvation Gini recovery | 0.80 → 0.69 | DEPLOYMENT.md:34 | root CSV | 0.79363 → 0.69210 (fork 0.79592 → 0.68777) | start point is fork-only |
| anti-starvation max wait | 125 → 87 ts | DEPLOYMENT.md:34 | root CSV | **122.65 → 88.15** (fork 124.8 → 87.15) | **87 is not obtainable from the pipeline artifact at all** |
| CI method | "bootstrap CI" | DEPLOYMENT.md:53 | benchmark_statistical.py:141-148 | `stats.t.ppf(0.975, df=n-1)` — Student-t. No bootstrap in that script | percentile bootstrap exists only in phase 22 |
| drift trigger multiplier | > 1.5× training-holdout MAE | DEPLOYMENT.md:61 | 03_models/concept_drift_detection.py:57 | `threshold = np.std(y_train) * 0.55` — no 1.5×, no holdout-MAE reference | NOT in the spec's defect list |
| drift rolling window | 50–100 jobs | DEPLOYMENT.md:61 | concept_drift_detection.py | `rolling_mae(..., win=80)` for the CSV; trigger test uses a **30**-job window (:97-102) | 30 is outside the stated range |
| starvation count definition | wait > 3× mean | DEPLOYMENT.md:64 | fairness_analysis.py (producer of the cited column) | that script uses 3× the job's own runtime | see verbatim quotes below |
| anti-starvation threshold + constant name | `fairness_analysis.py` (`STARVATION_THRESHOLD`-style), 3× mean wait | DEPLOYMENT.md:82 | fairness_analysis.py | no such constant exists (grep empty); bump age is 3× the job's own runtime | two errors in one row |
| drift window / multiplier row | rolling 50, 1.5× | DEPLOYMENT.md:85 | concept_drift_detection.py | rolling 80 (CSV) / 30 (trigger); multiplier 0.55×std(y_train) | neither number is in the script |
| annual savings | ≈ $78k/yr | DEPLOYMENT.md:90 | 05_results/roi/cost_benefit_analysis.csv | 78073.65 in the committed file, but its `mean_wait_improvement_pct` is the superseded 7.7144; re-running step [11/14] with the current 7.897862 yields **≈ $79,931** | stale input |
| first-year ROI | 86% | DEPLOYMENT.md:90 | same CSV | 85.8896 committed; re-run gives **≈ 90.3%** | stale input |
| benchmark size | 20-run **13**-scheduler | METHODOLOGY.md:42 | multi_scheduler_runs.csv | **14** unique schedulers × 20 runs | self-contradicts METHODOLOGY.md:32 ("14 as of v3.4") |
| output directory list | models, schedulers, scaling, fairness, shap, traces, roi | README_REPRODUCIBILITY.md:31 | 05_results/ | also contains **degeneracy/, trace_schedulers/, uncertainty/** — the first two hold the v3.4 headline artifacts | |
| what CI checks | `python -m compileall .` only | CONTRIBUTING.md:45 | .github/workflows/ci.yml | compileall over a subset (excludes tests/, conftest.py, vizstyle.py), plus a ruff gate and a pip dry-run job CONTRIBUTING never mentions | |

## ORPHAN (10) — value correct, producer absent from run_all_experiments.sh

From `05_results/traces/real_trace_validation.csv` (producer `build_real_trace_datasets.py` + `real_trace_validation.py`):
- DEPLOYMENT.md:16 LANL CM-5 −0.001 → −0.0007066100848580614
- DEPLOYMENT.md:16 SDSC SP2 0.072 → 0.07218296334259944
- DEPLOYMENT.md:16 retrained R²(log) 0.494 SDSC → 0.4942594144862178
- DEPLOYMENT.md:16 median baseline −0.694 → −0.6935626203556449

From `05_results/uncertainty/uncertainty_ood_benchmark.csv` (producer `uncertainty_scheduler_benchmark.py`):
- DEPLOYMENT.md:73 "every smart policy worse than FIFO (−2.0 to −4.8%)" → scenario arr0.5_nodes4: PROACTIVE −2.0134, UCB −2.9037, GUARDED −4.8260
- DEPLOYMENT.md:74 "spread guard fired on only 0.5% of ticks" → fallback_tick_pct 0.48608719935031564

From `05_results/fairness/budget_sweep.csv` (producer `fairness_budget_sweep.py`):
- DEPLOYMENT.md:83 default B = 60 ts → BUDGETS list at :57; B=60 is a swept point, not a coded default
- DEPLOYMENT.md:83 +7.4% of +13.1% → 7.379608223083591 / 13.143644249998536
- DEPLOYMENT.md:83 max wait 81 vs 138 → 81.3 / 138.1
- DEPLOYMENT.md:83 B ≤ 30 worse than FIFO → B=0 −0.8518, B=10 −0.6398, B=20 −1.4171, B=30 −0.1091; B=40 turns positive at +1.8714

## UNREPRODUCIBLE (4)

- **DEPLOYMENT.md:72** "68% empirical coverage vs 80% nominal". No CSV carries it; the value lives inside the pickle `03_models/wait_model_quantile.pkl` (`holdout_coverage = 0.6795454545454546`) and is printed to console. The "80% nominal" half MATCHes (`quantiles = [0.1, 0.5, 0.9]`).
- **METHODOLOGY.md:52** "`bash run_all_experiments.sh` regenerates the dataset, model, and **every result** deterministically on a fresh checkout." False: the script covers none of benchmark_results.csv, budget_sweep.csv, sensitivity_analysis_results.csv, real_trace_validation.csv, uncertainty_ood_benchmark.csv, xgboost_load_profile_performance.csv, xgboost_tuning_results.csv, nor anything under phases_22_30/.
- **README_REPRODUCIBILITY.md:7** "two runs produce identical datasets and results." **False by construction** for `05_results/scaling/scaling_analysis.csv`, whose `scheduler_overhead_sec`, `inference_time_sec` and `proactive_total_sim_time_sec` columns are `time.perf_counter()` wall-clock measurements (scaling_analysis.py:113,126,129-131,144). Same class of problem in phases_22_30/phase_26_scaling/scaling_benchmark.csv (latency columns).
- **CONTRIBUTING.md:39-41** "Anything that changes a reported number must be reproducible from run_all_experiments.sh on a clean checkout … Do not commit results that cannot be regenerated." Contradicted by the 7 orphan producers.

## HISTORICAL (2)

- DEPLOYMENT.md:111 warns never to mix legacy `wait_model.pkl` with the v2 pipeline. Accurate, but `03_models/wait_model.pkl` (741,082 bytes) is still committed and no pipeline step produces or consumes it.
- CITATION.cff:28-29 "a **30-phase** research pipeline". Directories stop at phase_28_manuscript; the v2 script runs only phases 22–27 while its own banner claims "Phases 01–30". Phases 29/30 exist only as prose labels.

## NOT-CLAIMED (5) — operator advice, no artifact to check
DEPLOYMENT.md:28 (2–4 weeks of data), :51 (promote only if holdout R² ≥ 0.7 — note the repo's own best real-trace retrained R² is 0.494, i.e. its own evidence would fail its own gate), :63 (Gini > 0.7 → tighten), :65 (> 5% of dispatch interval → batch; nearest artifact `throughput_overhead_pct` max 4.805), :86 (refresh every 3–6 months).

## No file anywhere states a wall-clock runtime for the pipeline
Repo-wide grep for "wall-clock", "minutes to run", "takes ~" returns only two unrelated hits. Nothing in METHODOLOGY.md, DEPLOYMENT.md, README_REPRODUCIBILITY.md, CONTRIBUTING.md, either pipeline script, or the Dockerfile says how long a run takes.

## The starvation definitions, verbatim

**1. `04_scheduler/fairness_analysis.py` — 3× the job's own RUNTIME** (lines 135-137 and 145):
```python
starving = [j for j in queue if (t - j.arrival_time) > (3 * j.runtime)]
non_starving = [j for j in queue if (t - j.arrival_time) <= (3 * j.runtime)]
queue = starving + non_starving
...
starvation_count = sum(1 for j in done if (j.start_time - j.arrival_time) > (3 * j.runtime))
```

**2. `04_scheduler/fairness_budget_sweep.py` — 3× the RUN-LEVEL MEAN WAIT** (lines 58 and 207, where `m = float(np.mean(w))`):
```python
STARVATION_FACTOR = 3.0                            # starved: wait > 3x run mean
...
'starvation': int(np.sum(w > STARVATION_FACTOR * m)),
```

**3. `DEPLOYMENT.md` — "3× mean", attributed to the wrong script** (lines 64 and 82):
```
| Starvation count (wait > 3× mean) | queue accounting | any sustained increase → investigate |
| Anti-starvation threshold | `fairness_analysis.py` (`STARVATION_THRESHOLD`-style bump age) | 3× mean wait | ... |
```
The wording matches `fairness_budget_sweep.py`; the attribution names `fairness_analysis.py`, which uses a different definition and contains no such constant. The two are not interchangeable: one is per-job relative (short jobs starve sooner), the other distribution-relative.


---

# manuscript.tex — 219 rows (every table cell counted): MATCH 212, STALE 3, ORPHAN 2, UNREPRODUCIBLE 1, HISTORICAL 1

Table cell counts: Table 1 = 24 cells, Table 2 = 12 numeric, Table 3 = 48 cells, Table 4 = 13 cells.

## Non-MATCH rows

| claim | stated | file:line | artifact | actual | VERDICT |
|---|---|---|---|---|---|
| Table 1, distinct scores/queue, Synthetic | 2.81 | manuscript.tex:310 | 05_results/degeneracy/ranking_degeneracy.csv `mean_distinct_predictions` | 2.8049917718047177 → rounds to **2.80** | STALE (last-digit rounding error; SDSC 3.09 and LANL 2.28 in same row are correct) |
| Zero-shot transfer R² ≈ 0 | ≈ 0 | manuscript.tex:273 | 05_results/traces/real_trace_validation.csv | lanl −0.00071, sdsc 0.0722 | ORPHAN (producer `real_trace_validation.py` not in run_all_experiments.sh) |
| Table 4, mean est/runtime, f-model column | 3.00 | manuscript.tex:495 | trace_estimate_quality.csv has only `fmodel_C5_ratio_median` | no f-model MEAN column exists; 3.00 is the analytic E[U(1,5)] | UNREPRODUCIBLE |
| "The reordering policies **all** buy mean wait with equity (Gini 0.88–0.89)" | 0.88–0.89 for all | manuscript.tex:474 | trace_scheduler_summary.csv (sdsc) | in band: SMALLEST_FIRST .8843, PROACTIVE .8859, SJF_USEREST .8823, SJF_ORACLE .8852, PROACTIVE_EST .8830. **Out of band: HRRN_USEREST .8649, SRPT_ORACLE .9476** | STALE (universal quantifier false) |
| BACKFILL mean wait across estimate qualities | 19.0–19.8 ts | manuscript.tex:512 | estimate_sensitivity_summary.csv | perfect 19.2477, f-model C=5 19.2277, modal 19.8309 → actual range **19.23–19.83** | STALE (lower bound understated) |
| "7.9% in the older 40-run paired study" | 7.9% | manuscript.tex:451 | benchmark_statistical_summary.csv | 7.897862 | HISTORICAL (explicitly labelled "older", contrasted with current 20-run 7.4% on same line — deliberate) |
| manuscript.tex + manuscript.pdf as artifacts | — | whole file | — | no pdflatex/latexmk/make step exists anywhere in the repo | ORPHAN BUILD |

## Answers to the spec's manuscript assumptions — MOST ARE ALREADY FALSE

1. **Monotone-fraction contradiction: REFUTED.** `:314` (Table 1) `63.1\% & 57.1\% & 56.9\%`; `:398-399` (§5) `57.1\%` SDSC vs `56.9\%` LANL; `:342-343`, `:347` (§4) same. CSV `pct_size_table_monotone` = 63.1377 / 57.1139 / 56.9014. **All four locations agree with each other and with the artifact.** The 51.9/61.5 pair exists only in the CHANGELOG v3.4 retraction. Spec item E1 is based on a stale observation.
2. **Tables citing artifacts: 4 of 4 do** (spec claimed none). `:319` full path; `:376-377` a glob (`the two *_equivalence.csv files under 05_results/`); `:436` and `:500` partial paths. **0 of 6 captions cite a producing command.** Figure captions `:326-327` and `:443-445` cite no source at all.
3. **`\includegraphics`: 2 present** (spec claimed none). `:325` → `../../05_results/degeneracy/ranking_degeneracy.png`; `:442` → `../../05_results/trace_schedulers/trace_scheduler_comparison.png`. Both files exist.
4. **Author block: FILLED** (spec asked to fill it). `:32-35` = `Rakshit Rameshbabu \\ \small Vellore Institute of Technology, Chennai`. No email/ORCID. `\date{\today}` at `:37` means the PDF carries a build-time date.
5. **Threats to Validity: EXISTS** at `:544` with 4 paragraphs (simulator fidelity, statistical power, covariate shift, trace vintage). **Related Work: EXISTS** at `:158` with 5 paragraphs.
6. **Citations: 30 `\cite` commands, 30 distinct keys, 30 `\bibitem`s, 0 undefined, 0 uncited.** Hand-rolled `thebibliography`, no `.bib`.
7. Section list: Introduction 73, Related Work 158, Experimental Setup 230, Ranking Degeneracy Measured 290, Sorting by Size Is Equivalent 352, Trace-Driven Re-Evaluation 406, Real Estimate Error vs f-Model 478, What Would Make Learning Matter 521, Threats to Validity 544, Conclusion 581, Artifact Availability 600.

## Integrity issues found in the manuscript that the spec did not list

- **`:514` selective non-adjustment.** "EASY +74% on LANL, p = 0.025" uses the RAW `ttest_p` (0.0248403). The Holm-adjusted value in the same CSV row is **0.2347 — not significant**. The paper insists on Holm at `:284`, `:540`, `:561`. Either quote the Holm p or say explicitly that this p is unadjusted and why.
- **`:340` semantic mislabel.** "in 18–27% of instants every queued job receives the same score, so the policy silently reduces to FCFS" is backed by the CSV column `pct_order_identical_to_arrival`, which measures *order equal to arrival order*, not *all scores tied*. No all-ties column exists in the CSV. Same mislabel appears in README.md:42,158. NEEDS CODE CHECK.
- **Table 1 rows 7+8 do not partition**: LANL 77.6% (order = size order) + 27.5% (order = arrival order) = 105.1%. Both cells match the CSV; the categories overlap, but the table layout implies exclusivity.
- **`:452,:459,:460,:453` percentage-base flip.** "beats FCFS by 20.4%" uses `pct_vs_ref` (base = Proactive) from the significance CSV; `trace_scheduler_equivalence.csv` reports the same comparison as −16.948% (base = FCFS). Internally consistent but two artifacts give two numbers for one comparison.
- The `19.0–19.8` claim at `:512` cites `estimate_sensitivity_summary.csv`, which is the known **not-regenerated** pre-v3.5 file.
- `:552` "recorded median of 19.3 min" is a **mean of the 20 per-window medians**, not a pooled median (median-of-medians is 7.23).


---

# phases_22_30 prose + txt artifacts — 157 rows: MATCH 124, STALE 13, ORPHAN 11, UNREPRODUCIBLE 6, NOT-CLAIMED 2, HISTORICAL 1

Files: COMPLETION_SUMMARY.md, PHASES_ROADMAP.md, phase_27_fairness/fairness_formal_analysis.md, phase_27_fairness/dropped_schedulers.txt, phase_24_extended_schedulers/novelty_claim.txt, phase_26_scaling/scaling_law_fit.txt.

**Cross-cutting:** none of these files' producers run in the root `run_all_experiments.sh`. The four generated artifacts (`baseline_comparison.csv`, `scaling_benchmark.csv` + `scaling_law_fit.txt`, `fairness_metrics.csv` + `sla_compliance.csv` + `dropped_schedulers.txt`) are reachable only via `run_all_experiments_v2.sh`; the three .md files have no producer at all. COMPLETION_SUMMARY's whole v3.2-Extensions section rests on artifacts orphaned in *both* pipelines (`real_trace_validation.csv`, `budget_sweep.csv`, `uncertainty_ood_benchmark.csv`, `wait_model_quantile.pkl`).

## Non-MATCH rows (33)

### COMPLETION_SUMMARY.md (hand-written)
| claim | stated | line | actual | verdict |
|---|---|---|---|---|
| Phase 24 Proactive mean wait | 15.71 | L41 | no artifact holds 15.71; current 16.6955 (ph27) / 15.9477 (multi) | STALE (traced by `git log -S` to v3.1 commit 69c733d, never refreshed) |
| Phase 24 FIFO mean wait | 16.65 | L41 | 18.2689 (ph27) / 17.2159 (multi) | STALE (same v3.1 origin) |
| Phase 24 FIFO Gini | 0.518 | L41 | 0.5160252 (multi) / 0.5261905 (ph27) | STALE |
| Phase 24 FIFO max wait | 53.9 ts | L41 | 54.65 (multi) / 57.85 (ph27) | STALE |
| SDSC SP2 cluster size/year | 128 nodes, 1998 | L51 | no CSV records node count or year | UNREPRODUCIBLE |
| SDSC p90 wait | ~15 h | L52 | no percentile column anywhere | UNREPRODUCIBLE |
| LANL CM-5 procs/years | 1024 procs, 1994–96 | L56 | not in any artifact | UNREPRODUCIBLE |
| LANL median wait | ~4 s | L56 | nearest is a train-split median note (0.05 min = 3.0 s), not the trace median | UNREPRODUCIBLE |
| Utilisation at all scales | > 99.7% | L65 | `scaling_benchmark.csv` XLarge = **99.69416** — strictly below | STALE (false as written) |
| Batched inference latency | 10–48 ms | L65 | measured **9.60–48.05 ms** | STALE (lower bound rounded up into truth) |
| Contended-cluster advantage | 14.4% | L66 | scaling_analysis.csv 14.5219 | STALE |

### PHASES_ROADMAP.md (hand-written)
| claim | stated | line | actual | verdict |
|---|---|---|---|---|
| 95% CI on all metrics (wait, fairness, speedup) | all metrics | L12 | stats_summary.csv has CIs only for fifo_mean_wait, proactive_mean_wait, wait_improvement_pct. No fairness or speedup CI exists | ORPHAN |
| SLURM backfill / Kubernetes QoS / Yarn FIFO, 5-scheduler benchmark | 5 named | L14 | baseline_comparison.csv has 14 schedulers, none of those three; superseded, recorded in dropped_schedulers.txt | STALE |
| cross_trace_mae.csv deliverable | cross-dataset MAE | L15 | file exists but holds ONE row, `synthetic_proxy_trace`, `mape_pct` exactly 200.0 (saturated placeholder). No cross-dataset MAE at all | ORPHAN |
| Alibaba traces | Alibaba | L15 | never delivered. Scaffolding only (ALIBABA_COLUMNS :163, `_map_alibaba()` :334, expected path `02_data/alibaba_2018.csv` :95). CHANGELOG:152 already records an earlier retraction of an Alibaba provenance claim | ORPHAN |
| Prove proactive ≥ FIFO on fairness | goal | L17 | contradicted by every artifact; withdrawn at COMPLETION_SUMMARY.md:72 but this file still lists it as the goal and marks Phase 27 complete at :32 | STALE |
| Zenodo DOI | DOI | L19, L108 | not assigned; checkbox unticked | ORPHAN |
| Phase effort estimates | 2–3d ×4, 2d ×2, 3–4d, 1–2d, 5–7d, ~3 weeks | L12-20, L97 | planning estimates, never measured | HISTORICAL |
| "verified on this machine, July 2026" | — | L105 | no run-log artifact | UNREPRODUCIBLE |

### fairness_formal_analysis.md (hand-written) — 9 of its ~12 claims are fabrications
| claim | stated | line | actual | verdict |
|---|---|---|---|---|
| SLA-1 | 95% of **jobs** within 2× geomean wait | L30 | code (fairness_sla_analysis.py:260-261) implements 95% of **runs** within 2× geomean of run-level waits | STALE (unit-of-analysis mismatch) |
| SLA-2 | no job starves: wait < 200 ts for 99% of jobs | L31 | never implemented. `grep 200` in fairness_sla_analysis.py returns nothing. Shipped SLA-2 is a continuous share (FIFO 0.7977, Proactive 0.8345, anti-starvation 0.7255) using wait > 3× runtime | STALE (doc and code measure different things, and disagree in direction) |
| Formal claim | Gini(Proactive) ≤ Gini(FIFO) + ε | L38 | 0.5262 → 0.7959; withdrawn at COMPLETION_SUMMARY.md:72 | STALE |
| Empirical Gini improvement | ~3.7% vs FIFO | L44 | **no artifact**. Every candidate: per-job 0.5262→0.7959 = 51.3% WORSE; multi-sched 0.5160→0.7934 = 53.76% worse; run-level 0.16951→0.16465 = 2.87% better (closest, still not 3.7%). novelty_claim.txt:109 states the opposite in plain text | ORPHAN (sign is wrong, not just magnitude) |
| Example fairness table, FIFO row | 18.27,0.420,0.876,127.3,2.1,60.62 | L62 | only 18.27 matches anything (fifo mean_wait 18.2689) | ORPHAN |
| Example fairness table, Proactive row | 16.72,0.368,0.901,116.4,2.3,50.61 | L63 | no value appears as the claimed quantity | ORPHAN |
| Example fairness table, SLURM row | 17.15,0.392,0.889,121.5,2.0,60.75 | L64 | scheduler itself was dropped; no values in any CSV | ORPHAN |
| Example SLA table, FIFO row | 0.94,0.875,0.82,0.885 | L70 | actual 1.0, 0.7977, 1.0, 0.9326 | ORPHAN |
| Example SLA table, Proactive row | 0.97,0.965,0.91,0.948 | L71 | actual 1.0, 0.8345, 1.0, 0.9448 | ORPHAN |
| Promised percentiles p50/p99/p99.9 | — | L8 | no percentile column in either delivered CSV | ORPHAN |
| Gini formula (L19), Jain formula (L23) | definitions | | no value asserted | NOT-CLAIMED ×2 |

Table headers the doc invents vs. what the CSV actually has:
```
doc:  scheduler,mean_wait,gini_coefficient,jain_fairness_index,max_wait,min_wait,ratio_max_min
real: scheduler,scheduler_name,data_source,n_runs,mean_wait,gini_run_level,jain_fairness_index_run_level,min_run_wait,max_run_wait,run_max_min_ratio,gini_per_job,max_job_wait,starvation_count,starvation_rate_pct
doc:  scheduler,sla1_95pct_2x_geomean,sla2_no_starvation_99pct,sla3_max_wait_le_150,compliance_score
real: ... sla2_no_starvation_share ...   (renamed precisely because the 99% target was never implemented)
```
All 26 table values were tested numerically against every numeric cell of every CSV under 05_results/ and phases_22_30/. **Exactly one — 18.27 — appears as the same quantity.**

### novelty_claim.txt (generated by scheduler_comparison.py)
| claim | stated | line | actual | verdict |
|---|---|---|---|---|
| benchmark run count in the header string | "15-run multi-scheduler benchmark" | L16 | multi_scheduler_runs.csv has exactly **20** runs per scheduler for all 14. Every number under the header is correct; the header string is stale. Conflicts with COMPLETION_SUMMARY:102 "20 paired runs" | STALE (fix the literal in the producer) |

### scaling_law_fit.txt (generated by scaling_benchmark.py)
| claim | stated | line | actual | verdict |
|---|---|---|---|---|
| VERDICT | "Inference latency is CONSTANT regardless of cluster size" | L35 | asserted over 48.05 → 9.60 → 28.40 → 19.49 ms: **non-monotone, 5.0× spread**, min and max adjacent in the ordering. Fit R² never computed | UNREPRODUCIBLE |

Measurement vs interpretation split for that file:
- **Measured, all verified MATCH:** scaling points L13-16 (32/64/128/256 GPUs on 4/8/16/32 nodes); metrics block L23-26 (wait 3743.83/3481.65/2797.88/1794.49; throughput 10.4/17.9/33.2/59.2; latency 48.05/9.60/28.40/19.49; overhead 4.81/0.96/2.84/1.95%); L40 "overhead < 5%" (max 4.805); L41 "non-trivial (48.05 ms)".
- **Fit / interpretation / extrapolation:** L31 model `latency = 64.547·n^−0.23`, L33 exponent −0.234 (re-ran polyfit: 64.5470550, −0.2341054 — reproduces exactly), L32 "O(1)", L35 VERDICT, L46-48 projections 14.98 / 12.74 / 9.21 ms at 512/1024/4096 GPUs (reproduce from the fit; pure extrapolation past the largest measured point).
- `scaling_benchmark.py:206` reads `"O(1)" if exponent < 0.1 else ...` — one-sided, so a strongly NEGATIVE exponent is labelled constant by fall-through. The file simultaneously claims latency is constant (L35) and non-trivial at 48.05 ms (L41), the latter taken from the smallest cluster.

## The withdrawn claim
"Proactive ≥ FIFO on fairness", asserted at PHASES_ROADMAP.md:17 and fairness_formal_analysis.md:4 and :38 (with a proof sketch at L40-44 and the fabricated 3.7% as support). Withdrawal is recorded at exactly one place — COMPLETION_SUMMARY.md:72, the only "withdraw" hit in the repo. **Neither source document was updated.** A reader of either file alone never learns the claim was retracted.

## COMPLETION_SUMMARY numbers taken from the phase-27 fork rather than 05_results/fairness/
All on line 70. Three of four are fork-specific:

| claim | phase_27 fork | 05_results/fairness | used |
|---|---|---|---|
| max wait 58 → **125** ts | 124.8 → 125 | 122.65 → 123 | fork |
| per-job Gini 0.53 → **0.80** | 0.7959223 → 0.80 | 0.7936274 → 0.79 | fork |
| anti-starvation max **87** ts | 87.15 → 87 | 88.15 → 88 | fork |
| anti-starvation Gini 0.69 | 0.6877748 | 0.6921033 | ambiguous (both round to 0.69) |
| FIFO Gini 0.53 | 0.5261905 | 0.5261905 | identical, not a tell |

The fork is consistently the more favourable source on both anti-starvation numbers. The two files also disagree on FIFO `mean_wait` (18.575 in 05_results/fairness vs 18.2689 in phase_27); COMPLETION_SUMMARY:29 and novelty_claim:86 both use the phase_27 value.
**`05_results/schedulers/fairness_metrics.csv` is byte-identical to `05_results/fairness/fairness_metrics.csv`** (duplicate write by fairness_analysis.py), so the real split is two-way, not three-way.


---

# docs/*.html — 207 rows: MATCH 116, STALE 43, ORPHAN 30, UNREPRODUCIBLE 6, HISTORICAL 10, NOT-CLAIMED 2

| File | Rows | MATCH | STALE | ORPHAN | UNREPRO | HIST | NOT-CLAIMED |
|---|---|---|---|---|---|---|---|
| docs/explanation.html | 106 | 79 | 6 | 13 | 3 | 5 | 0 |
| docs/project_report.html | 58 | 21 | 24 | 10 | 1 | 1 | 1 |
| docs/research_progress.html | 43 | 16 | 13 | 7 | 2 | 4 | 1 |

**No generator exists. All three files are hand-maintained.** Exhaustive grep of `*.py`/`*.sh` for html writes returns zero hits; neither pipeline script touches `docs/`. That explains the drift: explanation.html was hand-synced through v3.4/v3.5, the other two were last touched 21 Jul and are frozen at v3.2.

## explanation.html — every non-MATCH row

**STALE (6)**
- `:1423` `DATA.scaling.y = [14.4, 5.1, 0.0, 0.0]` — artifact is **14.5219**; the page's own figcaption at `:931` says +14.5%. Self-contradictory.
- `:1263` "the reservation price is measured (+45% mean wait for Gini 0.365)" — current **+11.8%** (19.25 vs 17.22), Gini **0.451**. The page retracts this itself at `:881`.
- `:791` "exactly the 33-of-40 split the paired benchmark measured" — `benchmark_statistical_results.csv` has 6 negative runs → **34/40**, which the page's own legend states at `:847`.
- `:1145`, `:1176`, `:1235` "12 steps" / "12/12 steps green" — the pipeline emits `[0/14]…[13/14]`. The 12-box flow at `:1159-1172` omits ranking-degeneracy [4/14], trace-driven benchmark [5/14] and estimate-sensitivity [3b/14], and calls step 4 a "5-scheduler benchmark" (it is 14).
- `:345`, `:390`, `:1318` version stamped "v3.4, July 2026" — repo is v3.5.

**UNREPRODUCIBLE (3)**
- `:725` 5-fold CV MAE 4.74 ± 0.42 — stdout only, no artifact.
- `:791` distilled browser model holdout MAE 4.64 / R² 0.842 — no artifact.
- `:1772` `var MODEL = {…"holdout":{"mae":4.64,"r2":0.842}}`, a 40-tree distillation powering the page's interactive demo. **No script in the repo produces it.**

**HISTORICAL (5)** — `:705` (smaller_jobs_in_queue held 82% importance), `:730` (in-sample R² ≈ 0.96), `:881` (earlier variant overstated at ~45%), `:1215` (R² 0.96 → honest 0.82–0.84), `:1217` (manuscript claimed Gini 0.37 vs 0.42).

**ORPHAN (13)** — value matches artifact, producer not in run_all_experiments.sh:
`:726` tuned R² 0.850 / MAE 4.57 and `:1099` 729 fits (tune_xgboost.py) · `:745-748` load-profile MAE/R² (generate_load_profiles.py) · `:917` + `DATA.pareto` budget sweep (fairness_budget_sweep.py) · `:957-960` uncertainty table, 131 vs 114 started, max 110 vs 122 (uncertainty_scheduler_benchmark.py) · `:965`, `:1247`, `:1273` 68% coverage (train_quantile_model.py) · `:983`, `:986`, `:993-996`, `:1261` real-trace numbers (real_trace_validation.py).

**Provenance defects counted MATCH:** `:848` "bootstrap 95% CI [4.88%, 10.91%]" is a Student-t interval; `:1036-1040` derives ROI from "the measured 7.9%" while the ROI CSV's own input field is the stale 7.7144.

**Soft rounding notes:** `:931` "utilisation > 99.7%" (min 99.694), "10–48 ms" (min 9.60), `:677` "25% of jobs start within 2 ts" (26.0%), `:1002` LANL "median wait ≈ 4 seconds" (per-window recorded median ≈ 2.05 s).

## project_report.html + research_progress.html

**CONFIRMED frozen v3.2 snapshots.** Both banner "v3.2 — Phases 01–30 Complete". Both carry 7.71% ± 8.87, p=1.4e-06, Wilcoxon 7.1e-06, 18.27→16.70, 33-of-40, CI [4.95%, 10.34%]. Both quote **PROACTIVE 16.10**, matching the un-regenerated `estimate_sensitivity_summary.csv`, not `multi_scheduler_benchmark.csv`. **Neither file contains the string `45,432`, `7.9%`, `15.95`, `14.5%`, `degenerac`, `TOST`, or `size sort`** — the entire headline result of the project is absent from both.

**Decisive question — does deleting them lose unique content?** Yes, technically: 11 claims appear nowhere else. But **none is a currently-valid result** — every one is stale, historical, or unverifiable:

1. Wilcoxon p = 7.1e-06 (current 2.674e-06) — STALE
2. "33/40 runs positive" (current 34/40) — STALE
3. 5-scheduler max-wait/Gini set: SJF 143.5/0.782, NN 134.2/0.796, PROACTIVE 125.7/0.790, PRIORITY 127.7/0.739 — STALE
4. OOD improvement range "−14% to +50%" (actual −9.66% to +52.57%) — STALE
5. PROACTIVE_BF 24.91/55.5 and BACKFILL 24.94/55.4 (current 19.20/53.8, 19.25/51.55) — STALE
6. Per-scenario uncertainty deltas +6.5/+6.1/+6.3, "−4.3% to −5.0%", "guard fired on 0.2% of ticks" (actual +6.64/+6.81/+6.46, −2.01/−2.90/−4.83, 0.486%) — STALE
7. Full 9-row budget-sweep table — mostly STALE
8. **Phase 03 classifier ROC-AUCs: LogReg 0.87 / RF 0.91 / XGBoost 0.93** (research_progress.html:204). Nowhere else in the repo. Producer `03_models/train_model_withxg.py` prints them, writes no artifact, is in no pipeline. **UNREPRODUCIBLE and genuinely unique.**
9. **Phase 04 leaky v1 regression MAE 5.25 ts / R² 0.833** (research_progress.html:225). Nowhere else. HISTORICAL, no artifact.
10. "Model v1 (with leakage, in-sample) apparent R² 0.83" — nowhere else. HISTORICAL.
11. "top 4 features contribute 15–23% each" (research_progress.html:254) — nowhere else, and **wrong**: actual gain importances 26.1/22.3/16.9/12.2%, which both files' own importance panels state correctly.

## Bootstrap CIs — the repo has two different 95% intervals on the same 40 runs, both called "bootstrap", plus one orphan

| Interval | where | provenance | verdict |
|---|---|---|---|
| [4.88%, 10.91%] | explanation.html:848 | `benchmark_statistical.py:141` Student-t → `benchmark_statistical_summary.csv` | value right, **label wrong** |
| [4.95%, 10.34%] | project_report.html:1092,1162; research_progress.html:382,410,979 | **no producer in the repo yields it.** Only other appearance is CHANGELOG.md:327 in the historical v3.1 block | ORPHAN / STALE |
| [+0.025, +0.228] ts | explanation.html:456 | multi_scheduler_equivalence.csv `ci_low/ci_high` — the 90% TOST CI in ts, not a % bootstrap | MATCH |
| [4.91%, 10.67%] | COMPLETION_SUMMARY.md:29 **only** | `stats_bootstrap.py` → `stats_summary.csv`. **The only genuine percentile bootstrap in the repo — and no HTML page quotes it** | MATCH |
| [4.9%, 10.9%] | README.md:151, RESULTS.md:154, CHANGELOG.md:36 | 1-dp rounding of the Student-t interval | value right, label wrong |

## Orphan-producer references (all three files present these as part of a reproducible pipeline; none flags that run_all_experiments.sh will not regenerate them)

- explanation.html: sensitivity_analysis.py :1115 · tune_xgboost.py :1099 · generate_load_profiles.py :1081 · benchmark_and_plot.py :1123 · train_quantile_model.py :1098 · real_trace_validation.py :1011,:1084 · fairness_budget_sweep.py :1117 · uncertainty_scheduler_benchmark.py :1117
- project_report.html: fairness_budget_sweep.py :1367,:1747 · train_quantile_model.py :1482,:1749 · real_trace_validation.py :1533,:1750 · benchmark_and_plot.py :1694 · uncertainty_scheduler_benchmark.py :1748
- research_progress.html: benchmark_and_plot.py :302 · real_trace_validation.py :723 · fairness_budget_sweep.py :848 · uncertainty_scheduler_benchmark.py :849 · train_quantile_model.py :850


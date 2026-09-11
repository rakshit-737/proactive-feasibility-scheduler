# Artefact verification

mode: full (worktree)
tree: C:\Users\Rakshit\AppData\Local\Temp\verify_artifacts_gbyixj80\tree

pipeline exit code: 0

| status | artefact | detail |
| --- | --- | --- |
| TIMING | `05_results/model_comparison_table1.csv` | wall-clock columns only, exempt by NONDETERMINISTIC_COLUMNS: training_time_sec (7 cells) |
| TIMING | `05_results/scaling/scaling_analysis.csv` | wall-clock columns only, exempt by NONDETERMINISTIC_COLUMNS: scheduler_overhead_sec (4 cells), inference_time_sec (4 cells), proactive_total_sim_time_sec (4 cells) |
| TIMING | `phases_22_30/phase_26_scaling/scaling_benchmark.csv` | wall-clock columns only, exempt by NONDETERMINISTIC_COLUMNS: inference_latency_ms (4 cells), throughput_overhead_pct (4 cells) |
| TIMING | `phases_22_30/phase_26_scaling/scaling_measurements.txt` | identical apart from 7 wall-clock line(s), exempt by NONDETERMINISTIC_TEXT |
| OK | `02_data/dataset.csv` | 110 rows x 7 cols identical |
| OK | `02_data/improved_wait_dataset.csv` | 2200 rows x 15 cols identical |
| OK | `03_models/feature_importance_v2-dark.png` | 124558 bytes (bytes not compared) |
| OK | `03_models/feature_importance_v2.png` | 126340 bytes (bytes not compared) |
| OK | `03_models/wait_model_quantile.pkl` | 2378058 bytes, unpickles (bytes not compared) |
| OK | `03_models/wait_model_v2.pkl` | 1154504 bytes, unpickles (bytes not compared) |
| OK | `05_results/ablation_importance-dark.png` | 149580 bytes (bytes not compared) |
| OK | `05_results/ablation_importance.png` | 152369 bytes (bytes not compared) |
| OK | `05_results/ablation_study_results.csv` | 12 rows x 24 cols identical |
| OK | `05_results/benchmark_statistical_results.csv` | 40 rows x 9 cols identical |
| OK | `05_results/benchmark_statistical_summary.csv` | 13 rows x 2 cols identical |
| OK | `05_results/degeneracy/feature_variation.csv` | 28 rows x 3 cols identical |
| OK | `05_results/degeneracy/non_degeneracy_sweep.csv` | 14 rows x 21 cols identical |
| OK | `05_results/degeneracy/non_degeneracy_utility.csv` | 14 rows x 13 cols identical |
| OK | `05_results/degeneracy/ranking_degeneracy-dark.png` | 237677 bytes (bytes not compared) |
| OK | `05_results/degeneracy/ranking_degeneracy.csv` | 3 rows x 13 cols identical |
| OK | `05_results/degeneracy/ranking_degeneracy.png` | 241284 bytes (bytes not compared) |
| OK | `05_results/degeneracy/ranking_degeneracy_totals.csv` | 1 rows x 10 cols identical |
| OK | `05_results/degeneracy/robustness_attack_utility.csv` | 6 rows x 10 cols identical |
| OK | `05_results/degeneracy/robustness_attacks.csv` | 14 rows x 13 cols identical |
| OK | `05_results/degeneracy/size_priority_table-dark.png` | 189394 bytes (bytes not compared) |
| OK | `05_results/degeneracy/size_priority_table.csv` | 74 rows x 4 cols identical |
| OK | `05_results/degeneracy/size_priority_table.png` | 193035 bytes (bytes not compared) |
| OK | `05_results/fairness/budget_pareto-dark.png` | 191255 bytes (bytes not compared) |
| OK | `05_results/fairness/budget_pareto.png` | 194549 bytes (bytes not compared) |
| OK | `05_results/fairness/budget_sweep.csv` | 9 rows x 8 cols identical |
| OK | `05_results/fairness/completion_by_size.csv` | 9 rows x 6 cols identical |
| OK | `05_results/fairness/fairness_metrics.csv` | 3 rows x 6 cols identical |
| OK | `05_results/fairness/wait_time_distribution_by_size-dark.png` | 126693 bytes (bytes not compared) |
| OK | `05_results/fairness/wait_time_distribution_by_size.png` | 128488 bytes (bytes not compared) |
| OK | `05_results/models/ablation_importance-dark.png` | 149580 bytes (bytes not compared) |
| OK | `05_results/models/ablation_importance.png` | 152369 bytes (bytes not compared) |
| OK | `05_results/models/ablation_study_results.csv` | 12 rows x 24 cols identical |
| OK | `05_results/models/concept_drift_results.csv` | 880 rows x 5 cols identical |
| OK | `05_results/models/concept_drift_triggers-dark.png` | 149619 bytes (bytes not compared) |
| OK | `05_results/models/concept_drift_triggers.png` | 152147 bytes (bytes not compared) |
| OK | `05_results/models/evaluation_splits.csv` | 12 rows x 9 cols identical |
| OK | `05_results/models/feature_collinearity.csv` | 12 rows x 18 cols identical |
| OK | `05_results/models/online_learning_results.csv` | 2 rows x 3 cols identical |
| OK | `05_results/scaling/scaling_curves-dark.png` | 161155 bytes (bytes not compared) |
| OK | `05_results/scaling/scaling_curves.png` | 163196 bytes (bytes not compared) |
| OK | `05_results/schedulers/estimate_sensitivity-dark.png` | 169731 bytes (bytes not compared) |
| OK | `05_results/schedulers/estimate_sensitivity.csv` | 320 rows x 13 cols identical |
| OK | `05_results/schedulers/estimate_sensitivity.png` | 172279 bytes (bytes not compared) |
| OK | `05_results/schedulers/estimate_sensitivity_summary.csv` | 16 rows x 9 cols identical |
| OK | `05_results/schedulers/fairness_metrics.csv` | 3 rows x 6 cols identical |
| OK | `05_results/schedulers/multi_scheduler_benchmark.csv` | 14 rows x 11 cols identical |
| OK | `05_results/schedulers/multi_scheduler_equivalence.csv` | 6 rows x 16 cols identical |
| OK | `05_results/schedulers/multi_scheduler_runs.csv` | 280 rows x 12 cols identical |
| OK | `05_results/schedulers/multi_scheduler_significance.csv` | 26 rows x 12 cols identical |
| OK | `05_results/schedulers/scheduler_comparison-dark.png` | 224129 bytes (bytes not compared) |
| OK | `05_results/schedulers/scheduler_comparison.png` | 227143 bytes (bytes not compared) |
| OK | `05_results/shap/shap_dependence_avg_free_per_node-dark.png` | 109703 bytes (bytes not compared) |
| OK | `05_results/shap/shap_dependence_avg_free_per_node.png` | 112341 bytes (bytes not compared) |
| OK | `05_results/shap/shap_dependence_can_fit_now-dark.png` | 82628 bytes (bytes not compared) |
| OK | `05_results/shap/shap_dependence_can_fit_now.png` | 84089 bytes (bytes not compared) |
| OK | `05_results/shap/shap_dependence_fragmentation-dark.png` | 111798 bytes (bytes not compared) |
| OK | `05_results/shap/shap_dependence_fragmentation.png` | 114504 bytes (bytes not compared) |
| OK | `05_results/shap/shap_dependence_gpu_fit_ratio-dark.png` | 94357 bytes (bytes not compared) |
| OK | `05_results/shap/shap_dependence_gpu_fit_ratio.png` | 96207 bytes (bytes not compared) |
| OK | `05_results/shap/shap_dependence_job_gpu-dark.png` | 98043 bytes (bytes not compared) |
| OK | `05_results/shap/shap_dependence_job_gpu.png` | 100320 bytes (bytes not compared) |
| OK | `05_results/shap/shap_dependence_max_free_node-dark.png` | 108393 bytes (bytes not compared) |
| OK | `05_results/shap/shap_dependence_max_free_node.png` | 111191 bytes (bytes not compared) |
| OK | `05_results/shap/shap_dependence_node_availability-dark.png` | 89849 bytes (bytes not compared) |
| OK | `05_results/shap/shap_dependence_node_availability.png` | 91842 bytes (bytes not compared) |
| OK | `05_results/shap/shap_dependence_queue_length-dark.png` | 109020 bytes (bytes not compared) |
| OK | `05_results/shap/shap_dependence_queue_length.png` | 111555 bytes (bytes not compared) |
| OK | `05_results/shap/shap_dependence_queue_pressure-dark.png` | 118640 bytes (bytes not compared) |
| OK | `05_results/shap/shap_dependence_queue_pressure.png` | 121335 bytes (bytes not compared) |
| OK | `05_results/shap/shap_dependence_running_jobs-dark.png` | 99847 bytes (bytes not compared) |
| OK | `05_results/shap/shap_dependence_running_jobs.png` | 101379 bytes (bytes not compared) |
| OK | `05_results/shap/shap_dependence_total_free-dark.png` | 97205 bytes (bytes not compared) |
| OK | `05_results/shap/shap_dependence_total_free.png` | 99168 bytes (bytes not compared) |
| OK | `05_results/shap/shap_dependence_variance_free-dark.png` | 106938 bytes (bytes not compared) |
| OK | `05_results/shap/shap_dependence_variance_free.png` | 108958 bytes (bytes not compared) |
| OK | `05_results/shap/shap_explained_rows.csv` | 400 rows x 1 cols identical |
| OK | `05_results/shap/shap_force_0-dark.png` | 139035 bytes (bytes not compared) |
| OK | `05_results/shap/shap_force_0.png` | 141396 bytes (bytes not compared) |
| OK | `05_results/shap/shap_force_1-dark.png` | 139071 bytes (bytes not compared) |
| OK | `05_results/shap/shap_force_1.png` | 141508 bytes (bytes not compared) |
| OK | `05_results/shap/shap_force_2-dark.png` | 141743 bytes (bytes not compared) |
| OK | `05_results/shap/shap_force_2.png` | 144182 bytes (bytes not compared) |
| OK | `05_results/shap/shap_provenance.csv` | 1 rows x 18 cols identical |
| OK | `05_results/shap/shap_summary-dark.png` | 326148 bytes (bytes not compared) |
| OK | `05_results/shap/shap_summary.png` | 335442 bytes (bytes not compared) |
| OK | `05_results/trace_schedulers/tost_power.csv` | 12 rows x 27 cols identical |
| OK | `05_results/trace_schedulers/trace_estimate_quality.csv` | 2 rows x 10 cols identical |
| OK | `05_results/trace_schedulers/trace_fidelity.csv` | 40 rows x 6 cols identical |
| OK | `05_results/trace_schedulers/trace_scheduler_comparison-dark.png` | 336757 bytes (bytes not compared) |
| OK | `05_results/trace_schedulers/trace_scheduler_comparison.png` | 341656 bytes (bytes not compared) |
| OK | `05_results/trace_schedulers/trace_scheduler_equivalence.csv` | 12 rows x 17 cols identical |
| OK | `05_results/trace_schedulers/trace_scheduler_significance.csv` | 88 rows x 15 cols identical |
| OK | `05_results/trace_schedulers/trace_scheduler_summary.csv` | 24 rows x 14 cols identical |
| OK | `05_results/trace_schedulers/trace_scheduler_windows.csv` | 480 rows x 18 cols identical |
| OK | `05_results/traces/real_feature_importance_lanl-dark.png` | 89304 bytes (bytes not compared) |
| OK | `05_results/traces/real_feature_importance_lanl.png` | 90235 bytes (bytes not compared) |
| OK | `05_results/traces/real_feature_importance_sdsc-dark.png` | 90084 bytes (bytes not compared) |
| OK | `05_results/traces/real_feature_importance_sdsc.png` | 91035 bytes (bytes not compared) |
| OK | `05_results/traces/real_pred_vs_actual_lanl-dark.png` | 268492 bytes (bytes not compared) |
| OK | `05_results/traces/real_pred_vs_actual_lanl.png` | 264057 bytes (bytes not compared) |
| OK | `05_results/traces/real_pred_vs_actual_sdsc-dark.png` | 434849 bytes (bytes not compared) |
| OK | `05_results/traces/real_pred_vs_actual_sdsc.png` | 424137 bytes (bytes not compared) |
| OK | `05_results/traces/real_trace_validation.csv` | 8 rows x 7 cols identical |
| OK | `05_results/traces/synthetic_proxy_validation_results.csv` | 2 rows x 4 cols identical |
| OK | `05_results/traces/synthetic_vs_real_comparison-dark.png` | 83106 bytes (bytes not compared) |
| OK | `05_results/traces/synthetic_vs_real_comparison.png` | 84226 bytes (bytes not compared) |
| OK | `05_results/uncertainty/censoring_analysis.csv` | 30 rows x 22 cols identical |
| OK | `05_results/uncertainty/uncertainty_ood_benchmark.csv` | 20 rows x 9 cols identical |
| OK | `05_results/uncertainty/uncertainty_summary-dark.png` | 114181 bytes (bytes not compared) |
| OK | `05_results/uncertainty/uncertainty_summary.png` | 115877 bytes (bytes not compared) |
| OK | `phases_22_30/phase_22_stats/ci_plots-dark.png` | 117697 bytes (bytes not compared) |
| OK | `phases_22_30/phase_22_stats/ci_plots.png` | 119681 bytes (bytes not compared) |
| OK | `phases_22_30/phase_22_stats/stats_summary.csv` | 5 rows x 10 cols identical |
| OK | `phases_22_30/phase_23_sensitivity/ood_failure_modes.csv` | 72 rows x 18 cols identical |
| OK | `phases_22_30/phase_23_sensitivity/ood_heatmap-dark.png` | 316174 bytes (bytes not compared) |
| OK | `phases_22_30/phase_23_sensitivity/ood_heatmap.png` | 319365 bytes (bytes not compared) |
| OK | `phases_22_30/phase_24_extended_schedulers/baseline_comparison.csv` | 14 rows x 10 cols identical |
| OK | `phases_22_30/phase_24_extended_schedulers/novelty_claim.txt` | 168 lines identical |
| OK | `phases_22_30/phase_24_extended_schedulers/scheduler_heatmap-dark.png` | 188772 bytes (bytes not compared) |
| OK | `phases_22_30/phase_24_extended_schedulers/scheduler_heatmap.png` | 191530 bytes (bytes not compared) |
| OK | `phases_22_30/phase_25_real_traces/real_vs_synthetic_comparison-dark.png` | 229692 bytes (bytes not compared) |
| OK | `phases_22_30/phase_25_real_traces/real_vs_synthetic_comparison.png` | 232563 bytes (bytes not compared) |
| OK | `phases_22_30/phase_25_real_traces/trace_inventory.csv` | 1 rows x 7 cols identical |
| OK | `phases_22_30/phase_26_scaling/inference_overhead_plot-dark.png` | 279108 bytes (bytes not compared) |
| OK | `phases_22_30/phase_26_scaling/inference_overhead_plot.png` | 283004 bytes (bytes not compared) |
| OK | `phases_22_30/phase_27_fairness/dropped_schedulers.txt` | 12 lines identical |
| OK | `phases_22_30/phase_27_fairness/fairness_metrics.csv` | 15 rows x 14 cols identical |
| OK | `phases_22_30/phase_27_fairness/sla_compliance.csv` | 15 rows x 7 cols identical |
| OK | `phases_22_30/phase_27_fairness/starvation_analysis-dark.png` | 116464 bytes (bytes not compared) |
| OK | `phases_22_30/phase_27_fairness/starvation_analysis.png` | 118220 bytes (bytes not compared) |

## Timing exemptions

These columns measure wall-clock time -- the machine, not the algorithm
-- and cannot reproduce across machines. A difference in one of them is
reported as TIMING and never fails the run. Every OTHER column of these
same files is compared strictly:

* `05_results/model_comparison_table1.csv`: training_time_sec
* `05_results/scaling/scaling_analysis.csv`: inference_time_sec, proactive_total_sim_time_sec, scheduler_overhead_sec
* `phases_22_30/phase_26_scaling/scaling_benchmark.csv`: inference_latency_ms, throughput_overhead_pct

The same exemption applies to the text reports that RENDER those
columns -- a wall-clock number does not stop being wall-clock because
a script printed it into a .txt. Only the named fields are masked;
every other field on those lines, and every other line, is compared
exactly:

* `phases_22_30/phase_26_scaling/scaling_measurements.txt`: latency and overhead columns of the metrics table
* `phases_22_30/phase_26_scaling/scaling_measurements.txt`: measured latency range
* `phases_22_30/phase_26_scaling/scaling_measurements.txt`: peak overhead
* `phases_22_30/phase_26_scaling/scaling_measurements.txt`: worst latency

135 artefacts: 4 TIMING, 131 OK -- 1368.1 s elapsed

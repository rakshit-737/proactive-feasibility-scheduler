# Reproducibility Guide

## One command

```bash
bash run_all_experiments.sh
```

Twenty steps, one entry point. There is no second script to remember: phases 22–27, which used
to live behind `phases_22_30/run_all_experiments_v2.sh`, now run as step 20, and the five studies
that no pipeline regenerated (the quantile model, the fairness budget sweep, the real-trace
datasets and their validation, the uncertainty benchmark) are steps 3, 6, 11 and 15. That is what
makes "`run_all_experiments.sh` regenerates every result" a true statement rather than an
aspiration. `phases_22_30/run_all_experiments_v2.sh` still exists and forwards here.

Step 1 regenerates `02_data/improved_wait_dataset.csv` and retrains `03_models/wait_model_v2.pkl`
before any analysis runs, so a clean checkout reproduces without relying on pre-committed
artifacts. Every script resolves paths relative to its own location and can be run from any
working directory. The pipeline exports `PYTHONUTF8=1` so Unicode console output survives a
Windows cp1252 shell.

### Choosing the interpreter

```bash
PY=/path/to/python bash run_all_experiments.sh
```

Set `PY` when you care which interpreter runs. Without it the script falls back to
`python3` then `python`, and on Windows a bare `python3` resolves to the Microsoft Store shim
rather than to an activated virtual environment — so an activated venv can be silently ignored.
`tools/verify_artifacts.py` always exports `PY=sys.executable` for this reason.

## What reproduces, and what cannot

Seeds are fixed throughout (global 42, training `42+i`, synthetic evaluation `1000+run`,
estimates `20000+run`, fairness `800+run`, scaling `4000+id+nodes`, budget `5000+i`,
uncertainty `7000+run`, traces `SEED=42`), and evaluation seeds are disjoint from training seeds.
On one machine the pipeline is deterministic: a fresh-clone run reproduced 153 of 164 tracked
artifacts identically, including every published number.

**Across machines the digits move, and the claims do not.** A 2-vCPU Linux runner reproduced
45,268 dispatch instants where this project's reference platform records 45,432 -- and **zero
equal-size / different-score violations on both**. The cause is XGBoost's histogram build, which
reduces floating point in parallel, so the fitted model depends on thread count and library
build; the model drives dispatch decisions, so one perturbation at the root reaches every
downstream count. Measured on the reference machine, same data and same seed: hold-out MAE
4.692659 / 4.669915 / 4.637280 / 4.693508 on 1 / 2 / 4 / 16 threads.

This is why there are **two** verifiers, and why neither replaces the other:

| tool | question it answers | scope |
|---|---|---|
| `tools/verify_artifacts.py` | does this tree regenerate itself, digit for digit? | reference platform |
| `tools/verify_claims.py` | do these artifacts support what the paper says? | any platform |

Quote counts with their scope. `reports/cross_platform_reproduction.md` has the full evidence.

Six columns cannot reproduce, on any machine, because they measure wall-clock time rather than
the algorithm:

| file | columns |
|---|---|
| `05_results/model_comparison_table1.csv` | `training_time_sec` |
| `05_results/scaling/scaling_analysis.csv` | `scheduler_overhead_sec`, `inference_time_sec`, `proactive_total_sim_time_sec` |
| `phases_22_30/phase_26_scaling/scaling_benchmark.csv` | `inference_latency_ms`, `throughput_overhead_pct` |

Between two runs on the same machine these moved by up to 84% while every other column was
bit-identical. No claim in the paper rests on them, and `tools/verify_artifacts.py` reports them
as `TIMING` rather than failing on them.

Runtime: about 9 minutes for the pre-v3.6 fourteen-step pipeline on a laptop; the twenty-step
pipeline is longer because it now includes the studies that were previously never run.

## Verifying that the committed artifacts match the code

```bash
python tools/verify_artifacts.py            # copy the tree, run the pipeline, diff everything
python tools/verify_artifacts.py --quick    # diff the working tree against HEAD, no re-run
python tools/verify_artifacts.py --smoke    # fast end-to-end; existence and CSV headers only
python tools/verify_claims.py               # do the artifacts support the claims? (seconds)
```

Off the reference platform, add `--expect claims`: digit differences are reported as `DRIFT`
and the verdict comes from the claim checks. That is what CI runs, because it is the question a
different machine can actually answer.

The full mode copies every tracked file into a scratch directory, runs the pipeline there, and
compares each regenerated artifact against its committed blob: CSVs cell by cell, text files line
by line, PNGs and pickles by existence and loadability. It exits non-zero on any mismatch. Run
`--quick` after regenerating in place and before committing, so that a number which moved is
something you decided rather than something you shipped.

If you have `make`: `make verify`, `make verify-quick`, `make smoke`, `make claims`, `make test`,
`make lint`.
`make` is optional and every target is a single command you can type directly.

### The tolerance policy

Comparison tolerances (`--rtol` 1e-9) exist to absorb CSV round-trip formatting, not behavioural
drift. Scheduling is discrete: a last-ulp difference in one prediction can flip a queue order and
move a published mean by percent, not by 1e-9. A cross-platform mismatch is therefore a finding to
investigate, never a reason to loosen the tolerance.

That investigation has now happened, and the tolerance was not loosened. The cross-platform
mismatch is real, its cause is understood, and the response was to add a verifier that checks the
claims rather than to widen a number until the check passed. One guard did change: the SHAP
split-reconstruction check now compares against the hold-out score recorded inside the model
bundle instead of a literal copied into its source, because the literal conflated "same split"
with "same machine". Its 5e-4 tolerance is unchanged.

## Environment

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

Python **3.14** — the interpreter the committed artifacts were generated with. The same version is
named in `requirements.txt`, `requirements-dev.txt`, the `Dockerfile` and CI, and
`tests/test_python_pin.py` fails if those four ever disagree again.

```bash
docker build -t proactive-scheduler .
docker run --rm -it proactive-scheduler
```

## Tests

```bash
python -m pytest
python -m ruff check .
```

The suite is fast by design: no test runs a full benchmark. Tests marked `artefact` skip when a
generated file is absent, so the suite passes on a fresh clone before the pipeline has run. The
golden-number tests are the exception — they read committed artifacts and fail rather than skip,
because a missing committed file means a broken checkout, not an un-run pipeline.

## Dashboard

```bash
streamlit run dashboard.py
```

## Where the artifacts land

`05_results/` holds `models/`, `schedulers/`, `degeneracy/`, `trace_schedulers/`, `scaling/`,
`fairness/`, `shap/`, `traces/`, `uncertainty/` and `roi/`. `phases_22_30/phase_2*/` holds the
phase-22 to phase-27 outputs. The degeneracy and trace-scheduler directories hold the headline
artifacts.

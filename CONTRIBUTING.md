# Contributing

Thanks for your interest in the Proactive Feasibility Scheduler. This is a research
codebase, so contributions are welcome but the bar is **reproducibility and honesty of
results** above all else.

## Development setup

```bash
git clone https://github.com/rakshit-737/proactive-feasibility-scheduler.git
cd proactive-feasibility-scheduler
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Reproduce the full study end-to-end (regenerates the dataset, trains the model, and runs
every experiment deterministically):

```bash
bash run_all_experiments.sh
```

A single run of the container is also supported: `docker build -t pfs . && docker run pfs`.

## Repository layout

The pipeline is organized by stage:

- `01_simulation/` — discrete-time cluster simulation
- `02_data/` — dataset generation and real-trace (SWF) loading
- `03_models/` — wait-time model training, ablation, SHAP, drift, online learning
- `04_scheduler/` — schedulers and benchmarks (FIFO, SJF, Priority, Proactive, NN, backfill)
- `05_results/` — generated figures, tables, and CSV outputs
- `phases_22_30/` — research-extension phases and roadmap
- `docs/` — HTML report and project documentation

## Standards

- **Reproducibility first.** Anything that changes a reported number must be reproducible
  from `run_all_experiments.sh` on a clean checkout. The pipeline is seeded end-to-end;
  keep it that way. Do not commit results that cannot be regenerated.
- **Honest metrics.** Report out-of-sample numbers (holdout / cross-validation), never
  in-sample. State trade-offs and negative results plainly, as the current `RESULTS.md`
  does.
- **Syntax must pass CI.** All Python must byte-compile cleanly (`python -m compileall .`);
  this is what the CI checks.
- **Keep large artifacts sensible.** Model pickles and generated CSVs are committed for
  reproducibility; do not add large binaries that the pipeline can regenerate.

## Proposing changes

Use small, focused commits with imperative messages (e.g. `Add Wilcoxon test to
benchmark`, `Fix seed handling in fairness sweep`). Open a pull request describing what
changed and, if it affects results, which command regenerates them. For substantial new
experiments, add a short note to `METHODOLOGY.md` and `RESULTS.md`.

## Questions

Open an issue for bugs, questions, or proposed experiments.

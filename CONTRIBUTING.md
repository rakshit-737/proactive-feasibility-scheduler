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
- `docs/` — the explanatory HTML report
- `tools/` — repository tooling (artifact verification)
- `reports/` — audit reports and the claim inventory

## Standards

- **Reproducibility first.** Anything that changes a reported number must be reproducible from
  `run_all_experiments.sh` on a clean checkout. The pipeline is seeded end-to-end; keep it that
  way, and do not change a seed or a seed formula without saying why in the changelog. Do not
  commit a result that no committed script regenerates: if you add a study, add its producer to
  `run_all_experiments.sh` in the same change, or the artifact does not belong in the repository.
- **Regenerate, never retype.** If a number moves, re-run the producing script and let the new
  value propagate. Never edit a number in prose to match a CSV, and never edit a CSV to match
  prose. `python tools/verify_artifacts.py --quick` tells you exactly which artifacts a change
  moved.
- **Honest metrics.** Report out-of-sample numbers (holdout / cross-validation), never
  in-sample. State trade-offs and negative results plainly, as the current `RESULTS.md`
  does.
- **CI runs the tests.** Every push and pull request byte-compiles the tree, runs
  `ruff check .`, runs `python -m pytest`, and runs a smoke reproduction of the pipeline
  (`python tools/verify_artifacts.py --smoke`) on Python 3.14. A weekly scheduled job runs the
  full artifact verification. Run `python -m pytest` and
  `python tools/verify_artifacts.py --quick` locally before pushing.
- **Keep large artifacts sensible.** Model pickles and generated CSVs are committed for
  reproducibility; do not add large binaries that the pipeline can regenerate.

## Proposing changes

Use small, focused commits with imperative messages (e.g. `Add Wilcoxon test to
benchmark`, `Fix seed handling in fairness sweep`). Open a pull request describing what
changed and, if it affects results, which command regenerates them. For substantial new
experiments, add a short note to `METHODOLOGY.md` and `RESULTS.md`.

## Questions

Open an issue for bugs, questions, or proposed experiments.

# make is OPTIONAL here. Every target below is a single python invocation that
# can be typed directly, because the project's own development environment is
# Windows and has no make: nothing in this file may become the only way to run
# something.
#
#   make verify        python tools/verify_artifacts.py --report verify_report.md
#   make verify-quick  python tools/verify_artifacts.py --quick
#   make smoke         python tools/verify_artifacts.py --smoke
#   make test          python -m pytest
#   make lint          python -m ruff check .

PY ?= python

.PHONY: verify verify-quick smoke test lint

# Full reproduction: copies the tree to scratch, runs run_all_experiments.sh
# there, diffs every artefact against the committed blob. ~9 minutes.
verify:
	$(PY) tools/verify_artifacts.py --report verify_report.md

# No pipeline run: diffs the working tree against HEAD. Seconds.
verify-quick:
	$(PY) tools/verify_artifacts.py --quick

# SMOKE=1 pipeline in scratch; checks that every artefact is produced and that
# CSV headers match. A smoke run writes REDUCED numbers, so values are not
# compared and its outputs must never be committed.
smoke:
	$(PY) tools/verify_artifacts.py --smoke

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

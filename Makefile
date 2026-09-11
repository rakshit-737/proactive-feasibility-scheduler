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
#   make paper         pdflatex x2 in phases_22_30/phase_28_manuscript
#   make paper-check   paper, then fail on any undefined reference

PY ?= python
PAPER_DIR ?= phases_22_30/phase_28_manuscript

.PHONY: verify verify-quick smoke test lint paper paper-check

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

# Build the manuscript. Two passes, because the paper uses \ref/\label
# cross-references and the first pass has not yet written the .aux they read;
# a one-pass build silently emits "??" where every reference should be.
#
# -halt-on-error rather than the default interactive prompt: a build that stops
# at the first error and returns non-zero is checkable, one that waits for input
# hangs a CI runner.
#
# LaTeX build artefacts (.aux, .log, .out, .toc) are gitignored; the PDF is
# tracked, because a reader who clones this repository should not need a TeX
# installation to read the paper.
paper:
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
	@echo "built $(PAPER_DIR)/manuscript.pdf"

# Fails if the paper still has unresolved cross-references or citations. Two
# passes should leave none; if this fires, a \label was renamed or a \cite has
# no matching \bibitem, and the PDF will contain a literal "??" or "[?]".
paper-check: paper
	@cd $(PAPER_DIR) && \
	  if grep -qE "LaTeX Warning: (Reference|Citation) .* undefined" manuscript.log; then \
	    echo "FAIL: undefined reference or citation:"; \
	    grep -E "LaTeX Warning: (Reference|Citation) .* undefined" manuscript.log; \
	    exit 1; \
	  else \
	    echo "OK: no undefined references or citations"; \
	  fi

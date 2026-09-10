#!/usr/bin/env bash
# Deprecated. Phases 22-27 now run from the root pipeline, so the repository
# has a single entry point and "run_all_experiments.sh regenerates every
# result" is a true statement. This shim forwards for backwards compatibility.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/run_all_experiments.sh" "$@"

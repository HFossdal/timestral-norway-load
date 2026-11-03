#!/usr/bin/env bash
# A fourth foundation model: Moirai 2.0 small (Salesforce), noon-before origin, same runs as
# run_timesfm.sh. uni2ts needs torch<2.5, so this uses its own environment: set MOIRAI_PY to
# that environment's python (default: ../timestral-venv-moirai, see README).
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONUNBUFFERED=1
PY=${MOIRAI_PY:-../timestral-venv-moirai/Scripts/python}

run() { echo "=== $(date +%H:%M:%S) $*"; "$PY" -m src.forecast --origin noon_before "$@" || echo "!!! FAILED: $*"; }

run --runs moirai2_temp --winters w2025 --context 2688
run --runs moirai2_temp_cold --winters w2025 --context 2688
run --runs moirai2_temp --winters w2025 --context 672
run --runs moirai2_temp_cold --winters w2025 --context 672
run --runs moirai2 --winters w2025 --context 2688
run --runs moirai2_temp --winters w2024 --context 2688
echo "=== $(date +%H:%M:%S) moirai runs done"

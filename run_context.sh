#!/usr/bin/env bash
# Calibration-winter checks after the 16-week gain was first noticed on the test winter.
# These runs confirm the pattern on 2022-25 and supply the 16-week bias fit; they do not
# constitute an independent, fully held-out choice of context length.
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONUNBUFFERED=1

run() { echo "=== $(date +%H:%M:%S) $*"; python -m src.forecast --origin noon_before "$@" || echo "!!! FAILED: $*"; }

run --runs chronos2_temp --winters w2024 --context 2688
run --runs timestral-209m_temp --winters w2024 --context 2688
run --runs chronos2_temp --winters w2022 w2023 --context 2688
run --runs timestral-209m_temp --winters w2023 w2022 --context 2688
echo "=== $(date +%H:%M:%S) context runs done"

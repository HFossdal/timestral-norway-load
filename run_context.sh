#!/usr/bin/env bash
# Context length chosen without the test winter: the 16-week (2688 h) runs on the calibration
# winters, so the 4 vs 16 week comparison (and the bias fit for the 16-week model) can be made
# on 2022-25 only. Calibration winter 2024-25 first, since it alone decides the choice.
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONUNBUFFERED=1

run() { echo "=== $(date +%H:%M:%S) $*"; python -m src.forecast --origin noon_before "$@" || echo "!!! FAILED: $*"; }

run --runs chronos2_temp --winters w2024 --context 2688
run --runs timestral-209m_temp --winters w2024 --context 2688
run --runs chronos2_temp --winters w2022 w2023 --context 2688
run --runs timestral-209m_temp --winters w2023 w2022 --context 2688
echo "=== $(date +%H:%M:%S) context runs done"

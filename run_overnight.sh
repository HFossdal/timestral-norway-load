#!/usr/bin/env bash
# Overnight robustness runs (noon-before origin), started after run_noon.sh has finished:
# 1. Chronos-2 + temperature on the extra winters -> test the bias fix on a second model
# 2. Noisy temperature on the calibration winters -> calibrate the realistic fix on noisy runs
# 3. Timestral sizes with temperature on the test winter -> model sizes in the main setting
# 4. Context-length sensitivity (8 and 16 weeks vs 4 weeks, the main setting at the time)
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONUNBUFFERED=1

run() { echo "=== $(date +%H:%M:%S) $*"; python -m src.forecast --origin noon_before "$@" || echo "!!! FAILED: $*"; }

run --runs chronos2_temp --winters w2022 w2023
run --runs chronos2_temp_noisy --winters w2022 w2023 w2024
run --runs timestral-209m_temp_noisy --winters w2022 w2023 w2024
run --runs timestral-5m_temp timestral-28m_temp timestral-123m_temp --winters w2025

# 4. Context-length sensitivity (test winter): 4 weeks (then the main setting) vs 8 and 16 weeks.
#    Separate cache files (ctx in the name), so the main results are untouched.
run --runs timestral-209m_temp --winters w2025 --context 2688
run --runs chronos2_temp --winters w2025 --context 2688
run --runs timestral-209m_temp --winters w2025 --context 1344

echo "=== $(date +%H:%M:%S) overnight runs done"

#!/usr/bin/env bash
# Temperature sensitivity probe: the same test-winter forecasts with the future temperature
# 3 °C colder, at 4 and 16 weeks of context. Load response (MW per °C) = (cold - base) / -3.
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONUNBUFFERED=1

run() { echo "=== $(date +%H:%M:%S) $*"; python -m src.forecast --origin noon_before --winters w2025 "$@" || echo "!!! FAILED: $*"; }

run --runs chronos2_temp_cold --context 672
run --runs chronos2_temp_cold --context 2688
run --runs timestral-209m_temp_cold --context 672
run --runs timestral-209m_temp_cold --context 2688
echo "=== $(date +%H:%M:%S) probe runs done"

# Replication on a second winter (calibration winter 2024-25): does the probe pattern hold?
# The plain temp runs for this winter already exist at both context lengths.
run24() { echo "=== $(date +%H:%M:%S) $*"; python -m src.forecast --origin noon_before --winters w2024 "$@" || echo "!!! FAILED: $*"; }

run24 --runs timestral-209m_temp_cold --context 672
run24 --runs timestral-209m_temp_cold --context 2688
run24 --runs chronos2_temp_cold --context 672
run24 --runs chronos2_temp_cold --context 2688
echo "=== $(date +%H:%M:%S) w2024 probe runs done"

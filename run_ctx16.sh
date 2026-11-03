#!/usr/bin/env bash
# Main setting switched to a 16-week context (2,688 h), chosen on the calibration winter 2024-25
# (README, section 3). Fill in the runs the main tables need at 16 weeks, priority first:
# noisy temperature (realistic setting), baselines, and load-only runs (temperature gain).
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONUNBUFFERED=1

run() { echo "=== $(date +%H:%M:%S) $*"; python -m src.forecast --origin noon_before --context 2688 "$@" || echo "!!! FAILED: $*"; }

run --runs timestral-209m_temp_noisy --winters w2025
run --runs chronos2_temp_noisy --winters w2025
run --runs snaive lgbm lgbm_temp --winters w2024 w2025
run --runs chronos2 --winters w2024 w2025
run --runs timestral-209m --winters w2025
echo "=== $(date +%H:%M:%S) ctx16 runs done"

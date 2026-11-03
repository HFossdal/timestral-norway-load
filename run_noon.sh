#!/usr/bin/env bash
# Main setting: noon-before origin (forecast made 12:00 local on D-1, lead 12-35 h),
# matching real day-ahead bidding. The midnight-origin runs (run_experiments.sh) are
# the secondary comparison. One model per process; cached runs are skipped.
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONUNBUFFERED=1

run() { echo "=== $(date +%H:%M:%S) $*"; python -m src.forecast --origin noon_before "$@" || echo "!!! FAILED: $*"; }

CAL_TEST="--winters w2024 w2025"

# Ordered so the runs that decide the calibration fix land first (calibration + test winters).
run --runs snaive lgbm lgbm_temp $CAL_TEST              # baselines (cheap)
run --runs timestral-209m_temp $CAL_TEST
run --runs chronos2_temp $CAL_TEST
run --runs timestral-209m $CAL_TEST
run --runs chronos2 $CAL_TEST

# Robustness (priority): noisy future temperature, test winter
run --runs chronos2_temp_noisy --winters w2025
run --runs timestral-209m_temp_noisy --winters w2025

# Extra winters for Timestral: 3-winter conformal sensitivity run and 4-winter Christmas
run --runs timestral-209m_temp --winters w2022 w2023
run --runs timestral-209m --winters w2022 w2023

# Holiday indicator covariate. Timestral on all 4 winters so the Christmas effect is judged on 4 events.
run --runs chronos2_temp_holiday $CAL_TEST
run --runs timestral-209m_temp_holiday                  # all winters

echo "=== $(date +%H:%M:%S) noon runs done"

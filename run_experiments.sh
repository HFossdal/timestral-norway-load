#!/usr/bin/env bash
# Runs every forecast experiment, one model per process so memory is freed between models.
# Finished runs are cached in results/cache/forecasts/ and skipped on rerun.
# Winters: w2024 = calibration, w2025 = test, w2022/w2023 = extra calibration winters (sensitivity run).
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONUNBUFFERED=1

run() { echo "=== $(date +%H:%M:%S) $*"; python -m src.forecast "$@" || echo "!!! FAILED: $*"; }

CAL_TEST="--winters w2024 w2025"

# Baselines and zero-shot models, load history only
run --runs snaive lgbm                                   # all winters
run --runs chronos2 $CAL_TEST
run --runs timestral-209m                                # all winters (conformal sensitivity)

# Temperature covariate (observed temperature, an oracle)
run --runs lgbm_temp $CAL_TEST
run --runs chronos2_temp $CAL_TEST
run --runs timestral-209m_temp                           # all winters (conformal sensitivity)

# Extras: holiday indicator, noisy temperature (robustness, test winter only)
run --runs chronos2_temp_holiday $CAL_TEST
run --runs timestral-209m_temp_holiday $CAL_TEST
run --runs chronos2_temp_noisy --winters w2025
run --runs timestral-209m_temp_noisy --winters w2025

# Timestral model sizes (test winter)
run --runs timestral-5m timestral-28m timestral-123m --winters w2025

echo "=== $(date +%H:%M:%S) all done"

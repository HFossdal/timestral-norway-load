#!/usr/bin/env bash
# "How much history is enough?": 32 weeks (5,376 h) on the calibration and test winters, to see
# whether the 4 -> 8 -> 16 week gain keeps going or levels off. Timestral first (the main model).
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONUNBUFFERED=1

run() { echo "=== $(date +%H:%M:%S) $*"; python -m src.forecast --origin noon_before --context 5376 "$@" || echo "!!! FAILED: $*"; }

# First: the real archived MET Nordic weather forecasts (temp_fc) in the main 16-week setting,
# test + calibration winter (so recalibration can be fitted on real-forecast runs), both models.
# Answers the oracle-temperature limitation, so it goes before 32 weeks.
python -m src.data_weather_forecast --winters w2025 w2024   # fills any runs the first pass missed
run_fc() { echo "=== $(date +%H:%M:%S) $*"; python -m src.forecast --origin noon_before --context 2688 "$@" || echo "!!! FAILED: $*"; }
run_fc --runs timestral-209m_temp_fc --winters w2025 w2024
run_fc --runs chronos2_temp_fc --winters w2025 w2024

# Fifth winter 2021-22, robustness only (energy-price crisis and COVID measures that winter)
run_w() { echo "=== $(date +%H:%M:%S) $*"; python -m src.forecast --origin noon_before "$@" || echo "!!! FAILED: $*"; }
run_w --context 2688 --runs timestral-209m_temp chronos2_temp --winters w2021
run_w --context 672 --runs timestral-209m_temp chronos2_temp --winters w2021
echo "=== $(date +%H:%M:%S) w2021 runs done"

# Several weather stations per zone, population-weighted (data_weather_multi.py), 16 weeks,
# test winter. Before the 32-week runs: it answers a stated limitation, 32 weeks is a nice-to-have.
run_w --context 2688 --runs timestral-209m_temp_multi chronos2_temp_multi --winters w2025
echo "=== $(date +%H:%M:%S) multi-station test runs done"

run --runs timestral-209m_temp --winters w2024

# Real weather forecasts on two more winters (2022-23, 2023-24), so the real-forecast
# recalibration can be fitted on 3 winters like the rest of the project. Downloads the
# archive first (skips files already cached).
python -m src.data_weather_forecast --winters w2022 w2023
run_fc --runs timestral-209m_temp_fc --winters w2023 w2022
run_fc --runs chronos2_temp_fc --winters w2023 w2022
echo "=== $(date +%H:%M:%S) real-forecast calibration winters done"

# Synthetic load with a known temperature slope (src/synthetic.py): does the probe response
# track the true slope, and does the range of temperatures in the window matter?
echo "=== $(date +%H:%M:%S) synthetic probe"
python -m src.synthetic --models timestral-209m chronos2 || echo "!!! FAILED: synthetic"
echo "=== $(date +%H:%M:%S) synthetic done"

run --runs timestral-209m_temp --winters w2025
run --runs chronos2_temp --winters w2024 w2025
echo "=== $(date +%H:%M:%S) ctx32 runs done"

# Probe extensions (test winter): 8 weeks for Timestral-209M (completes the curve),
# and Timestral-123M at 4 and 16 weeks (does the context dependence hold for a smaller model?)
run_w --context 1344 --runs timestral-209m_temp_cold --winters w2025
run_w --context 672 --runs timestral-123m_temp_cold --winters w2025
run_w --context 2688 --runs timestral-123m_temp timestral-123m_temp_cold --winters w2025
echo "=== $(date +%H:%M:%S) probe extensions done"

# Probe at 32 weeks (Timestral-209M, test winter): the 4/8/16/32 curve
run --runs timestral-209m_temp_cold --winters w2025
echo "=== $(date +%H:%M:%S) 32-week probe done"

# Optional: the calibration winter with several stations (only needed to refit corrections)
run_w --context 2688 --runs timestral-209m_temp_multi chronos2_temp_multi --winters w2024
echo "=== $(date +%H:%M:%S) multi-station runs done"

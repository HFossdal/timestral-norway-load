#!/usr/bin/env bash
# A third foundation model: TimesFM 3.0 (Google), noon-before origin. Priority order: the main
# comparison and the temperature probe on the test winter at 16 and 4 weeks, then load only,
# then the calibration winter.
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONUNBUFFERED=1

run() { echo "=== $(date +%H:%M:%S) $*"; python -m src.forecast --origin noon_before "$@" || echo "!!! FAILED: $*"; }

run --runs timesfm3_temp --winters w2025 --context 2688
run --runs timesfm3_temp_cold --winters w2025 --context 2688
run --runs timesfm3_temp --winters w2025 --context 672
run --runs timesfm3_temp_cold --winters w2025 --context 672
run --runs timesfm3 --winters w2025 --context 2688
run --runs timesfm3_temp --winters w2024 --context 2688

# Second batch: real MET weather forecasts on the test winter (ranking under realistic conditions),
# and two more calibration winters (fit for the one-sided recalibration, 4-winter accuracy).
run --runs timesfm3_temp_fc --winters w2025 --context 2688
# probe on a second winter (calibration winter 2024-25), as for Timestral and Chronos-2 in run_probe.sh
run --runs timesfm3_temp --winters w2024 --context 672
run --runs timesfm3_temp_cold --winters w2024 --context 672
run --runs timesfm3_temp_cold --winters w2024 --context 2688
run --runs timesfm3_temp --winters w2023 --context 2688
run --runs timesfm3_temp --winters w2022 --context 2688

# Third batch: the remaining main runs Timestral and Chronos-2 have. Real forecasts on the
# calibration winters (recalibration on real-forecast runs), the fifth winter 2021-22, 4 weeks on
# the remaining winters (16 vs 4 weeks in every winter), several weather stations per zone.
run --runs timesfm3_temp_fc --winters w2024 --context 2688
run --runs timesfm3_temp_fc --winters w2023 --context 2688
run --runs timesfm3_temp_fc --winters w2022 --context 2688
run --runs timesfm3_temp --winters w2021 --context 2688
run --runs timesfm3_temp --winters w2023 --context 672
run --runs timesfm3_temp --winters w2022 --context 672
run --runs timesfm3_temp --winters w2021 --context 672
run --runs timesfm3_temp_multi --winters w2025 --context 2688
run --runs timesfm3_temp_multi --winters w2024 --context 2688
echo "=== $(date +%H:%M:%S) timesfm runs done"

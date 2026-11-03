# Results: what is where

Everything here is produced by `python -m src.report`, `python -m src.headline` and `python -m src.synthetic --report` from the cached forecasts in `results/cache/` (not in the repo). Each table is saved twice: `.csv` (full precision) and `.md` (rounded, readable).

**Naming.** `<table>_<origin>_ctx<hours>`:
- `noon_before_ctx2688`: main setting (forecast at 12:00 the day before, 16 weeks of history)
- `noon_before_ctx672`: 4 weeks of history (the diagnosis setting)
- `midnight_ctx672`: midnight origin, 4 weeks (secondary comparison)

Tables without a suffix apply to all settings (thresholds, station lists).

## Figures (`figures/`)

| File | Shows | README |
|---|---|---|
| `probe_response.png` | Response to 3 °C colder / historical slope, 4/8/16/32 weeks, Timestral-209M, Timestral-123M, TimesFM 3.0, Chronos-2, Moirai 2.0 | Key figure, section 1 |
| `thumbnail.png` | Link-preview card (1200 x 630): probe response at 4 vs 16 weeks, mean of the five zones, four foundation models | |
| `probe_context.png` | Probe response vs the range of temperatures in the history window (Timestral, TimesFM 3.0, Chronos-2) | Section 1 |
| `headline_cold_misses.png` | Share of cold hours above / below Timestral's 80% band, 4 weeks vs fixes | Section 2 |
| `real_forecast_bands.png` | Timestral-209M, 16 weeks: band misses with measured temperature, real MET forecast, and after recalibration | Section 5 |
| `synthetic_probe.png` | Synthetic load with a known slope: response vs true slope (TimesFM 3.0 on β = 50 only) and vs window range (Timestral, Chronos-2) | Section 1 |
| `fan_cold_snap_NO1_*.png` | Timestral: example fan chart in a cold snap, before / after the fix | |
| `fix_before_after_*.png` | Timestral: coverage before / after the conformal fix, per setting | |
| `error_by_regime_*.png` | MAE by regime and bias by temperature: Timestral, TimesFM 3.0 (noon-before only), Chronos-2, LightGBM | |
| `sizes_*.png` | Timestral 5M-209M: accuracy and coverage vs CPU time | Appendix F |
| `sanity_*.png` | Data checks: load and temperature per zone | |

## Tables (`tables/`), by README section

| README | Tables |
|---|---|
| 1. Mechanism | `probe`, `probe_summary` (mean over zones and 4 vs 16 week change, with CIs; `probe_summary_w2024` = the same on the calibration winter), `probe_context` (+ `probe_context_rows`, one row per zone and origin), `context_novelty`, `historical_slopes`, `synthetic_probe`, `synthetic_confounders` (the added real-data features), `synthetic_length_change` (4 vs 16 weeks, paired) |
| 2. Cold misses | `main_table`, `fix_timestral-209m_temp` (rows "none") |
| 3. More history | `context` (each longer context vs 4 weeks, per winter) |
| 4. One-sided recalibration | `cold_cqr`, `cold_cqr_corrections`, `conformal_*`, `aci` |
| 5. Real weather forecasts | `weather_forecast_error`, `covariates`, `fix_*_temp_fc`, `cold_cqr_fc` (Timestral, Chronos-2, TimesFM 3.0), `cold_cqr_fc_corrections` |
| 6. Comparison | `main_table`, `timestral_vs_chronos2`, `foundation_models_stats`, `foundation_models_diff` (all foundation models, every winter, 4 and 16 weeks, and the real-forecast runs), `cold_cqr_timesfm3`, `covariates`, `peaks`, `reserve`; Statnett's day-ahead forecast: `statnett_stats`, `statnett_diff` (our models minus Statnett, paired), `statnett_scale` (Elhub / ENTSO-E level per winter and zone), `statnett_checks` (timing shift, level offset, correlation per winter and zone) |
| 7. Robustness | `robust_w2021_*` (fifth winter), `multi_station_diff`, `multi_station_stats`, `stations_multi` |
| Appendix | `covariates` (noisy temperature), `fix_*`, `fix_lowo_*`, `fix_params_*` (bias correction), `christmas_*`, `holiday_flag_4winters`, `holiday_fix`, `sizes*`, `ensemble_*` |
| Setup | `thresholds` (regime thresholds), `regime_hours` (hours per regime, winter and zone) |

**Glossary of column names.** `cov80` / `cov90`: share of hours inside the 80% / 90% band. `above80` / `below80`: share above q0.90 / below q0.10 (target 0.10 each). `width80`: mean band width (MW). `bias`: actual minus median (positive = under-forecast). `dMAE`, `dPinball`: paired difference b - a (negative = b better), with `_lo` / `_hi` 95% day-level bootstrap bounds. `_temp_fc`: real MET Nordic weather forecast; `_temp_noisy`: synthetic ±1.5 °C error; `_temp_multi`: several stations per zone; `_temp_cold`: probe run (future 3 °C colder).

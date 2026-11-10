#### Synthetic load with a known slope (corrected technical-review rerun)

Synthetic load combines a daily/weekly profile, a known heating-temperature effect and AR(1) noise, driven by real station temperatures. The original generator restarted noise and slow drift at the beginning of each context, so the overlapping four weeks differed between the two history lengths. Those results have been superseded.

The corrected generator creates one common maximum-history process per zone and noise seed, then slices it by origin and context length. With unmodified temperature, the four-week load and temperature inputs equal the final four weeks of the sixteen-week inputs exactly. For range experiments, temperatures are still squeezed/stretched around each window's own mean; the noise remains matched and future temperatures stay unchanged.

All results below were rerun with five matched noise seeds (0–4), 15 weekly origins from December 2025 to March 2026, and three zones (NO1, NO3, NO4): 225 forecasts per cell. Context differences are paired by model, zone, origin and seed. The 95% intervals use 2,000 crossed seed/origin bootstrap draws, with all zones kept together for each origin. Seed variability is reported separately, rather than treating the five copies of an origin as independent dates. Full per-seed results are in `results/tables/synthetic_seed_summary.*`.

![Corrected synthetic responses to the known slope and window temperature range; five matched seeds with seed/origin bootstrap intervals](results/figures/synthetic_probe.png)

Response divided by the known slope (TimesFM ran β = 50 only):

| Model / history       | β = 25   |   β = 50 | β = 100   | β = 50, range ×0.5 / ×1.5   |
|:----------------------|:---------|---------:|:----------|:----------------------------|
| Timestral, 4 weeks    | 0.99     |     1.01 | 1.00      | 0.92 / 1.00                 |
| Timestral, 16 weeks   | 1.01     |     1.02 | 1.00      | 0.94 / 1.02                 |
| Chronos-2, 4 weeks    | 0.92     |     0.9  | 0.91      | 0.92 / 0.90                 |
| Chronos-2, 16 weeks   | 0.91     |     0.89 | 0.86      | 0.86 / 0.87                 |
| TimesFM 3.0, 4 weeks  | n/a      |     0.96 | n/a       | n/a / n/a                   |
| TimesFM 3.0, 16 weeks | n/a      |     0.93 | n/a       | n/a / n/a                   |

#### Adding real-data features to the matched synthetic histories

At β = 50, the added features are a slow daily AR(1) drift (stationary SD 100 MW), a 10% load dip on 24 December–1 January, a delayed temperature effect, and a mismatch between the single station supplied to the model and the several-station temperature driving load. These are controlled simulations, not estimates of the true load-generating process.

Response divided by the known response for each forecast, across five seeds. The Christmas multiplier also reduces the true temperature response on holiday target hours; this analytical reference is corrected per origin, without changing the inferred model outputs:

| Model / history       |   Clean |   Slow drift |   Christmas dip |
|:----------------------|--------:|-------------:|----------------:|
| Timestral, 4 weeks    |    1.01 |         1    |            1.01 |
| Timestral, 16 weeks   |    1.02 |         1.01 |            1.02 |
| Chronos-2, 4 weeks    |    0.9  |         0.86 |            0.87 |
| Chronos-2, 16 weeks   |    0.89 |         0.85 |            0.86 |
| TimesFM 3.0, 4 weeks  |    0.96 |         0.96 |            0.96 |
| TimesFM 3.0, 16 weeks |    0.93 |         0.91 |            0.91 |

For the delayed effect, load follows exponentially smoothed temperature with α = 1/24 per hour. The correct mean response to a sustained 3 °C drop during the 24-hour horizon is 0.3867 times β, not β. The table normalises by that smaller correct response (the exact value is computed in `src/synthetic.py`). For the station condition, a coherent zone-wide temperature change is assumed.

| Model / condition           |   4 weeks |   16 weeks | Change [95% CI]      | Change across seeds: min / max (SD)   |
|:----------------------------|----------:|-----------:|:---------------------|:--------------------------------------|
| Timestral, delayed effect   |      0.52 |       1.02 | +0.51 [+0.45, +0.57] | +0.50 / +0.52 (0.005)                 |
| Timestral, one station      |      0.6  |       0.7  | +0.10 [+0.07, +0.13] | +0.09 / +0.10 (0.005)                 |
| Chronos-2, delayed effect   |      0.91 |       0.9  | -0.01 [-0.05, +0.03] | -0.02 / -0.00 (0.007)                 |
| Chronos-2, one station      |      0.6  |       0.62 | +0.02 [-0.00, +0.04] | +0.00 / +0.02 (0.008)                 |
| TimesFM 3.0, delayed effect |      0.99 |       0.99 | +0.00 [-0.06, +0.07] | -0.03 / +0.02 (0.021)                 |
| TimesFM 3.0, one station    |      0.66 |       0.64 | -0.02 [-0.05, +0.02] | -0.03 / -0.00 (0.009)                 |

With matched histories, Timestral's delayed-effect response rises from 0.52 to 1.02 of the correct response. This is evidence for a possible mechanism behind the real-data context effect: a delayed covariate effect can be harder to infer from a short window. It does not establish that thermal inertia caused the Norwegian result. Station mismatch also produces a smaller positive context effect (+0.10 [+0.07, +0.13]). Several artificial features can therefore produce context dependence; the simulations leave multiple possible mechanisms for real load.

![Changes in normalised response with longer context, for five features and three models; five seeds with zones clustered by origin](results/figures/synthetic_mechanism.png)

The smoothing time constant is an assumption. The simulations use only one winter's temperatures, three zones and 15 weekly origins with overlapping histories; five noise seeds are not five independent weather histories. A real-data test with a smoothed-temperature input is still needed. Legacy synthetic caches are excluded by the `matched-v2` path and a complete-grid validation; the report refuses to render from partial or mixed caches.

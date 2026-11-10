"""Render the corrected synthetic README section from generated, validated tables.

Run only after python -m src.synthetic --report and python -m src.review.
This deliberately fails if the complete five-seed result is unavailable.
"""
import json
import re

import pandas as pd

from src import ROOT


def one(frame, **filters):
    for key, value in filters.items():
        frame = frame[frame[key] == value]
    if len(frame) != 1:
        raise ValueError(f"expected one documented result for {filters}, got {len(frame)}")
    return frame.iloc[0]


def replace_table(text, marker, rows):
    if text.count(marker) != 1:
        raise ValueError(f"missing or ambiguous README marker: {marker}")
    pos = text.index(marker)
    match = re.search(r"(?m)^\|[^\n]*(?:\n\|[^\n]*)+\n?", text[pos:])
    if match is None:
        raise ValueError(f"no table after {marker}")
    start, end = pos + match.start(), pos + match.end()
    return text[:start] + pd.DataFrame(rows).to_markdown(index=False) + "\n" + text[end:]


def replace_paragraph(text, marker, body):
    if text.count(marker) != 1:
        raise ValueError(f"missing or ambiguous README paragraph: {marker}")
    start = text.index(marker)
    end = text.index("\n\n", start)
    return text[:start] + body + text[end:]


def secondary_tables(text, tables):
    """Refresh secondary prose too: pooled one-day intervals also changed."""
    def interval(r):
        return f"{r.dMAE:+.1f} MW [{r.dMAE_lo:+.1f}, {r.dMAE_hi:+.1f}]"

    stats = pd.read_csv(tables / "multi_station_stats_noon_before_ctx2688.csv")
    diff = pd.read_csv(tables / "multi_station_diff_noon_before_ctx2688.csv")
    rows = []
    for model, name in [("timestral-209m", "Timestral"), ("chronos2", "Chronos-2"), ("timesfm3", "TimesFM 3.0")]:
        for winter, label in [("w2025", "test winter"), ("w2024", "calibration winter")]:
            run = model + "_temp"
            a, b = [one(stats, winter=winter, run=r, regime="all") for r in [run, run + "_multi"]]
            r, cold = [one(diff, winter=winter, comparison=f"{run}_multi vs {run}", regime=g)
                       for g in ["all", "cold"]]
            rows.append({"16 weeks": f"{name}, {label}", "MAE, one station → several": f"{a.MAE:.1f} → {b.MAE:.1f} MW",
                         "Change [95% CI]": f"{r.dMAE_pct:+.1f}% ({interval(r)})",
                         "Cold snaps": f"{cold.dMAE_pct:+.1f}% ({interval(cold)})",
                         "80% coverage": f"{a.cov80:.2f} → {b.cov80:.2f}"})
    text = replace_table(text, "#### Several weather stations per zone", rows)

    cov = pd.read_csv(tables / "covariates_noon_before_ctx2688.csv")
    noisy = one(cov, comparison="timestral-209m_temp_noisy vs timestral-209m_temp", regime="all")
    text = re.sub(r"At 16 weeks it costs Timestral [^\n]+? and drops coverage",
                  f"At 16 weeks it costs Timestral {interval(noisy)} and drops coverage", text)

    peaks = pd.read_csv(tables / "peaks_noon_before_ctx2688.csv")
    peaks4 = pd.read_csv(tables / "peaks_noon_before_ctx672.csv")
    peak_rows = [one(peaks4 if short else peaks, run=run, regime="cold") for run, short in
                 [("timestral-209m_temp", False), ("timestral-209m_temp", True),
                  ("chronos2_temp", False), ("lgbm_temp", False), ("timesfm3_temp", False)]]
    t, t4, c, lgb, tf = peak_rows
    text = replace_paragraph(text, "**B. Daily peaks on cold-snap days", (
        f"**B. Daily peaks on cold-snap days ({int(t.zone_days)} zone-days).** Share of days with the actual peak-hour "
        f"load above q0.90 (target 10%): Timestral 16 weeks {100*t.short_at_peak:.0f}% "
        f"[{100*t.short_at_peak_lo:.0f}, {100*t.short_at_peak_hi:.0f}], 4 weeks {100*t4.short_at_peak:.0f}%, "
        f"Chronos-2 {100*c.short_at_peak:.0f}%, LightGBM {100*lgb.short_at_peak:.0f}%, TimesFM 3.0 "
        f"{100*tf.short_at_peak:.0f}% [{100*tf.short_at_peak_lo:.0f}, {100*tf.short_at_peak_hi:.0f}]. Peak bias: "
        + ", ".join(f"{r.peak_bias:+.0f}" for r in peak_rows) + f" MW. Timestral gets the peak hour right "
        f"(within ±1 h) on {100*t.hour_hit:.0f}% of days (`peaks`)."))

    t, c = [one(pd.read_csv(tables / f"fix_lowo_{model}_temp_noon_before_ctx672.csv"), regime="cold")
            for model in ["timestral-209m", "chronos2"]]
    text = replace_paragraph(text, "Fitting on the single mild calibration winter", (
        "Fitting on the single mild calibration winter over-corrects (Timestral cold bias -43 MW), which is why "
        f"the 3-winter fit became the main one. With retrospective leave-one-winter-out (4 winters, {int(t.zone_days)} "
        f"cold zone-days), Timestral at 4 weeks goes from {t.bias_before:+.0f} to {t.bias_after:+.0f} MW "
        f"(cold-snap MAE {interval(t)}) and Chronos-2 from {c.bias_before:+.0f} to {c.bias_after:+.0f} MW "
        f"({interval(c)}). Chronos-2's interval includes zero after the pairing and within-winter bootstrap corrections. "
        "At 16 weeks Timestral's fitted slope is near zero, so there is little bias left to fix. The five-winter "
        "version gives the same qualitative picture; these folds can fit on later winters."))

    flag = one(pd.read_csv(tables / "holiday_flag_4winters_noon_before_ctx672.csv"),
               comparison="timestral-209m_temp_holiday vs timestral-209m_temp", regime="holiday")
    shift = one(pd.read_csv(tables / "holiday_fix_noon_before_ctx2688.csv"), model="timestral-209m_temp", variant="by type")
    xmas = one(pd.read_csv(tables / "christmas_per_winter_timestral-209m_temp_noon_before_ctx2688.csv"), winter="w2025")
    text = replace_paragraph(text, "**E. Holidays.**", (
        f"**E. Holidays.** Over Christmas and New Year the models over-forecast (Timestral, 16 weeks, test winter: "
        f"bias {xmas.bias:+.0f} MW, coverage {xmas.cov80:.2f}). A 0/1 holiday covariate doesn't help on holidays "
        f"(4 weeks, {int(flag.n_days)} holiday days over four winters: {interval(flag)}). A retrospective shift by "
        "holiday type learned from other winters (Christmas, romjul, New Year, other) changes holiday MAE by "
        f"{shift.dMAE_pct:+.1f}% ({interval(shift)}). With effectively four Christmases the interval includes zero, "
        "so this is suggestive but not established."))

    ens = one(pd.read_csv(tables / "ensemble_diff_noon_before_ctx2688.csv"), winter="w2025",
              comparison="ensemble vs timestral-209m_temp", regime="all")
    text = replace_paragraph(text, "**G. A simple ensemble doesn't help.**", (
        "**G. A simple ensemble doesn't help.** Averaging Timestral's and Chronos-2's quantiles beats Chronos-2 "
        f"in every winter but not Timestral alone ({interval(ens)} on the test winter), and it brings back part "
        "of Chronos-2's cold bias."))
    return text


def real_tables(text, tables):
    """Keep the headline Markdown numbers tied to the regenerated CSVs."""
    review = pd.read_csv(tables / "review_bootstrap.csv")
    review = review[review.block_days == 1]
    labels = {"w2022": "2022-23", "w2023": "2023-24", "w2024": "2024-25 (calibration)",
              "w2025": "2025-26 (test)", "w2021": "2021-22 (robustness)"}
    def interval(r):
        return f"{r.dMAE:+.1f} [{r.dMAE_lo:+.1f}, {r.dMAE_hi:+.1f}]"
    rows = []
    for w, label in labels.items():
        row = {"Winter": label}
        for m, column in [("timestral-209m", "Timestral, all"), ("chronos2", "Chronos-2, all"),
                          ("timesfm3", "TimesFM 3.0, all")]:
            comparison = f"{m}_temp@ctx2688 vs {m}_temp"
            r = one(review, kind="context", winter=w, comparison=comparison, regime="all")
            row[column] = (f"{r.dMAE_pct:+.0f}% ({interval(r)} MW)" if m == "timestral-209m"
                           else f"{r.dMAE_pct:+.0f}%")
            if m == "timestral-209m":
                cold = one(review, kind="context", winter=w, comparison=comparison, regime="cold")
                row["Timestral, cold snaps"] = f"{cold.dMAE_pct:+.0f}%"
        rows.append(row)
    text = replace_table(text, "Change in MAE, 16 vs 4 weeks of history (paired, per winter).", rows)

    head = pd.read_csv(tables / "timestral_vs_chronos2_noon_before_ctx2688.csv")
    rows = []
    for w, label in list(labels.items()) + [("w2025 real forecast", "2025-26, real MET forecast")]:
        r, c = one(head, winter=w, regime="all"), one(head, winter=w, regime="cold")
        rows.append({"Winter": label, "MAE, all hours": f"{r.dMAE_pct:+.1f}% ({interval(r)} MW)",
                     "MAE, cold snaps": f"{c.dMAE_pct:+.1f}% ({interval(c)} MW)"})
    text = replace_table(text, "Timestral vs Chronos-2, paired, 16 weeks, per winter:", rows)

    rows = []
    for w, label in labels.items():
        row = {"TimesFM 3.0 minus Timestral, MW [95% CI]": label}
        for ctx, suffix, col in [(672, "_temp", "4 weeks"), (2688, "_temp", "16 weeks"),
                                 (2688, "_temp_fc", "16 weeks, real MET forecast")]:
            if w == "w2021" and suffix == "_temp_fc":
                row[col] = "no forecast archive"
            else:
                r = one(review, kind="model", winter=w, context_h=ctx, temperature=suffix,
                        comparison=f"timesfm3{suffix} vs timestral-209m{suffix}", regime="all")
                row[col] = interval(r)
        rows.append(row)
    text = replace_table(text, "With 4 weeks of history, the original one-day intervals", rows)

    fc = pd.read_csv(tables / "cold_cqr_fc_noon_before_ctx2688.csv")
    fc = fc[fc.model == "timestral-209m_temp_fc"]
    rows = []
    cases = [("Test winter, no fix", "3 winters", "none"),
             ("Test winter, one-sided, fit on 2022-25", "3 winters", "CQR one-sided, temp bins"),
             ("4 winters (leave one out), no fix", "lowo", "none"),
             ("4 winters (leave one out), one-sided", "lowo", "CQR one-sided, temp bins")]
    for label, cal, variant in cases:
        r, cold, deep = [one(fc, calibration=cal, variant=variant, regime=reg)
                         for reg in ["all", "cold", "below_m10"]]
        upper = f"{100*cold.above80:.1f}%"
        if cal == "lowo" and variant != "none":
            upper += f" [{100*cold.above_lo:.0f}, {100*cold.above_hi:.0f}]"
        lower = f"{100*deep.above80:.0f}%"
        if variant != "none":
            lower += f" [{100*deep.above_lo:.0f}, {100*deep.above_hi:.0f}]"
        rows.append({"Timestral, real MET forecast": label, "80% coverage": f"{r.cov80:.2f}",
                     "Above / below, cold snaps": upper + f" / {100*cold.below80:.1f}%",
                     "Above / below, below -10 °C": lower + f" / {100*deep.below80:.0f}%",
                     "Band width": f"{r.width80:.0f} MW"})
    text = replace_table(text, "#### Recalibrating on real-forecast runs", rows)
    deep = one(fc, calibration="lowo", variant="CQR one-sided, temp bins", regime="below_m10")
    text = re.sub(r"and on \d+% \[[^]]+\] of hours below -10 °C",
                  f"and on {100*deep.above80:.0f}% [{100*deep.above_lo:.0f}, {100*deep.above_hi:.0f}] of hours below -10 °C", text)

    cal = pd.read_csv(tables / "review_calibration_bootstrap.csv")
    rows = []
    for suffix, regime, label in [("_temp", "cold", "Observed temperature, cold snaps"),
                                  ("_temp_fc", "cold", "Real forecast, cold snaps"),
                                  ("_temp_fc", "below_m10", "Real forecast, below -10 °C")]:
        for block in [1, 3, 7]:
            r = one(cal, model="timestral-209m" + suffix, context_h=2688, calibration="lowo",
                    variant="CQR one-sided, temp bins", regime=regime, block_days=block)
            rows.append({"Timestral, 16 weeks, four left-out winters": label, "Block days": block,
                         "Above band, % [95% CI]": f"{100*r.above80:.1f} [{100*r.above_lo:.1f}, {100*r.above_hi:.1f}]",
                         "Below band, % [95% CI]": f"{100*r.below80:.1f} [{100*r.below_lo:.1f}, {100*r.below_hi:.1f}]"})
    marker = "#### Calibration intervals with longer day blocks"
    section = (marker + "\n\nThe miss-rate conclusions also survive 1-, 3- and 7-day blocks. After one-sided "
               "temperature-bin recalibration, the observed-temperature cold-snap tail intervals contain the 10% "
               "target for all three models at 16 weeks. That is compatibility with the target, not a guarantee "
               "of calibration on unseen weather. With real forecasts, cold-snap and deep-cold upper misses "
               "remain above 10% at all three block lengths, for Timestral, Chronos-2 and TimesFM.\n\n" +
               pd.DataFrame(rows).to_markdown(index=False) + "\n\n" +
               "The complete `results/tables/review_calibration_bootstrap.*` has 567 rows, including both context "
               "lengths with observed weather and the test-winter and retrospective leave-one-winter-out fits. "
               "Draws retain models/zones together and stay within winters. These intervals condition on the "
               "existing fitted corrections; calibration coefficients are not refitted in each replicate.\n\n")
    end = text.index("### 6. Comparison with the other models")
    start = text.index(marker) if marker in text else end
    text = text[:start] + section + text[end:]

    stats = pd.read_csv(tables / "statnett_stats_noon_before_ctx2688.csv")
    diff = pd.read_csv(tables / "statnett_diff_noon_before_ctx2688.csv")
    comparison = "timestral-209m_temp_fc vs statnett"
    rows = []
    for w in ["w2022", "w2023", "w2024", "w2025"]:
        row = {"Winter, MAE (MW), real weather forecasts": labels[w]}
        for run, name in [("statnett", "Statnett"), ("timestral-209m_temp_fc", "Timestral"),
                          ("chronos2_temp_fc", "Chronos-2"), ("timesfm3_temp_fc", "TimesFM 3.0")]:
            row[name] = f"{one(stats, winter=w, run=run, zone='ALL', regime='all').MAE:.1f}"
        row["Timestral minus Statnett [95% CI]"] = interval(one(diff, winter=w, comparison=comparison, regime="all"))
        rows.append(row)
    text = replace_table(text, "The fixed 2024-25 scaling factors", rows)
    pooled = one(diff, winter="all winters", comparison=comparison, regime="cold")
    text = re.sub(r"and in cold snaps pooled over the four winters \([^)]+\)",
                  f"and in cold snaps pooled over the four winters ({interval(pooled)} MW)", text)
    cold = one(diff, winter="w2023", comparison=comparison, regime="cold")
    text = re.sub(r"though not in the cold snaps of 2023-24 \([^)]+\)",
                  f"though not in the cold snaps of 2023-24 ({interval(cold)} MW)", text)
    return secondary_tables(text, tables)


def main():
    tables = ROOT / "results/tables"
    manifest = json.loads((tables / "synthetic_manifest.json").read_text(encoding="utf-8"))
    if manifest["cache_version"] != "matched-v2" or manifest["seeds"] != list(range(5)) or manifest["rows"] != 11250:
        raise ValueError("complete matched-v2 results required before updating claims")
    d = pd.read_csv(tables / "synthetic_probe.csv")
    ch = pd.read_csv(tables / "synthetic_length_change.csv").set_index(["model", "confounder"])
    names = {"timestral-209m": "Timestral", "chronos2": "Chronos-2", "timesfm3": "TimesFM 3.0"}

    def ratio(m, T, beta=50, k=1., conf="none"):
        r = d[(d.model == m) & (d.context_h == T) & (d.beta == beta) & (d.k == k) & (d.confounder == conf)]
        return f"{r.iloc[0].ratio:.2f}" if len(r) else "n/a"

    clean, features, mechanisms = [], [], []
    for m, name in names.items():
        for T in [672, 2688]:
            clean.append({"Model / history": f"{name}, {T // 168} weeks",
                          **{f"β = {b}": ratio(m, T, b) for b in [25, 50, 100]},
                          "β = 50, range ×0.5 / ×1.5": f"{ratio(m, T, k=.5)} / {ratio(m, T, k=1.5)}"})
            features.append({"Model / history": f"{name}, {T // 168} weeks",
                             "Clean": ratio(m, T), "Slow drift": ratio(m, T, conf="drift"),
                             "Christmas dip": ratio(m, T, conf="christmas")})
        for conf, label in [("lag", "delayed effect"), ("station", "one station")]:
            r = ch.loc[m, conf]
            mechanisms.append({"Model / condition": f"{name}, {label}", "4 weeks": f"{r.ratio_4w:.2f}",
                               "16 weeks": f"{r.ratio_16w:.2f}",
                               "Change [95% CI]": f"{r.change:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]",
                               "Change across seeds: min / max (SD)": f"{r.seed_min:+.2f} / {r.seed_max:+.2f} ({r.seed_sd:.3f})"})
    lag = ch.loc["timestral-209m", "lag"]
    if lag.lo > 0:
        conclusion = (f"With matched histories, Timestral's delayed-effect response rises from {lag.ratio_4w:.2f} "
                      f"to {lag.ratio_16w:.2f} of the correct response. This is evidence for a possible mechanism "
                      "behind the real-data context effect: a delayed covariate effect can be harder to infer "
                      "from a short window. It does not establish that thermal inertia caused the Norwegian result.")
    else:
        conclusion = (f"After matching histories, Timestral's delayed-effect response changes by {lag.change:+.2f} "
                      f"[{lag.lo:+.2f}, {lag.hi:+.2f}]. The earlier claim that this simulation reproduced the "
                      "real-data context effect is withdrawn. A delayed response remains a possible mechanism, "
                      "but these corrected results do not establish a positive context effect for it.")
    station = ch.loc["timestral-209m", "station"]
    if station.lo > 0:
        conclusion += (f" Station mismatch also produces a smaller positive context effect "
                       f"({station.change:+.2f} [{station.lo:+.2f}, {station.hi:+.2f}]). "
                       "Several artificial features can therefore produce context dependence; the simulations "
                       "leave multiple possible mechanisms for real load.")
    section = """#### Synthetic load with a known slope (corrected technical-review rerun)

Synthetic load combines a daily/weekly profile, a known heating-temperature effect and AR(1) noise, driven by real station temperatures. The original generator restarted noise and slow drift at the beginning of each context, so the overlapping four weeks differed between the two history lengths. Those results have been superseded.

The corrected generator creates one common maximum-history process per zone and noise seed, then slices it by origin and context length. With unmodified temperature, the four-week load and temperature inputs equal the final four weeks of the sixteen-week inputs exactly. For range experiments, temperatures are still squeezed/stretched around each window's own mean; the noise remains matched and future temperatures stay unchanged.

All results below were rerun with five matched noise seeds (0–4), 15 weekly origins from December 2025 to March 2026, and three zones (NO1, NO3, NO4): 225 forecasts per cell. Context differences are paired by model, zone, origin and seed. The 95% intervals use 2,000 crossed seed/origin bootstrap draws, with all zones kept together for each origin. Seed variability is reported separately, rather than treating the five copies of an origin as independent dates. Full per-seed results are in `results/tables/synthetic_seed_summary.*`.

![Corrected synthetic responses to the known slope and window temperature range; five matched seeds with seed/origin bootstrap intervals](results/figures/synthetic_probe.png)

Response divided by the known slope (TimesFM ran β = 50 only):

""" + pd.DataFrame(clean).to_markdown(index=False) + """

#### Adding real-data features to the matched synthetic histories

At β = 50, the added features are a slow daily AR(1) drift (stationary SD 100 MW), a 10% load dip on 24 December–1 January, a delayed temperature effect, and a mismatch between the single station supplied to the model and the several-station temperature driving load. These are controlled simulations, not estimates of the true load-generating process.

Response divided by the known response for each forecast, across five seeds. The Christmas multiplier also reduces the true temperature response on holiday target hours; this analytical reference is corrected per origin, without changing the inferred model outputs:

""" + pd.DataFrame(features).to_markdown(index=False) + """

For the delayed effect, load follows exponentially smoothed temperature with α = 1/24 per hour. The correct mean response to a sustained 3 °C drop during the 24-hour horizon is 0.3906 times β, not β. The table normalises by that smaller correct response (the exact value is computed in `src/synthetic.py`). For the station condition, a coherent zone-wide temperature change is assumed.

""" + pd.DataFrame(mechanisms).to_markdown(index=False) + "\n\n" + conclusion + """

![Changes in normalised response with longer context, for five features and three models; five seeds with zones clustered by origin](results/figures/synthetic_mechanism.png)

The smoothing time constant is an assumption. The simulations use only one winter's temperatures, three zones and 15 weekly origins with overlapping histories; five noise seeds are not five independent weather histories. A real-data test with a smoothed-temperature input is still needed. Legacy synthetic caches are excluded by the `matched-v2` path and a complete-grid validation; the report refuses to render from partial or mixed caches.

"""
    # Keep the documented normaliser in sync with the defining equation without
    # importing models or their inference dependencies into this renderer.
    factor = 1 - sum((1 - 1 / 24) ** h for h in range(1, 25)) / 24
    section = section.replace("0.3906", f"{factor:.4f}")
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    start = text.index("#### Synthetic load with a known slope")
    end = text.index("### 2. Short history", start)
    text = text[:start] + section + text[end:]
    text = real_tables(text, tables)
    path.write_text(text, encoding="utf-8")
    (tables / "synthetic_review_summary.md").write_text(section.rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

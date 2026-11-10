"""Technical-review bootstrap sensitivity from cached forecasts; no model inference.

Run: python -m src.review
All 1/3/7-day results are reported, with no method selected by its outcome.
"""
import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd

from src import ROOT, load_config
from src.report import (ALL_WINTERS, MAIN_WINTERS, COLORS, INK2, fc_missing_days,
                        load_ctx, paired_diff, probe_summary, save, save_fig, style)

MODELS = ["timestral-209m", "chronos2", "timesfm3"]
BLOCKS = [1, 3, 7]


def probe_sensitivity(cfg, load, temp):
    """Check the headline covariate-response effect with the same block lengths."""
    parts = []
    for w in ["w2025", "w2024"]:
        for block in BLOCKS:
            d = probe_summary(cfg, "noon_before", load, temp, [w], block_days=block, model_contrasts=True)
            have = set(d.loc[d.stat == "change from 4 weeks", "model"])
            if not set(MODELS) <= have:
                raise ValueError(f"missing paired 4/16-week probe caches for {w}: {set(MODELS) - have}")
            parts.append(d.assign(winter=w, block_days=block))
            print(f"probe sensitivity: {w}, {block}-day blocks complete", flush=True)
    d = pd.concat(parts, ignore_index=True)
    save(d.set_index(["winter", "block_days", "model", "context_h", "stat"]), "review_probe_bootstrap", cfg)
    plot = d[(d.stat == "change from 4 weeks") & d.model.isin(MODELS)]
    fig, ax = plt.subplots(figsize=(9, 4.7))
    labels = []
    names = {"timestral-209m": "Timestral", "chronos2": "Chronos-2", "timesfm3": "TimesFM 3.0"}
    for i, ((winter, model), g) in enumerate(plot.groupby(["winter", "model"])):
        labels.append(f"{names[model]}, {winter[1:]}-{int(winter[1:]) + 1}")
        for j, block in enumerate(BLOCKS):
            r = g[g.block_days == block].iloc[0]
            y = i + (j - 1) * .22
            ax.hlines(y, r.lo, r.hi, color=COLORS[j], lw=1.5)
            ax.plot(r.value, y, "o", color=COLORS[j], label=f"{block}-day blocks" if i == 0 else None)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Change in normalised temperature response, 16 minus 4 weeks")
    ax.legend(frameon=False)
    style(ax, "Temperature probe: 95% paired intervals, all five zones together")
    fig.tight_layout()
    save_fig(fig, "probe_bootstrap_sensitivity", cfg)
    manifest = {"resamples": cfg["bootstrap_n"], "seed": 0, "block_days": BLOCKS, "rows": len(d),
                "winters": ["w2025", "w2024"], "missing_primary_probe_caches": [],
                "pairing": "winter, zone, origin and target hour; models and zones share resampled days",
                "model_interactions": "Timestral 16-minus-4-week change minus each comparison model's paired change",
                "inference_rerun": False}
    (ROOT / "results/tables/review_probe_bootstrap_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(d[d.stat != "mean ratio"].to_string(index=False))


def refresh_existing(cfg, load, temp):
    """Refresh affected one-day paired tables, without replacing unrelated outputs."""
    from src import report as r
    def write_existing(frame, name):
        if (ROOT / "results/tables" / (name + ".csv")).exists() and not frame.empty:
            save(frame, name, cfg)
            print("refreshed", name, flush=True)
    for origin, ctx in [("noon_before", 2688), ("noon_before", 672), ("midnight", 672)]:
        local = {**cfg, "context_hours": ctx}
        suffix = f"_{origin}_ctx{ctx}"
        m = r.prepare(r.load_runs(local, origin), load, temp, local)
        write_existing(r.covariate_table(m, local), "covariates" + suffix)
        write_existing(r.context_table(local, origin, load, temp), "context" + suffix)
        for base, fn in [("timestral_vs_chronos2", r.head_to_head_table),
                         ("holiday_fix", r.holiday_fix_table)]:
            if (ROOT / "results/tables" / (base + suffix + ".csv")).exists():
                write_existing(fn(local, origin, load, temp, ctx), base + suffix)
        if (ROOT / "results/tables" / ("foundation_models_diff" + suffix + ".csv")).exists():
            stats, diff = r.foundation_models_table(local, origin, load, temp)
            write_existing(stats, "foundation_models_stats" + suffix)
            write_existing(diff, "foundation_models_diff" + suffix)
        if (ROOT / "results/tables" / ("statnett_diff" + suffix + ".csv")).exists():
            stats, diff, scale, checks = r.statnett_table(local, origin, load, temp, ctx)
            for frame, base in [(stats, "statnett_stats"), (diff, "statnett_diff"),
                                (scale, "statnett_scale"), (checks, "statnett_checks")]:
                write_existing(frame, base + suffix)
        if (ROOT / "results/tables" / ("ensemble_diff" + suffix + ".csv")).exists():
            diff, stats = r.ensemble_table(local, origin, load, temp, ctx)
            write_existing(stats, "ensemble_stats" + suffix)
            write_existing(diff, "ensemble_diff" + suffix)
        if (ROOT / "results/tables" / ("multi_station_diff" + suffix + ".csv")).exists():
            diff, stats = r.multi_station_table(local, origin, load, temp, ctx)
            write_existing(diff, "multi_station_diff" + suffix)
            write_existing(stats, "multi_station_stats" + suffix)
    refresh_probes(cfg, load, temp)


def refresh_probes(cfg, load, temp):
    """Refresh origin-paired probes and novelty tables after the DST pairing fix."""
    from src import report as r
    for origin, ctx in [("noon_before", 2688), ("noon_before", 672), ("midnight", 672)]:
        local = {**cfg, "context_hours": ctx}
        suffix = f"_{origin}_ctx{ctx}"
        for prefix, fn in [("probe", r.probe_table), ("probe_summary", r.probe_summary),
                           ("context_novelty", r.novelty_table)]:
            if (ROOT / "results/tables" / (prefix + suffix + ".csv")).exists():
                save(fn(local, origin, load, temp), prefix + suffix, cfg)
                print("refreshed", prefix + suffix, flush=True)
        name = "probe_summary_w2024" + suffix
        if (ROOT / "results/tables" / (name + ".csv")).exists():
            save(r.probe_summary(local, origin, load, temp, ["w2024"]), name, cfg)
    for origin in ["noon_before", "midnight"]:
        name = "robust_w2021_novelty_" + origin
        if (ROOT / "results/tables" / (name + ".csv")).exists():
            save(r.novelty_table(cfg, origin, load, temp, winters=ALL_WINTERS), name, cfg)


def refresh_forecast_calibration(cfg, load, temp):
    """Recompute only tables affected by the available-weather grouping fix."""
    from src import report as r
    for origin, ctx in [("noon_before", 2688), ("noon_before", 672), ("midnight", 672)]:
        local = {**cfg, "context_hours": ctx}
        suffix = f"_{origin}_ctx{ctx}"
        m = r.prepare(r.load_runs(local, origin), load, temp, local)
        m = r.add_noisy_day_temp(m, local)
        if m.run.str.endswith("_temp_fc").any():
            m = r.add_fc_day_temp(m, local)
            m = m[~(m.run.str.endswith("_temp_fc") & m.target_day.isin(r.fc_missing_days(m)))]
        runs = set(m.run)
        for run in ["timestral-209m_temp_fc", "chronos2_temp_fc"]:
            if run in runs:
                tab, corr = r.conformal_table(m, run, local)
                save(tab.set_index(["variant", "calibration", "regime"]), f"conformal_{run}{suffix}", local)
                save(corr, f"conformal_corrections_{run}{suffix}", local)
        for name, run, cal_run, tcol in r.FIX_RUNS:
            if tcol == "day_temp" or not {run, cal_run} <= runs:
                continue
            tab, params = r.fix_table(m[m.run.isin({run, cal_run})], run, cal_run, local, tcol)
            if params.empty:
                continue
            save(tab.set_index(["variant", "calibration", "regime"]), f"fix_{run}{name}{suffix}", local)
            save(params, f"fix_params_{run}{name}{suffix}", local)
            if run == cal_run and m.loc[m.run == run, "winter"].nunique() == 4:
                save(r.lowo_table(m, run, local, tcol), f"fix_lowo_{run}{suffix}", local)
            print("refreshed available-weather calibration", run, name, suffix, flush=True)


def calibration_sensitivity(cfg, load, temp):
    """Check one-sided miss-rate intervals too, without repeating model inference."""
    from src import report as r
    runs = [m + "_temp" for m in MODELS]
    forecast = r.load_ctx(cfg, "noon_before", MAIN_WINTERS, 2688, r.FM_FC_RUNS, load, temp)
    forecast = r.add_fc_day_temp(forecast, cfg)
    forecast = forecast[~forecast.target_day.isin(r.fc_missing_days(forecast))]
    rows = []
    for block in BLOCKS:
        observed, _ = r.cold_cqr_table(cfg, "noon_before", load, temp, runs=runs, block_days=block)
        observed = observed.rename(columns={"eval": "calibration"}).assign(temperature="_temp", block_days=block)
        actual, _ = r.cold_cqr_fc_table(forecast, cfg, block_days=block)
        actual = actual.assign(temperature="_temp_fc", context_h=2688, block_days=block)
        rows.extend([observed, actual])
        print("completed calibration miss-rate sensitivity", block, "day blocks", flush=True)
    d = pd.concat(rows, ignore_index=True)
    save(d.set_index(["temperature", "model", "context_h", "calibration", "variant", "regime", "block_days"]),
         "review_calibration_bootstrap", cfg)
    meta = {"resamples": cfg["bootstrap_n"], "seed": 0, "block_days": BLOCKS, "rows": len(d),
            "inference_rerun": False,
            "method": "consecutive calendar-day blocks within winters; shared draws across models and zones",
            "conditioning": "existing fitted corrections; calibration fits are not refitted in bootstrap replicates"}
    (ROOT / "results/tables/review_calibration_bootstrap_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--refresh-existing", action="store_true", help="regenerate affected original one-day tables")
    p.add_argument("--refresh-probes", action="store_true", help="regenerate only origin-paired probes and novelty tables")
    p.add_argument("--probe-only", action="store_true", help="run 1/3/7-day sensitivity for the temperature-response probe only")
    p.add_argument("--calibration-only", action="store_true", help="refresh forecast/noisy calibration groups from available weather")
    p.add_argument("--calibration-bootstrap", action="store_true", help="check one-sided calibration miss rates with 1/3/7-day blocks")
    args = p.parse_args()
    cfg = load_config()
    load = pd.read_parquet(ROOT / "data/processed/load.parquet")
    temp = pd.read_parquet(ROOT / "data/processed/temperature.parquet")
    if args.calibration_bootstrap:
        calibration_sensitivity(cfg, load, temp)
        return
    if args.calibration_only:
        refresh_forecast_calibration(cfg, load, temp)
        return
    if args.probe_only:
        probe_sensitivity(cfg, load, temp)
        return
    if args.refresh_probes:
        refresh_probes(cfg, load, temp)
        return
    if args.refresh_existing:
        refresh_existing(cfg, load, temp)
        return
    missing, tables, cached = [], [], {}
    cache_dir = ROOT / "results/cache/forecasts"
    for w in ALL_WINTERS:
        for ctx, suffix in [(672, "_temp"), (2688, "_temp"), (2688, "_temp_fc")]:
            if suffix == "_temp_fc" and w not in MAIN_WINTERS:
                continue
            names = [m + suffix for m in MODELS]
            absent = [f"{r}__noon_before__ctx{ctx}__{w}.parquet" for r in names
                      if not (cache_dir / f"{r}__noon_before__ctx{ctx}__{w}.parquet").exists()]
            missing.extend(absent)
            if absent:
                continue
            d = load_ctx(cfg, "noon_before", [w], ctx, names, load, temp)
            if suffix == "_temp_fc":
                d = d[~d.target_day.isin(fc_missing_days(d))]
            cached[w, ctx, suffix] = d

    def compare(d, a, b, kind, winter, ctx, suffix):
        for block in BLOCKS:
            t = paired_diff(d, a, b, cfg["bootstrap_n"], sorted(d.winter.unique()), block_days=block)
            t = t.reset_index()
            t = t[t.regime.isin(["all", "cold"])]
            tables.append(t.assign(kind=kind, winter=winter, context_h=ctx,
                                   temperature=suffix, block_days=block))

    for ctx, suffix in [(672, "_temp"), (2688, "_temp"), (2688, "_temp_fc")]:
        available = [w for w in ALL_WINTERS if (w, ctx, suffix) in cached]
        groups = [(w, [w]) for w in available]
        if set(available) == set(MAIN_WINTERS if suffix == "_temp_fc" else ALL_WINTERS):
            groups.append(("pooled4" if suffix == "_temp_fc" else "pooled5", available))
        for name, winters in groups:
            d = pd.concat([cached[w, ctx, suffix] for w in winters], ignore_index=True)
            for m in MODELS[1:]:
                compare(d, MODELS[0] + suffix, m + suffix, "model", name, ctx, suffix)

    available = [w for w in ALL_WINTERS if all((w, ctx, "_temp") in cached for ctx in [672, 2688])]
    groups = [(w, [w]) for w in available]
    if set(available) == set(ALL_WINTERS):
        groups.append(("pooled5", available))
    for name, winters in groups:
        for m in MODELS:
            a = pd.concat([cached[w, 672, "_temp"] for w in winters])
            b = pd.concat([cached[w, 2688, "_temp"] for w in winters])
            run = m + "_temp"
            d = pd.concat([a[a.run == run], b[b.run == run].assign(run=run + "@ctx2688")])
            compare(d, run, run + "@ctx2688", "context", name, 2688, "_temp")

    result = pd.concat(tables, ignore_index=True)
    result["excludes_zero"] = (result.dMAE_lo > 0) | (result.dMAE_hi < 0)
    save(result.set_index(["kind", "winter", "context_h", "temperature", "comparison", "regime", "block_days"]),
         "review_bootstrap", cfg)
    # Small table directly supporting the headline claims, keeping all three methods.
    headline = result[(result.winter == "w2025") & (result.regime == "all") &
                      ((result.kind == "context") | (result.context_h == 2688))]
    save(headline.set_index(["kind", "temperature", "comparison", "block_days"]), "review_bootstrap_headline", cfg)
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = []
    for i, ((kind, suffix, comparison), g) in enumerate(headline.groupby(["kind", "temperature", "comparison"])):
        label = comparison.replace("_temp_fc", " + forecast").replace("_temp", "").replace("@ctx2688", " 16w")
        labels.append(label)
        for j, block in enumerate(BLOCKS):
            r = g[g.block_days == block].iloc[0]
            ax.errorbar(r.dMAE, i + (j - 1) * .22,
                        xerr=[[r.dMAE - r.dMAE_lo], [r.dMAE_hi - r.dMAE]],
                        fmt="o", color=COLORS[j], capsize=3, label=f"{block}-day blocks" if i == 0 else None)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel("Paired MAE difference (MW): second forecast minus reference")
    ax.legend(frameon=False)
    style(ax, "Test winter: 95% intervals with 1-, 3- and 7-day blocks")
    fig.tight_layout()
    save_fig(fig, "bootstrap_sensitivity", cfg)
    meta = {"resamples": cfg["bootstrap_n"], "seed": 0, "block_days": BLOCKS,
            "method": "non-circular moving calendar-day blocks, stratified within winters; shared across models and zones",
            "missing_caches": missing, "rows": len(result), "inference_rerun": False}
    (ROOT / "results/tables/review_bootstrap_manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(headline[["kind", "temperature", "comparison", "block_days", "dMAE", "dMAE_lo", "dMAE_hi"]].to_string(index=False))
    print("Missing caches:", missing)
    probe_sensitivity(cfg, load, temp)
    calibration_sensitivity(cfg, load, temp)


if __name__ == "__main__":
    main()

"""Tables and figures from the cached forecasts.

Usage:
    python -m src.report                                   # main: origin and context from config.yaml
    python -m src.report --origin noon_before --context 672   # the original 4-week setting
    python -m src.report --origin midnight --context 672      # secondary comparison
Outputs go to results/tables/ (CSV + Markdown) and results/figures/, suffixed with origin and context.

Tables:
- main_table: every run on the test winter, per zone and regime (all / cold / peak /
  holiday / normal), with day-level bootstrap 95% CIs for MAE and 80% coverage.
- covariates: paired test-winter comparisons (load only vs + temperature, oracle vs noisy
  temperature, + holiday indicator), as differences with bootstrap CIs.
- christmas_<run>: Christmas (24 Dec - 1 Jan) vs other days pooled over all 4 winters,
  per winter, and per calendar day; Easter reported separately.
- conformal_<run>: coverage and width before vs after CQR, for each variant and for
  calibration on the single 2024-25 winter vs three winters (2022-25, the main fit).
"""
import argparse
import glob
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import ROOT, load_config
from src import conformal
from src.bootstrap import day_draws, bootstrap_margin_totals
from src.forecast import COLD_SHIFT, noise_offset
from src.metrics import add_row_metrics, mase_scales, pinball, summarize
from src.regimes import LOCAL_TZ, is_holiday, label, thresholds, set_forecast_groups

REGIMES = ["all", "cold", "peak", "holiday", "normal"]
TEST, CAL, CAL3 = "w2025", ["w2024"], ["w2022", "w2023", "w2024"]
# The four winters planned from the start. 2021-22 was added late as a robustness check (energy-
# price crisis and COVID measures that winter), so it is kept out of every main number and only
# enters through robust_w2021() and the per-winter context table.
MAIN_WINTERS = CAL3 + [TEST]
ALL_WINTERS = ["w2021"] + MAIN_WINTERS
FIX_RUNS = [("", "timestral-209m_temp", "timestral-209m_temp", "day_temp"),
            ("_caloracle", "timestral-209m_temp_noisy", "timestral-209m_temp", "day_temp_noisy"),
            ("", "timestral-209m_temp_noisy", "timestral-209m_temp_noisy", "day_temp_noisy"),
            ("", "chronos2_temp", "chronos2_temp", "day_temp"),
            ("", "chronos2_temp_noisy", "chronos2_temp_noisy", "day_temp_noisy"),
            ("", "timestral-209m_temp_fc", "timestral-209m_temp_fc", "day_temp_fc"),
            ("", "chronos2_temp_fc", "chronos2_temp_fc", "day_temp_fc")]
SIZES = {"timestral-5m": 5, "timestral-28m": 28, "timestral-123m": 123, "timestral-209m": 209}  # million params

# Figure style: categorical slots 1-4 of the dataviz reference palette (checked for
# colour-blind separation); text in neutral ink, recessive grid, direct labels.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def setting(sfx: str) -> str:
    """'_noon_before_ctx2688' -> 'noon-before origin, 16 weeks of history' (figure subtitles)."""
    origin, ctx = sfx.strip("_").rsplit("_ctx", 1)
    return f"{origin.replace('_', '-')} origin, {int(ctx) // 168} weeks of history"


def style(ax, title=None):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=INK)


def save_fig(fig, name, cfg):
    out = ROOT / cfg["paths"]["results"] / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_runs(cfg, origin, winters=None, ctx=None) -> pd.DataFrame:
    """All cached forecasts for one origin mode and context length (default: the configured one),
    on the given winters (default: the four main winters)."""
    ctx = ctx or cfg["context_hours"]
    winters = MAIN_WINTERS if winters is None else winters
    pattern = ROOT / cfg["paths"]["results"] / "cache" / "forecasts" / f"*__{origin}__ctx{ctx}__*.parquet"
    files = [f for f in glob.glob(str(pattern)) if winters is None or Path(f).stem.split("__")[-1] in winters]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def prepare(fc: pd.DataFrame, load, temp, cfg) -> pd.DataFrame:
    thr = thresholds(load, temp, cfg["regime_threshold_end"])
    return add_row_metrics(label(fc, temp, thr), mase_scales(load, cfg["regime_threshold_end"]))


def bootstrap_ci(m: pd.DataFrame, by: list[str], cols: list[str], n_boot: int, seed=0,
                 block_days=1, calendar=None) -> pd.DataFrame:
    """Day-level block bootstrap of the mean of each column within each group.

    Whole target days are resampled (hours within a day are correlated); the same set of
    resampled days is used for every group so CIs are comparable across models.
    """
    days, draws = day_draws(m if calendar is None else calendar, n_boot, block_days, seed)
    if "winter" not in m:
        m = m.assign(winter="single")
    rows = []
    for key, g in m.groupby(by):
        # per-day sums and counts -> resampled means are ratios of summed sums to summed counts
        daily = g.groupby(["winter", "target_day"])[cols].agg(["sum", "count"]).reindex(days, fill_value=0)
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        for c in cols:
            if not daily[(c, "count")].sum():
                rec[f"{c}_lo"] = rec[f"{c}_hi"] = np.nan
                continue
            s, n = daily[(c, "sum")].values[draws].sum(1), daily[(c, "count")].values[draws].sum(1)
            with np.errstate(invalid="ignore"):
                est = s / n
            rec[f"{c}_lo"], rec[f"{c}_hi"] = np.nanpercentile(est, [2.5, 97.5])
        rows.append(rec)
    return pd.DataFrame(rows).set_index(by)


def by_regime(m: pd.DataFrame, by: list[str], n_boot=0) -> pd.DataFrame:
    """Summaries for each (overlapping) regime, stacked with a 'regime' column."""
    parts = []
    for r in REGIMES:
        sub = m if r == "all" else m[m[r]]
        if sub.empty:
            continue
        t = summarize(sub, by)
        if n_boot:
            sub = sub.assign(in80=sub["in80"].astype(float))
            t = t.join(bootstrap_ci(sub, by, ["ae", "in80"], n_boot))
        parts.append(t.assign(regime=r).set_index("regime", append=True))
    return pd.concat(parts)


def main_table(m: pd.DataFrame, cfg) -> pd.DataFrame:
    test = m[m["winter"] == TEST]
    overall = by_regime(test.assign(zone="ALL"), ["run", "zone"], cfg["bootstrap_n"])
    per_zone = by_regime(test, ["run", "zone"])
    return pd.concat([overall, per_zone]).sort_index()


def paired_diff(m: pd.DataFrame, a: str, b: str, n_boot: int, winters=(TEST,), block_days=1) -> pd.DataFrame:
    """Run b minus a, matched on winter, zone, forecast origin and target hour, per regime.

    Negative MAE / pinball differences mean b is better. CIs come from the day-level
    bootstrap, so they account for the pairing (both runs see the same resampled days).
    """
    test = m[m["winter"].isin(winters)]
    # Fixed 24-hour target windows overlap at spring DST: target hour alone is
    # not unique. Match the forecast origin as well, rather than cross-joining.
    keys = ["winter", "zone", "origin_utc", "time_utc"]
    flags = ["target_day", "cold", "peak", "holiday", "normal"]
    A = test[test["run"] == a].set_index(keys)
    B = test[test["run"] == b].set_index(keys)
    if not A.index.is_unique or not B.index.is_unique:
        raise ValueError("duplicate forecast keys in paired comparison")
    matched = A.index.intersection(B.index)
    A, B = A.loc[matched], B.loc[matched]
    d = A[flags].join(B[["ae", "pinball", "in80"]].astype(float) - A[["ae", "pinball", "in80"]].astype(float))
    d = d.reset_index().assign(comparison=f"{b} vs {a}")
    rows = []
    for r in REGIMES:
        sub = d if r == "all" else d[d[r]]
        if sub.empty:
            continue
        t = sub.groupby("comparison")[["ae", "pinball", "in80"]].mean().rename(
            columns={"ae": "dMAE", "pinball": "dPinball", "in80": "dCov80"})
        t = t.join(bootstrap_ci(sub, ["comparison"], ["ae", "pinball"], n_boot,
                               block_days=block_days, calendar=d).rename(
            columns=lambda c: "d" + c.replace("ae", "MAE").replace("pinball", "Pinball")))
        t["n_days"] = sub["target_day"].nunique()
        # Relative MAE change, against run a's MAE on the same rows
        t["dMAE_pct"] = 100 * t["dMAE"] / A.loc[sub.set_index(keys).index, "ae"].mean()
        rows.append(t.assign(regime=r).set_index("regime", append=True))
    return pd.concat(rows)


def add_fc_day_temp(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Day temperature as the real-forecast runs saw it: the mean of the archived MET Nordic
    forecast over the 24 target hours (NaN where the archive has no run)."""
    from src.forecast import forecast_temps
    o = pd.DatetimeIndex(sorted(df["origin_utc"].unique()))
    H = int(df["lead_h"].max()) + 1
    fc = forecast_temps(o, cfg["zones"], H)
    keys = pd.concat([pd.DataFrame({"origin_utc": o, "zone": z, "day_temp_fc": np.nanmean(fc[z][:, -24:], axis=1)
                                    if not np.isnan(fc[z]).all() else np.nan}) for z in cfg["zones"]])
    out = df.merge(keys, on=["origin_utc", "zone"], how="left")
    return set_forecast_groups(out, out["run"].str.endswith("_temp_fc"), "day_temp_fc")


def fc_missing_days(m: pd.DataFrame) -> set:
    """Target days whose origin had no archived MET Nordic forecast (forecast.py filled them
    with observed temperature), so they are dropped from every temp_fc comparison. All winters
    (the test winter has 2 such days, 2023-24 one)."""
    from src.forecast import forecast_temps
    o = pd.DatetimeIndex(sorted(m["origin_utc"].unique()))
    if not len(o):
        return set()
    zones = sorted(m["zone"].unique())
    forecasts = forecast_temps(o, zones, 36)
    miss = np.logical_or.reduce([np.isnan(forecasts[z]).any(axis=1) for z in zones])
    days = m.loc[m["origin_utc"].isin(o[miss]), "target_day"]
    return set(days.unique())


def covariate_table(m: pd.DataFrame, cfg) -> pd.DataFrame:
    """All covariate comparisons that are cached for this origin."""
    have = set(m.loc[m["winter"] == TEST, "run"])
    pairs = []
    for base in ["lgbm", "chronos2", "timestral-209m", "timesfm3", "moirai2"]:
        pairs += [(base, f"{base}_temp"),                   # does temperature help?
                  (f"{base}_temp", f"{base}_temp_noisy"),   # cost of a synthetic noisy forecast
                  (f"{base}_temp", f"{base}_temp_fc"),      # cost of the real archived forecast
                  (f"{base}_temp_noisy", f"{base}_temp_fc"),
                  (f"{base}_temp", f"{base}_temp_multi"),   # several stations per zone vs one
                  (f"{base}_temp", f"{base}_temp_holiday")] # does a holiday flag help?
    skip = fc_missing_days(m) if any(r.endswith("_temp_fc") for r in have) else set()
    parts = []
    for a, b in pairs:
        if a in have and b in have:
            mm = m[~m["target_day"].isin(skip)] if b.endswith("_temp_fc") else m
            parts.append(paired_diff(mm, a, b, cfg["bootstrap_n"]))
    return pd.concat(parts) if parts else pd.DataFrame()


def weather_error_table(m: pd.DataFrame, temp: pd.DataFrame, cfg) -> pd.DataFrame:
    """How wrong were the real temperature forecasts? Archived MET Nordic forecast minus the
    observed station temperature over the 24 target hours, test winter, per zone and on
    cold-snap days. Compared with the synthetic run: one N(0, temp_noise_sd) offset per day."""
    from src.forecast import forecast_temps
    t = m[(m["winter"] == TEST)].drop_duplicates(["zone", "time_utc"])
    o = pd.DatetimeIndex(sorted(t["origin_utc"].unique()))
    H = int(t["lead_h"].max()) + 1
    fc = forecast_temps(o, cfg["zones"], H)
    rows = []
    for zi, z in enumerate(cfg["zones"]):
        e = pd.DataFrame(fc[z][:, -24:], index=o)  # the 24 target hours of each origin
        times = np.stack([pd.DatetimeIndex(oo + pd.to_timedelta(np.arange(H - 24, H), "h")) for oo in o])
        obs = temp[z].reindex(pd.DatetimeIndex(times.ravel())).values.reshape(len(o), 24)
        err = e.values - obs
        day_err = np.nanmean(err, axis=1)          # error of the day-mean temperature
        cold = t[t["zone"] == z].groupby("origin_utc")["cold"].first().reindex(o).fillna(False).values
        for reg, mask in [("all", np.ones(len(o), bool)), ("cold", cold)]:
            ok = mask & ~np.isnan(day_err)
            rows.append({"zone": z, "regime": reg, "days": int(ok.sum()),
                         "hourly_bias": np.nanmean(err[ok]), "hourly_sd": np.nanstd(err[ok]),
                         "day_mean_bias": day_err[ok].mean(), "day_mean_sd": day_err[ok].std(),
                         "day_mean_MAE": np.abs(day_err[ok]).mean()})
    return pd.DataFrame(rows).set_index(["zone", "regime"])


def christmas_period(day) -> str:
    """'christmas' for 24 Dec - 1 Jan (the holiday regime in every winter), else 'other'.
    The caller marks the remaining holidays (Easter 2024, inside w2023) as 'other_holiday'."""
    if (day.month == 12 and day.day >= 24) or (day.month == 1 and day.day == 1):
        return "christmas"
    return "other"


def christmas_table(m: pd.DataFrame, run: str, n_boot: int) -> dict[str, pd.DataFrame]:
    """Christmas vs other days for one run over every cached winter (4 events, not 1)."""
    r = m[m["run"] == run].copy()
    r["period"] = [christmas_period(d) for d in r["target_day"]]
    r.loc[(r["period"] == "other") & r["holiday"], "period"] = "other_holiday"
    r = r.assign(in80=r["in80"].astype(float))

    pooled = summarize(r, ["period"]).join(bootstrap_ci(r, ["period"], ["ape", "in80", "bias"], n_boot))
    pooled.insert(0, "n_days", r.groupby("period")["target_day"].nunique())
    pooled.insert(1, "n_winters", r.groupby("period")["winter"].nunique())

    xmas = r[r["period"] == "christmas"]
    per_winter = summarize(xmas, ["winter"])
    per_winter.insert(0, "n_days", xmas.groupby("winter")["target_day"].nunique())

    # Per calendar day: mean bias each winter, and in how many winters it had the same sign
    xmas = xmas.assign(cal_day=[f"{d.day:02d} {'Dec' if d.month == 12 else 'Jan'}" for d in xmas["target_day"]])
    w = xmas.groupby(["cal_day", "winter"])["bias"].mean().unstack()
    by_day = pd.DataFrame({"mean_bias": w.mean(1), "winters_under": (w > 0).sum(1), "winters_over": (w < 0).sum(1),
                           "MAPE": xmas.groupby("cal_day")["ape"].mean()})
    by_day = by_day.loc[sorted(by_day.index, key=lambda s: (s.endswith("Jan"), s))]
    return {"pooled": pooled, "per_winter": per_winter, "by_day": by_day}


def conformal_table(m: pd.DataFrame, run: str, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coverage/width on the test winter before and after CQR, per variant and calibration set."""
    r = m[m["run"] == run]
    test = r[r["winter"] == TEST]
    rows, corrections = [], []
    settings = [("none", None, None)]
    for cal_name, winters in [("1 winter", CAL), ("3 winters", CAL3)]:
        cal = r[r["winter"].isin(winters)]
        if cal["winter"].nunique() < len(winters):
            continue  # extra winters not cached for this run
        for v in ["pooled", "zone", "mondrian", "mondrian_zone"]:
            settings.append((v, cal_name, cal))
    for variant, cal_name, cal in settings:
        if variant == "none":
            adj = test
        else:
            adj, tables = conformal.recalibrate(cal, test, variant, cfg["conformal_min_n"])
            for level, t in tables.items():
                corrections.append(t.assign(variant=variant, calibration=cal_name, level=level))
        adj = adj.assign(
            in80=(adj["y_true"] >= adj["q0.10"]) & (adj["y_true"] <= adj["q0.90"]),
            in90=(adj["y_true"] >= adj["q0.05"]) & (adj["y_true"] <= adj["q0.95"]),
            width80=adj["q0.90"] - adj["q0.10"], width90=adj["q0.95"] - adj["q0.05"])
        for reg in REGIMES:
            sub = adj if reg == "all" else adj[adj[reg]]
            rows.append({"variant": variant, "calibration": cal_name or "-", "regime": reg, "n": len(sub),
                         "cov80": sub["in80"].mean(), "width80": sub["width80"].mean(),
                         "cov90": sub["in90"].mean(), "width90": sub["width90"].mean()})
    return pd.DataFrame(rows), pd.concat(corrections, ignore_index=True) if corrections else pd.DataFrame()


def add_noisy_day_temp(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Day temperature as the noisy-temperature run saw it: observed + the same per-forecast
    offset forecast.py added to the model's future temperatures."""
    zi = {z: i for i, z in enumerate(cfg["zones"])}
    keys = df[["origin_utc", "zone"]].drop_duplicates()
    keys["offset"] = [noise_offset(o, zi[z], cfg["temp_noise_sd"]) for o, z in zip(keys["origin_utc"], keys["zone"])]
    out = df.merge(keys, on=["origin_utc", "zone"], how="left")
    out = out.assign(day_temp_noisy=out["day_temp"] + out["offset"])
    return set_forecast_groups(out, out["run"].str.endswith("_temp_noisy"), "day_temp_noisy")


def fix_variants(cal: pd.DataFrame, test: pd.DataFrame, cfg, temp_col="day_temp",
                 cal_temp_col="day_temp") -> tuple[dict, tuple]:
    """The fix variants (bias correction, CQR) of the test forecasts, all fitted on `cal` only.

    The bias is fitted against cal_temp_col on the calibration rows and applied with temp_col
    on the test rows (day_temp_noisy for the realistic version: the temperature forecast the
    model was given). Mondrian groups must already use the weather available to
    that run (oracle, forecast or noisy); they stay fixed during the bias shift.
    Returns ({name: corrected test}, (a, b)).
    """
    a, b = conformal.fit_bias(cal, cal_temp_col)
    cal_b = conformal.apply_bias(cal, a, b, cal_temp_col)
    test_b = conformal.apply_bias(test, a, b, temp_col)
    n_min = cfg["conformal_min_n"]
    return {
        "none": test,
        "CQR mondrian": conformal.recalibrate(cal, test, "mondrian", n_min)[0],
        "bias only": test_b,
        "bias + CQR pooled": conformal.recalibrate(cal_b, test_b, "pooled")[0],
        "bias + CQR mondrian": conformal.recalibrate(cal_b, test_b, "mondrian", n_min)[0],
    }, (a, b)


def interval_stats(d: pd.DataFrame) -> dict:
    """Accuracy and interval metrics for one set of rows. above80 / below80 = share of hours
    above q0.90 / below q0.10 (target 0.10 each)."""
    err = d["y_true"] - d["q0.50"]
    out = {"n": len(d), "MAE": err.abs().mean(), "bias": err.mean(), "pinball": pinball(d).mean(),
           "cov80": ((d["y_true"] >= d["q0.10"]) & (d["y_true"] <= d["q0.90"])).mean(),
           "above80": (d["y_true"] > d["q0.90"]).mean(), "below80": (d["y_true"] < d["q0.10"]).mean(),
           "width80": (d["q0.90"] - d["q0.10"]).mean(),
           "cov90": ((d["y_true"] >= d["q0.05"]) & (d["y_true"] <= d["q0.95"])).mean(),
           "width90": (d["q0.95"] - d["q0.05"]).mean()}
    if d["q0.05"].isna().all():  # decile-only models (TimesFM 3.0, Moirai 2.0): no 90% band, no comparable pinball
        out.update(pinball=np.nan, cov90=np.nan, width90=np.nan)
    return out


FIX_REGIMES = ["all", "cold", "below_m10", "holiday"]


def fix_table(m: pd.DataFrame, run: str, cal_run: str, cfg, temp_col="day_temp") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test-winter results of CQR alone vs bias correction (+ CQR), per calibration set.

    cal_run gives the calibration rows. If it is the run itself, the bias is fitted against the
    same temperature column it is applied with (temp_col); if it is another run (the noisy run
    calibrated on the oracle run), the fit uses observed temperature.
    """
    cal_temp_col = temp_col if cal_run == run else "day_temp"
    test = m[(m["run"] == run) & (m["winter"] == TEST)]
    c = m[m["run"] == cal_run]
    rows, params = [], []
    for cal_name, winters in [("1 winter", CAL), ("3 winters", CAL3)]:
        cal = c[c["winter"].isin(winters)]
        if cal["winter"].nunique() < len(winters):
            continue
        variants, (a, b) = fix_variants(cal, test, cfg, temp_col, cal_temp_col)
        params.append({"calibration": cal_name, "a": a, "b": b, "n_cal": len(cal)})
        for name, d in variants.items():
            if name == "none" and cal_name != "1 winter":
                continue  # identical for every calibration set
            d = d.assign(below_m10=d["day_temp"] < -10)
            for reg in FIX_REGIMES:
                rows.append({"variant": name, "calibration": "-" if name == "none" else cal_name, "regime": reg,
                             **interval_stats(d if reg == "all" else d[d[reg]])})
    return pd.DataFrame(rows), pd.DataFrame(params)


def lowo_table(m: pd.DataFrame, run: str, cfg, temp_col="day_temp") -> pd.DataFrame:
    """Robustness check of the bias correction: leave one winter out.

    For each of the 4 winters, fit (a, b) on the other three and correct that winter, then pool
    the 4 corrected winters (4 winters of cold days instead of 1). This is a cross-check, not
    the main result: when another winter is held out, the test winter 2025-26 is among the
    fitting winters. The main result stays "fit on 2022-25, test on 2025-26".
    Differences are corrected minus original (negative = better), with day-level bootstrap CIs.
    """
    r = m[m["run"] == run]
    parts = []
    for w in sorted(r["winter"].unique()):
        cal, test = r[r["winter"] != w], r[r["winter"] == w]
        a, b = conformal.fit_bias(cal, temp_col)
        tb = conformal.apply_bias(test, a, b, temp_col)
        e0, e1 = test["y_true"] - test["q0.50"], tb["y_true"] - tb["q0.50"]
        parts.append(test[["winter", "target_day", "cold", "day_temp"]].assign(
            slope=b, bias_before=e0, bias_after=e1, dMAE=e1.abs() - e0.abs(),
            dPinball=pinball(tb).values - pinball(test).values,
            above_before=(test["y_true"] > test["q0.90"]).astype(float),
            below_before=(test["y_true"] < test["q0.10"]).astype(float),
            above_after=(tb["y_true"] > tb["q0.90"]).astype(float),
            below_after=(tb["y_true"] < tb["q0.10"]).astype(float)))
    d = pd.concat(parts).assign(below_m10=lambda x: x["day_temp"] < -10, pooled=f"{r['winter'].nunique()} winters")
    cols = ["bias_before", "bias_after", "dMAE", "dPinball", "above_before", "below_before",
            "above_after", "below_after"]
    rows = []
    for reg in ["all", "cold", "below_m10"]:
        s = d if reg == "all" else d[d[reg]]
        ci = bootstrap_ci(s, ["pooled"], ["dMAE", "dPinball"], cfg["bootstrap_n"]).iloc[0]
        rows.append({"regime": reg, "zone_days": len(s) / 24, **s[cols].mean(), **ci,
                     "slope_min": d["slope"].min(), "slope_max": d["slope"].max()})
    return pd.DataFrame(rows).set_index("regime")


def sizes_table(m: pd.DataFrame, suffix="") -> pd.DataFrame:
    """The four Timestral sizes on the test winter (suffix "" = load only, "_temp").

    sec_per_forecast is wall-clock CPU time per (zone, day) forecast as logged by forecast.py,
    i.e. batch time / batch size with batches of 25 and symmetric averaging on.
    """
    t = m[(m["winter"] == TEST) & m["run"].isin([s + suffix for s in SIZES])]
    out = summarize(t, ["run"]).join(t.groupby("run")["sec_per_forecast"].mean())
    out.insert(0, "params_M", out.index.str.removesuffix(suffix).map(SIZES))
    return out.sort_values("params_M")


def fig_sizes(t: pd.DataFrame, cfg, name="sizes",
              title="Timestral sizes, test winter 2025-26, midnight origin, load history only"):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    # 123M and 209M take about the same CPU time (209M slightly less), so their labels go on
    # opposite sides: 209M left/below, 123M right/above
    offsets = {5: (8, 4), 28: (8, 4), 123: (6, 6), 209: (-34, -14)}
    for ax, col, label in [(axes[0], "MAE", "MAE (MW)"), (axes[1], "cov80", "80% interval coverage")]:
        ax.plot(t["sec_per_forecast"], t[col], color=COLORS[0], lw=2, marker="o", ms=8)
        for _, r in t.iterrows():
            ax.annotate(f"{int(r['params_M'])}M", (r["sec_per_forecast"], r[col]), textcoords="offset points",
                        xytext=offsets[int(r["params_M"])], fontsize=9, color=INK)
        ax.set_xscale("log")
        ticks = [x for x in [0.03, 0.1, 0.3, 0.5, 1, 2, 3]
                 if t["sec_per_forecast"].min() * 0.7 <= x <= t["sec_per_forecast"].max() * 1.25]
        ax.set_xticks(ticks, [f"{x:g}" for x in ticks])
        ax.minorticks_off()
        ax.set_xlabel("CPU seconds per forecast (log scale)", color=INK2, fontsize=9)
        style(ax, label)
    axes[1].set_ylim(0.70, 0.90)  # wide enough that sampling noise doesn't look like a trend
    axes[1].axhline(0.80, color=INK2, lw=1, ls="--")
    axes[0].set_ylim(t["MAE"].min() - 4, t["MAE"].max() + 1.5)
    axes[1].annotate("target 0.80", (0.06, 0.80), xytext=(0, -12), textcoords="offset points",
                     fontsize=8, color=INK2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout()
    save_fig(fig, name, cfg)


def fig_fix(tab: pd.DataFrame, cfg, sfx: str, cal_name="3 winters"):
    """Key figure: share of hours above / below the 80% band (target 10% each), before and after."""
    variants = [("none", "-", "No fix"), ("CQR mondrian", cal_name, "Conformal"),
                ("bias + CQR mondrian", cal_name, "Bias fix +\nconformal")]
    regimes = [("all", "All hours"), ("cold", "Cold-snap days"), ("below_m10", "Days below −10 °C")]
    t = tab.set_index(["variant", "calibration", "regime"])
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
    x = np.arange(len(variants))
    for ax, (reg, title) in zip(axes, regimes):
        above = [t.loc[(v, c, reg), "above80"] for v, c, _ in variants]
        below = [t.loc[(v, c, reg), "below80"] for v, c, _ in variants]
        n_days = t.loc[("none", "-", reg), "n"] / 24  # rows are zone-hours
        for vals, dx, color, label in [(above, -0.19, COLORS[1], "Load above the band (under-forecast)"),
                                       (below, 0.19, COLORS[0], "Load below the band (over-forecast)")]:
            bars = ax.bar(x + dx, vals, width=0.36, color=color, label=label, edgecolor="white", linewidth=2)
            ax.bar_label(bars, labels=[f"{100 * v:.0f}%" for v in vals], fontsize=8, color=INK, padding=2)
        ax.axhline(0.10, color=INK2, lw=1, ls="--")
        ax.set_xticks(x, [lab for _, _, lab in variants], fontsize=9)
        style(ax, f"{title} ({n_days:.0f} zone-days)" if reg != "all" else title)
    axes[0].annotate("target 10%", (0.5, 0.10), xytext=(0, 4), textcoords="offset points", fontsize=8,
                     color=INK2, ha="center")  # in the gap between the first two bar groups
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].set_ylabel("share of hours outside the 80% band", color=INK2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=2)
    fig.suptitle(f"Timestral-209M + temperature, test winter 2025-26 ({setting(sfx)}, "
                 f"corrections fitted on {cal_name})", x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_fig(fig, f"fix_before_after{sfx}", cfg)


def fig_regimes(m: pd.DataFrame, cfg, sfx: str):
    """Error by regime (MAE) and median bias against the day's temperature, test winter.
    TimesFM 3.0 appears where it was run (not at the midnight origin). Each model keeps the
    colour it has in the thumbnail."""
    runs = [("timestral-209m_temp", "Timestral-209M", COLORS[0]), ("timesfm3_temp", "TimesFM 3.0", COLORS[1]),
            ("chronos2_temp", "Chronos-2", COLORS[2]), ("lgbm_temp", "LightGBM", COLORS[3])]
    test = m[(m["winter"] == TEST) & m["run"].isin([r for r, *_ in runs])]
    runs = [r for r in runs if r[0] in set(test["run"])]
    if len(runs) < 3:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    regs = [("normal", "Normal"), ("cold", "Cold snap"), ("peak", "Peak hour"), ("holiday", "Christmas")]
    x = np.arange(len(regs))
    w = 0.8 / len(runs)
    for i, (run, name, color) in enumerate(runs):
        r = test[test["run"] == run]
        vals = [r.loc[r[g], "ae"].mean() for g, _ in regs]
        ax1.bar(x + (i - (len(runs) - 1) / 2) * w, vals, width=w * 0.92, color=color, label=name,
                edgecolor="white", linewidth=2)
    ax1.set_xticks(x, [lab for _, lab in regs], fontsize=9)
    ax1.legend(frameon=False, fontsize=8)
    style(ax1, "MAE by regime (MW), with temperature")

    bins = [-30, -10, -5, 0, 5, 20]
    mids = ["< −10", "−10 to −5", "−5 to 0", "0 to 5", "> 5"]
    for run, name, color in runs:
        r = test[test["run"] == run]
        b = r.groupby(pd.cut(r["day_temp"], bins), observed=False)["bias"].mean()
        ax2.plot(mids, b.values, color=color, lw=2, marker="o", ms=7)  # colours as in the legend on the left
    ax2.axhline(0, color=INK2, lw=1)
    ax2.set_xlabel("daily mean temperature of the target day (°C)", color=INK2, fontsize=9)
    style(ax2, "Median bias (actual − forecast, MW): + = under-forecast")
    fig.suptitle(f"Test winter 2025-26, {setting(sfx)}", x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout()
    save_fig(fig, f"error_by_regime{sfx}", cfg)


def fig_fan(before: pd.DataFrame, after: pd.DataFrame, cfg, sfx: str, zone="NO1", days=7):
    """Fan chart for the coldest week of the test winter in one zone, before and after the fix."""
    b, a = before[before["zone"] == zone], after[after["zone"] == zone]
    daily = b.groupby("target_day")["day_temp"].first()
    start = daily.rolling(days).mean().idxmin()  # last day of the coldest `days`-day window
    window = sorted(daily.index[(daily.index <= start)])[-days:]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, sharey=True)
    for ax, d, title in [(axes[0], b, "Timestral-209M + temperature, as is"),
                         (axes[1], a, "After bias fix + Mondrian conformal (fitted on 3 earlier winters)")]:
        d = d[d["target_day"].isin(window)].sort_values("time_utc")
        t = d["time_utc"].dt.tz_convert("Europe/Oslo")
        ax.fill_between(t, d["q0.05"], d["q0.95"], color="#b7d3f6", lw=0, label="90% band")
        ax.fill_between(t, d["q0.10"], d["q0.90"], color="#86b6ef", lw=0, label="80% band")
        ax.plot(t, d["q0.50"], color="#1c5cab", lw=1.5, label="forecast (median)")
        ax.plot(t, d["y_true"], color=INK, lw=1.5, label="actual load")
        style(ax, f"{title}: {100 * (d['y_true'] > d['q0.90']).mean():.0f}% of hours above the 80% band, "
                  f"{100 * (d['y_true'] < d['q0.10']).mean():.0f}% below")
        ax.set_ylabel("MW", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    temps = ", ".join(f"{daily[w]:.0f}" for w in window)
    fig.suptitle(f"{zone}, coldest week of the test winter ({window[0]} to {window[-1]}; daily mean °C: {temps}), "
                 f"{setting(sfx)}", x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout()
    save_fig(fig, f"fan_cold_snap_{zone}{sfx}", cfg)


CONTEXT_RUNS = ["timestral-209m_temp", "chronos2_temp"]


def load_ctx(cfg, origin, winters, ctx, runs, load, temp) -> pd.DataFrame:
    """Prepared forecasts of the given runs at one context length (empty if nothing cached)."""
    try:
        d = load_runs(cfg, origin, winters, ctx)
    except ValueError:  # nothing cached
        return pd.DataFrame()
    d = d[d["run"].isin(runs)]
    return prepare(d, load, temp, cfg) if len(d) else d


def context_table(cfg, origin, load, temp) -> pd.DataFrame:
    """Context-length sensitivity: each longer-context run vs the 4-week (672 h) run of the same
    model, as paired differences (negative = longer context better), per winter.
    The effect was first noticed on the test winter, then checked on the calibration
    and other winters; this was not a fully held-out context selection. Runs are renamed
    '<run>@ctx<T>' so paired_diff can match them."""
    parts = []
    for winter in sorted(cfg["evaluation_winters"]):
        base = load_ctx(cfg, origin, [winter], 672, CONTEXT_RUNS, load, temp)
        for ctx in [1344, 2688, 5376]:
            alt = load_ctx(cfg, origin, [winter], ctx, CONTEXT_RUNS, load, temp)
            for run in sorted(set(alt["run"]) & set(base["run"])) if len(alt) and len(base) else []:
                both = pd.concat([base[base["run"] == run], alt[alt["run"] == run].assign(run=f"{run}@ctx{ctx}")])
                d = paired_diff(both, run, f"{run}@ctx{ctx}", cfg["bootstrap_n"], winters=(winter,))
                sec = alt.loc[alt["run"] == run, "sec_per_forecast"].mean()
                parts.append(d.assign(winter=winter, model=run, context_h=ctx, sec_per_forecast=sec))
    return pd.concat(parts) if parts else pd.DataFrame()


ACI_GAMMAS = [0.005, 0.02, 0.05]  # per-day step sizes; no tuning on the test winter, all reported


def aci_table(cfg, origin, load, temp) -> pd.DataFrame:
    """Online recalibration (adaptive conformal inference) vs static per-zone CQR, both starting
    from the calibration winter 2024-25, on the test winter, at 4 and 16 weeks of context.
    ACI tracks the overall miss rate as the winter goes; it cannot fix a one-sided miss pattern,
    because it moves both ends of the band together."""
    rows = []
    for ctx in [672, 2688]:
        r = load_ctx(cfg, origin, CAL + [TEST], ctx, CONTEXT_RUNS, load, temp)
        for run in sorted(set(r["run"])) if len(r) else []:
            x = r[r["run"] == run]
            cal, test = x[x["winter"].isin(CAL)], x[x["winter"] == TEST]
            if cal.empty or test.empty:
                continue
            variants = {"none": test, "CQR zone (static)": conformal.recalibrate(cal, test, "zone")[0]}
            for g in ACI_GAMMAS:
                out = test
                for level in conformal.INTERVALS:
                    out = conformal.aci(cal, out, level, g)
                out[conformal.QCOLS] = np.sort(out[conformal.QCOLS].values, axis=1)
                variants[f"ACI gamma={g}"] = out
            for name, d in variants.items():
                for reg in ["all", "cold"]:
                    rows.append({"model": run, "context_h": ctx, "variant": name, "regime": reg,
                                 **interval_stats(d if reg == "all" else d[d["cold"]])})
    return pd.DataFrame(rows)


PEAK_RUNS = ["snaive", "lgbm_temp", "chronos2_temp", "timestral-209m_temp", "timestral-209m_temp_noisy", "timesfm3_temp"]


def peak_table(m: pd.DataFrame, cfg) -> pd.DataFrame:
    """The operator's view: each day's peak hour (test winter, per zone-day).

    peak_err      actual daily peak minus forecast daily peak (max of the median over the 24 h);
                  positive = the peak was under-forecast
    short_at_peak at the hour of the actual peak, load was above the forecast's q0.90, i.e. a
                  plan that holds reserves up to the upper 80% band would have been caught short
                  (target 10%). Uses the forecast's own band at that hour, so it is a fair test.
    hour_hit      forecast peak hour within +/- 1 h of the actual peak hour
    With day-level bootstrap CIs on the absolute peak error and the short-at-peak rate."""
    t = m[(m["winter"] == TEST) & m["run"].isin(PEAK_RUNS)]
    t = t.sort_values(["run", "zone", "target_day", "time_utc"])
    g = t.groupby(["run", "zone", "target_day"])
    i_true = g["y_true"].idxmax()
    i_pred = g["q0.50"].idxmax()
    d = t.loc[i_true, ["run", "zone", "target_day", "cold", "y_true", "q0.90", "time_utc"]].reset_index(drop=True)
    d["pred_peak"] = t.loc[i_pred, "q0.50"].values
    d["pred_time"] = t.loc[i_pred, "time_utc"].to_numpy()  # keeps the UTC timezone
    d["peak_err"] = d["y_true"] - d["pred_peak"]
    d["abs_peak_err"] = d["peak_err"].abs()
    d["short_at_peak"] = (d["y_true"] > d["q0.90"]).astype(float)
    d["hour_hit"] = ((d["time_utc"] - d["pred_time"]).abs() <= pd.Timedelta(hours=1)).astype(float)
    rows = []
    for reg in ["all", "cold"]:
        s = d if reg == "all" else d[d["cold"]]
        gr = s.groupby("run")
        tab = pd.DataFrame({"zone_days": gr.size(), "peak_bias": gr["peak_err"].mean(),
                            "peak_MAE": gr["abs_peak_err"].mean(), "short_at_peak": gr["short_at_peak"].mean(),
                            "hour_hit": gr["hour_hit"].mean()})
        tab = tab.join(bootstrap_ci(s, ["run"], ["abs_peak_err", "short_at_peak"], cfg["bootstrap_n"]))
        rows.append(tab.assign(regime=reg).set_index("regime", append=True))
    return pd.concat(rows)


def historical_slopes(load, temp, end) -> pd.Series:
    """Each zone's load-temperature sensitivity from history (MW per °C, negative = more load
    when colder): OLS of daily mean load on daily mean temperature with day-of-week and
    month fixed effects, winter months (Nov-Mar) before `end`, holidays excluded. The month
    effects absorb most of the season/daylight pattern that moves with temperature, so the
    slope is closer to a pure temperature effect (still observational, not causal)."""
    day = load.index.tz_convert(LOCAL_TZ).normalize().tz_localize(None)
    L, T = load.groupby(day).mean(), temp.groupby(day).mean()
    hol = is_holiday(load.index).groupby(day).max().values.astype(bool)
    keep = (L.index < end) & L.index.month.isin([11, 12, 1, 2, 3]) & ~hol
    idx = L.index[keep]
    fe = pd.get_dummies(pd.DataFrame({"dow": idx.dayofweek, "month": idx.month}).astype(str)).values.astype(float)
    slopes = {}
    for z in L.columns:
        X = np.column_stack([T.loc[keep, z].values, fe])
        slopes[z] = np.linalg.lstsq(X, L.loc[keep, z].values, rcond=None)[0][0]
    return pd.Series(slopes, name="historical_MW_per_C")


def probe_table(cfg, origin, load, temp) -> pd.DataFrame:
    """Temperature sensitivity probe (test winter): how much does the median forecast change
    when the future temperature is COLD_SHIFT °C colder (temp_cold run) than in the plain temp
    run, in MW per °C, per model, context length and zone, vs the historical slope.
    ratio = model response / historical slope (1 = the model reacts as strongly as the data
    says load does; < 1 = it under-reacts, which would show up as cold-snap under-forecasting).
    Every (model, context) pair with both runs cached is included: 4/8/16/32 weeks for
    Timestral-209M, 4/16 weeks for Timestral-123M, Chronos-2, TimesFM 3.0 and Moirai 2.0."""
    hist = historical_slopes(load, temp, cfg["regime_threshold_end"])
    rows = []
    for ctx in [672, 1344, 2688, 5376]:
        for model in ["timestral-209m", "timestral-123m", "chronos2", "timesfm3", "moirai2"]:
            base = load_ctx(cfg, origin, [TEST], ctx, [f"{model}_temp"], load, temp)
            cold = load_ctx(cfg, origin, [TEST], ctx, [f"{model}_temp_cold"], load, temp)
            if base.empty or cold.empty:
                continue
            keys = ["winter", "zone", "origin_utc", "time_utc"]
            d = base.set_index(keys)[["cold", "q0.50"]].join(
                cold.set_index(keys)[["q0.50"]].rename(columns={"q0.50": "q_cold"}), how="inner")
            d["response"] = (d["q_cold"] - d["q0.50"]) / -COLD_SHIFT
            for reg in ["all", "cold"]:
                s = d if reg == "all" else d[d["cold"]]
                r = s.groupby(level="zone")["response"].mean()
                for z, v in r.items():
                    rows.append({"model": model, "context_h": ctx, "regime": reg, "zone": z,
                                 "model_MW_per_C": v, "historical_MW_per_C": hist[z], "ratio": v / hist[z]})
    return pd.DataFrame(rows)


def probe_summary(cfg, origin, load, temp, winters=(TEST,), block_days=1, model_contrasts=False) -> pd.DataFrame:
    """The probe with uncertainty: per model and context length, the mean ratio over the five
    zones (every zone has the same hours, so the mean over hours weights the zones equally),
    and each model's change from 4 to 16 weeks paired on the same hours. 95% CIs from the
    day-level bootstrap (same resampled days at both lengths)."""
    hist = historical_slopes(load, temp, cfg["regime_threshold_end"])
    keys = ["winter", "zone", "origin_utc", "time_utc"]
    rows, changes = [], {}
    for model in ["timestral-209m", "timestral-123m", "chronos2", "timesfm3", "moirai2"]:
        per_ctx = {}
        for ctx in [672, 1344, 2688, 5376]:
            base = load_ctx(cfg, origin, list(winters), ctx, [f"{model}_temp"], load, temp)
            cold = load_ctx(cfg, origin, list(winters), ctx, [f"{model}_temp_cold"], load, temp)
            if base.empty or cold.empty:
                continue
            d = base.set_index(keys)[["target_day", "q0.50"]].join(
                cold.set_index(keys)[["q0.50"]].rename(columns={"q0.50": "q_cold"}), how="inner")
            slope = d.index.get_level_values("zone").map(hist).values
            d["r"] = (d["q_cold"] - d["q0.50"]) / -COLD_SHIFT / slope
            per_ctx[ctx] = d
            ci = bootstrap_ci(d.reset_index().assign(g=1), ["g"], ["r"], cfg["bootstrap_n"], block_days=block_days).iloc[0]
            rows.append({"model": model, "context_h": ctx, "stat": "mean ratio", "value": d["r"].mean(),
                         "lo": ci["r_lo"], "hi": ci["r_hi"]})
        if 672 in per_ctx and 2688 in per_ctx:
            j = per_ctx[2688][["r", "target_day"]].join(per_ctx[672][["r"]], rsuffix="_4w", how="inner")
            j["diff"] = j["r"] - j["r_4w"]
            ci = bootstrap_ci(j.reset_index().assign(g=1), ["g"], ["diff"], cfg["bootstrap_n"], block_days=block_days).iloc[0]
            changes[model] = j[["target_day", "diff"]]
            rows.append({"model": model, "context_h": 2688, "stat": "change from 4 weeks", "value": j["diff"].mean(),
                         "lo": ci["diff_lo"], "hi": ci["diff_hi"]})
    if model_contrasts and "timestral-209m" in changes:
        for other in ["chronos2", "timesfm3"]:
            if other not in changes:
                continue
            j = changes["timestral-209m"].join(changes[other][["diff"]], rsuffix="_other", how="inner")
            j["interaction"] = j["diff"] - j["diff_other"]
            ci = bootstrap_ci(j.reset_index().assign(g=1), ["g"], ["interaction"], cfg["bootstrap_n"],
                              block_days=block_days).iloc[0]
            rows.append({"model": f"timestral-209m minus {other}", "context_h": 2688,
                         "stat": "difference in context change", "value": j.interaction.mean(),
                         "lo": ci.interaction_lo, "hi": ci.interaction_hi})
    return pd.DataFrame(rows)


def probe_context(cfg, origin, load, temp, n_boot=500) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Is the probe response set by the window's length or by the temperatures it contains?
    (Plan fixed before looking.)

    Per zone and origin (test winter): ratio = mean probe response over the 24 target hours /
    the zone's historical slope; range = max - min of daily-mean temperature in the model's
    history window. OLS of the ratio on zone fixed effects plus, in turn:
      [w16]                     16-week dummy only
      [w16, range]              + window temperature range
      [w16, range, day_temp]    + target day temperature
    If range carries the effect, the w16 coefficient shrinks towards zero. Day-level bootstrap
    CIs. Returns (per zone-origin rows, coefficient table). Planned for Timestral and Chronos-2;
    TimesFM 3.0 added with the same specification once its probe runs existed."""
    hist = historical_slopes(load, temp, cfg["regime_threshold_end"])
    parts = []
    for ctx in [672, 2688]:
        for model in ["timestral-209m", "chronos2", "timesfm3"]:
            base = load_ctx(cfg, origin, [TEST], ctx, [f"{model}_temp"], load, temp)
            cold = load_ctx(cfg, origin, [TEST], ctx, [f"{model}_temp_cold"], load, temp)
            if base.empty or cold.empty:
                continue
            keys = ["zone", "origin_utc", "time_utc"]
            d = base.set_index(keys)[["q0.50", "day_temp", "target_day"]].join(
                cold.set_index(keys)[["q0.50"]].rename(columns={"q0.50": "q_cold"}), how="inner").reset_index()
            d["response"] = (d["q_cold"] - d["q0.50"]) / -COLD_SHIFT
            g = d.groupby(["zone", "origin_utc"]).agg(response=("response", "mean"), day_temp=("day_temp", "first"),
                                                     target_day=("target_day", "first")).reset_index()
            rng = []
            for z, o in zip(g["zone"], g["origin_utc"]):
                daily = temp[z].loc[o - pd.Timedelta(hours=ctx): o - pd.Timedelta(hours=1)].resample("D").mean()
                rng.append(daily.max() - daily.min())
            g["window_range"] = rng
            g["ratio"] = g["response"] / g["zone"].map(hist)
            parts.append(g.assign(model=model, context_h=ctx))
    if not parts:
        return pd.DataFrame(), pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    d["w16"] = (d["context_h"] == 2688).astype(float)

    def ols(x, cols):
        X = np.column_stack([pd.get_dummies(x["zone"]).values.astype(float)] + [x[c].values for c in cols])
        return np.linalg.lstsq(X, x["ratio"].values, rcond=None)[0][-len(cols):]

    rng_ = np.random.default_rng(0)
    rows = []
    for model, x in d.groupby("model"):
        days = x["target_day"].unique()
        by_day = {k: g for k, g in x.groupby("target_day")}
        draws = [rng_.choice(days, len(days)) for _ in range(n_boot)]
        for spec in [["w16"], ["w16", "window_range"], ["w16", "window_range", "day_temp"]]:
            est = ols(x, spec)
            boot = np.array([ols(pd.concat([by_day[p] for p in pick]), spec) for pick in draws])
            for i, c in enumerate(spec):
                rows.append({"model": model, "spec": " + ".join(spec), "term": c, "coef": est[i],
                             "lo": np.percentile(boot[:, i], 2.5), "hi": np.percentile(boot[:, i], 97.5)})
    return d, pd.DataFrame(rows)


def novelty_table(cfg, origin, load, temp, run="timestral-209m_temp", winters=None) -> pd.DataFrame:
    """Why does a 16-week context help? Split the 4 vs 16 week comparison (all 4 winters) by
    whether the target day is colder than every day the model saw in its context:
      seen         a day at least as cold is inside the 4-week context
      new_to_4w    colder than all of the last 4 weeks, but not of the last 16
      new_to_both  colder than all of the last 16 weeks (the model must extrapolate either way)
    Days = local-day mean temperature at the zone's station. dMAE = 16 weeks minus 4 weeks,
    with day-level bootstrap CIs."""
    a = load_ctx(cfg, origin, winters, 672, [run], load, temp)
    b = load_ctx(cfg, origin, winters, 2688, [run], load, temp)
    keys = ["winter", "zone", "origin_utc", "time_utc"]
    d = a.set_index(keys)[["target_day", "day_temp", "cold", "y_true", "q0.50"]].join(
        b.set_index(keys)[["q0.50"]].rename(columns={"q0.50": "q16"}), how="inner").reset_index()
    d = d.assign(e4=d["y_true"] - d["q0.50"], e16=d["y_true"] - d["q16"])

    # Coldest daily mean temperature in the 4 and 16 weeks before each origin
    local = temp.index.tz_convert(LOCAL_TZ)
    daily = temp.groupby(local.normalize().tz_localize(None)).mean()
    o = d[["zone", "origin_utc"]].drop_duplicates()
    oday = pd.DatetimeIndex(o["origin_utc"]).tz_convert(LOCAL_TZ).normalize().tz_localize(None)
    for weeks in (4, 16):
        o[f"min{weeks}"] = [daily.loc[t - pd.Timedelta(weeks=weeks): t - pd.Timedelta(days=1), z].min()
                            for z, t in zip(o["zone"], oday)]
    d = d.merge(o, on=["zone", "origin_utc"])
    d["context"] = np.select([d["day_temp"] >= d["min4"], d["day_temp"] >= d["min16"]],
                             ["seen", "new_to_4w"], "new_to_both")
    d["dMAE"] = d["e16"].abs() - d["e4"].abs()

    rows = []
    for reg in ["all", "cold"]:
        s = d if reg == "all" else d[d["cold"]]
        g = s.groupby("context")
        t = pd.DataFrame({"zone_days": g.size() / 24, "bias_4w": g["e4"].mean(), "bias_16w": g["e16"].mean(),
                          "MAE_4w": g["e4"].apply(lambda x: x.abs().mean()),
                          "MAE_16w": g["e16"].apply(lambda x: x.abs().mean())})
        t["dMAE_pct"] = 100 * (t["MAE_16w"] / t["MAE_4w"] - 1)
        t = t.join(bootstrap_ci(s, ["context"], ["dMAE"], cfg["bootstrap_n"]))
        rows.append(t.assign(regime=reg).set_index("regime", append=True))
    return pd.concat(rows)


def cold_cqr_table(cfg, origin, load, temp, winters=None, evals=("test", "lowo"),
                   runs=CONTEXT_RUNS, block_days=1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fix for the one-sided cold misses: CQR per temperature bin, two-sided vs one-sided
    (each end of the band calibrated on its own, see conformal.fit_asym).

    Two evaluations, at 4 and 16 weeks of context:
      test  fit on the three winters 2022-25, apply to the test winter 2025-26 (main result)
      lowo  leave one winter out: fit on 3 winters, apply to the 4th, pool the 4 (4x the very
            cold days; the test winter is fitting data in 3 of the 4 folds)
    above80 / below80 get day-level bootstrap CIs (the < -10 °C group is only ~75 zone-days
    over 4 winters, so expect wide intervals). Returns (results, test-fit correction tables).
    For decile-only models only the 80% band is recalibrated (the 90% one doesn't exist)."""
    n_min = cfg["conformal_min_n"]
    variants = {"none": None, "CQR two-sided, temp bins": True, "CQR one-sided, temp bins": False}
    rows, qtabs = [], []
    for ctx in [672, 2688]:
        r = load_ctx(cfg, origin, winters, ctx, runs, load, temp)
        for run in sorted(set(r["run"])) if len(r) else []:
            x = r[r["run"] == run]
            folds = {"test": [(x[x["winter"].isin(CAL3)], x[x["winter"] == TEST])],
                     "lowo": [(x[x["winter"] != w], x[x["winter"] == w]) for w in sorted(x["winter"].unique())]}
            for ev, pairs in folds.items():
                if ev not in evals:
                    continue
                for name, two_sided in variants.items():
                    parts = []
                    for cal, test in pairs:
                        if two_sided is None:
                            parts.append(test)
                            continue
                        out, tabs = conformal.recalibrate_asym(cal, test, two_sided=two_sided, min_n=n_min)
                        parts.append(out)
                        if ev == "test":
                            qtabs += [t.assign(model=run, context_h=ctx, variant=name, level=lv) for lv, t in tabs.items()]
                    d = pd.concat(parts)
                    d = d.assign(below_m10=d["day_temp"] < -10, g=1,
                                 above=(d["y_true"] > d["q0.90"]).astype(float),
                                 below=(d["y_true"] < d["q0.10"]).astype(float))
                    for reg in ["all", "cold", "below_m10"]:
                        sub = d if reg == "all" else d[d[reg]]
                        ci = bootstrap_ci(sub, ["g"], ["above", "below"], cfg["bootstrap_n"], block_days=block_days).iloc[0]
                        rows.append({"model": run, "context_h": ctx, "eval": ev, "variant": name, "regime": reg,
                                     **interval_stats(sub), **ci})
    return pd.DataFrame(rows), pd.concat(qtabs) if qtabs else pd.DataFrame()


def cold_cqr_fc_table(m: pd.DataFrame, cfg, block_days=1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The one-sided temperature-bin CQR on the real-forecast runs, binned by the *forecast*
    day temperature (what the operator knows at noon the day before). The reported regimes
    (cold, below -10) still use observed temperature. Three calibrations:
      1 winter   fit on 2024-25, apply to the test winter
      3 winters  fit on 2022-25, apply to the test winter (main, as elsewhere in the project)
      lowo       leave one of the 4 real-forecast winters out, pool (test winter is fitting
                 data in 3 of 4 folds; ~4x the deep-cold days)"""
    rows, qtabs = [], []
    for run in FM_FC_RUNS:
        x = m[m["run"] == run]
        if x[x["winter"] == TEST].empty:
            continue
        folds = {"1 winter": [(x[x["winter"].isin(CAL)], x[x["winter"] == TEST])]}
        if set(CAL3) <= set(x["winter"]):
            folds["3 winters"] = [(x[x["winter"].isin(CAL3)], x[x["winter"] == TEST])]
            folds["lowo"] = [(x[x["winter"] != w], x[x["winter"] == w]) for w in sorted(x["winter"].unique())]
        for cal_name, pairs in folds.items():
            variants = {"none": pd.concat([t for _, t in pairs])}
            for name, two_sided in [("CQR two-sided, temp bins", True), ("CQR one-sided, temp bins", False)]:
                parts = []
                for cal, test in pairs:
                    out, tabs = conformal.recalibrate_asym(cal, test, temp_col="day_temp_fc",
                                                           two_sided=two_sided, min_n=cfg["conformal_min_n"])
                    parts.append(out)
                    if cal_name != "lowo":
                        qtabs += [t.assign(model=run, calibration=cal_name, variant=name, level=lv)
                                  for lv, t in tabs.items()]
                variants[name] = pd.concat(parts)
            for name, d in variants.items():
                d = d.assign(below_m10=d["day_temp"] < -10, g=1,
                             above=(d["y_true"] > d["q0.90"]).astype(float),
                             below=(d["y_true"] < d["q0.10"]).astype(float))
                for reg in ["all", "cold", "below_m10"]:
                    sub = d if reg == "all" else d[d[reg]]
                    ci = bootstrap_ci(sub, ["g"], ["above", "below"], cfg["bootstrap_n"], block_days=block_days).iloc[0]
                    rows.append({"model": run, "calibration": cal_name, "variant": name, "regime": reg,
                                 "zone_days": len(sub) / 24, **interval_stats(sub), **ci})
    return pd.DataFrame(rows), pd.concat(qtabs) if qtabs else pd.DataFrame()


def robust_w2021(cfg, origin, load, temp) -> dict[str, pd.DataFrame]:
    """The fifth winter 2021-22 as a robustness check. That winter had record power prices in
    southern Norway (NO1, NO2, NO5) and COVID measures until Feb 2022, so households and
    industry likely used less than the weather alone implies. It is not used for any main
    number; here we ask whether the main findings hold when it is added:
      w2021      accuracy and band misses of both models on 2021-22 alone, 4 and 16 weeks
      novelty    the 4 vs 16 week split by how new the cold is, over 5 winters
      cold_cqr   one-sided temperature-bin CQR, leave one winter out over 5 winters
      bias_lowo  the temperature bias correction, leave one winter out over 5 winters"""
    out = {}
    rows = []
    runs5 = CONTEXT_RUNS + ["timesfm3_temp"]  # TimesFM 3.0 (added later) joins the accuracy and CQR checks
    for ctx in [672, 2688]:
        d = load_ctx(cfg, origin, ["w2021"], ctx, runs5, load, temp)
        for run in sorted(set(d["run"])) if len(d) else []:
            x = d[d["run"] == run].assign(below_m10=lambda v: v["day_temp"] < -10)
            for reg in ["all", "cold", "below_m10"]:
                sub = x if reg == "all" else x[x[reg]]
                rows.append({"model": run, "context_h": ctx, "regime": reg, **interval_stats(sub)})
    if not rows:
        return out
    out["w2021"] = pd.DataFrame(rows)
    out["novelty"] = novelty_table(cfg, origin, load, temp, winters=ALL_WINTERS)
    out["cold_cqr"], _ = cold_cqr_table(cfg, origin, load, temp, winters=ALL_WINTERS, evals=("lowo",), runs=runs5)
    lowo = []
    for ctx in [672, 2688]:
        d = load_ctx(cfg, origin, ALL_WINTERS, ctx, CONTEXT_RUNS, load, temp)
        for run in sorted(set(d["run"])) if len(d) else []:
            if d.loc[d["run"] == run, "winter"].nunique() == len(ALL_WINTERS):
                lowo.append(lowo_table(d, run, cfg).assign(model=run, context_h=ctx))
    out["bias_lowo"] = pd.concat(lowo) if lowo else pd.DataFrame()
    return out


RESERVE_RUNS = ["snaive", "lgbm_temp", "chronos2_temp", "timestral-209m_temp", "timestral-209m_temp_noisy",
                "timestral-209m_temp_fc", "chronos2_temp_fc", "timesfm3_temp", "timesfm3_temp_fc"]


def reserve_table(cfg, origin, load, temp, ctx_main=2688, cover=0.90) -> pd.DataFrame:
    """Retrospective test-period estimate of the margin covering `cover` of daily peaks.

    Per zone: required margin = the 90th percentile over test days of
        actual daily peak - forecast daily peak (max of the median over the 24 h).
    This uses test-period errors to estimate the margin and does not validate a reserve
    policy on unseen days. Summed over the five zones
    (each zone holds its own margin). Computed on all test days and on each zone's cold-snap
    days (25-34 days per zone, so the cold numbers are rough). Total gets a day-level
    bootstrap CI. Timestral at 4 weeks is added as 'timestral-209m_temp@4w'."""
    main = load_ctx(cfg, origin, [TEST], ctx_main, RESERVE_RUNS, load, temp)
    short = load_ctx(cfg, origin, [TEST], 672, ["timestral-209m_temp"], load, temp)
    t = pd.concat([main, short.assign(run="timestral-209m_temp@4w")])
    # real-forecast runs: drop the 2 days the archive has no forecast for
    t = t[~(t["run"].str.endswith("_temp_fc") & t["target_day"].isin(fc_missing_days(t)))]
    sn, _ = statnett_run(cfg, origin, load, temp, [TEST], ctx_main)  # the professional benchmark, same days
    if len(sn):
        t = pd.concat([t, sn[sn["run"] == "statnett"]], ignore_index=True)
    g = t.groupby(["run", "zone", "target_day"])
    d = pd.DataFrame({"actual": g["y_true"].max(), "forecast": g["q0.50"].max(), "cold": g["cold"].first()}).reset_index()
    d["err"] = d["actual"] - d["forecast"]
    rng = np.random.default_rng(0)
    days = np.array(sorted(d["target_day"].unique()))
    draws = rng.integers(0, len(days), size=(cfg["bootstrap_n"], len(days)))
    rows = []
    for reg in ["all", "cold"]:
        s = d if reg == "all" else d[d["cold"]]
        for run, r in s.groupby("run"):
            per_zone = r.groupby("zone")["err"].quantile(cover)
            # bootstrap the total: resample days, recompute each zone's quantile, sum
            boot = bootstrap_margin_totals(r, days, draws, cover)
            rows.append({"regime": reg, "run": run, **{z: v for z, v in per_zone.items()},
                         "total": per_zone.sum(), "total_lo": np.nanpercentile(boot, 2.5),
                         "total_hi": np.nanpercentile(boot, 97.5), "zone_days": len(r)})
    return pd.DataFrame(rows).set_index(["regime", "run"])


def ensemble_table(cfg, origin, load, temp, ctx=2688) -> pd.DataFrame:
    """Simple ensemble: average Timestral's and Chronos-2's quantiles hour by hour (equal
    weights, nothing fitted, so no calibration data needed). Compared with each model alone
    as paired differences on every winter (negative = ensemble better)."""
    r = load_ctx(cfg, origin, None, ctx, CONTEXT_RUNS, load, temp)
    if r.empty:
        return pd.DataFrame(), pd.DataFrame()
    # only winters where both models were run (the midnight origin lacks some)
    both_w = r.groupby("winter")["run"].nunique()
    r = r[r["winter"].isin(both_w[both_w == len(CONTEXT_RUNS)].index)]
    if r.empty:
        return pd.DataFrame(), pd.DataFrame()
    keys = ["zone", "origin_utc", "time_utc"]  # origin too: the spring DST hour is in two forecasts
    a = r[r["run"] == "timestral-209m_temp"].set_index(keys).sort_index()
    b = r[r["run"] == "chronos2_temp"].set_index(keys).reindex(a.index)
    ens = a.copy()
    ens[conformal.QCOLS] = (a[conformal.QCOLS].values + b[conformal.QCOLS].values) / 2
    ens["run"] = "ensemble"
    ens = add_row_metrics(ens.reset_index(), mase_scales(load, cfg["regime_threshold_end"]))
    both = pd.concat([r, ens], ignore_index=True)
    parts = []
    for w in sorted(both["winter"].unique()):
        for base in CONTEXT_RUNS:
            parts.append(paired_diff(both, base, "ensemble", cfg["bootstrap_n"], winters=(w,)).assign(winter=w))
    stats = [{"winter": w, "run": run, "regime": reg, **interval_stats(x if reg == "all" else x[x[reg]])}
             for (w, run), x in both[both["winter"] == TEST].groupby(["winter", "run"]) for reg in ["all", "cold"]]
    return pd.concat(parts), pd.DataFrame(stats)


def holiday_day_type(day) -> str:
    """Holiday days grouped by how load behaves: Christmas Eve/Day/Boxing Day, the days
    between Christmas and New Year ("romjul"), New Year's Day, and other public holidays."""
    if day.month == 12 and day.day in (24, 25, 26):
        return "christmas"
    if day.month == 12:
        return "romjul"
    if day.month == 1 and day.day == 1:
        return "new_year"
    return "other"


FM_RUNS = ["timestral-209m_temp", "chronos2_temp", "timesfm3_temp", "moirai2_temp"]
FM_FC_RUNS = ["timestral-209m_temp_fc", "chronos2_temp_fc", "timesfm3_temp_fc"]  # Moirai 2.0 can't use temperature


def foundation_models_table(cfg, origin, load, temp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The zero-shot foundation models with temperature, every cached winter (2021-22 included)
    at 4 and 16 weeks, plus the real-weather-forecast runs at 16 weeks.
    stats: accuracy and one-sided misses per winter, run, context and regime (all / cold).
    diff:  paired MAE differences, each model vs Timestral-209M at the same context and winter
           ("<run> vs timestral-209m_temp", or vs timestral-209m_temp_fc for the real-forecast
           runs; negative = the other model better), and each model's 16 vs 4 weeks per winter
           ("<run>@ctx2688 vs <run>", negative = 16 weeks better).
    The real-forecast runs leave out the days without an archived forecast (fc_missing_days).
    TimesFM 3.0 and Moirai 2.0 predict deciles only (no q0.05 / q0.95), so their pinball loss
    and 90% band are left empty rather than computed from missing tails."""
    stats, diffs, by_ctx = [], [], {}
    sets = [(w, ctx, FM_RUNS) for w in ALL_WINTERS for ctx in [672, 2688]] + [(w, 2688, FM_FC_RUNS) for w in MAIN_WINTERS]
    for w, ctx, runs in sets:
        m = load_ctx(cfg, origin, [w], ctx, runs, load, temp)
        if m.empty:
            continue
        if runs is FM_FC_RUNS:
            m = m[~m["target_day"].isin(fc_missing_days(m))]
        else:
            by_ctx[(w, ctx)] = m
        for run, x in m.groupby("run"):
            for reg in ["all", "cold"]:
                st = interval_stats(x if reg == "all" else x[x["cold"]])
                stats.append({"winter": w, "run": run, "context_h": ctx, "regime": reg, **st})
        if runs[0] in set(m["run"]):
            for run in sorted(set(m["run"]) - {runs[0]}):
                diffs.append(paired_diff(m, runs[0], run, cfg["bootstrap_n"], [w])
                             .assign(winter=w, context_h=ctx))
    for w in ALL_WINTERS:
        if (w, 672) in by_ctx and (w, 2688) in by_ctx:
            a, b = by_ctx[(w, 672)], by_ctx[(w, 2688)]
            for run in sorted(set(a["run"]) & set(b["run"])):
                both = pd.concat([a[a["run"] == run], b[b["run"] == run].assign(run=f"{run}@ctx2688")])
                diffs.append(paired_diff(both, run, f"{run}@ctx2688", cfg["bootstrap_n"], [w])
                             .assign(winter=w, context_h=2688))
    if not stats:
        return pd.DataFrame(), pd.DataFrame()
    return (pd.DataFrame(stats).set_index(["winter", "run", "context_h", "regime"]),
            pd.concat(diffs) if diffs else pd.DataFrame())


def statnett_run(cfg, origin, load, temp, winters, ctx=2688) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Statnett's day-ahead load forecast (ENTSO-E Transparency Platform, data_entsoe.py) as two
    pseudo-runs on the target hours of Timestral's real-forecast run (days without an archived
    weather forecast dropped, so every real-forecast comparison uses the same hours).

    Statnett forecasts ENTSO-E total load, which includes grid losses; our target, Elhub
    consumption, doesn't. So the forecast is scaled to Elhub's level per zone, with
    k = Elhub / ENTSO-E actual load summed over the target hours of:
      statnett           the calibration winter 2024-25 (fitted before the test winter)
      statnett_insample  the evaluated winter itself (uses that winter's actuals, favours Statnett)
    Point forecast only: every quantile column holds the forecast, so only MAE-type metrics
    mean anything. The raw forecast and ENTSO-E actual stay in columns statnett_raw / entsoe_actual.
    origin_utc labels our evaluation window, not Statnett's issue time; the retrieved
    series does not reconstruct publication/revision vintages at that origin.
    Returns (prepared rows of both runs, the scale factors). Empty if the data isn't downloaded."""
    P = ROOT / cfg["paths"]["processed"]
    if not (P / "entsoe_actual.parquet").exists():
        return pd.DataFrame(), pd.DataFrame()
    r = load_ctx(cfg, origin, sorted(set(winters) | set(CAL)), ctx, ["timestral-209m_temp_fc"], load, temp)
    if r.empty:
        return pd.DataFrame(), pd.DataFrame()
    r = r[~r["target_day"].isin(fc_missing_days(r))]
    s = r[["winter", "zone", "origin_utc", "time_utc", "lead_h", "y_true"]].copy()
    at = pd.MultiIndex.from_arrays([s["time_utc"], s["zone"]])
    for col, f in [("statnett_raw", "entsoe_forecast"), ("entsoe_actual", "entsoe_actual")]:
        s[col] = pd.read_parquet(P / f"{f}.parquet").stack().reindex(at).values
    s = s.dropna(subset=["statnett_raw", "entsoe_actual"])
    level = s.groupby(["winter", "zone"])[["y_true", "entsoe_actual"]].sum()
    level = (level["y_true"] / level["entsoe_actual"]).rename("k").reset_index()
    k_cal = level[level["winter"].isin(CAL)].set_index("zone")["k"]
    k_own = level.set_index(["winter", "zone"])["k"]
    runs = []
    for run, k in [("statnett", s["zone"].map(k_cal).values),
                   ("statnett_insample", k_own.reindex(pd.MultiIndex.from_arrays([s["winter"], s["zone"]])).values)]:
        x = s.assign(run=run, **{q: s["statnett_raw"] * k for q in conformal.QCOLS})
        runs.append(x[x["winter"].isin(winters)])
    return prepare(pd.concat(runs, ignore_index=True), load, temp, cfg), level


def statnett_checks(cfg, st: pd.DataFrame) -> pd.DataFrame:
    """Is Statnett's error a data artefact? Per winter and zone, on the benchmark hours, scored
    against its own target (ENTSO-E actual load, unscaled):
      MAE             as published
      MAE_best_shift  with the forecast moved by the shift that fits best (best_shift_h, -3..+3 h,
                      positive = moved later), in case of a timestamp convention problem
      MAE_debiased    after removing the zone's mean error in that winter (uses that winter's
                      actuals, so it favours Statnett), in case the error is just a level offset
      corr            hourly correlation of forecast and actual
    Zone ALL is the mean over zones (they have the same hours)."""
    fc = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "entsoe_forecast.parquet")
    rows = []
    for (w, z), x in st[st["run"] == "statnett"].groupby(["winter", "zone"]):
        t, act = pd.DatetimeIndex(x["time_utc"]), x["entsoe_actual"].values
        mae = {h: np.nanmean(np.abs(fc[z].shift(h).reindex(t).values - act)) for h in range(-3, 4)}
        best = min(mae, key=mae.get)
        err = act - x["statnett_raw"].values
        rows.append({"winter": w, "zone": z, "MAE": mae[0], "best_shift_h": best, "MAE_best_shift": mae[best],
                     "MAE_debiased": np.abs(err - err.mean()).mean(), "corr": np.corrcoef(x["statnett_raw"], act)[0, 1]})
    t = pd.DataFrame(rows)
    allz = t.groupby("winter")[["MAE", "MAE_best_shift", "MAE_debiased", "corr"]].mean().reset_index().assign(zone="ALL")
    return pd.concat([t, allz]).set_index(["winter", "zone"])


def statnett_table(cfg, origin, load, temp, ctx=2688) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The professional benchmark: Statnett's day-ahead forecast vs our models with real weather
    forecasts (and Timestral with measured temperature), each winter with real-forecast runs, on
    identical hours. Statnett's forecast is published by 10:00 CET the day before, so it is
    available at our noon-before origin, and it also uses real weather forecasts.
    stats: MAE, MAPE and bias per winter, run, zone and regime (all / cold); 'statnett_own_target'
           scores the unscaled forecast against ENTSO-E's actual load (its own target).
    diff:  paired MAE differences, each model minus Statnett (scaled on the calibration winter),
           per winter and pooled over the winters (negative = our model better).
    scale: Elhub / ENTSO-E actual load per winter and zone.
    checks: statnett_checks (timing, level offset, correlation)."""
    st, level = statnett_run(cfg, origin, load, temp, MAIN_WINTERS, ctx)
    if st.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    ours = load_ctx(cfg, origin, MAIN_WINTERS, ctx, FM_FC_RUNS + ["timestral-209m_temp"], load, temp)
    keys = ["winter", "zone", "origin_utc", "time_utc"]
    hours = st.loc[st["run"] == "statnett", keys]
    m = pd.concat([ours.merge(hours, on=keys, validate="many_to_one"), st], ignore_index=True)
    own = st[st["run"] == "statnett"].assign(run="statnett_own_target")
    own["ae"] = (own["entsoe_actual"] - own["statnett_raw"]).abs()
    own["ape"] = 100 * own["ae"] / own["entsoe_actual"]
    own["bias"] = own["entsoe_actual"] - own["statnett_raw"]
    rows = []
    for (w, run), x in pd.concat([m, own], ignore_index=True).groupby(["winter", "run"]):
        for zone, xz in [("ALL", x)] + list(x.groupby("zone")):
            for reg in ["all", "cold"]:
                sub = xz if reg == "all" else xz[xz["cold"]]
                rows.append({"winter": w, "run": run, "zone": zone, "regime": reg, "MAE": sub["ae"].mean(),
                             "MAPE": sub["ape"].mean(), "bias": sub["bias"].mean(), "zone_days": len(sub) / 24})
    diffs = []
    winters = sorted(set(m["winter"]))
    for run in sorted(set(ours["run"])):
        for ws, label in [([w], w) for w in winters] + [(winters, "all winters")]:
            if m.loc[m["winter"].isin(ws) & (m["run"] == run)].empty:
                continue
            diffs.append(paired_diff(m, "statnett", run, cfg["bootstrap_n"], ws).assign(winter=label))
    return (pd.DataFrame(rows).set_index(["winter", "run", "zone", "regime"]), pd.concat(diffs),
            level.set_index(["winter", "zone"]), statnett_checks(cfg, st))


def head_to_head_table(cfg, origin, load, temp, ctx=2688) -> pd.DataFrame:
    """Timestral vs Chronos-2 (both with temperature), paired, per winter including 2021-22,
    plus the test winter with real MET forecasts. Negative dMAE = Timestral better."""
    parts = []
    for w in ALL_WINTERS:
        m = load_ctx(cfg, origin, [w], ctx, ["timestral-209m_temp", "chronos2_temp"], load, temp)
        if len(m) and m["run"].nunique() == 2:
            parts.append(paired_diff(m, "chronos2_temp", "timestral-209m_temp", cfg["bootstrap_n"],
                                     winters=(w,)).assign(winter=w))
    fc = load_ctx(cfg, origin, [TEST], ctx, ["timestral-209m_temp_fc", "chronos2_temp_fc"], load, temp)
    if len(fc) and fc["run"].nunique() == 2:
        fc = fc[~fc["target_day"].isin(fc_missing_days(fc))]
        parts.append(paired_diff(fc, "chronos2_temp_fc", "timestral-209m_temp_fc",
                                 cfg["bootstrap_n"]).assign(winter=f"{TEST} real forecast"))
    return pd.concat(parts) if parts else pd.DataFrame()


def multi_station_table(cfg, origin, load, temp, ctx=2688) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Several stations per zone (temp_multi) vs one (temp), calibration and test winter:
    paired differences and interval stats for each model that has both runs in that winter."""
    bases = ["timestral-209m", "chronos2", "timesfm3"]
    runs = [f"{b}{s}" for b in bases for s in ["_temp", "_temp_multi"]]
    diffs, stats = [], []
    for w in CAL + [TEST]:
        m = load_ctx(cfg, origin, [w], ctx, runs, load, temp)
        have = [b for b in bases if {f"{b}_temp", f"{b}_temp_multi"} <= set(m["run"] if len(m) else [])]
        if not have:
            continue
        m = m[m["run"].isin([f"{b}{s}" for b in have for s in ["_temp", "_temp_multi"]])]
        diffs += [paired_diff(m, f"{b}_temp", f"{b}_temp_multi", cfg["bootstrap_n"], [w]).assign(winter=w)
                  for b in have]
        stats += [{"winter": w, "run": r, "regime": reg, **interval_stats(x if reg == "all" else x[x["cold"]])}
                  for r, x in m.groupby("run") for reg in ["all", "cold"]]
    if not diffs:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(diffs), pd.DataFrame(stats).set_index(["winter", "run", "regime"])


def holiday_fix_table(cfg, origin, load, temp, ctx=2688) -> pd.DataFrame:
    """Fix for holiday over-forecasting: shift the forecast on holiday days by the mean relative
    error of the median on holidays in other winters (all quantiles shifted, as apply_bias).
    Holidays are known in advance, so the shift is ex-ante.
      constant   one shift for all holiday days
      by type    one shift per holiday_day_type (Christmas / romjul / New Year / other)
    Leave one winter out over 4 winters (each winter has ~9-12 holiday days, so the evidence is
    about 4 events, and the CIs say so). dMAE = corrected minus original (negative = better)."""
    r = load_ctx(cfg, origin, None, ctx, CONTEXT_RUNS, load, temp)
    r = r[r["holiday"]].copy()
    r["htype"] = [holiday_day_type(d) for d in pd.to_datetime(r["target_day"])]
    r["rel"] = (r["y_true"] - r["q0.50"]) / r["q0.50"]
    rows = []
    for run in CONTEXT_RUNS:
        x = r[r["run"] == run]
        for variant, by in [("constant", None), ("by type", "htype")]:
            parts = []
            for w in sorted(x["winter"].unique()):
                cal, test = x[x["winter"] != w], x[x["winter"] == w].copy()
                if by is None:
                    shift = np.full(len(test), cal["rel"].mean())
                else:
                    shift = test[by].map(cal.groupby(by)["rel"].mean()).fillna(cal["rel"].mean()).values
                fixed = test.copy()
                for c in conformal.QCOLS:
                    fixed[c] = test[c].values + shift * test["q0.50"].values
                e0, e1 = test["y_true"] - test["q0.50"], fixed["y_true"] - fixed["q0.50"]
                parts.append(test[["winter", "target_day"]].assign(
                    bias_before=e0, bias_after=e1, dMAE=e1.abs() - e0.abs(),
                    dPinball=pinball(fixed).values - pinball(test).values,
                    cov_before=((test["y_true"] >= test["q0.10"]) & (test["y_true"] <= test["q0.90"])).astype(float),
                    cov_after=((fixed["y_true"] >= fixed["q0.10"]) & (fixed["y_true"] <= fixed["q0.90"])).astype(float),
                    below_before=(test["y_true"] < test["q0.10"]).astype(float),
                    below_after=(fixed["y_true"] < fixed["q0.10"]).astype(float), g=1))
            d = pd.concat(parts)
            ci = bootstrap_ci(d, ["g"], ["dMAE", "dPinball"], cfg["bootstrap_n"]).iloc[0]
            mae0 = d["bias_before"].abs().mean()
            rows.append({"model": run, "variant": variant, "days": d["target_day"].nunique(),
                         **d[["bias_before", "bias_after", "dMAE", "dPinball", "cov_before", "cov_after",
                              "below_before", "below_after"]].mean(), "dMAE_pct": 100 * d["dMAE"].mean() / mae0, **ci})
    return pd.DataFrame(rows).set_index(["model", "variant"])


def save(df: pd.DataFrame, name: str, cfg):
    out = ROOT / cfg["paths"]["results"] / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv")
    (out / f"{name}.md").write_text(df.round(3).to_markdown(), encoding="utf-8")


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--origin", default=cfg["origin"], choices=["midnight", "noon_before"])
    p.add_argument("--context", type=int, default=cfg["context_hours"], help="context length (hours) of the main runs")
    args = p.parse_args()
    origin, cfg["context_hours"] = args.origin, args.context
    sfx = f"_{origin}_ctx{args.context}"  # 4-week (672) and 16-week (2688) outputs side by side

    load = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "load.parquet")
    temp = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "temperature.parquet")
    fc = load_runs(cfg, origin)
    fc = fc[~fc["run"].str.endswith("_temp_cold")]  # probe runs are only used in probe_table
    m = prepare(fc, load, temp, cfg)
    m = add_noisy_day_temp(m, cfg)  # the temperature forecast the noisy runs were given (all rows)
    if any(m["run"].str.endswith("_temp_fc")):
        m = add_fc_day_temp(m, cfg)
        # Days without an archived forecast (temp_fc fell back to observed temperature): drop them
        # from the real-forecast runs so those runs are evaluated on real forecasts only
        skip = fc_missing_days(m)
        m = m[~(m["run"].str.endswith("_temp_fc") & m["target_day"].isin(skip))]
        print(f"temp_fc runs: dropped {len(skip)} target days without an archived forecast: {sorted(skip)}")
    runs = set(m["run"])
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(f"origin = {origin}; cached runs: {sorted(runs)}")

    thr = thresholds(load, temp, cfg["regime_threshold_end"])
    save(thr, "thresholds", cfg)
    save(historical_slopes(load, temp, cfg["regime_threshold_end"]).to_frame().join(
        load.mean().rename("mean_load_MW")), "historical_slopes", cfg)
    counts = m[m["run"] == "snaive"].groupby(["winter", "zone"])[["cold", "peak", "holiday", "normal"]].sum() // 1
    save(counts, "regime_hours" + sfx, cfg)

    main = main_table(m, cfg)
    save(main, "main_table" + sfx, cfg)
    print(main.xs("ALL", level="zone").round(3).to_string())

    if any(r.endswith("_temp_fc") for r in runs) or (ROOT / "data" / "processed" / "temp_forecast.parquet").exists():
        werr = weather_error_table(m, temp, cfg)
        save(werr, "weather_forecast_error" + sfx, cfg)
        print("\nReal temperature forecast error (forecast - observed, °C), test winter\n", werr.round(2).to_string())

    cov = covariate_table(m, cfg)
    if not cov.empty:
        save(cov, "covariates" + sfx, cfg)
        print("\nPaired differences (b - a), test winter\n", cov.round(3).to_string())

    for run in ["timestral-209m", "timestral-209m_temp", "timestral-209m_temp_holiday"]:
        if run in runs and m.loc[m["run"] == run, "winter"].nunique() == 4:
            for name, t in christmas_table(m, run, cfg["bootstrap_n"]).items():
                save(t, f"christmas_{name}_{run}{sfx}", cfg)
                print(f"\nChristmas, {run}, {name}\n", t.round(3).to_string())

    for run in ["timestral-209m", "timestral-209m_temp", "timestral-209m_temp_fc", "chronos2_temp_fc"]:
        if run in runs:
            tab, corr = conformal_table(m, run, cfg)
            save(tab.set_index(["variant", "calibration", "regime"]), f"conformal_{run}{sfx}", cfg)
            save(corr, f"conformal_corrections_{run}{sfx}", cfg)
            print(f"\n{run}\n", tab[tab["regime"].isin(["all", "cold"])].round(3).to_string(index=False))

    for suffix, what in [("", "load history only"), ("_temp", "with temperature")]:
        if all(r + suffix in runs for r in SIZES):
            t = sizes_table(m, suffix)
            save(t, f"sizes{suffix}{sfx}", cfg)
            fig_sizes(t, cfg, f"sizes{suffix}{sfx}",
                      f"Timestral sizes, test winter 2025-26, {origin.replace('_', '-')} origin, {what}")
            print(f"\nTimestral sizes, {what}\n", t[["params_M", "MAE", "MAPE", "pinball", "cov80",
                                                             "cov90", "sec_per_forecast"]].round(3).to_string())

    # Holiday flag judged on every cached winter (4 Christmases) instead of the test winter only
    a, b = "timestral-209m_temp", "timestral-209m_temp_holiday"
    if all(m.loc[m["run"] == r, "winter"].nunique() == 4 for r in (a, b)):
        hol = paired_diff(m, a, b, cfg["bootstrap_n"], winters=sorted(m["winter"].unique()))
        save(hol, "holiday_flag_4winters" + sfx, cfg)
        print("\nHoliday flag over 4 winters (b - a)\n", hol.round(3).to_string())

    ctx = context_table(cfg, origin, load, temp)
    if not ctx.empty:
        save(ctx, "context" + sfx, cfg)
        print("\nContext length vs 4 weeks (b - a)\n", ctx.round(3).to_string())
        if origin == "noon_before":
            nov = novelty_table(cfg, origin, load, temp)
            save(nov, "context_novelty" + sfx, cfg)
            print("\nWhy 16 weeks helps: target day colder than the context?\n", nov.round(2).to_string())
    acit = aci_table(cfg, origin, load, temp)
    if not acit.empty:
        save(acit.set_index(["model", "context_h", "variant", "regime"]), "aci" + sfx, cfg)
        print("\nOnline recalibration (ACI) vs static CQR\n", acit[["model", "context_h", "variant", "regime", "cov80",
              "above80", "below80", "width80", "cov90", "pinball"]].round(3).to_string(index=False))

    cq, cq_tabs = cold_cqr_table(cfg, origin, load, temp)
    if not cq.empty:
        save(cq.set_index(["model", "context_h", "eval", "variant", "regime"]), "cold_cqr" + sfx, cfg)
        save(cq_tabs, "cold_cqr_corrections" + sfx, cfg)
        print("\nOne-sided CQR by temperature bin\n", cq[["model", "context_h", "eval", "variant", "regime", "cov80",
              "above80", "below80", "width80"]].round(3).to_string(index=False))
    if cfg["context_hours"] == 2688 and origin == "noon_before":
        # TimesFM 3.0 (added later): at each history length where all three fitting winters are cached
        full = [c for c in [672, 2688]
                if load_ctx(cfg, origin, CAL3, c, ["timesfm3_temp"], load, temp).get("winter", pd.Series()).nunique() == len(CAL3)]
        if full:
            cq_tf, _ = cold_cqr_table(cfg, origin, load, temp, runs=["timesfm3_temp"])
            cq_tf = cq_tf[cq_tf["context_h"].isin(full)]
            save(cq_tf.set_index(["model", "context_h", "eval", "variant", "regime"]), "cold_cqr_timesfm3" + sfx, cfg)

    if any(r.endswith("_temp_fc") for r in runs):
        cqf, cqf_tabs = cold_cqr_fc_table(m, cfg)
        if not cqf.empty:
            save(cqf.set_index(["model", "calibration", "variant", "regime"]), "cold_cqr_fc" + sfx, cfg)
            save(cqf_tabs, "cold_cqr_fc_corrections" + sfx, cfg)
            print("\nOne-sided CQR by forecast temperature bin, real weather forecasts\n", cqf[["model", "calibration", "variant",
                  "regime", "cov80", "above80", "below80", "width80"]].round(3).to_string(index=False))

    res = reserve_table(cfg, origin, load, temp, ctx_main=cfg["context_hours"])
    if not res.empty:
        save(res, "reserve" + sfx, cfg)
        print("\nRetrospective peak-error margins (sum of zone-wise test-period 90th percentiles, MW)\n",
              res.round(0).to_string())

    ens_d, ens_s = ensemble_table(cfg, origin, load, temp, ctx=cfg["context_hours"])
    save(ens_d, "ensemble_diff" + sfx, cfg)
    save(ens_s, "ensemble_stats" + sfx, cfg)
    print("\nEnsemble (mean of Timestral and Chronos-2 quantiles) vs each model\n", ens_d.round(2).to_string())

    hol = holiday_fix_table(cfg, origin, load, temp, ctx=cfg["context_hours"])
    save(hol, "holiday_fix" + sfx, cfg)
    print("\nHoliday shift correction, leave one winter out\n", hol.round(3).to_string())

    # Main-setting extras (16 weeks): head-to-head per winter, several stations, fifth winter
    if cfg["context_hours"] == 2688 and origin == "noon_before":
        h2h = head_to_head_table(cfg, origin, load, temp)
        save(h2h, "timestral_vs_chronos2" + sfx, cfg)
        print("\nTimestral vs Chronos-2 per winter (b - a)\n", h2h.round(2).to_string())
        fm_stats, fm_diff = foundation_models_table(cfg, origin, load, temp)
        if not fm_stats.empty:
            save(fm_stats, "foundation_models_stats" + sfx, cfg)
            save(fm_diff, "foundation_models_diff" + sfx, cfg)
            print("\nFoundation models, test winter\n", fm_stats.round(3).to_string())
        sn_stats, sn_diff, sn_scale, sn_checks = statnett_table(cfg, origin, load, temp)
        if not sn_stats.empty:
            save(sn_stats, "statnett_stats" + sfx, cfg)
            save(sn_diff, "statnett_diff" + sfx, cfg)
            save(sn_scale, "statnett_scale" + sfx, cfg)
            save(sn_checks, "statnett_checks" + sfx, cfg)
            print("\nOur models minus Statnett's day-ahead forecast (b - a)\n",
                  sn_diff.xs("all", level="regime")[["dMAE", "dMAE_lo", "dMAE_hi", "dMAE_pct", "winter"]].round(2).to_string())
        ms_diff, ms_stats = multi_station_table(cfg, origin, load, temp)
        if not ms_diff.empty:
            save(ms_diff, "multi_station_diff" + sfx, cfg)
            save(ms_stats, "multi_station_stats" + sfx, cfg)
            print("\nSeveral stations vs one\n", ms_diff.round(2).to_string())
        for k, v in robust_w2021(cfg, origin, load, temp).items():
            save(v, f"robust_w2021_{k}_{origin}", cfg)
            print(f"\nRobustness, 2021-22: {k}\n", v.round(3).to_string())

    peaks = peak_table(m, cfg)
    save(peaks, "peaks" + sfx, cfg)
    print("\nDaily peak hour, test winter\n", peaks.round(3).to_string())

    probe = probe_table(cfg, origin, load, temp)
    if not probe.empty:
        save(probe.set_index(["model", "context_h", "regime", "zone"]), "probe" + sfx, cfg)
        save(probe_summary(cfg, origin, load, temp).set_index(["model", "context_h", "stat"]), "probe_summary" + sfx, cfg)
        # replication on the calibration winter 2024-25 (run_probe.sh, second block)
        rep = probe_summary(cfg, origin, load, temp, winters=CAL)
        if not rep.empty:
            save(rep.set_index(["model", "context_h", "stat"]), "probe_summary_w2024" + sfx, cfg)
        print("\nTemperature sensitivity probe (MW per °C, model vs history)\n", probe.round(2).to_string(index=False))
        pc, pc_coef = probe_context(cfg, origin, load, temp)
        save(pc, "probe_context_rows" + sfx, cfg)
        save(pc_coef, "probe_context" + sfx, cfg)
        print("\nProbe: window length vs window temperature range\n", pc_coef.round(3).to_string(index=False))

    # The fix: bias correction + CQR (oracle temperature, and the realistic noisy-temperature run)
    # Name suffix, run, calibration run, temperature column. "_caloracle": the noisy run corrected
    # with a fit on the oracle run (all we had before the noisy calibration winters were run).
    for name_sfx, run, cal_run, tcol in FIX_RUNS:
        if run in runs and cal_run in runs:
            tab, params = fix_table(m[m["run"].isin({run, cal_run})], run, cal_run, cfg, tcol)
            if params.empty:
                continue  # calibration winters not cached
            save(tab.set_index(["variant", "calibration", "regime"]), f"fix_{run}{name_sfx}{sfx}", cfg)
            save(params, f"fix_params_{run}{name_sfx}{sfx}", cfg)
            print(f"\nFix, {run}{name_sfx}\n", params.round(4).to_string(index=False), "\n",
                  tab[tab["regime"] != "holiday"].round(3).to_string(index=False))
            if run == cal_run and m.loc[m["run"] == run, "winter"].nunique() == 4:
                lowo = lowo_table(m, run, cfg, tcol)
                save(lowo, f"fix_lowo_{run}{sfx}", cfg)
                print(f"\nLeave-one-winter-out check of the bias correction, {run}\n", lowo.round(3).to_string())
            if run == "timestral-209m_temp" and "3 winters" in set(params["calibration"]):
                fig_fix(tab, cfg, sfx)
                r = m[m["run"] == run]
                v, _ = fix_variants(r[r["winter"].isin(CAL3)], r[r["winter"] == TEST], cfg)
                fig_fan(v["none"], v["bias + CQR mondrian"], cfg, sfx)

    fig_regimes(m, cfg, sfx)


if __name__ == "__main__":
    main()

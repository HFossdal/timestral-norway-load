"""Rolling-origin day-ahead forecasts for every model, cached to parquet.

Usage:
    python -m src.forecast --runs snaive lgbm timestral-209m chronos2
    python -m src.forecast --runs timestral-209m_temp --winters w2024 w2025
    python -m src.forecast --list

Setup (see config.yaml):
- One forecast per target day D in each evaluation winter. The target window is the
  24 UTC hours starting at local midnight of D. On the spring DST day (23 local hours)
  this window therefore ends at 00:00 local on D+1; we keep 24 hours so every forecast
  has the same shape (logged once per run).
- origin "midnight": the forecast is made at local midnight of D (lead 0-23 h).
  origin "noon_before": made at 12:00 local on D-1; the model forecasts 36 h and we keep
  the last 24 (lead 12-35 h).
- Context = the `context_hours` hours strictly before the origin. Nothing at or after the
  origin is used, except covariates, whose horizon values are observed temperatures
  (oracle) or the 0/1 holiday calendar (truly known in advance).
"""
import argparse
import logging
import time

import numpy as np
import pandas as pd

from src import ROOT, load_config
from src.models import QUANTILES, Chronos2, LightGBMQuantile, Moirai2, SeasonalNaive, Timestral, TimesFM3
from src.regimes import LOCAL_TZ, is_holiday

log = logging.getLogger("forecast")
QCOLS = [f"q{q:.2f}" for q in QUANTILES]
COLD_SHIFT = 3.0  # °C, the temperature sensitivity probe (temp_cold runs)


def make_runs(cfg) -> dict:
    """Name -> constructor. Constructors are lazy so only the requested models are loaded."""
    runs = {
        "snaive": lambda: SeasonalNaive(),
        "lgbm": lambda: LightGBMQuantile(use_temp=False),
        "lgbm_temp": lambda: LightGBMQuantile(use_temp=True),
    }
    for covs in [(), ("temp",), ("temp", "holiday"), ("temp_noisy",), ("temp_cold",), ("temp_fc",), ("temp_multi",)]:
        suffix = "".join(f"_{c}" for c in covs)
        runs[f"chronos2{suffix}"] = lambda covs=covs: Chronos2(covariates=covs)
        runs[f"timesfm3{suffix}"] = lambda covs=covs: TimesFM3(covariates=covs)
        runs[f"moirai2{suffix}"] = lambda covs=covs: Moirai2(covariates=covs)
        for size in cfg["timestral_sizes"]:
            runs[f"timestral-{size.lower()}{suffix}"] = lambda size=size, covs=covs: Timestral(size, covariates=covs)
    return runs


def load_data(cfg):
    load = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "load.parquet")
    temp = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "temperature.parquet")
    assert load.index.equals(temp.index), "load and temperature must share one hourly UTC index"
    assert not load.isna().any().any() and not temp.isna().any().any(), "unexpected NaN in inputs"
    return load, temp


def forecast_days(winter: dict, origin_mode: str):
    """Target-window starts (UTC), origins (UTC) and prediction length H for one winter."""
    local_midnights = pd.date_range(winter["start"], winter["end"], freq="D", inclusive="left", tz=LOCAL_TZ)
    target_start = local_midnights.tz_convert("UTC")
    if origin_mode == "midnight":
        origins = target_start
    elif origin_mode == "noon_before":
        origins = (local_midnights - pd.Timedelta(hours=12)).tz_convert("UTC")
    else:
        raise ValueError(origin_mode)
    lead0 = (target_start - origins) / pd.Timedelta(hours=1)
    assert (lead0 == lead0[0]).all(), "lead time must be constant within a winter"
    return target_start, origins, int(lead0[0]) + 24


def noise_offset(origin, zone_index: int, sd: float) -> float:
    """The temperature-forecast error for one forecast: N(0, sd), seeded by (origin hour, zone)
    so every model, and the bias correction in conformal.py, sees the same draw."""
    return np.random.default_rng([int(origin.timestamp()) // 3600, zone_index]).normal(0, sd)


def forecast_temps(origins, zones, H) -> dict:
    """Archived MET Nordic temperature forecasts over the horizon of each origin (see
    data_weather_forecast.py): zone -> (n_origins, H) array, NaN where not downloaded.
    Run used = the 06 UTC run of the origin's UTC day, at least 5 h before the origin
    (published in time: the noon-before origin is 10-11 UTC, the midnight origin 22-23 UTC)."""
    path = ROOT / "data" / "processed" / "temp_forecast.parquet"
    if not path.exists():
        return {z: np.full((len(origins), H), np.nan) for z in zones}
    wide = pd.read_parquet(path).pivot_table(index=["run_utc", "time_utc"], columns="zone", values="temp_fc")
    runs = (origins - pd.Timedelta(hours=5)).normalize() + pd.Timedelta(hours=6)
    keys = pd.MultiIndex.from_arrays([np.repeat(runs, H),
                                      np.concatenate([o + pd.to_timedelta(np.arange(H), "h") for o in origins])])
    vals = wide.reindex(keys)
    return {z: vals[z].values.reshape(len(origins), H) if z in vals else np.full((len(origins), H), np.nan)
            for z in zones}


def multi_station_temps(index) -> pd.DataFrame | None:
    """Population-weighted zone temperature from several stations (data_weather_multi.py),
    on the load index; None if it has not been built."""
    path = ROOT / "data" / "processed" / "temperature_multi.parquet"
    if not path.exists():
        return None
    tm = pd.read_parquet(path).reindex(index)
    assert not tm.isna().any().any(), "NaN in temperature_multi"
    return tm


def build_batch(load, temp, holiday, zones, origins, T, H, noise_sd):
    """Stack every (zone, origin) problem into arrays. Rows are ordered zone-major."""
    pos = load.index.get_indexer(origins)
    assert (pos >= T).all() and (pos + H <= len(load)).all(), "context or horizon outside the data"
    past = np.stack([np.arange(p - T, p) for p in pos])        # (n_origins, T) row positions
    future = np.stack([np.arange(p, p + H) for p in pos])      # (n_origins, H)
    covs = ["temp", "holiday", "temp_noisy", "temp_cold", "temp_fc", "temp_multi"]
    fc = forecast_temps(origins, zones, H)
    tm = multi_station_temps(load.index)
    tm = temp if tm is None else tm  # placeholder so the batch always has the key; only temp_multi runs use it
    b = {"context": [], "cov_past": {k: [] for k in covs}, "cov_future": {k: [] for k in covs},
         "target_times": [], "origin": [], "zone": []}
    hol = holiday.values
    for zi, z in enumerate(zones):
        y, t = load[z].values, temp[z].values
        b["context"].append(y[past])
        b["cov_past"]["temp"].append(t[past])
        b["cov_future"]["temp"].append(t[future])
        b["cov_past"]["holiday"].append(hol[past])
        b["cov_future"]["holiday"].append(hol[future])
        # Noisy-temperature robustness: past = observed, horizon = observed + one N(0, sd) offset
        # per forecast (a day-ahead weather forecast tends to be off by a similar amount all day).
        # Seeded by (origin hour, zone) so every model sees the same noise.
        offsets = np.array([noise_offset(o, zi, noise_sd) for o in origins])
        b["cov_past"]["temp_noisy"].append(t[past])
        b["cov_future"]["temp_noisy"].append(t[future] + offsets[:, None])
        # Sensitivity probe: the same forecast, but the future is COLD_SHIFT °C colder. Comparing
        # it with the plain temp run gives the model's load response in MW per °C (report.py).
        b["cov_past"]["temp_cold"].append(t[past])
        b["cov_future"]["temp_cold"].append(t[future] - COLD_SHIFT)
        # Real weather forecast: past = observed, horizon = the archived MET Nordic forecast
        b["cov_past"]["temp_fc"].append(t[past])
        # The archive has gaps (no runs 16 Mar 12 UTC - 18 Mar 06 UTC 2026): those origins get the
        # observed temperature instead, and report.py drops their target days from every
        # comparison involving temp_fc (fc_missing_days).
        b["cov_future"]["temp_fc"].append(np.where(np.isnan(fc[z]), t[future], fc[z]))
        # Several stations per zone, population-weighted (past and future, observed like "temp")
        b["cov_past"]["temp_multi"].append(tm[z].values[past])
        b["cov_future"]["temp_multi"].append(tm[z].values[future])
        b["target_times"].append(load.index.values[future])
        b["origin"].extend(origins)
        b["zone"].extend([z] * len(origins))
    b["context"] = np.concatenate(b["context"])
    for part in ("cov_past", "cov_future"):
        b[part] = {k: np.concatenate(v) for k, v in b[part].items()}
    b["target_times"] = np.concatenate(b["target_times"])
    return b


def run_one(model, run_name, winter_name, cfg, load, temp, holiday, overwrite=False):
    origin_mode, T = cfg["origin"], cfg["context_hours"]
    out_dir = ROOT / cfg["paths"]["results"] / "cache" / "forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_name}__{origin_mode}__ctx{T}__{winter_name}.parquet"
    if path.exists() and not overwrite:
        log.info("cached: %s", path.name)
        return

    winter = cfg["evaluation_winters"][winter_name]
    target_start, origins, H = forecast_days(winter, origin_mode)
    zones = cfg["zones"]

    if isinstance(model, LightGBMQuantile):
        # Train only on data strictly before this winter's first origin
        model.fit(load, temp, train_end=origins[0], origin_hour_local=-12 if origin_mode == "noon_before" else 0, H=H)

    batch = build_batch(load, temp, holiday, zones, origins, T, H, cfg["temp_noise_sd"])
    if run_name.endswith("_temp_fc"):
        missing = np.isnan(forecast_temps(origins, zones, H)[zones[0]]).any(axis=1)
        assert missing.mean() < 0.05, "temperature forecasts missing: run data_weather_forecast"
        log.info("temp_fc: %d of %d origins had no archived forecast and use observed temperature "
                 "(excluded in report.py): %s", missing.sum(), len(origins), list(origins[missing].date))
    t0 = time.perf_counter()
    q = model.predict(batch, H)                                 # (n, H, Q)
    sec_per_forecast = (time.perf_counter() - t0) / len(q)
    q = q[:, -24:, :]                                           # keep the 24 target hours
    times = batch["target_times"][:, -24:]

    n = len(q)
    df = pd.DataFrame({
        "run": run_name, "winter": winter_name,
        "zone": np.repeat(batch["zone"], 24),
        "origin_utc": np.repeat(pd.DatetimeIndex(batch["origin"]), 24),
        "time_utc": pd.DatetimeIndex(times.ravel()).tz_localize("UTC"),
        "lead_h": np.tile(np.arange(H - 24, H), n),
    })
    for j, c in enumerate(QCOLS):
        df[c] = q[:, :, j].ravel()
    df["y_true"] = np.concatenate([load[z].reindex(pd.DatetimeIndex(t).tz_localize("UTC")).values
                                   for z, t in zip(batch["zone"], times)])
    df["sec_per_forecast"] = sec_per_forecast
    df.to_parquet(path)
    log.info("%s %s: %d forecasts, %.3f s/forecast -> %s", run_name, winter_name, n, sec_per_forecast, path.name)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    runs = make_runs(cfg)
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=[])
    p.add_argument("--winters", nargs="+", default=list(cfg["evaluation_winters"]))
    p.add_argument("--origin", choices=["midnight", "noon_before"], help="override config 'origin'")
    p.add_argument("--context", type=int, help="override config 'context_hours' (sensitivity run)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    if args.list:
        print("\n".join(runs))
        return
    if args.origin:
        cfg["origin"] = args.origin
    if args.context:
        cfg["context_hours"] = args.context

    load, temp = load_data(cfg)
    holiday = is_holiday(load.index)
    log.info("origin=%s, context=%dh. DST: the spring-forward day's window is 24 UTC hours "
             "(ends 00:00 local next day); all data is UTC so contexts have no gaps.",
             cfg["origin"], cfg["context_hours"])
    for name in args.runs:
        model = runs[name]()
        for w in args.winters:
            run_one(model, name, w, cfg, load, temp, holiday, args.overwrite)


if __name__ == "__main__":
    main()

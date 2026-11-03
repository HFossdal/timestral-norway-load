"""Hourly air temperature per bidding zone from the MET Norway Frost API.

Usage:
    python -m src.data_weather --find-stations   # list candidate stations, pick IDs for config.yaml
    python -m src.data_weather                   # download + clean -> data/processed/temperature.parquet

API facts used (from frost.met.no/howto.html, concepts2.html and swagger.json):
- Auth: HTTP basic auth with the client ID as username and an empty password.
- Times are UTC; intervals "t0/t1" include t0 and exclude t1.
- Hourly temperature: elements=air_temperature, timeresolutions=PT1H.
  timeoffsets=default and levels=default pick the standard series (2 m height).
"""
import argparse
import logging
import os

import pandas as pd
import requests
from dotenv import load_dotenv

from src import ROOT, load_config

FROST = "https://frost.met.no"
log = logging.getLogger("data_weather")


def frost_get(endpoint: str, params: dict) -> dict:
    """GET a Frost endpoint. Returns the parsed JSON, or {} if Frost reports no data (404)."""
    load_dotenv(ROOT / ".env")
    client_id = os.environ.get("FROST_CLIENT_ID")
    if not client_id:
        raise RuntimeError("FROST_CLIENT_ID is missing. Copy .env.example to .env and fill it in.")
    r = requests.get(f"{FROST}/{endpoint}", params=params, auth=(client_id, ""), timeout=120)
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    return r.json()


def find_stations(cfg: dict) -> None:
    """Print stations matching each zone's name filter that have air temperature since history_start."""
    for zone, name in cfg["weather"]["station_name_search"].items():
        res = frost_get("sources/v0.jsonld", {
            "types": "SensorSystem",
            "name": name,
            "elements": "air_temperature",
            "validtime": f"{cfg['history_start']}/now",
            "fields": "id,name,validFrom,validTo,geometry,masl",
        })
        print(f"\n{zone} (search '{name}'):")
        for s in res.get("data", []):
            lon, lat = s.get("geometry", {}).get("coordinates", [None, None])
            print(f"  {s['id']:<10} {s['name']:<35} lat={lat} lon={lon} masl={s.get('masl')} "
                  f"valid {s.get('validFrom', '')[:10]} -> {s.get('validTo', 'now')[:10]}")


def download_station(station: str, start: str, end: str) -> pd.DataFrame:
    """Download hourly air temperature for one station, one calendar year per request.

    Raw responses are cached per year in data/raw/frost/ so reruns don't hit the API.
    Returns a DataFrame with UTC 'time', 'temp' and 'quality'.
    """
    cache_dir = ROOT / "data" / "raw" / "frost"
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        t0 = max(pd.Timestamp(start), pd.Timestamp(f"{year}-01-01"))
        t1 = min(pd.Timestamp(end), pd.Timestamp(f"{year + 1}-01-01"))
        if t0 >= t1:
            continue
        cache = cache_dir / f"{station}_{t0:%Y%m%d}_{t1:%Y%m%d}.parquet"
        if cache.exists():
            frames.append(pd.read_parquet(cache))
            continue
        res = frost_get("observations/v0.jsonld", {
            "sources": station,
            "elements": "air_temperature",
            "timeresolutions": "PT1H",
            "timeoffsets": "default",
            "levels": "default",
            "referencetime": f"{t0:%Y-%m-%d}/{t1:%Y-%m-%d}",
        })
        rows = []
        for item in res.get("data", []):
            obs = item["observations"]
            if len(obs) > 1:  # should not happen with the default filters; keep the first and say so
                log.warning("%s %s: %d observations at one time, keeping first", station, item["referenceTime"], len(obs))
            rows.append({"time": item["referenceTime"], "temp": obs[0]["value"], "quality": obs[0].get("qualityCode")})
        df = pd.DataFrame(rows, columns=["time", "temp", "quality"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.to_parquet(cache)
        log.info("%s %s: %d hourly values", station, year, len(df))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame, index: pd.DatetimeIndex, max_gap: int, label: str) -> pd.Series:
    """Put one station's temperatures on the full hourly UTC index.

    - Frost quality code 7 ("erroneous") is set to NaN. Codes 0-6 are kept; code 6 means
      Frost filled a missing value automatically, so we count and log those.
    - Duplicate timestamps: keep the first (logged).
    - Gaps of up to max_gap hours: linear interpolation (temperature is smooth on that scale).
    - Longer gaps: left as NaN and logged; both models accept NaN as 'missing'.
    """
    df = df.copy()
    df.loc[df["quality"] == 7, "temp"] = float("nan")
    log.info("%s: %d values flagged erroneous (code 7) set to NaN, %d model-filled by Frost (code 6)",
             label, int((df["quality"] == 7).sum()), int((df["quality"] == 6).sum()))
    s = df.set_index("time")["temp"]
    n_dup = s.index.duplicated().sum()
    if n_dup:
        log.warning("%s: dropping %d duplicate timestamps", label, n_dup)
        s = s[~s.index.duplicated()]
    s = s.reindex(index)
    is_na = s.isna()
    n_missing = int(is_na.sum())
    # Length of the NaN run each hour belongs to (run id increments whenever is_na flips)
    run_id = is_na.ne(is_na.shift()).cumsum()
    run_len = is_na.groupby(run_id).transform("size")
    s = s.interpolate(limit_area="inside")
    s[is_na & (run_len > max_gap)] = float("nan")  # long gaps stay fully missing
    log.info("%s: %d missing hours (%.2f%%), %d filled by interpolation, %d left NaN in gaps > %dh",
             label, n_missing, 100 * n_missing / len(index), n_missing - int(s.isna().sum()),
             int(s.isna().sum()), max_gap)
    return s


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--find-stations", action="store_true")
    args = parser.parse_args()
    cfg = load_config()

    if args.find_stations:
        find_stations(cfg)
        return

    stations = cfg["weather"]["stations"]
    missing = [z for z, s in stations.items() if not s]
    if missing:
        raise SystemExit(f"No station ID set for {missing} in config.yaml. Run with --find-stations first.")

    index = pd.date_range(cfg["history_start"], cfg["data_end"], freq="h", tz="UTC", inclusive="left")
    out = {}
    for zone in cfg["zones"]:
        raw = download_station(stations[zone], cfg["history_start"], cfg["data_end"])
        out[zone] = clean(raw, index, cfg["weather"]["max_interpolate_hours"], f"{zone} ({stations[zone]})")

    temps = pd.DataFrame(out, index=index)
    temps.index.name = "time_utc"
    path = ROOT / cfg["paths"]["processed"] / "temperature.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    temps.to_parquet(path)
    log.info("Wrote %s with shape %s", path, temps.shape)


if __name__ == "__main__":
    main()

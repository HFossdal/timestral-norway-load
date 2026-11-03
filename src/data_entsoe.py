"""Statnett's day-ahead load forecast and ENTSO-E "actual total load" per bidding zone, from the
ENTSO-E Transparency Platform. Used only as the professional benchmark; the target stays Elhub
consumption (data_load.py).

Usage:
    python -m src.data_entsoe    # -> data/processed/entsoe_forecast.parquet, entsoe_actual.parquet

Needs ENTSOE_API_TOKEN in .env.

API behaviour (entsoe-py 0.8.1, read from its source on 2026-11-02):
- query_load_forecast(area, start, end) requests documentType A65 (system total load) with
  processType A01 (day ahead); query_load requests the same document with A16 (realised).
  Both split the period into monthly requests and return one column in local time.
- start/end must be timezone-aware. Neighbouring monthly requests can both return the
  boundary timestamp, so duplicates are dropped.
- The resolution changes from 60 to 15 minutes on 18 Mar 2025 (both series, all zones).
  Everything is averaged to hourly MW, and the resolution found in each year is logged.
- By EU rules (Regulation 543/2013, art. 6.1(b)) the day-ahead forecast is published at the
  latest two hours before day-ahead gate closure, so by 10:00 CET the day before: it is
  available at our noon-before origin.
- ENTSO-E total load includes grid losses; Elhub consumption doesn't, so the two differ in level.
"""
import logging
import os

import pandas as pd
import requests
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

from src import ROOT, load_config

AREA = {"NO1": "NO_1", "NO2": "NO_2", "NO3": "NO_3", "NO4": "NO_4", "NO5": "NO_5"}
log = logging.getLogger("data_entsoe")


def download_year(client, kind: str, zone: str, year: int, end: pd.Timestamp) -> pd.Series:
    """One zone, one local calendar year (cut at `end`), hourly MW in UTC, cached as raw parquet."""
    cache_dir = ROOT / "data" / "raw" / "entsoe"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{kind}_{zone}_{year}.parquet"
    if cache.exists():
        return pd.read_parquet(cache).iloc[:, 0]

    start = pd.Timestamp(f"{year}-01-01", tz="Europe/Oslo")
    stop = min(pd.Timestamp(f"{year + 1}-01-01", tz="Europe/Oslo"), end)
    query = client.query_load_forecast if kind == "forecast" else client.query_load
    try:
        df = query(AREA[zone], start=start, end=stop)
    except requests.HTTPError as e:
        # the error message holds the request URL, which contains the token: report the status only
        raise RuntimeError(f"ENTSO-E {kind} {zone} {year}: HTTP {e.response.status_code}") from None
    s = df.iloc[:, 0].astype(float)
    s = s[~s.index.duplicated()].tz_convert("UTC")
    step = s.index.to_series().diff().mode()
    log.info("%s %s %d: %d values, resolution %s", kind, zone, year, len(s), step.iloc[0] if len(step) else "?")
    s = s.resample("h").mean()  # MW is an average power, so averaging 15-minute values gives the hour
    s.to_frame(kind).to_parquet(cache)
    return s


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    load_dotenv(ROOT / ".env")
    token = os.environ.get("ENTSOE_API_TOKEN")
    if not token:
        raise RuntimeError("ENTSOE_API_TOKEN is missing. Copy .env.example to .env and fill it in.")
    client = EntsoePandasClient(api_key=token)

    index = pd.date_range(cfg["history_start"], cfg["data_end"], freq="h", tz="UTC", inclusive="left")
    end = pd.Timestamp(cfg["data_end"], tz="Europe/Oslo")
    years = range(index[0].year, index[-1].year + 1)
    for kind in ["forecast", "actual"]:
        out = pd.DataFrame(index=index)
        for zone in cfg["zones"]:
            s = pd.concat([download_year(client, kind, zone, y, end) for y in years])
            out[zone] = s[~s.index.duplicated()].reindex(index)
            n_na = int(out[zone].isna().sum())
            log.info("%s %s: %d missing hours (%.2f%%)", kind, zone, n_na, 100 * n_na / len(index))
        out.index.name = "time_utc"
        path = ROOT / cfg["paths"]["processed"] / f"entsoe_{kind}.parquet"
        out.to_parquet(path)
        log.info("Wrote %s with shape %s (MW)", path, out.shape)


if __name__ == "__main__":
    main()

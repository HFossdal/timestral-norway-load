"""Archived MET Nordic temperature forecasts at our five weather stations.

Replaces the synthetic noise run with what an operator really had at the noon-before origin.

Source: MET Norway's operational MET Nordic forecast archive (the forecast behind yr.no),
1 km grid, runs at 00/06/12/18 UTC, 58-64 h long, bias-corrected against observations
(https://github.com/metno/NWPdocs/wiki/MET-Nordic-dataset). Read via OPeNDAP, so only the
five grid points are transferred, not the full 1 km fields.

Which run: the origin is 12:00 local on D-1 (11:00 UTC in winter). We use the 06 UTC run of
D-1, which is published within a couple of hours, so it is available at the origin. It covers
the whole 36 h horizon (lead 5-41 h). Grid point = nearest 1 km cell to the station.

Usage:
    python -m src.data_weather_forecast --winters w2025 w2024
Output: data/processed/temp_forecast.parquet, long format
    run_utc, time_utc, zone, temp_fc (°C)
"""
import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import netCDF4
import numpy as np
import pandas as pd
import pyproj

from src import ROOT, load_config
from src.regimes import LOCAL_TZ

log = logging.getLogger("weather_forecast")
URL = "https://thredds.met.no/thredds/dodsC/metpparchive/{t:%Y/%m/%d}/met_forecast_1_0km_nordic_{t:%Y%m%dT%H}Z.nc"
RUN_HOUR = 6  # UTC
# Station coordinates (lon, lat) from the Frost sources endpoint
STATIONS = {"NO1": (10.72, 59.9423), "NO2": (8.0767, 58.2), "NO3": (10.4538, 63.4107),
            "NO4": (18.9368, 69.6537), "NO5": (5.3327, 60.383)}


def grid_index(ds) -> dict:
    """Nearest grid cell (row, col) to each station, in the file's Lambert projection."""
    tr = pyproj.Transformer.from_crs("EPSG:4326", pyproj.CRS.from_proj4(ds["projection_lcc"].proj4), always_xy=True)
    x, y = ds["x"][:], ds["y"][:]
    out = {}
    for z, (lon, lat) in STATIONS.items():
        px, py = tr.transform(lon, lat)
        out[z] = (int(np.abs(y - py).argmin()), int(np.abs(x - px).argmin()))
    return out


def read_points(url: str, idx: dict) -> pd.DataFrame:
    ds = netCDF4.Dataset(url)
    try:
        times = pd.to_datetime(ds["time"][:], unit="s", utc=True)
        return pd.concat([pd.DataFrame({"time_utc": times, "zone": z,
                                        "temp_fc": np.asarray(ds["air_temperature_2m"][:, i, j]) - 273.15})
                          for z, (i, j) in idx.items()])
    finally:
        ds.close()


def fetch_run(run: pd.Timestamp, idx: dict) -> pd.DataFrame | None:
    """Station-point temperatures for one origin's run, or None if nothing could be read.

    Tries the 06 UTC run three times (the server sometimes answers "DAP failure"), then falls
    back to the 00 UTC run of the same day, which is older but also available at the origin.
    run_utc is the nominal 06 UTC slot (the lookup key); source_run is the run actually used.
    """
    last = None
    for source in [run, run - pd.Timedelta(hours=RUN_HOUR)]:
        for attempt in range(3):
            try:
                d = read_points(URL.format(t=source), idx)
                return d.assign(run_utc=run, source_run=source)
            except (OSError, RuntimeError) as e:  # missing file or a server/connection error
                last = e
                time.sleep(5 * (attempt + 1))
    log.warning("no forecast for run %s (%s)", run, last)
    return None


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--winters", nargs="+", default=["w2025"])
    args = p.parse_args()

    out = ROOT / cfg["paths"]["processed"] / "temp_forecast.parquet"
    have = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    runs = []
    for w in args.winters:
        winter = cfg["evaluation_winters"][w]
        # one run per target day D: the 06 UTC run on D-1
        days = pd.date_range(winter["start"], winter["end"], freq="D", inclusive="left", tz=LOCAL_TZ)
        runs += [(d.tz_localize(None) - pd.Timedelta(days=1) + pd.Timedelta(hours=RUN_HOUR)).tz_localize("UTC")
                 for d in days]
    if len(have):
        runs = [r for r in runs if r not in set(have["run_utc"])]
    log.info("%d runs to fetch", len(runs))
    if not runs:
        return

    idx = grid_index(netCDF4.Dataset(URL.format(t=runs[0])))
    log.info("grid cells: %s", idx)
    # Two parallel connections (processes: the netCDF library is not thread-safe). Saved after
    # every chunk, so an interruption loses at most one chunk.
    fetch = partial(fetch_run, idx=idx)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for c in range(0, len(runs), 20):
            got = [d for d in pool.map(fetch, runs[c:c + 20]) if d is not None]
            if got:
                have = pd.concat([have] + got, ignore_index=True)
                have.to_parquet(out)
            log.info("%d / %d runs done", min(c + 20, len(runs)), len(runs))
    n_fallback = (have["source_run"] != have["run_utc"]).groupby(have["run_utc"]).any().sum()
    log.info("have %d runs (%d from the 00 UTC fallback)", have["run_utc"].nunique(), n_fallback)
    log.info("saved %s", out)


if __name__ == "__main__":
    main()

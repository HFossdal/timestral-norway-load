"""Hourly electricity consumption per bidding zone from Elhub's open energy-data API.

Usage:
    python -m src.data_load      # download + clean -> data/processed/load.parquet

Why Elhub and not ENTSO-E: Elhub (Norway's national metering hub) needs no API key.
Its consumption is metered end-use per price area, so it excludes grid losses and is a
few percent below ENTSO-E's "actual total load". The daily, weekly and temperature
patterns are the same.

API behaviour (checked by hand on 2026-10-31):
- Dataset CONSUMPTION_PER_GROUP_MBA_HOUR returns hourly kWh per price area and per
  consumption group (household, cabin, primary, secondary, tertiary).
- At most one month per request. startDate/endDate are read as Oslo wall-clock times
  (any offset is ignored) and the end is exclusive, so we request whole local months.
- Returned startTime values carry their UTC offset (+01:00 / +02:00); during the autumn
  DST change the repeated 02:00 hour appears twice with different offsets. Converting
  to UTC therefore gives unique, gap-free hours.
"""
import logging

import pandas as pd
import requests

from src import ROOT, load_config

URL = "https://api.elhub.no/energy-data/v0/price-areas"
DATASET = "CONSUMPTION_PER_GROUP_MBA_HOUR"
log = logging.getLogger("data_load")


def download_month(month: pd.Timestamp) -> pd.DataFrame:
    """All price areas and groups for one local calendar month, cached as raw parquet.

    Months that may still be revised (the current and previous month) are not cached.
    """
    cache_dir = ROOT / "data" / "raw" / "elhub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{month:%Y-%m}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    nxt = month + pd.offsets.MonthBegin(1)
    r = requests.get(URL, params={"dataset": DATASET, "startDate": f"{month:%Y-%m-%d}",
                                  "endDate": f"{nxt:%Y-%m-%d}"}, timeout=300)
    r.raise_for_status()
    rows = [x for area in r.json()["data"] for x in area["attributes"]["consumptionPerGroupMbaHour"]]
    df = pd.DataFrame(rows)[["startTime", "priceArea", "consumptionGroup", "quantityKwh", "meteringPointCount"]]
    df["time"] = pd.to_datetime(df["startTime"], utc=True)  # offset-aware strings -> exact UTC
    df = df.drop(columns="startTime")
    if nxt < pd.Timestamp.now() - pd.DateOffset(months=2):
        df.to_parquet(cache)
    log.info("%s: %d rows", f"{month:%Y-%m}", len(df))
    return df


def to_zone_load(raw: pd.DataFrame, zones: list[str], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Sum consumption groups per zone and hour -> MW (kWh in one hour / 1000), on the full UTC index.

    Hours where not all 5 groups are present are set to NaN and logged, because a missing
    group would show up as a fake drop in load.
    """
    raw = raw[raw["priceArea"].isin(zones)]
    n_groups = raw["consumptionGroup"].nunique()
    g = raw.groupby(["time", "priceArea"])
    load = g["quantityKwh"].sum().unstack() / 1000.0
    complete = g["consumptionGroup"].nunique().unstack() == n_groups
    n_dup = raw.duplicated(["time", "priceArea", "consumptionGroup"]).sum()
    if n_dup:
        log.warning("%d duplicate (hour, zone, group) rows were summed; check the raw data", n_dup)
    load = load.where(complete).reindex(index)[zones]
    for z in zones:
        n_na = int(load[z].isna().sum())
        log.info("%s: %d missing or incomplete hours (%.2f%%)", z, n_na, 100 * n_na / len(index))
    return load


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    index = pd.date_range(cfg["history_start"], cfg["data_end"], freq="h", tz="UTC", inclusive="left")

    # Local months covering the UTC index (first month starts one hour before 2021-01-01 00:00 UTC)
    months = pd.date_range(pd.Timestamp(cfg["history_start"]).replace(day=1), cfg["data_end"], freq="MS")
    raw = pd.concat([download_month(m) for m in months], ignore_index=True)

    load = to_zone_load(raw, cfg["zones"], index)
    load.index.name = "time_utc"
    path = ROOT / cfg["paths"]["processed"] / "load.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    load.to_parquet(path)
    log.info("Wrote %s with shape %s (MW)", path, load.shape)


if __name__ == "__main__":
    main()

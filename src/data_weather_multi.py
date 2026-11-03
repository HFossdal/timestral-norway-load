"""Population-weighted zone temperature from several Frost stations per bidding zone.

Addresses the "one weather station per zone" limitation. A zone's load responds to the
temperature where people (and buildings) are, so each zone gets 3-5 towns, each standing for
its surrounding region, weighted by that region's population.

Station choice (a fixed rule, decided before any model run):
  for each town, the nearest MET station (road-weather stations "E6 ...", "RV ...", "FV ..."
  excluded) within 25 km that has air temperature over the whole period with
  < 2% missing hours and < 2% values filled by Frost (quality code 6).
Weights are approximate regional populations (thousands, rounded, 2024), e.g. NO1's Oslo point
stands for Oslo + Akershus. Rough is fine here: the point is to spread the weight over the zone.

Zone temperature per hour = weighted mean over the stations that have a value that hour
(weights renormalised if one is missing). The regime labels (cold snaps) keep using the
original single stations, so only the models' temperature input changes.

Usage:
    python -m src.data_weather_multi
Output:
    data/processed/temperature_multi.parquet   (same shape as temperature.parquet)
    results/tables/stations_multi.csv          (chosen station per town, with quality numbers)
"""
import logging
import re

import numpy as np
import pandas as pd
import requests

from src import ROOT, load_config
from src.data_weather import clean, download_station, frost_get

log = logging.getLogger("data_weather_multi")

# zone -> [(town, lon, lat, regional population in thousands)]
TOWNS = {
    "NO1": [("Oslo", 10.75, 59.91, 1430), ("Drammen", 10.20, 59.74, 270),
            ("Fredrikstad", 10.93, 59.22, 310), ("Hamar", 11.07, 60.79, 370)],
    "NO2": [("Stavanger", 5.73, 58.97, 490), ("Kristiansand", 7.99, 58.15, 190),
            ("Arendal", 8.77, 58.46, 120), ("Skien", 9.61, 59.21, 180)],
    "NO3": [("Trondheim", 10.40, 63.43, 350), ("Steinkjer", 11.50, 64.01, 130),
            ("Ålesund", 6.15, 62.47, 150), ("Molde", 7.16, 62.74, 120)],
    "NO4": [("Tromsø", 18.96, 69.65, 120), ("Harstad", 16.54, 68.80, 50), ("Bodø", 14.40, 67.28, 120),
            ("Mo i Rana", 14.14, 66.31, 120), ("Alta", 23.27, 69.97, 75)],
    "NO5": [("Bergen", 5.32, 60.39, 450), ("Voss", 6.42, 60.63, 50),
            ("Førde", 5.85, 61.45, 100), ("Sogndal", 7.10, 61.23, 50)],
}
ROAD = re.compile(r"^(E\d+|RV\s?\d+|FV\s?\d+)\b")
MAX_KM, MAX_MISSING, MAX_FILLED = 25, 0.02, 0.02


def candidates(lon: float, lat: float, since: str) -> list[dict]:
    res = frost_get("sources/v0.jsonld", {
        "types": "SensorSystem", "geometry": f"nearest(POINT({lon} {lat}))", "nearestmaxcount": 10,
        "elements": "air_temperature", "validtime": f"{since}/now",
        "fields": "id,name,validFrom,validTo,geometry,masl,distance"})
    return [s for s in res.get("data", []) if not ROAD.match(s["name"]) and s.get("distance", 99) <= MAX_KM]


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    start, end = cfg["history_start"], cfg["data_end"]
    index = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    chosen, series = [], {}
    for zone, towns in TOWNS.items():
        for town, lon, lat, pop in towns:
            pick = None
            for s in candidates(lon, lat, start):
                try:
                    raw = download_station(s["id"], start, end)
                except requests.HTTPError as e:  # Frost 412: no hourly series matching the filters
                    log.info("%s %s: %s %s has no usable hourly series (%s) -> skip", zone, town, s["id"], s["name"],
                             e.response.status_code)
                    continue
                raw = raw[(raw["time"] >= index[0]) & (raw["time"] <= index[-1])]
                missing = 1 - raw["time"].nunique() / len(index)
                filled = float((raw["quality"] == 6).mean()) if len(raw) else 1.0
                ok = missing < MAX_MISSING and filled < MAX_FILLED
                log.info("%s %s: %s %s, %.1f km, missing %.1f%%, Frost-filled %.1f%% -> %s", zone, town, s["id"],
                         s["name"], s["distance"], 100 * missing, 100 * filled, "OK" if ok else "skip")
                if ok:
                    pick = {"zone": zone, "town": town, "station": s["id"], "name": s["name"], "km": round(s["distance"], 1),
                            "masl": s.get("masl"), "missing_pct": round(100 * missing, 2),
                            "filled_pct": round(100 * filled, 2), "weight_k": pop}
                    series[(zone, town)] = clean(raw, index, cfg["weather"]["max_interpolate_hours"], f"{zone} {town} ({s['id']})")
                    break
            if pick is None:
                log.warning("%s %s: no station passed the rule; town dropped", zone, town)
            else:
                chosen.append(pick)
    st = pd.DataFrame(chosen)
    out = {}
    for zone, g in st.groupby("zone"):
        vals = np.column_stack([series[(zone, t)].values for t in g["town"]])
        w = np.where(np.isnan(vals), 0.0, g["weight_k"].values.astype(float))
        out[zone] = np.nansum(np.nan_to_num(vals) * w, axis=1) / w.sum(axis=1)
    temps = pd.DataFrame(out, index=index)[cfg["zones"]]
    temps.index.name = "time_utc"
    log.info("hours with no station at all: %s", temps.isna().sum().to_dict())
    temps.to_parquet(ROOT / cfg["paths"]["processed"] / "temperature_multi.parquet")
    st.to_csv(ROOT / cfg["paths"]["results"] / "tables" / "stations_multi.csv", index=False)
    single = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "temperature.parquet")
    cmp = pd.DataFrame({"corr": [temps[z].corr(single[z]) for z in cfg["zones"]],
                        "mean_diff_C": [(temps[z] - single[z]).mean() for z in cfg["zones"]],
                        "sd_diff_C": [(temps[z] - single[z]).std() for z in cfg["zones"]]}, index=cfg["zones"])
    print(st.to_string(index=False))
    print("\nWeighted zone temperature vs the single station\n", cmp.round(3).to_string())


if __name__ == "__main__":
    main()

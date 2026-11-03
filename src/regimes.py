"""Regime labels: holidays, cold snaps and peak hours.

All thresholds are computed per zone on history before the test winter (config
`regime_threshold_end`), never on the test winter itself.

- cold snap: the target day's local daily mean temperature is below the zone's 5th
  percentile of daily mean temperature. Uses observed temperature, i.e. the same oracle
  as the covariate (in operation you would use the temperature forecast).
- peak (reporting): realised load above the zone's 95th percentile of hourly load.
  This uses the outcome, so it is only for *reporting* errors.
- peak_exante: the model's own median forecast above that same threshold. Known at
  forecast time, so it can be used to *choose* a conformal correction.
- holiday: Norwegian public holidays plus 24-31 December.
"""
import datetime as dt

import holidays
import numpy as np
import pandas as pd

LOCAL_TZ = "Europe/Oslo"


def holiday_dates(years) -> set[dt.date]:
    """Norwegian public holidays plus 24-31 December.

    The days between Christmas and New Year are not official holidays, but much of Norway
    is off work and load behaves like a holiday ("romjul"), so we treat them as holidays.
    Sundays are not counted (include_sundays=False is set explicitly).
    """
    days = set(holidays.Norway(years=list(years), include_sundays=False).keys())
    for y in years:
        days |= {dt.date(y, 12, d) for d in range(24, 32)}
    return days


def is_holiday(times_utc: pd.DatetimeIndex) -> pd.Series:
    """1.0 if the local calendar day of each UTC timestamp is a holiday, else 0.0."""
    local_days = times_utc.tz_convert(LOCAL_TZ).date
    days = holiday_dates(range(times_utc.min().year - 1, times_utc.max().year + 2))
    return pd.Series([float(d in days) for d in local_days], index=times_utc)


def daily_mean_temp(temp: pd.DataFrame) -> pd.DataFrame:
    """Mean temperature per local calendar day (index: datetime.date)."""
    local = temp.tz_convert(LOCAL_TZ)
    return local.groupby(local.index.date).mean()


def thresholds(load: pd.DataFrame, temp: pd.DataFrame, end: str, cold_q=0.05, peak_q=0.95) -> pd.DataFrame:
    """Per-zone cold-snap (°C) and peak-load (MW) thresholds from history before `end` (UTC date)."""
    cut = pd.Timestamp(end, tz="UTC")
    dmt = daily_mean_temp(temp.loc[temp.index < cut])
    return pd.DataFrame({
        "cold_temp": dmt.quantile(cold_q),
        "peak_load": load.loc[load.index < cut].quantile(peak_q),
    })


def label(fc: pd.DataFrame, temp: pd.DataFrame, thr: pd.DataFrame) -> pd.DataFrame:
    """Add target_day, regime flags and the exclusive Mondrian group to a forecast table."""
    fc = fc.copy()
    # The target day is the local date of the first hour of each forecast window
    first_hour = fc.groupby(["run", "zone", "origin_utc"])["time_utc"].transform("min")
    fc["target_day"] = first_hour.dt.tz_convert(LOCAL_TZ).dt.date

    dmt = daily_mean_temp(temp).stack().rename("day_temp")  # (date, zone) -> °C
    fc["day_temp"] = dmt.reindex(pd.MultiIndex.from_arrays([fc["target_day"], fc["zone"]])).values
    fc["cold"] = fc["day_temp"] < fc["zone"].map(thr["cold_temp"])
    fc["peak"] = fc["y_true"] > fc["zone"].map(thr["peak_load"])
    fc["peak_exante"] = fc["q0.50"] > fc["zone"].map(thr["peak_load"])
    fc["holiday"] = is_holiday(pd.DatetimeIndex(fc["time_utc"])).values.astype(bool)

    # Reporting regime (realised): overlapping flags, "normal" = none of them
    fc["normal"] = ~(fc["cold"] | fc["peak"] | fc["holiday"])
    # Mondrian group (known at forecast time, exclusive): cold day > forecast peak > other
    fc["group"] = np.select([fc["cold"], fc["peak_exante"]], ["cold", "peak_exante"], default="other")
    return fc

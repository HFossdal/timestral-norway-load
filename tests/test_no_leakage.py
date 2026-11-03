"""Leakage checks: a forecast may only see data from before its origin, and the regime
thresholds may only use history before the test winter.

Run: python -m pytest tests -q
"""
import numpy as np
import pandas as pd

from src import forecast, regimes


def hour_coded_data(zones=("NO1", "NO2")):
    """Load and temperature where every value is its own row number, so each array in a
    forecast batch shows exactly which hours it was built from."""
    idx = pd.date_range("2025-09-01", "2026-01-01", freq="h", tz="UTC", inclusive="left")
    load = pd.DataFrame({z: np.arange(len(idx), dtype=float) for z in zones}, index=idx)
    return load, load.copy(), pd.Series(np.zeros(len(idx)), index=idx)


def test_forecast_batch_only_sees_the_past(monkeypatch):
    # the archived weather forecasts and multi-station temperatures live in data/, not in the
    # repo: stub them so the test is the same on a fresh clone
    monkeypatch.setattr(forecast, "forecast_temps", lambda o, z, H: {k: np.full((len(o), H), np.nan) for k in z})
    monkeypatch.setattr(forecast, "multi_station_temps", lambda index: None)
    load, temp, holiday = hour_coded_data()
    zones = list(load.columns)
    T = 672
    for mode in ["noon_before", "midnight"]:
        target_start, origins, H = forecast.forecast_days({"start": "2025-11-01", "end": "2025-12-01"}, mode)
        b = forecast.build_batch(load, temp, holiday, zones, origins, T, H, noise_sd=1.5)
        pos = np.tile(load.index.get_indexer(origins), len(zones))  # rows are zone-major
        # the history is exactly the T hours before the origin, never the origin hour itself
        assert (b["context"].min(axis=1) == pos - T).all()
        assert (b["context"].max(axis=1) == pos - 1).all()
        for name, past in b["cov_past"].items():
            if name != "holiday":
                assert (past.max(axis=1) < pos).all(), name
        # known future covariates cover the horizon only, starting at the origin
        assert (b["cov_future"]["temp"][:, 0] == pos).all()
        assert (b["cov_future"]["temp"][:, -1] == pos + H - 1).all()
        # the target day lies inside the horizon: noon-before = 12 h ahead, 36 h horizon
        lead = (target_start - origins) / pd.Timedelta(hours=1)
        assert (lead == (12 if mode == "noon_before" else 0)).all()
        assert H == lead[0] + 24


def test_thresholds_ignore_data_after_the_cutoff():
    load, temp, _ = hour_coded_data()
    cut = "2025-11-01"
    before = regimes.thresholds(load, temp, cut)
    # make everything from the test winter on absurdly cold and high: nothing may change
    later = temp.index >= pd.Timestamp(cut, tz="UTC")
    temp2, load2 = temp.copy(), load.copy()
    temp2.loc[later] = -50.0
    load2.loc[later] = 1e9
    pd.testing.assert_frame_equal(regimes.thresholds(load2, temp2, cut), before)

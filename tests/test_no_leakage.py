"""Leakage checks: a forecast may only see data from before its origin, and the regime
thresholds may only use history before the test winter.

Run: python -m pytest tests -q
"""
import numpy as np
import pandas as pd
import pytest

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


@pytest.mark.parametrize("suffix,temp_col", [("_temp_fc", "day_temp_fc"), ("_temp_noisy", "day_temp_noisy")])
def test_realistic_correction_groups_do_not_see_future_outcomes(suffix, temp_col):
    times = pd.date_range("2025-01-01", "2025-01-04", freq="h", tz="UTC")
    local_days = times.tz_convert("Europe/Oslo").day
    observed = pd.DataFrame({"NO1": np.where(local_days == 2, -12., 3.)}, index=times)
    target = pd.to_datetime(["2025-01-02", "2025-01-03"], utc=True)
    fc = pd.DataFrame({"run": "timestral-209m" + suffix, "zone": "NO1", "time_utc": target,
                       "origin_utc": target - pd.Timedelta(hours=12), "y_true": 100., "q0.50": 100.})
    thr = pd.DataFrame({"cold_temp": [-5.], "peak_load": [150.]}, index=["NO1"])
    a = regimes.label(fc, observed, thr)
    assert not a.group_available.any()
    a[temp_col] = [-1., -8.]
    a = regimes.set_forecast_groups(a, a.run.str.endswith(suffix), temp_col)
    b = regimes.label(fc.assign(y_true=10000.), -observed, thr)
    b[temp_col] = [-1., -8.]
    b = regimes.set_forecast_groups(b, b.run.str.endswith(suffix), temp_col)
    assert list(a["group"]) == list(b["group"]) == ["other", "cold"]
    assert list(a.cold) != list(b.cold)
    assert not a.peak.any() and b.peak.all()


def test_missing_weather_in_any_zone_excludes_the_forecast_day(monkeypatch):
    from src import report
    origins = pd.date_range("2025-01-01", periods=2, tz="UTC")
    def forecasts(o, zones, H):
        values = {z: np.zeros((len(o), H)) for z in zones}
        if "NO2" in values:
            values["NO2"][1, 0] = np.nan
        return values
    monkeypatch.setattr(forecast, "forecast_temps", forecasts)
    frame = pd.DataFrame({"zone": ["NO1", "NO2", "NO1", "NO2"],
                          "origin_utc": origins.repeat(2), "target_day": list(origins.date.repeat(2))})
    assert report.fc_missing_days(frame) == {origins[1].date()}

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.bootstrap import day_draws, bootstrap_margin_totals
from src.report import bootstrap_ci, paired_diff


@pytest.mark.parametrize("block", [1, 3, 7])
def test_blocks_are_consecutive_and_stay_in_their_winter(block):
    f = pd.DataFrame({"winter": np.repeat(["a", "b"], 20),
                      "target_day": list(pd.date_range("2023-11-01", periods=20).date) +
                                    list(pd.date_range("2025-11-01", periods=20).date)})
    calendar, draws = day_draws(f, 50, block)
    for offset in [0, 20]:
        picks = draws[:, offset:offset + 20]
        assert ((picks >= offset) & (picks < offset + 20)).all()
        for j in range(0, 20, block):
            assert (np.diff(picks[:, j:min(j + block, 20)], axis=1) == 1).all()
    assert len(calendar) == 40


def test_missing_calendar_day_does_not_join_nonconsecutive_days():
    f = pd.DataFrame({"target_day": [dt.date(2025, 1, 1), dt.date(2025, 1, 3)]})
    calendar, _ = day_draws(f, 5, 3)
    assert calendar.get_level_values("target_day")[1] == dt.date(2025, 1, 2)


def test_one_day_matches_original_iid_draws():
    f = pd.DataFrame({"target_day": pd.date_range("2025-01-01", periods=10).date})
    _, draws = day_draws(f, 30, 1)
    np.testing.assert_array_equal(draws, np.random.default_rng(0).integers(0, 10, (30, 10)))


def test_zone_dependence_is_preserved():
    days = pd.date_range("2025-01-01", periods=30).date
    f = pd.DataFrame({"target_day": list(days) * 2, "model": "a",
                      "error": np.r_[np.arange(30), -np.arange(30)]})
    for block in [1, 3, 7]:
        t = bootstrap_ci(f, ["model"], ["error"], 100, block_days=block)
        assert t.loc["a", "error_lo"] == t.loc["a", "error_hi"] == 0


def test_paired_comparison_uses_only_common_hours():
    days = pd.date_range("2025-01-01", periods=20, tz="UTC")
    a = pd.DataFrame({"winter": "w2025", "zone": "NO1", "time_utc": days, "origin_utc": days,
                      "target_day": days.date, "cold": False, "peak": False, "holiday": False,
                      "normal": True, "ae": np.arange(20) + 10., "pinball": 0., "in80": True, "run": "a"})
    b = a.iloc[1:].assign(run="b", ae=lambda x: x.ae - 2)
    for block in [1, 3, 7]:
        t = paired_diff(pd.concat([a, b]), "a", "b", 100, block_days=block)
        r = t.loc[("b vs a", "all")]
        assert r.dMAE == r.dMAE_lo == r.dMAE_hi == -2
        assert r.n_days == 19


def test_overlapping_dst_target_hours_match_on_origin():
    days = pd.date_range("2026-03-01", periods=30, tz="UTC")
    a = pd.DataFrame({"winter": "w2025", "zone": "NO1", "time_utc": days, "origin_utc": days,
                      "target_day": days.date, "cold": False, "peak": False, "holiday": False,
                      "normal": True, "ae": np.arange(30) + 10., "pinball": 0., "in80": True, "run": "a"})
    a.loc[29, "time_utc"] = a.loc[28, "time_utc"]  # same hour, different forecast origins
    b = a.assign(run="b", ae=lambda x: x.ae - 2)
    t = paired_diff(pd.concat([a, b]), "a", "b", 100, block_days=7)
    assert t.loc[("b vs a", "all"), "dMAE"] == -2


def test_vectorised_peak_margin_matches_scalar_bootstrap_with_missing_days():
    days = pd.date_range("2025-01-01", periods=12).date
    errors = pd.DataFrame({"zone": ["a"] * 12 + ["b"] * 9,
                           "target_day": list(days) + list(days[3:]),
                           "err": list(np.arange(12.) ** 2) + list(np.arange(9.) * -3)})
    draws = np.random.default_rng(0).integers(0, len(days), (50, len(days)))
    draws[0] = 0  # no observations from zone b in this replicate
    reference = []
    for pick in draws:
        total = 0.
        for _, g in errors.groupby("zone"):
            values = g.set_index("target_day")["err"].reindex(days[pick]).dropna()
            total += np.quantile(values, .9) if len(values) else np.nan
        reference.append(total)
    np.testing.assert_allclose(bootstrap_margin_totals(errors, days, draws), reference, equal_nan=True)


@pytest.mark.parametrize("block", [1, 3, 7])
def test_probe_model_interaction_cancels_shared_fluctuations(monkeypatch, block):
    from src import report
    monkeypatch.setattr(report, "historical_slopes", lambda *a: pd.Series({"NO1": -1., "NO2": -1.}))
    days = pd.date_range("2025-01-01", periods=30, tz="UTC")
    def cached(cfg, origin, winters, ctx, runs, load, temp):
        model = runs[0].split("_temp")[0]
        base = {"timestral-209m": 1., "chronos2": 4., "timesfm3": 2.}
        if model not in base or ctx not in [672, 2688]:
            return pd.DataFrame()
        rows = []
        for zi, zone in enumerate(["NO1", "NO2"]):
            ratio = base[model] + .1 * np.cos(np.arange(30))
            if ctx == 2688:
                # All models share a variable context effect; pairing must make
                # it cancel in the difference between their context changes.
                ratio += (2. if model == "timestral-209m" else 1.) + .5 * np.sin(np.arange(30) / 3) + .2 * zi
            rows.append(pd.DataFrame({"winter": winters[0], "zone": zone, "origin_utc": days,
                                      "time_utc": days + pd.Timedelta(hours=12), "target_day": days.date,
                                      "q0.50": 100 + (3 * ratio if runs[0].endswith("_cold") else np.zeros(30))}))
        return pd.concat(rows, ignore_index=True)
    monkeypatch.setattr(report, "load_ctx", cached)
    d = report.probe_summary({"regime_threshold_end": "2025-11-01", "bootstrap_n": 100},
                             "noon_before", None, None, block_days=block, model_contrasts=True)
    interactions = d[d.stat == "difference in context change"]
    assert len(interactions) == 2
    np.testing.assert_allclose(interactions[["value", "lo", "hi"]].values, 1., atol=1e-12)

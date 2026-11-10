"""Tests for src/conformal.py on synthetic data where the right answer is known.

Run: python -m pytest tests -q
"""
import numpy as np
import pandas as pd
import pytest

from src import conformal

Z90 = 1.2816  # standard normal 90th percentile


def forecasts(n, rng, temps=None, shift_cold=0.0, sd_model=1.0):
    """y ~ N(mu, 1); the model's quantiles assume sd = sd_model (too narrow if < 1). In bins
    below 0 °C the truth is shifted up by shift_cold (a one-sided cold miss)."""
    t = rng.uniform(-15, 5, n) if temps is None else temps
    mu = rng.normal(100, 10, n)
    y = mu + rng.normal(0, 1, n) + shift_cold * (t < 0)
    q = {"q0.05": mu - 1.6449 * sd_model, "q0.10": mu - Z90 * sd_model, "q0.50": mu,
         "q0.90": mu + Z90 * sd_model, "q0.95": mu + 1.6449 * sd_model}
    return pd.DataFrame({"y_true": y, "day_temp": t, **q})


def miss(d, lo="q0.10", hi="q0.90"):
    return (d["y_true"] > d[hi]).mean(), (d["y_true"] < d[lo]).mean()


def test_correction_is_the_conformal_order_statistic():
    # Romano et al. 2019: Q = the ceil((n+1)(1-alpha))-th smallest score
    s = pd.Series(np.arange(1, 10, dtype=float))        # n = 9, alpha = 0.2 -> 8th smallest
    assert conformal.correction(s, 80) == 8.0
    s = pd.Series(np.arange(1, 241, dtype=float))       # n = 240, alpha = 0.1 -> 217th smallest
    assert conformal.correction(s, 90) == 217.0
    s = pd.Series(np.arange(1, 5, dtype=float))         # too few points: capped at the maximum
    assert conformal.correction(s, 90) == 4.0


def test_cqr_restores_marginal_coverage():
    rng = np.random.default_rng(0)
    cal = forecasts(20000, rng, sd_model=0.6)            # bands far too narrow
    test = forecasts(20000, rng, sd_model=0.6)
    before = 1 - sum(miss(test))
    out, _ = conformal.recalibrate(cal, test, "pooled")
    after = 1 - sum(miss(out))
    assert before < 0.6
    assert after == pytest.approx(0.80, abs=0.01)
    assert 1 - sum(miss(out, "q0.05", "q0.95")) == pytest.approx(0.90, abs=0.01)


def test_one_sided_bins_balance_a_one_sided_cold_miss():
    rng = np.random.default_rng(1)
    cal = forecasts(30000, rng, shift_cold=1.0)
    test = forecasts(30000, rng, shift_cold=1.0)
    cold = test["day_temp"] < 0
    above, below = miss(test[cold])
    assert above > 0.3 and below < 0.05                  # the failure: misses on one side
    two, _ = conformal.recalibrate_asym(cal, test, two_sided=True)
    one, _ = conformal.recalibrate_asym(cal, test, two_sided=False)
    a2, b2 = miss(two[cold])
    a1, b1 = miss(one[cold])
    assert a2 - b2 > 0.05                                # two-sided per bin stays lopsided
    assert a1 == pytest.approx(0.10, abs=0.015)          # one-sided: 10% on each side
    assert b1 == pytest.approx(0.10, abs=0.015)
    a, b = miss(one[~cold])                              # mild bins untouched in spirit
    assert a == pytest.approx(0.10, abs=0.015) and b == pytest.approx(0.10, abs=0.015)


def test_small_bin_falls_back_to_pooled():
    rng = np.random.default_rng(2)
    t = np.concatenate([rng.uniform(-4, 5, 5000), rng.uniform(-15, -11, 100)])  # only 100 h below -10
    cal = forecasts(len(t), rng, temps=t)
    table = conformal.fit_asym(cal, 80, min_n=240).set_index("bin")
    pooled = conformal.fit_asym(cal.assign(day_temp=1.0), 80, min_n=1).set_index("bin").loc[">= 0"]
    assert table.loc["< -10", "fallback"]
    assert table.loc["< -10", "n"] == 100
    assert table.loc["< -10", "Q_hi"] == pooled["Q_hi"]
    assert not table.loc["-5 to 0", "fallback"]


def test_recalibrated_quantiles_stay_ordered():
    rng = np.random.default_rng(3)
    cal, test = forecasts(5000, rng, shift_cold=2.0), forecasts(5000, rng, shift_cold=2.0)
    for out, _ in [conformal.recalibrate(cal, test, "pooled"), conformal.recalibrate_asym(cal, test)]:
        v = out[conformal.QCOLS].values
        assert (np.diff(v, axis=1) >= 0).all()


def test_decile_only_model_keeps_its_columns():
    # TimesFM 3.0 / Moirai 2.0 have no q0.05 / q0.95: recalibration must fix the 80% band,
    # leave the median alone and not shift the other columns into the empty ones
    rng = np.random.default_rng(5)
    cal, test = (forecasts(20000, rng, shift_cold=1.0).assign(**{"q0.05": np.nan, "q0.95": np.nan})
                 for _ in range(2))
    out, _ = conformal.recalibrate_asym(cal, test, two_sided=False)
    assert (out["q0.50"] == test["q0.50"]).all()
    assert out["q0.05"].isna().all() and out["q0.95"].isna().all()
    cold = test["day_temp"] < 0
    a, b = miss(out[cold])
    assert a == pytest.approx(0.10, abs=0.015) and b == pytest.approx(0.10, abs=0.015)


def test_bias_correction_recovers_known_parameters():
    rng = np.random.default_rng(4)
    t = rng.uniform(-20, 10, 5000)
    q50 = rng.uniform(1000, 5000, 5000)
    a, b = 0.01, 0.004                                   # rel. error = a + b * max(0, 0 - T)
    y = q50 * (1 + a + b * np.maximum(0, -t))
    cal = pd.DataFrame({"y_true": y, "q0.50": q50, "day_temp": t})
    a_hat, b_hat = conformal.fit_bias(cal)
    assert a_hat == pytest.approx(a, abs=1e-9) and b_hat == pytest.approx(b, abs=1e-9)


def test_mondrian_requires_available_groups_but_pooled_does_not():
    d = pd.DataFrame({"group": ["unavailable"], "group_available": [False],
                      "y_true": [1.], "q0.10": [0.], "q0.50": [1.], "q0.90": [2.]})
    with pytest.raises(ValueError, match="available at the forecast origin"):
        conformal.fit(d, "mondrian", 80)
    with pytest.raises(ValueError, match="available at the forecast origin"):
        conformal.apply(d, pd.DataFrame({"group": ["other"], "Q": [0.], "n": [1]}), 80)
    assert len(conformal.fit(d, "pooled", 80)) == 1

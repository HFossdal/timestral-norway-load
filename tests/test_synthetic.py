import numpy as np
import pandas as pd
import pytest

from src import synthetic as s


@pytest.fixture
def temperatures():
    idx = pd.date_range(s.ORIGINS.min() - pd.Timedelta(hours=max(s.CONTEXTS) + 48),
                        s.ORIGINS.max() + pd.Timedelta(hours=s.H), freq="h")
    x = np.arange(len(idx))
    return pd.DataFrame({z: 2 + zi + 8 * np.sin(x / 149) + 2 * np.cos(x / 12)
                         for zi, z in enumerate(s.ZONES)}, index=idx)


@pytest.mark.parametrize("conf", ["none", "drift", "christmas", "lag", "station"])
def test_short_history_is_exact_suffix_of_long(temperatures, conf):
    for seed in [0, 3]:
        short = s.build(temperatures, 50, 1, 672, conf, temperatures * .8, seed)
        long = s.build(temperatures, 50, 1, 2688, conf, temperatures * .8, seed)
        np.testing.assert_array_equal(short["context"], long["context"][:, -672:])
        np.testing.assert_array_equal(short["past"], long["past"][:, -672:])
        np.testing.assert_array_equal(short["future"], long["future"])


def test_default_unmodified_history_is_exact_suffix(temperatures):
    # This call also works with the original API: it fails numerically on the
    # old generator, rather than failing because a new seed argument is absent.
    short = s.build(temperatures, 50, 1, 672)
    long = s.build(temperatures, 50, 1, 2688)
    np.testing.assert_array_equal(short["context"], long["context"][:, -672:])


def test_range_manipulation_preserves_noise_and_future(temperatures):
    plain = s.build(temperatures, 50, 1, 672, seed=2)
    for k in [.5, 1.5]:
        changed = s.build(temperatures, 50, k, 672, seed=2)
        delta = 50 * (np.maximum(0, 15 - changed["past"]) - np.maximum(0, 15 - plain["past"]))
        np.testing.assert_allclose(changed["context"] - plain["context"], delta, atol=1e-10)
        np.testing.assert_allclose(changed["past"].mean(1), plain["past"].mean(1))
        np.testing.assert_allclose(np.ptp(changed["past"], axis=1), k * np.ptp(plain["past"], axis=1))
        np.testing.assert_array_equal(changed["future"], plain["future"])
        np.testing.assert_array_equal(changed["true_response"], plain["true_response"])


def test_seeds_change_noise_and_are_reproducible(temperatures):
    a = s.build(temperatures, 50, 1, 672, seed=0)
    b = s.build(temperatures, 50, 1, 672, seed=1)
    np.testing.assert_array_equal(a["context"], s.build(temperatures, 50, 1, 672, seed=0)["context"])
    assert not np.array_equal(a["context"], b["context"])


def test_overlapping_origins_share_one_underlying_drift_process(temperatures):
    b = s.build(temperatures, 50, 1, 2688, conf="drift", seed=4)
    gap = int((s.ORIGINS[1] - s.ORIGINS[0]) / pd.Timedelta(hours=1))
    np.testing.assert_array_equal(b["context"][0, gap:], b["context"][1, :-gap])


def test_known_response_accounts_for_delay_and_holiday_multiplier():
    holiday = pd.Timestamp("2025-12-29", tz="UTC")
    assert s.known_response(50, "christmas", holiday) == pytest.approx(45.)
    assert s.known_response(50, "none", holiday) == 50.
    assert s.known_response(50, "lag", holiday) == pytest.approx(50 * s.LAG_FACTOR)
    # At 00 UTC on Dec 23, only the last forecast hour is a local holiday.
    assert s.known_response(50, "christmas", pd.Timestamp("2025-12-23", tz="UTC")) == pytest.approx(50 * (23 + .9) / 24)


def test_bootstrap_keeps_zones_together():
    g = pd.DataFrame([{"seed": seed, "origin": origin, "response": seed + origin}
                      for seed in range(5) for origin in range(15)])
    a = s.cluster_summary(g)
    b = s.cluster_summary(pd.concat([g, g], ignore_index=True))
    for key in ["response", "lo", "hi", "seed_sd"]:
        assert a[key] == b[key]


def test_legacy_and_partial_caches_are_rejected():
    with pytest.raises(ValueError, match="mixed cache"):
        s.validate_cache(pd.DataFrame({"cache_version": ["old"]}))
    row = dict(cache_version=s.CACHE_VERSION, model="timestral-209m", context_h=672,
               beta=50, k=1., confounder="none", seed=0, zone="NO1", origin=s.ORIGINS[0])
    with pytest.raises(ValueError, match="incomplete"):
        s.validate_cache(pd.DataFrame([row]))

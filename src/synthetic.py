"""Controlled test of the probe finding: does the model learn the load-temperature slope from
its context, and does the range of temperatures it has seen matter, at fixed window length?

Synthetic load with a KNOWN temperature effect, built on real station temperatures (so the
weather has realistic dynamics):
    load_t = 3000 MW * daily/weekly profile + beta * (15 °C - T_t) + AR(1) noise (sd ~30 MW)
Below 15 °C (almost every winter hour) this is a linear slope of -beta MW per °C.

Each forecast is made twice, with the real future temperature and with it 3 °C colder (the
same probe as report.probe_table); response = change in the median / 3 °C, averaged over the
24 h. Four tests, each planned before it was run:
  A  beta in {0, 25, 50, 100}, window temperatures as observed (k = 1), 4 and 16 weeks.
     Learning in context => response tracks beta.
  B  beta = 50, window temperatures squeezed/stretched around their mean by k in {0.5, 1.5}
     (load built from the changed temperatures, so the relation stays exact), 4 and 16 weeks.
     The future temperatures stay real. Range matters causally => response rises with k.
  C  beta = 50, k = 1, plus one real-data feature the model is not told about (added after A
     and B): "drift" = slow level changes independent of temperature
     (daily AR(1), phi 0.95, sd 100 MW); "christmas" = load x 0.90 on 24 Dec - 1 Jan (cold and
     low-load at once). Does it make a short window under-react?
  D  the same for two more suspects: "lag" = load follows temperature smoothed with a 24 h time
     constant while the model sees hourly temperature (the correct response to a 3 °C colder
     day is then LAG_FACTOR * beta, not beta); "station" = load built from the several-stations
     zone temperature, model given the single station.
Cells share k = 1, beta = 50. 15 origins (weekly from 1 Dec 2025, 00 UTC) x 3 temperature
series (NO1, NO3, NO4 stations) = 45 problems per cell.

Usage:
    python -m src.synthetic --models timestral-209m chronos2   # run A + B (cached per model)
    python -m src.synthetic --models timestral-209m chronos2 --confounders   # run C
    python -m src.synthetic --models timestral-209m chronos2 --mechanism     # run D
    python -m src.synthetic --report                           # tables + figure
"""
import argparse
import hashlib
import importlib.metadata
import json
import logging
import time

import numpy as np
import pandas as pd

from src import ROOT, load_config
from src.forecast import COLD_SHIFT

log = logging.getLogger("synthetic")
ZONES = ["NO1", "NO3", "NO4"]
ORIGINS = pd.date_range("2025-12-01", periods=15, freq="7D", tz="UTC")
CELLS = [(b, 1.0, "none") for b in (0, 25, 50, 100)] + [(50, 0.5, "none"), (50, 1.5, "none")]  # (beta, k, confounder)
CONF_CELLS = [(50, 1.0, "drift"), (50, 1.0, "christmas")]
MECH_CELLS = [(50, 1.0, "lag"), (50, 1.0, "station")]
CLEAN_CELL = [(50, 1.0, "none")]  # the clean baseline, for models added after tests A and B (TimesFM 3.0)
CONTEXTS = [672, 2688]
H = 24
CACHE_VERSION = "matched-v2"
SEEDS = (0, 1, 2, 3, 4)
OUT = ROOT / "results" / "cache" / "synthetic" / CACHE_VERSION
LAG_ALPHA = 1 / 24  # exponential smoothing per hour: time constant about 24 h
# Share of a sustained 3 °C drop that reaches the smoothed temperature, averaged over the 24
# forecast hours: 1 - mean of (1 - alpha)^h for h = 1..24 (about 0.39)
LAG_FACTOR = 1 - np.mean((1 - LAG_ALPHA) ** np.arange(1, H + 1))


def christmas(times: pd.DatetimeIndex) -> np.ndarray:
    """True on 24 Dec - 1 Jan, local time."""
    local = times.tz_convert("Europe/Oslo")
    return ((local.month == 12) & (local.day >= 24)) | ((local.month == 1) & (local.day == 1))


def known_response(beta, conf, origin):
    """Exact colder-future response while heating is active (checked in build())."""
    factor = LAG_FACTOR if conf == "lag" else 1.
    if conf == "christmas":
        # The Christmas multiplier applies to the temperature term too, and a
        # 24-hour UTC window can contain only part of a local holiday.
        factor *= np.where(christmas(pd.date_range(origin, periods=H, freq="h")), .9, 1.).mean()
    return beta * factor


def synthetic_load(times: pd.DatetimeIndex, t: np.ndarray, beta: float, seed: int | np.random.SeedSequence,
                   conf: str = "none") -> np.ndarray:
    """Generate one common calendar history; build() slices it for all contexts.

    A seed alone cannot align calls with different start dates. Callers must use
    the same reference calendar, as build() does, before taking suffixes.
    """
    local = times.tz_convert("Europe/Oslo")
    h, dow = local.hour.values, local.dayofweek.values
    profile = 1 + 0.08 * np.sin(2 * np.pi * (h - 6) / 24) + 0.04 * np.sin(2 * np.pi * (h - 14) / 12)
    profile = profile * np.where(dow >= 5, 0.95, 1.0)
    streams = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    noise_seed, drift_seed = streams.spawn(2)
    rng = np.random.default_rng(noise_seed)
    e = np.zeros(len(t))
    shocks = rng.normal(0, 30 * np.sqrt(1 - 0.9 ** 2), len(t))
    for i in range(1, len(t)):
        e[i] = 0.9 * e[i - 1] + shocks[i]
    y = 3000 * profile + beta * np.maximum(0, 15 - t) + e
    if conf == "drift":
        # slow level changes: one AR(1) value per local day (phi 0.95, stationary sd 100 MW)
        day = pd.factorize(local.normalize())[0]
        drift = np.zeros(day.max() + 1)
        shocks = np.random.default_rng(drift_seed).normal(0, 100 * np.sqrt(1 - 0.95 ** 2), len(drift))
        for i in range(1, len(drift)):
            drift[i] = 0.95 * drift[i - 1] + shocks[i]
        y = y + drift[day]
    elif conf == "christmas":
        y = np.where(christmas(times), 0.90 * y, y)
    return y


def build(temp: pd.DataFrame, beta: float, k: float, T: int, conf: str = "none",
          temp_multi: pd.DataFrame | None = None, seed: int = 0) -> dict:
    """One batch per cell: context load + temperatures, future temperatures plain and colder.
    The model always sees the single-station temperature; for "lag" and "station" the load is
    driven by the smoothed or the several-stations temperature instead."""
    if T not in CONTEXTS:
        raise ValueError(f"unsupported context {T}")
    if conf == "station" and temp_multi is None:
        raise ValueError("station condition requires temperature_multi.parquet")
    driver = {"lag": temp.ewm(alpha=LAG_ALPHA, adjust=False).mean(), "station": temp_multi}.get(conf)
    # Generate one underlying winter per zone/seed, including the longest history.
    # Origins and context lengths slice this same process on the same calendar.
    common = pd.date_range(ORIGINS.min() - pd.Timedelta(hours=max(CONTEXTS)),
                           ORIGINS.max() - pd.Timedelta(hours=1), freq="h")
    b = {"context": [], "past": [], "future": [], "zone": [], "origin": [], "xmas_in_window": [], "true_response": []}
    for zi, z in enumerate(ZONES):
        base_temp = (temp if driver is None else driver).loc[common, z].to_numpy()
        process_seed = np.random.SeedSequence([seed, zi])
        base_load = synthetic_load(common, base_temp, beta, process_seed, conf)
        for oi, o in enumerate(ORIGINS):
            past_t = temp.index[(temp.index >= o - pd.Timedelta(hours=T)) & (temp.index < o)]
            tp = temp.loc[past_t, z].values
            loc = common.get_indexer(past_t)
            if len(loc) != T or (loc < 0).any():
                raise ValueError("incomplete common synthetic history")
            y = base_load[loc].copy()
            if k != 1.0:
                # Keep the intended window-specific mean/range manipulation. Only
                # the temperature contribution changes; the noise stays matched.
                if conf != "none":
                    raise ValueError("range manipulations are defined for the clean condition only")
                changed = tp.mean() + k * (tp - tp.mean())
                y += beta * (np.maximum(0, 15 - changed) - np.maximum(0, 15 - tp))
                tp = changed
            b["context"].append(y)
            b["xmas_in_window"].append(christmas(past_t).any())
            b["past"].append(tp)
            future_times = pd.date_range(o, periods=H, freq="h")
            future_driver = (temp if driver is None else driver).loc[future_times, z].values
            if not np.isfinite(future_driver).all() or (future_driver > 15).any():
                raise ValueError("known-response normalisation requires complete, heating-active future drivers")
            b["future"].append(temp.loc[future_times, z].values)
            b["true_response"].append(known_response(beta, conf, o))
            b["zone"].append(z)
            b["origin"].append(o)
    return {k_: np.array(v) for k_, v in b.items()}


def run(model_name: str, temp: pd.DataFrame, cells=CELLS, tag: str = "", temp_multi=None, seeds=SEEDS):
    from src.models import Chronos2, Timestral, TimesFM3
    constructors = {"timestral-209m": lambda: Timestral("209M", covariates=("temp",)),
             "chronos2": lambda: Chronos2(covariates=("temp",)),
             "timesfm3": lambda: TimesFM3(covariates=("temp",))}
    model = None
    out = OUT / f"{model_name}.parquet"
    # saved after every cell, and finished cells are skipped, so an interrupted run resumes
    rows = [pd.read_parquet(out)] if out.exists() else []
    keys = ["context_h", "beta", "k", "confounder", "seed"]
    if rows:
        if set(rows[0].cache_version) != {CACHE_VERSION}:
            raise ValueError("incompatible synthetic cache")
        counts = rows[0].groupby(keys).size()
        if (not counts.eq(len(ZONES) * len(ORIGINS)).all() or
                rows[0].duplicated(keys + ["zone", "origin"]).any()):
            raise ValueError("incomplete cached cell")
    done = set(map(tuple, rows[0][keys].drop_duplicates().values)) if rows else set()
    OUT.mkdir(parents=True, exist_ok=True)
    for T in CONTEXTS:
        for beta, k, conf, seed in [(beta, k, conf, seed) for beta, k, conf in cells for seed in seeds]:
            digest = hashlib.sha256(pd.util.hash_pandas_object(temp, index=True).values.tobytes())
            if conf == "station":
                digest.update(pd.util.hash_pandas_object(temp_multi, index=True).values.tobytes())
            data_hash = digest.hexdigest()
            if (T, beta, k, conf, seed) in done:
                old = rows[0][(rows[0].context_h == T) & (rows[0].beta == beta) & (rows[0].k == k) &
                              (rows[0].confounder == conf) & (rows[0].seed == seed)]
                if not old.temperature_hash.eq(data_hash).all():
                    raise ValueError("temperature data changed: use a new cache version")
                continue
            if model is None:
                model = constructors[model_name]()
            started = time.perf_counter()
            b = build(temp, beta, k, T, conf, temp_multi, seed)
            med = {}
            for name, shift in [("plain", 0.0), ("cold", COLD_SHIFT)]:
                batch = {"context": b["context"], "cov_past": {"temp": b["past"]},
                         "cov_future": {"temp": b["future"] - shift}}
                med[name] = model.predict(batch, H)[:, :, 2]          # median, (n, H)
            resp = (med["cold"] - med["plain"]).mean(axis=1) / -COLD_SHIFT
            span = [pd.Series(p).groupby(np.arange(len(p)) // 24).mean().agg(lambda s: s.max() - s.min())
                    for p in b["past"]]
            rows.append(pd.DataFrame({"model": model_name, "context_h": T, "beta": beta, "k": k,
                                      "confounder": conf, "true_response": b["true_response"],
                                      "zone": b["zone"], "origin": b["origin"],
                                      "response": resp, "window_range": span,
                                      "xmas_in_window": b["xmas_in_window"], "seed": seed,
                                      "cache_version": CACHE_VERSION, "temperature_hash": data_hash}))
            log.info("%s T=%d beta=%s k=%s %s seed=%d: response %.2f MW/°C, %.1fs", model_name, T, beta, k,
                     conf, seed, -resp.mean(), time.perf_counter() - started)
            pending = out.with_suffix(".parquet.tmp")
            pd.concat(rows).to_parquet(pending)
            pending.replace(out)


def cluster_summary(g, value="response", n_boot=2000):
    """Crossed seed/origin bootstrap; zones stay together within each origin.

    Average zones first. Resample seeds and origins independently, sharing each
    origin draw across all seeds. Context differences are formed before this call.
    """
    wide = g.groupby(["seed", "origin"])[value].mean().unstack("origin")
    if wide.isna().any().any():
        raise ValueError("unmatched seed/origin grid")
    v = wide.to_numpy()
    rng = np.random.default_rng(0)
    si = rng.integers(len(v), size=(n_boot, len(v)))
    oi = rng.integers(v.shape[1], size=(n_boot, v.shape[1]))
    boot = v[si[:, :, None], oi[:, None, :]].mean(axis=(1, 2))
    seed_means = v.mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(n=len(g), n_origins=v.shape[1], n_seeds=len(v), response=v.mean(), lo=lo, hi=hi,
                seed_sd=seed_means.std(ddof=1), seed_min=seed_means.min(), seed_max=seed_means.max())


def expected_cells(model):
    return (CLEAN_CELL if model == "timesfm3" else CELLS) + CONF_CELLS + MECH_CELLS


def validate_cache(d):
    """Fail closed: reports may only use the complete corrected five-seed grid."""
    keys = ["model", "context_h", "beta", "k", "confounder", "seed", "zone", "origin"]
    if set(d.cache_version) != {CACHE_VERSION} or d.duplicated(keys).any():
        raise ValueError("mixed cache versions or duplicate synthetic forecasts")
    expected = {(m, T, beta, k, conf, seed, z, o)
                for m in ["timestral-209m", "chronos2", "timesfm3"]
                for T in CONTEXTS for beta, k, conf in expected_cells(m)
                for seed in SEEDS for z in ZONES for o in ORIGINS}
    actual = set(d[keys].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError(f"synthetic grid incomplete: {len(expected - actual)} missing, {len(actual - expected)} unexpected rows")
    if d[d.confounder != "station"].temperature_hash.nunique() != 1 or d[d.confounder == "station"].temperature_hash.nunique() != 1:
        raise ValueError("mixed temperature inputs across synthetic caches")


def report(cfg):
    """Tables: response / true beta per cell (test A and B), with a bootstrap over the 45
    problems. Figure: response vs beta (A), and vs window temperature range (B)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.report import COLORS, INK, INK2, save, save_fig, style

    files = sorted(OUT.glob("*.parquet"))
    if not files:
        raise ValueError(f"no corrected synthetic caches in {OUT}")
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    validate_cache(d)
    d["response"] = -d["response"]  # MW of extra load per °C colder (positive = more load)
    d["confounder"] = d["confounder"].fillna("none") if "confounder" in d else "none"  # A + B files predate C
    # Recompute the analytical reference for every cached row. Earlier matched-v2
    # checkpoints used base beta for Christmas target hours; correcting reference
    # metadata does not require repeating inference or changing any inputs.
    old_reference = d["true_response"].to_numpy().copy()
    d["true_response"] = [known_response(b, c, o) for b, c, o in zip(d.beta, d.confounder, d.origin)]
    changed_references = int((~np.isclose(old_reference, d.true_response)).sum())
    # does the window [origin - T, origin) contain 24 Dec - 1 Jan? (same rule for every cell)
    d["xmas_in_window"] = [christmas(pd.date_range(o - pd.Timedelta(hours=T), periods=T, freq="h")).any()
                           for o, T in zip(d["origin"], d["context_h"])]
    # Small auditable output: reviewers can check seed variability and paired
    # statistics from per-origin responses without repeating model inference.
    archive = ROOT / cfg["paths"]["results"] / "tables" / "synthetic_responses_matched_v2.csv.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    order = ["model", "context_h", "beta", "k", "confounder", "seed", "zone", "origin"]
    d.sort_values(order).rename(columns={"response": "response_mw_per_degree_colder"}).to_csv(
        archive, index=False, compression={"method": "gzip", "mtime": 0})

    def summary(x, by):
        rows = []
        for key, g in x.groupby(by):
            stats = cluster_summary(g)
            true = g["true_response"].mean()
            ratio = (g["response"] / g["true_response"].replace(0, np.nan)).mean()
            rows.append({**dict(zip(by, key)), **stats, "true_response": true,
                         "ratio": ratio,
                         "window_range": g["window_range"].mean()})
        return pd.DataFrame(rows)

    t = summary(d, ["model", "context_h", "beta", "k", "confounder"])
    save(t.set_index(["model", "context_h", "beta", "k", "confounder"]), "synthetic_probe", cfg)
    print(t.round(2).to_string(index=False))
    seed_table = d.assign(ratio=d.response / d.true_response.replace(0, np.nan)).groupby(
        ["model", "context_h", "beta", "k", "confounder", "seed"]).agg(
        response=("response", "mean"), true_response=("true_response", "mean"),
        ratio=("ratio", "mean"), n=("response", "size"))
    save(seed_table, "synthetic_seed_summary", cfg)
    # Test C: beta = 50, k = 1, by confounder, overall and split by Christmas in the window
    c = d[(d["beta"] == 50) & (d["k"] == 1.0)]
    tc = pd.concat([summary(c, ["model", "context_h", "confounder"]).assign(windows="all"),
                    summary(c, ["model", "context_h", "confounder", "xmas_in_window"]).assign(
                        windows=lambda x: np.where(x["xmas_in_window"], "with Christmas", "without"))
                    .drop(columns="xmas_in_window")])
    save(tc.set_index(["model", "context_h", "confounder", "windows"]).sort_index(), "synthetic_confounders", cfg)
    print(tc.round(2).to_string(index=False))
    # Paired change in response / correct response from 4 to 16 weeks, per model and cell (beta =
    # 50, k = 1; the same 45 problems at both lengths), with a bootstrap over the problems
    rows = []
    for (m, conf), g in c.groupby(["model", "confounder"]):
        r = g.assign(r=g["response"] / g["true_response"]).pivot_table(
            index=["zone", "origin", "seed"], columns="context_h", values="r")
        if not {672, 2688} <= set(r.columns):
            continue
        diff = (r[2688] - r[672]).rename("response").reset_index()
        stats = cluster_summary(diff)
        rows.append({"model": m, "confounder": conf, **{k: v for k, v in stats.items() if k != "response"},
                     "ratio_4w": r[672].mean(), "ratio_16w": r[2688].mean(), "change": stats["response"]})
    ch = pd.DataFrame(rows)
    save(ch.set_index(["model", "confounder"]), "synthetic_length_change", cfg)
    print(ch.round(2).to_string(index=False))
    # Pair model interactions explicitly; separate model intervals cannot test
    # whether the context effect differs between models.
    wide = c.assign(r=c.response / c.true_response).pivot(
        index=["confounder", "zone", "origin", "seed"], columns=["model", "context_h"], values="r")
    interactions = []
    for other in ["chronos2", "timesfm3"]:
        delta = ((wide["timestral-209m", 2688] - wide["timestral-209m", 672]) -
                 (wide[other, 2688] - wide[other, 672])).rename("response").reset_index()
        for conf, g in delta.groupby("confounder"):
            stats = cluster_summary(g)
            interactions.append({"comparison": f"Timestral context change minus {other}", "confounder": conf,
                                 **{("change" if k == "response" else k): v for k, v in stats.items()}})
    save(pd.DataFrame(interactions).set_index(["comparison", "confounder"]), "synthetic_context_interactions", cfg)
    t = t[t["confounder"] == "none"]  # the figure shows tests A and B

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    models = [("timestral-209m", "Timestral"), ("chronos2", "Chronos-2")]
    ax = axes[0]
    for (m, name), ls in zip(models, ["-", ":"]):
        for T, c in [(672, COLORS[1]), (2688, COLORS[0])]:
            s = t[(t["model"] == m) & (t["context_h"] == T) & (t["k"] == 1.0)]
            ax.errorbar(s["beta"], s["response"], yerr=[s["response"] - s["lo"], s["hi"] - s["response"]],
                        color=c, ls=ls, marker="o", ms=4, capsize=2, label=f"{name}, {T // 168} weeks")
    # TimesFM ran beta=50 only. Hollow squares keep the true x coordinate while
    # remaining distinguishable from the other models' circular markers.
    for T, c in [(672, COLORS[1]), (2688, COLORS[0])]:
        s = t[(t["model"] == "timesfm3") & (t["context_h"] == T) & (t["k"] == 1.0)]
        ax.errorbar(s["beta"], s["response"], yerr=[s["response"] - s["lo"], s["hi"] - s["response"]],
                    color=c, ls="none", marker="s", mfc="none", ms=5, capsize=2,
                    label=f"TimesFM 3.0, {T // 168} weeks (β = 50 only)")
    ax.plot([0, 100], [0, 100], color=INK, lw=1, ls="--", label="true slope")
    ax.set_xlabel("true slope in the synthetic data (MW per °C)", color=INK2, fontsize=8)
    ax.set_ylabel("model response (MW per °C colder)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    style(ax, "A. Does the response track the true slope?")
    ax = axes[1]
    for (m, name), ls in zip(models, ["-", ":"]):
        for T, c in [(672, COLORS[1]), (2688, COLORS[0])]:
            s = t[(t["model"] == m) & (t["context_h"] == T) & (t["beta"] == 50)].sort_values("k")
            ax.errorbar(s["window_range"], s["response"], yerr=[s["response"] - s["lo"], s["hi"] - s["response"]],
                        color=c, ls=ls, marker="o", ms=4, capsize=2, label=f"{name}, {T // 168} weeks")
    ax.axhline(50, color=INK, lw=1, ls="--")
    ax.set_xlabel("range of daily temperatures in the window (°C), squeezed or stretched", color=INK2, fontsize=8)
    style(ax, "B. True slope 50: does the range seen matter?")
    fig.suptitle("Synthetic load with a known temperature effect", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    save_fig(fig, "synthetic_probe", cfg)

    fig, ax = plt.subplots(figsize=(9, 4))
    for mi, (m, name) in enumerate(models + [("timesfm3", "TimesFM 3.0")]):
        s = ch[(ch.model == m)].set_index("confounder").reindex(["none", "drift", "christmas", "lag", "station"])
        x = np.arange(len(s)) + (mi - 1) * 0.20
        ax.errorbar(x, s.change, yerr=[s.change - s.lo, s.hi - s.change],
                    fmt="o", color=COLORS[mi], capsize=3, label=name)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(np.arange(5), ["Clean", "Slow drift", "Christmas", "Delayed effect", "One station"])
    ax.set_ylabel("Change in normalised response\n(16 minus 4 weeks)")
    ax.legend(frameon=False)
    style(ax, "Matched histories: five seeds; 95% seed/origin bootstrap intervals")
    fig.tight_layout()
    save_fig(fig, "synthetic_mechanism", cfg)
    from huggingface_hub.constants import HF_HUB_CACHE
    from pathlib import Path
    repos = {"timestral-209m": "mistralai--Timestral-209M", "chronos2": "amazon--chronos-2",
             "timesfm3": "google--timesfm-3.0-pytorch"}
    revisions = {}
    for m, repo in repos.items():
        ref = Path(HF_HUB_CACHE) / ("models--" + repo) / "refs/main"
        revisions[m] = ref.read_text().strip() if ref.exists() else "unavailable"
    manifest = {"cache_version": CACHE_VERSION, "seeds": list(SEEDS), "rows": len(d),
                "origins": [str(o) for o in ORIGINS], "zones": ZONES, "contexts": CONTEXTS,
                "temperature_hashes": sorted(d.temperature_hash.unique()),
                "bootstrap": "2000 crossed seed/origin draws, zones clustered; seed 0",
                "model_revisions_in_local_cache": revisions,
                "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "auditable_responses": archive.name,
                "auditable_responses_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "reference_response_version": "holiday-aware-v1",
                "cached_reference_rows_corrected": changed_references,
                "packages": {p: importlib.metadata.version(p) for p in
                             ["numpy", "pandas", "torch", "timestral", "chronos-forecasting", "timesfm"]}}
    dest = ROOT / cfg["paths"]["results"] / "tables" / "synthetic_manifest.json"
    dest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--confounders", action="store_true", help="run test C instead of A + B")
    p.add_argument("--mechanism", action="store_true", help="run test D instead of A + B")
    p.add_argument("--clean", action="store_true", help="run only the clean beta = 50 cell")
    p.add_argument("--all", action="store_true", help="all review cells (TimesFM: beta=50 only)")
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--report", action="store_true")
    args = p.parse_args()
    temp = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "temperature.parquet")
    for m in args.models:
        if args.all:
            multi = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "temperature_multi.parquet")
            run(m, temp, expected_cells(m), temp_multi=multi, seeds=args.seeds)
        elif args.clean:
            run(m, temp, CLEAN_CELL, "_clean", seeds=args.seeds)
        elif args.mechanism:
            multi = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "temperature_multi.parquet")
            run(m, temp, MECH_CELLS, "_mechanism", multi, seeds=args.seeds)
        elif args.confounders:
            run(m, temp, CONF_CELLS, "_confounders", seeds=args.seeds)
        else:
            run(m, temp, seeds=args.seeds)
    if args.report:
        report(cfg)


if __name__ == "__main__":
    main()

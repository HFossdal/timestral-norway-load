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
import logging

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
OUT = ROOT / "results" / "cache" / "synthetic"
LAG_ALPHA = 1 / 24  # exponential smoothing per hour: time constant about 24 h
# Share of a sustained 3 °C drop that reaches the smoothed temperature, averaged over the 24
# forecast hours: 1 - mean of (1 - alpha)^h for h = 1..24 (about 0.39)
LAG_FACTOR = 1 - np.mean((1 - LAG_ALPHA) ** np.arange(1, H + 1))


def christmas(times: pd.DatetimeIndex) -> np.ndarray:
    """True on 24 Dec - 1 Jan, local time."""
    local = times.tz_convert("Europe/Oslo")
    return ((local.month == 12) & (local.day >= 24)) | ((local.month == 1) & (local.day == 1))


def synthetic_load(times: pd.DatetimeIndex, t: np.ndarray, beta: float, seed: int,
                   conf: str = "none") -> np.ndarray:
    """Profile + beta * heating degrees + AR(1) noise (+ the confounder of test C). Same seed ->
    same noise, so the plain and the colder forecast share an identical context."""
    local = times.tz_convert("Europe/Oslo")
    h, dow = local.hour.values, local.dayofweek.values
    profile = 1 + 0.08 * np.sin(2 * np.pi * (h - 6) / 24) + 0.04 * np.sin(2 * np.pi * (h - 14) / 12)
    profile = profile * np.where(dow >= 5, 0.95, 1.0)
    rng = np.random.default_rng(seed)
    e = np.zeros(len(t))
    shocks = rng.normal(0, 30 * np.sqrt(1 - 0.9 ** 2), len(t))
    for i in range(1, len(t)):
        e[i] = 0.9 * e[i - 1] + shocks[i]
    y = 3000 * profile + beta * np.maximum(0, 15 - t) + e
    if conf == "drift":
        # slow level changes: one AR(1) value per local day (phi 0.95, stationary sd 100 MW)
        day = pd.factorize(local.normalize())[0]
        drift = np.zeros(day.max() + 1)
        shocks = np.random.default_rng(seed + 10**6).normal(0, 100 * np.sqrt(1 - 0.95 ** 2), len(drift))
        for i in range(1, len(drift)):
            drift[i] = 0.95 * drift[i - 1] + shocks[i]
        y = y + drift[day]
    elif conf == "christmas":
        y = np.where(christmas(times), 0.90 * y, y)
    return y


def build(temp: pd.DataFrame, beta: float, k: float, T: int, conf: str = "none",
          temp_multi: pd.DataFrame | None = None) -> dict:
    """One batch per cell: context load + temperatures, future temperatures plain and colder.
    The model always sees the single-station temperature; for "lag" and "station" the load is
    driven by the smoothed or the several-stations temperature instead."""
    driver = {"lag": temp.ewm(alpha=LAG_ALPHA, adjust=False).mean(), "station": temp_multi}.get(conf)
    b = {"context": [], "past": [], "future": [], "zone": [], "origin": [], "xmas_in_window": []}
    for zi, z in enumerate(ZONES):
        for oi, o in enumerate(ORIGINS):
            past_t = temp.index[(temp.index >= o - pd.Timedelta(hours=T)) & (temp.index < o)]
            tp = temp.loc[past_t, z].values
            tp = tp.mean() + k * (tp - tp.mean())   # squeeze / stretch the window's temperatures
            drv = tp if driver is None else driver.loc[past_t, z].values
            b["context"].append(synthetic_load(past_t, drv, beta, seed=1000 * zi + oi, conf=conf))
            b["xmas_in_window"].append(christmas(past_t).any())
            b["past"].append(tp)
            b["future"].append(temp.loc[o: o + pd.Timedelta(hours=H - 1), z].values)
            b["zone"].append(z)
            b["origin"].append(o)
    return {k_: np.array(v) for k_, v in b.items()}


def run(model_name: str, temp: pd.DataFrame, cells=CELLS, tag: str = "", temp_multi=None):
    from src.models import Chronos2, Timestral, TimesFM3
    model = {"timestral-209m": lambda: Timestral("209M", covariates=("temp",)),
             "chronos2": lambda: Chronos2(covariates=("temp",)),
             "timesfm3": lambda: TimesFM3(covariates=("temp",))}[model_name]()
    out = OUT / f"{model_name}{tag}.parquet"
    # saved after every cell, and finished cells are skipped, so an interrupted run resumes
    rows = [pd.read_parquet(out)] if out.exists() else []
    if rows and "confounder" not in rows[0]:  # A + B files predate the confounder column
        rows[0]["confounder"] = "none"
    done = set(zip(rows[0]["context_h"], rows[0]["beta"], rows[0]["k"], rows[0]["confounder"])) if rows else set()
    OUT.mkdir(parents=True, exist_ok=True)
    for T in CONTEXTS:
        for beta, k, conf in cells:
            if (T, beta, k, conf) in done:
                continue
            b = build(temp, beta, k, T, conf, temp_multi)
            med = {}
            for name, shift in [("plain", 0.0), ("cold", COLD_SHIFT)]:
                batch = {"context": b["context"], "cov_past": {"temp": b["past"]},
                         "cov_future": {"temp": b["future"] - shift}}
                med[name] = model.predict(batch, H)[:, :, 2]          # median, (n, H)
            resp = (med["cold"] - med["plain"]).mean(axis=1) / -COLD_SHIFT
            span = [pd.Series(p).groupby(np.arange(len(p)) // 24).mean().agg(lambda s: s.max() - s.min())
                    for p in b["past"]]
            rows.append(pd.DataFrame({"model": model_name, "context_h": T, "beta": beta, "k": k,
                                      "confounder": conf, "true_response": beta * (LAG_FACTOR if conf == "lag" else 1),
                                      "zone": b["zone"], "origin": b["origin"],
                                      "response": resp, "window_range": span,
                                      "xmas_in_window": b["xmas_in_window"]}))
            log.info("%s T=%d beta=%s k=%s %s: mean response %.1f MW/°C (true %s)", model_name, T, beta, k,
                     conf, -resp.mean(), beta)
            pd.concat(rows).to_parquet(out)


def report(cfg):
    """Tables: response / true beta per cell (test A and B), with a bootstrap over the 45
    problems. Figure: response vs beta (A), and vs window temperature range (B)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.report import COLORS, INK, INK2, save, save_fig, style

    d = pd.concat([pd.read_parquet(f) for f in OUT.glob("*.parquet")])
    d["response"] = -d["response"]  # MW of extra load per °C colder (positive = more load)
    d["confounder"] = d["confounder"].fillna("none") if "confounder" in d else "none"  # A + B files predate C
    d["true_response"] = d["true_response"].fillna(d["beta"]) if "true_response" in d else d["beta"]
    # does the window [origin - T, origin) contain 24 Dec - 1 Jan? (same rule for every cell)
    d["xmas_in_window"] = [christmas(pd.date_range(o - pd.Timedelta(hours=T), periods=T, freq="h")).any()
                           for o, T in zip(d["origin"], d["context_h"])]
    rng = np.random.default_rng(0)

    def summary(x, by):
        rows = []
        for key, g in x.groupby(by):
            r = g["response"].values
            boot = [rng.choice(r, len(r)).mean() for _ in range(1000)]
            true = g["true_response"].iloc[0]  # beta, or the smaller correct response under "lag"
            rows.append({**dict(zip(by, key)), "n": len(r), "response": r.mean(),
                         "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5), "true_response": true,
                         "ratio": r.mean() / true if true else np.nan, "window_range": g["window_range"].mean()})
        return pd.DataFrame(rows)

    t = summary(d, ["model", "context_h", "beta", "k", "confounder"])
    save(t.set_index(["model", "context_h", "beta", "k", "confounder"]), "synthetic_probe", cfg)
    print(t.round(2).to_string(index=False))
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
            index=["zone", "origin"], columns="context_h", values="r")
        if not {672, 2688} <= set(r.columns):
            continue
        diff = (r[2688] - r[672]).dropna().values
        rb = np.random.default_rng(0)
        boot = [rb.choice(diff, len(diff)).mean() for _ in range(2000)]
        rows.append({"model": m, "confounder": conf, "n": len(diff), "ratio_4w": r[672].mean(),
                     "ratio_16w": r[2688].mean(), "change": diff.mean(),
                     "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5)})
    ch = pd.DataFrame(rows)
    save(ch.set_index(["model", "confounder"]), "synthetic_length_change", cfg)
    print(ch.round(2).to_string(index=False))
    t = t[t["confounder"] == "none"]  # the figure shows tests A and B

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    models = [("timestral-209m", "Timestral"), ("chronos2", "Chronos-2")]
    ax = axes[0]
    for (m, name), ls in zip(models, ["-", ":"]):
        for T, c in [(672, COLORS[1]), (2688, COLORS[0])]:
            s = t[(t["model"] == m) & (t["context_h"] == T) & (t["k"] == 1.0)]
            ax.errorbar(s["beta"], s["response"], yerr=[s["response"] - s["lo"], s["hi"] - s["response"]],
                        color=c, ls=ls, marker="o", ms=4, capsize=2, label=f"{name}, {T // 168} weeks")
    # TimesFM 3.0 ran only the beta = 50 cell: squares, drawn 3 MW/°C to the right so they don't hide the others
    for T, c in [(672, COLORS[1]), (2688, COLORS[0])]:
        s = t[(t["model"] == "timesfm3") & (t["context_h"] == T) & (t["k"] == 1.0)]
        ax.errorbar(s["beta"] + 3, s["response"], yerr=[s["response"] - s["lo"], s["hi"] - s["response"]],
                    color=c, ls="none", marker="s", ms=5, capsize=2, label=f"TimesFM 3.0, {T // 168} weeks (β = 50 only)")
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


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--confounders", action="store_true", help="run test C instead of A + B")
    p.add_argument("--mechanism", action="store_true", help="run test D instead of A + B")
    p.add_argument("--clean", action="store_true", help="run only the clean beta = 50 cell")
    p.add_argument("--report", action="store_true")
    args = p.parse_args()
    temp = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "temperature.parquet")
    for m in args.models:
        if args.clean:
            run(m, temp, CLEAN_CELL, "_clean")
        elif args.mechanism:
            multi = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "temperature_multi.parquet")
            run(m, temp, MECH_CELLS, "_mechanism", multi)
        elif args.confounders:
            run(m, temp, CONF_CELLS, "_confounders")
        else:
            run(m, temp)
    if args.report:
        report(cfg)


if __name__ == "__main__":
    main()

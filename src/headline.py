"""README key figure: where do Timestral's 80% bands miss in cold weather?

Reads the saved fix tables (no forecasting), so run it after `python -m src.report` for
both context lengths. Test winter 2025-26, noon-before origin, observed temperature.
Bars = share of hours above the band (under-forecast) and below it (over-forecast);
a calibrated 80% band misses 10% on each side.

Second figure (real_forecast_bands): the same misses with the archived MET Nordic weather
forecast instead of observed temperature, before and after the one-sided temperature-bin
recalibration fitted on the 2022-25 real-forecast runs, plus leave-one-winter-out checks.

Usage: python -m src.headline
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import ROOT, load_config
from src.report import COLORS, GRID, INK, INK2, save_fig, style

# (label, table, variant, calibration). The correction is the 2-parameter temperature bias
# correction fitted on the three winters before the calibration cut-off (never the test winter).
SETUPS = [("4 weeks of history", "ctx672", "none", "-"),
          ("4 weeks + temperature\ncorrection", "ctx672", "bias only", "3 winters"),
          ("16 weeks of history", "ctx2688", "none", "-")]
PANELS = [("cold", "Cold-snap days (zone's coldest 5%)"), ("below_m10", "Days below −10 °C")]


def main():
    cfg = load_config()
    tab = ROOT / cfg["paths"]["results"] / "tables"
    rows = []
    for label, ctx, variant, cal in SETUPS:
        f = pd.read_csv(tab / f"fix_timestral-209m_temp_noon_before_{ctx}.csv")
        f = f[(f["variant"] == variant) & (f["calibration"] == cal)].set_index("regime")
        for reg, _ in PANELS:
            rows.append({"setup": label, "regime": reg, "n": f.loc[reg, "n"],
                         "above": 100 * f.loc[reg, "above80"], "below": 100 * f.loc[reg, "below80"]})
    d = pd.DataFrame(rows)
    print(d.round(1).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    x = range(len(SETUPS))
    w = 0.36
    for ax, (reg, title) in zip(axes, PANELS):
        s = d[d["regime"] == reg]
        b1 = ax.bar([i - w / 2 - 0.01 for i in x], s["above"], w, color=COLORS[1],
                    label="load above the band (under-forecast)")
        b2 = ax.bar([i + w / 2 + 0.01 for i in x], s["below"], w, color=COLORS[0],
                    label="load below the band (over-forecast)")
        for bars in (b1, b2):
            ax.bar_label(bars, fmt="%.0f%%", fontsize=8, color=INK2, padding=2)
        ax.axhline(10, color=INK, lw=1, ls="--", label="target: 10% on each side")
        ax.set_xticks(list(x), [lbl for lbl, *_ in SETUPS], fontsize=9)
        style(ax, f"{title}, {int(s['n'].iloc[0]):,} hours")
    axes[0].set_ylabel("share of hours outside the 80% band (%)", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Timestral-209M's misses in cold weather are one-sided; more history or a "
                 "temperature correction mostly rebalances them", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    save_fig(fig, "headline_cold_misses", cfg)
    weather_figure(cfg)
    probe_figure(cfg)
    probe_context_figure(cfg)
    thumbnail(cfg)


def thumbnail(cfg):
    """Link-preview / website card (1200 x 630 px): the probe result for the four foundation
    models in one picture, mean ratio over the five zones at 4 and 16 weeks. Colours are four
    categorical slots checked for colour-blind separation; every line is labelled directly.
    Saved without save_fig so the pixel size stays exact."""
    tab = ROOT / cfg["paths"]["results"] / "tables"
    p = pd.read_csv(tab / "probe_noon_before_ctx2688.csv")
    mean = p[p["regime"] == "all"].groupby(["model", "context_h"])["ratio"].mean()
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    models = [("timestral-209m", "Timestral", "#2a78d6"), ("timesfm3", "TimesFM 3.0", "#eb6834"),
              ("chronos2", "Chronos-2", "#1baf7a"), ("moirai2", "Moirai 2.0", "#eda100")]
    fig, ax = plt.subplots(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    for key, name, color in models:
        if (key, 672) not in mean or (key, 2688) not in mean:
            continue
        y = [mean[(key, 672)], mean[(key, 2688)]]
        ax.plot([0, 1], y, color=color, lw=3, solid_capstyle="round", zorder=2)
        ax.scatter([0, 1], y, s=110, color=color, edgecolors=surface, linewidths=2.5, zorder=3)
        ax.text(-0.04, y[0], f"{y[0]:.2f}", ha="right", va="center", fontsize=13, color=ink2)
        ax.text(1.04, y[1], f"{name}  {y[1]:.2f}", ha="left", va="center", fontsize=15, color=ink,
                fontweight="bold" if key == "timestral-209m" else "normal")
    ax.set_xlim(-0.25, 1.55)
    ax.set_ylim(-0.05, 0.8)
    ax.set_xticks([0, 1], ["4 weeks of history", "16 weeks of history"], fontsize=14, color=ink)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.tick_params(axis="y", labelsize=12, colors=ink2, length=0)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.grid(axis="y", color=grid, lw=1)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.set_axisbelow(True)
    fig.text(0.04, 0.925, "Give Timestral more history and it reacts more strongly to cold", fontsize=21,
             color=ink, fontweight="bold", ha="left")
    fig.text(0.04, 0.865, "Forecast response to future temperatures 3 °C colder, divided by each zone's estimated historical\n"
                          "load–temperature slope (1.0 = matches that reference). Test winter 2025-26, five bidding zones.",
             fontsize=13, color=ink2, ha="left", va="top", linespacing=1.4)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.74, bottom=0.12)
    fig.savefig(ROOT / cfg["paths"]["results"] / "figures" / "thumbnail.png", facecolor=surface)
    plt.close(fig)


def probe_context_figure(cfg):
    """Probe response vs the range of temperatures in the history window (report.probe_context):
    one dot per zone and origin, lines = mean per 2 °C bin (bins with at least 20 dots)."""
    tab = ROOT / cfg["paths"]["results"] / "tables"
    d = pd.read_csv(tab / "probe_context_rows_noon_before_ctx2688.csv")
    models = [(m, t) for m, t in [("timestral-209m", "Timestral-209M"), ("timesfm3", "TimesFM 3.0"),
                                  ("chronos2", "Chronos-2")] if (d["model"] == m).any()]
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3.8), sharey=True, sharex=True, squeeze=False)
    axes = axes[0]
    for ax, (model, title) in zip(axes, models):
        for ctx, color, lbl in [(672, COLORS[1], "4 weeks of history"), (2688, COLORS[0], "16 weeks of history")]:
            x = d[(d["model"] == model) & (d["context_h"] == ctx)]
            ax.scatter(x["window_range"], x["ratio"], s=6, color=color, alpha=0.25, linewidths=0)
            b = x.groupby((x["window_range"] // 2) * 2 + 1)["ratio"].agg(["mean", "size"])
            b = b[b["size"] >= 20]
            ax.plot(b.index, b["mean"], color=color, lw=2, marker="o", ms=4, label=lbl)
        ax.set_ylim(-0.1, 1.2)
        ax.set_xlabel("range of daily temperatures in the history window (°C)", color=INK2, fontsize=8)
        style(ax, title)
    axes[0].set_ylabel("response / historical slope", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    # TimesFM's window-range coefficient includes zero (probe_context table), so the title credits it
    # with the length effect only
    fig.suptitle("Timestral reacts more strongly when the window has seen more cold, and more again with a longer "
                 "window; TimesFM gains from length only, Chronos-2 from neither",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    save_fig(fig, "probe_context", cfg)


def probe_figure(cfg):
    """The mechanism: how strongly each model reacts to colder future temperature, as a share
    of the historical load-temperature slope, for every history length run (probe_table in
    report.py: same test-winter forecasts with all future temperatures 3 °C colder). One panel
    per model; dots light to dark = 4, 8, 16, 32 weeks."""
    tab = ROOT / cfg["paths"]["results"] / "tables"
    p = pd.read_csv(tab / "probe_noon_before_ctx2688.csv")
    p = p[p["regime"] == "all"]
    zones = cfg["zones"]
    shades = {672: "#b7d3f2", 1344: "#7fb0e8", 2688: "#2a78d6", 5376: "#16477f"}  # one hue, light -> dark
    models = [(m, t) for m, t in [("timestral-209m", "Timestral-209M"), ("timestral-123m", "Timestral-123M"),
                                  ("timesfm3", "TimesFM 3.0"), ("chronos2", "Chronos-2"), ("moirai2", "Moirai 2.0")]
              if (p["model"] == m).any()]
    ncols = min(3, len(models))
    nrows = -(-len(models) // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.3 * nrows), sharey=True, sharex=True, squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(models):]:  # unused grid cells
        ax.axis("off")
    for ax, (model, title) in zip(axes, models):
        r = p[p["model"] == model].pivot(index="zone", columns="context_h", values="ratio").reindex(zones)
        y = range(len(zones))
        for i, z in enumerate(zones):
            ax.plot([r.loc[z].min(), r.loc[z].max()], [i, i], color=GRID, lw=3, zorder=1)
        for ctx in sorted(r.columns):
            ax.scatter(r[ctx], y, s=60, color=shades[ctx], edgecolors="white", linewidths=1.5, zorder=2,
                       label=f"{ctx // 168} weeks of history")
        ax.axvline(1, color=INK, lw=1, ls="--", label="historical slope")
        ax.set_yticks(list(y), zones, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(min(0, p["ratio"].min() - 0.05), 1.1)  # room for a model that doesn't react
        ax.set_xlabel("response to 3 °C colder / historical slope", color=INK2, fontsize=8)
        ax.tick_params(labelbottom=True)  # shared x: keep the numbers on every panel, not just the bottom row
        style(ax, title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("How strongly each model reacts to colder weather, by length of history: Timestral gains most "
                 "from a longer window",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=len(labels))
    save_fig(fig, "probe_response", cfg)


def bars(ax, d, labels):
    """Above / below bars per setup with the 10% target line (shared by both figures)."""
    x = range(len(labels))
    w = 0.36
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], d["above"], w, color=COLORS[1],
                label="load above the band (under-forecast)")
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], d["below"], w, color=COLORS[0],
                label="load below the band (over-forecast)")
    for b in (b1, b2):
        ax.bar_label(b, fmt="%.0f%%", fontsize=8, color=INK2, padding=2)
    ax.axhline(10, color=INK, lw=1, ls="--", label="target: 10% on each side")
    ax.set_xticks(list(x), labels, fontsize=9)


def weather_figure(cfg):
    """Timestral, 16 weeks, cold-snap days: measured temperature vs the real weather forecast,
    before and after the one-sided recalibration fitted on real-forecast runs. Left: the test
    winter (fit on 2022-25). Right: all four real-forecast winters, leave one winter out."""
    tab = ROOT / cfg["paths"]["results"] / "tables"
    obs = pd.read_csv(tab / "cold_cqr_noon_before_ctx2688.csv")
    obs = obs[(obs["model"] == "timestral-209m_temp") & (obs["context_h"] == 2688)
              & (obs["variant"] == "none") & (obs["regime"] == "cold")].set_index("eval")
    fc = pd.read_csv(tab / "cold_cqr_fc_noon_before_ctx2688.csv")
    fc = fc[(fc["model"] == "timestral-209m_temp_fc") & (fc["regime"] == "cold")].set_index(["calibration", "variant"])
    labels = ["measured temperature\n(perfect forecast)", "real MET weather\nforecast",
              "real forecast +\none-sided recalibration"]
    panels = [("Cold-snap days, test winter", obs.loc["test"], fc.loc["3 winters"]),
              ("Cold-snap days, 4 winters (leave one winter out)", obs.loc["lowo"], fc.loc["lowo"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (title, o, f) in zip(axes, panels):
        rows = [o, f.loc["none"], f.loc["CQR one-sided, temp bins"]]
        d = pd.DataFrame([{"above": 100 * r["above80"], "below": 100 * r["below80"]} for r in rows])
        bars(ax, d, labels)
        style(ax, title)
    axes[0].set_ylabel("share of hours outside the 80% band (%)", color=INK2, fontsize=9)
    axes[0].set_ylim(0, 30)  # headroom for the legend
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Timestral-209M, 16 weeks: recalibration narrows the cold misses from real weather forecasts, "
                 "but cannot fully see cold the forecast missed", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    save_fig(fig, "real_forecast_bands", cfg)


if __name__ == "__main__":
    main()

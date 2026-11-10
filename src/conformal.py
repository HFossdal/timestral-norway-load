"""Conformalized quantile regression (CQR, Romano et al. 2019) on the forecast intervals.

For a nominal (1 - alpha) interval [q_lo, q_hi]:
  1. On the calibration rows, nonconformity score E = max(q_lo - y, y - q_hi).
     E > 0 means y fell outside the interval (by E MW); E < 0 means inside.
  2. Correction Q = the ceil((n + 1)(1 - alpha)) / n empirical quantile of E
     (np.quantile with method="higher", so it is an actual score, as in the paper).
  3. Test interval = [q_lo - Q, q_hi + Q]. Q > 0 widens, Q < 0 narrows.

Variants (which calibration rows give the correction for a test row):
  pooled          one Q for everything
  zone            one Q per zone
  mondrian        one Q per Mondrian group (cold / peak_exante / other, see regimes.py), zones pooled
  mondrian_zone   one Q per (group, zone); falls back to the group's pooled Q if n < min_n

Caveat: hours within a day are strongly correlated, so the effective calibration sample
is closer to the number of days than the number of hours, and the finite-sample guarantee
assumes exchangeability, which a mild-to-cold winter shift breaks. That is exactly what
we measure on the test winter.

Temperature-dependent bias correction (step before CQR):
At noon-before, Timestral under-forecasts more the colder the target day is, so misses in
cold snaps are one-sided (above the band). CQR widens both sides, so we first shift the
forecast. On the calibration rows we fit the relative error of the median
    (y - q0.50) / q0.50 = a + b * max(0, HINGE - T)
where T is the target day's mean temperature (°C) and HINGE = 0 °C: above freezing the
bias is about zero, below it grows roughly linearly (seen in the calibration winters).
Relative (%) rather than MW so one fit pools zones of very different size. All five
quantiles are shifted by the same amount (a + b * ...) * q0.50, so the band keeps its
width; CQR then fixes the width on the shifted forecast.
"""
import numpy as np
import pandas as pd

INTERVALS = {80: ("q0.10", "q0.90"), 90: ("q0.05", "q0.95")}
QCOLS = ["q0.05", "q0.10", "q0.50", "q0.90", "q0.95"]
HINGE = 0.0  # °C


def fit_bias(cal: pd.DataFrame, temp_col: str = "day_temp") -> tuple[float, float]:
    """Least-squares (a, b) for the relative median error against degrees below HINGE."""
    x = np.maximum(0.0, HINGE - cal[temp_col].values)
    y = ((cal["y_true"] - cal["q0.50"]) / cal["q0.50"]).values
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def apply_bias(df: pd.DataFrame, a: float, b: float, temp_col: str = "day_temp") -> pd.DataFrame:
    """Shift every quantile by (a + b * max(0, HINGE - T)) * q0.50."""
    out = df.copy()
    shift = (a + b * np.maximum(0.0, HINGE - df[temp_col].values)) * df["q0.50"].values
    for c in QCOLS:
        out[c] = df[c].values + shift
    return out


def scores(df: pd.DataFrame, level: int) -> pd.Series:
    lo, hi = INTERVALS[level]
    return np.maximum(df[lo] - df["y_true"], df["y_true"] - df[hi])


def correction(s: pd.Series, level: int) -> float:
    """CQR correction (Romano et al. 2019): the ceil((n+1)(1-alpha))-th smallest score, capped
    at the largest when n is too small. (Until 1 Nov 2026 this took one order statistic higher
    via np.quantile(method="higher"), slightly too conservative; caught by tests/.)"""
    n = len(s)
    # (n+1)(1-alpha) with 1-alpha = level/100; the tiny epsilon stops float error (e.g.
    # 8.000000000000002) from pushing ceil() one rank up
    k = min(int(np.ceil((n + 1) * level / 100 - 1e-9)), n)
    return float(np.sort(np.asarray(s, dtype=float))[k - 1])


def require_known_groups(frame: pd.DataFrame, keys) -> None:
    if "group" in keys and "group_available" in frame and not frame["group_available"].fillna(False).all():
        raise ValueError("Mondrian grouping requires temperature information available at the forecast origin")


def fit(cal: pd.DataFrame, variant: str, level: int, min_n: int = 240) -> pd.DataFrame:
    """Table of corrections: one row per key, with columns Q and n (calibration hours)."""
    keys = {"pooled": [], "zone": ["zone"], "mondrian": ["group"], "mondrian_zone": ["group", "zone"]}[variant]
    require_known_groups(cal, keys)
    s = scores(cal, level)
    if not keys:
        return pd.DataFrame({"Q": [correction(s, level)], "n": [len(s)]})
    table = s.groupby([cal[k] for k in keys]).agg(Q=lambda x: correction(x, level), n="size").reset_index()
    table.columns = keys + ["Q", "n"]
    table["fallback"] = False
    if variant == "mondrian_zone":
        # Include (group, zone) pairs that never occurred in calibration (n = 0) so they fall back too
        full = pd.MultiIndex.from_product([sorted(cal["group"].unique()), sorted(cal["zone"].unique())],
                                          names=["group", "zone"])
        table = table.set_index(["group", "zone"]).reindex(full).reset_index()
        table["n"] = table["n"].fillna(0).astype(int)
        table["fallback"] = False
        pooled = fit(cal, "mondrian", level).set_index("group")["Q"]
        small = table["n"] < min_n
        table.loc[small, "Q"] = table.loc[small, "group"].map(pooled)
        table.loc[small, "fallback"] = True
    return table


def apply(test: pd.DataFrame, table: pd.DataFrame, level: int) -> pd.DataFrame:
    """Return test with corrected interval columns for this level.

    A test group that never occurred in calibration gets Q = 0 (no correction) and a warning.
    """
    lo, hi = INTERVALS[level]
    keys = [c for c in table.columns if c not in ("Q", "n", "fallback")]
    require_known_groups(test, keys)
    if keys:
        Q = test[keys].merge(table, on=keys, how="left")["Q"].values
    else:
        Q = np.full(len(test), table["Q"].iloc[0])
    missing = np.isnan(Q)
    if missing.any():
        print(f"warning: {missing.sum()} test rows had no calibration group; using Q = 0")
        Q = np.where(missing, 0.0, Q)
    out = test.copy()
    # Keep the median inside the interval if a negative Q would make it cross
    out[lo] = np.minimum(test[lo].values - Q, test["q0.50"].values)
    out[hi] = np.maximum(test[hi].values + Q, test["q0.50"].values)
    return out


def sort_quantiles(out: pd.DataFrame) -> pd.DataFrame:
    """Re-sort the quantile columns per row so they stay ordered after correction. Columns a
    model doesn't predict (all NaN, e.g. q0.05 / q0.95 of TimesFM 3.0) are left out, since
    np.sort would move the NaNs to the end and shift every other column."""
    cols = [c for c in QCOLS if out[c].notna().any()]
    out[cols] = np.sort(out[cols].values, axis=1)
    return out


def recalibrate(cal: pd.DataFrame, test: pd.DataFrame, variant: str, min_n: int = 240):
    """Apply CQR to both the 80% and 90% intervals. Returns (corrected test, correction tables)."""
    out, tables = test, {}
    for level in INTERVALS:
        tables[level] = fit(cal, variant, level, min_n)
        out = apply(out, tables[level], level)
    # The 80% and 90% corrections differ, so re-sort the 5 quantiles to keep them ordered
    return sort_quantiles(out), tables


def aci(cal: pd.DataFrame, test: pd.DataFrame, level: int, gamma: float, delay_days: int = 2) -> pd.DataFrame:
    """Adaptive conformal inference (Gibbs & Candès 2021) on top of CQR, run online per zone.

    Each test day d gets the correction Q_d = the (1 - alpha_d) quantile of the nonconformity
    scores known by the origin: the calibration winter plus the test days up to d - delay_days.
    (At the noon-before origin the last fully observed day is d - 2.) After the forecast,
    alpha moves with the miss rate of the newest known day:
        alpha <- alpha + gamma * (alpha_target - miss rate of day d - delay_days)
    so a run of misses lowers alpha, which widens the band, and vice versa.
    Steps are daily (one forecast per day), so gamma is per day, not per hour.
    Returns test with this level's interval corrected and the alpha used per row.
    """
    lo, hi = INTERVALS[level]
    target = 1 - level / 100
    parts = []
    for zone, t in test.groupby("zone"):
        t = t.sort_values("time_utc").copy()
        base = scores(cal[cal["zone"] == zone], level).values
        s_test = scores(t, level).values
        days = t["target_day"].values
        uniq = pd.unique(days)
        alpha, Q, alphas = target, np.zeros(len(t)), np.zeros(len(t))
        Q_day = {}  # correction used on each test day
        for k, d in enumerate(uniq):
            if k >= delay_days:
                # newest fully observed day: was it covered by the band we gave it?
                newest = days == uniq[k - delay_days]
                miss = (s_test[newest] > Q_day[k - delay_days]).mean()
                alpha += gamma * (target - miss)
            known = np.isin(days, uniq[:max(0, k - delay_days + 1)])
            pool = np.concatenate([base, s_test[known]])
            a = min(max(alpha, 1e-3), 1.0)  # alpha <= 0 would mean an infinite band
            Q_day[k] = float(np.quantile(pool, 1 - a, method="higher"))
            cur = days == d
            Q[cur], alphas[cur] = Q_day[k], a
        t[lo] = np.minimum(t[lo].values - Q, t["q0.50"].values)
        t[hi] = np.maximum(t[hi].values + Q, t["q0.50"].values)
        t[f"alpha{level}"] = alphas
        parts.append(t)
    return pd.concat(parts)


# --- One-sided (asymmetric) CQR by temperature bin ---------------------------------------
# Plain CQR adds the same Q to both ends of the band, so a one-sided miss pattern (load above
# the band in deep cold, rarely below) gets a wider band on both sides instead of a shifted
# one. The asymmetric version (Romano et al. 2019, sec. on asymmetric intervals) calibrates
# each end separately at alpha/2:
#     Q_hi = (1 - alpha/2) conformal quantile of  y - q_hi   (how far load goes above the band)
#     Q_lo = (1 - alpha/2) conformal quantile of  q_lo - y   (how far it goes below)
# and does this per bin of the target day's mean temperature (Mondrian), because the size of
# the miss depends on how cold it is. Bins with too few calibration hours fall back to the
# pooled value. The temperature is the same one the forecast was given (observed for the
# oracle runs), so the bin is known at the origin.
TEMP_BINS = [-np.inf, -10.0, -5.0, 0.0, np.inf]
TEMP_LABELS = ["< -10", "-10 to -5", "-5 to 0", ">= 0"]


def temp_bin(t: pd.Series) -> pd.Series:
    return pd.cut(t, TEMP_BINS, right=False, labels=TEMP_LABELS).astype(str)


def fit_asym(cal: pd.DataFrame, level: int, temp_col: str = "day_temp", two_sided: bool = False,
             min_n: int = 240) -> pd.DataFrame:
    """Per temperature bin: Q_lo, Q_hi and n (calibration hours). two_sided=True gives plain
    CQR per bin (Q_lo = Q_hi from the max score) for comparison."""
    lo, hi = INTERVALS[level]
    half = 100 - (100 - level) / 2  # e.g. 80% band -> each end at 90%
    s_lo, s_hi = cal[lo] - cal["y_true"], cal["y_true"] - cal[hi]
    bins = temp_bin(cal[temp_col])

    def q(mask):
        if two_sided:
            v = correction(np.maximum(s_lo[mask], s_hi[mask]), level)
            return v, v
        return correction(s_lo[mask], half), correction(s_hi[mask], half)

    pooled = q(np.ones(len(cal), bool))
    rows = []
    for b in TEMP_LABELS:
        mask = (bins == b).values
        n = int(mask.sum())
        (q_lo, q_hi), fb = (q(mask), False) if n >= min_n else (pooled, True)
        rows.append({"bin": b, "Q_lo": q_lo, "Q_hi": q_hi, "n": n, "fallback": fb})
    return pd.DataFrame(rows)


def apply_asym(test: pd.DataFrame, table: pd.DataFrame, level: int, temp_col: str = "day_temp") -> pd.DataFrame:
    lo, hi = INTERVALS[level]
    t = table.set_index("bin")
    b = temp_bin(test[temp_col])
    out = test.copy()
    out[lo] = np.minimum(test[lo].values - b.map(t["Q_lo"]).values, test["q0.50"].values)
    out[hi] = np.maximum(test[hi].values + b.map(t["Q_hi"]).values, test["q0.50"].values)
    return out


def recalibrate_asym(cal: pd.DataFrame, test: pd.DataFrame, temp_col: str = "day_temp",
                     cal_temp_col: str | None = None, two_sided: bool = False, min_n: int = 240):
    """Both intervals, as recalibrate(). Returns (corrected test, {level: table})."""
    out, tables = test, {}
    for level in INTERVALS:
        tables[level] = fit_asym(cal, level, cal_temp_col or temp_col, two_sided, min_n)
        out = apply_asym(out, tables[level], level, temp_col)
    return sort_quantiles(out), tables

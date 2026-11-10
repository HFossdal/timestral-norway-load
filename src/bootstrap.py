"""Moving calendar-day blocks, shared by all models/zones and stratified by winter."""
import numpy as np
import pandas as pd
import warnings


def bootstrap_margin_totals(errors, days, draws, cover=.9):
    """Sum zone-wise peak-error quantiles on shared day draws.

    Equivalent to reindexing and dropping missing days in every scalar replicate,
    without creating thousands of temporary pandas groups. A replicate missing
    every observation in a zone remains NaN, matching the scalar calculation.
    """
    totals = np.zeros(len(draws))
    for _, g in errors.groupby("zone"):
        values = g.set_index("target_day")["err"].reindex(days).to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            totals += np.nanquantile(values[draws], cover, axis=1)
    return totals


def day_draws(frame, n_boot, block_days=1, seed=0):
    """Return the winter/day calendar and resampled positions on that calendar.

    Non-circular moving blocks are sampled within each winter, then truncated to
    that winter's original length. Missing days retain a calendar position (zero
    observations), so a gap is never mistaken for consecutive observed days.
    The same draws must be used for every model and zone in a comparison.
    """
    if block_days < 1 or n_boot < 1:
        raise ValueError("block_days and n_boot must be positive")
    f = frame[["target_day"]].copy()
    f["winter"] = frame["winter"].values if "winter" in frame else "single"
    rng, calendars, draws, offset = np.random.default_rng(seed), [], [], 0
    for winter, g in f.groupby("winter", sort=True):
        days = pd.date_range(min(g.target_day), max(g.target_day), freq="D").date
        n = len(days)
        if block_days > n:
            raise ValueError(f"{winter}: {n} days cannot support {block_days}-day blocks")
        starts = rng.integers(0, n - block_days + 1,
                              size=(n_boot, (n + block_days - 1) // block_days))
        pick = (starts[:, :, None] + np.arange(block_days)).reshape(n_boot, -1)[:, :n]
        draws.append(pick + offset)
        calendars.extend((winter, day) for day in days)
        offset += n
    if not calendars:
        raise ValueError("no days to resample")
    return pd.MultiIndex.from_tuples(calendars, names=["winter", "target_day"]), np.concatenate(draws, axis=1)

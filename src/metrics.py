"""Forecast metrics on the long forecast tables written by forecast.py.

Each row is one (zone, target hour) with quantile columns q0.05 ... q0.95 and y_true.
- MAE, MAPE: on the median q0.50.
- MASE: MAE divided by the in-sample seasonal-naive MAE (weekly lag) of that zone on
  history before the test winter, so it is comparable across zones.
- pinball: mean pinball (quantile) loss over the 5 quantiles, in MW.
- cov80 / cov90: share of hours inside [q0.10, q0.90] / [q0.05, q0.95]; targets 0.80 / 0.90.
- width80 / width90: average interval width in MW.
"""
import numpy as np
import pandas as pd

from src.models import QUANTILES

QCOLS = [f"q{q:.2f}" for q in QUANTILES]


def mase_scales(load: pd.DataFrame, end: str) -> pd.Series:
    """Per-zone mean |y_t - y_{t-168}| over all history before `end`."""
    hist = load.loc[:pd.Timestamp(end, tz="UTC") - pd.Timedelta(hours=1)]
    return (hist - hist.shift(168)).abs().mean()


def pinball(df: pd.DataFrame) -> pd.Series:
    """Mean pinball loss over the quantiles, one value per row."""
    y = df["y_true"].values
    losses = [np.maximum(q * (y - df[c].values), (q - 1) * (y - df[c].values)) for q, c in zip(QUANTILES, QCOLS)]
    return pd.Series(np.mean(losses, axis=0), index=df.index)


def add_row_metrics(df: pd.DataFrame, scales: pd.Series) -> pd.DataFrame:
    """Per-row error terms; summary metrics are then plain means over any grouping."""
    df = df.copy()
    err = df["y_true"] - df["q0.50"]
    df["ae"] = err.abs()
    df["ape"] = 100 * df["ae"] / df["y_true"]
    df["ase"] = df["ae"] / df["zone"].map(scales)
    df["pinball"] = pinball(df)
    df["in80"] = (df["y_true"] >= df["q0.10"]) & (df["y_true"] <= df["q0.90"])
    # NaN (not False) where a model has no q0.05 / q0.95 (TimesFM 3.0), so cov90 is n/a, not 0
    df["in90"] = ((df["y_true"] >= df["q0.05"]) & (df["y_true"] <= df["q0.95"])).astype(float).where(df["q0.05"].notna())
    df["width80"] = df["q0.90"] - df["q0.10"]
    df["width90"] = df["q0.95"] - df["q0.05"]
    df["bias"] = err  # positive = under-forecast
    return df


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean of the per-row terms within each group, with readable column names and n (hours)."""
    out = df.groupby(by).agg(
        n=("ae", "size"), MAE=("ae", "mean"), MAPE=("ape", "mean"), MASE=("ase", "mean"),
        pinball=("pinball", "mean"), cov80=("in80", "mean"), cov90=("in90", "mean"),
        width80=("width80", "mean"), width90=("width90", "mean"), bias=("bias", "mean"),
    )
    return out

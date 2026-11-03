"""Model wrappers. Every model maps a batch of forecast problems to quantile forecasts.

A batch is a dict built by forecast.build_batch():
    context      (B, T)    load history before the origin (MW)
    cov_past     dict name -> (B, T)   covariate values over the context
    cov_future   dict name -> (B, H)   covariate values over the horizon
    target_times (B, H)    UTC timestamps of the forecast hours
    origin       (B,)      UTC forecast origin (first hour NOT observed)
    zone         (B,)      bidding zone
predict(batch, H) returns an array (B, H, len(QUANTILES)), sorted along the last axis.

Interfaces used (checked against the installed source):
- Timestral: TimestralForecaster.forecast(context (B,V,T), prediction_length,
  known_covariates (B,C,T+H)) -> Forecast with .quantile(level) of shape (B,V,H).
- Chronos-2: Chronos2Pipeline.predict_quantiles(list of dicts with "target",
  "past_covariates", "future_covariates", prediction_length, quantile_levels)
  -> (list of (n_variates, H, n_q) tensors, list of means).
- TimesFM 3.0: TimesFM3Forecaster.predict_batch(contexts, horizon, past_future_covariates
  (list of (C, T+H) arrays), return_quantiles=True) -> ForecastOutput per series with
  .quantiles of shape (H, 9) at levels 0.1, 0.2, ..., 0.9. Covariates go through the model's
  variate attention (native, no separate regression step).
- Moirai 2.0: Moirai2Forecast(module=Moirai2Module.from_pretrained(...), prediction_length,
  context_length, target_dim=1, feat_dynamic_real_dim=C, past_feat_dynamic_real_dim=0), called
  with the tensors its describe_inputs() lists (past_target (B,T,1), past_observed_target,
  past_is_pad, feat_dynamic_real (B,T+H,C), observed_feat_dynamic_real) -> (B, 9, H) at levels
  0.1, ..., 0.9. Known covariates enter as extra variates. Needs its own environment (torch<2.5).
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from src.regimes import LOCAL_TZ, is_holiday

QUANTILES = [0.05, 0.10, 0.50, 0.90, 0.95]


class SeasonalNaive:
    """Point forecast = load at the same hour one week earlier.

    Quantiles = point + empirical quantiles of the weekly-difference residuals
    y[s] - y[s-168] over the context (the last T-168 hours, e.g. 3 weeks for T=672),
    pooled over all hours of the day. Deliberately simple.
    """
    name = "snaive"

    def predict(self, batch, H):
        ctx = batch["context"]
        T = ctx.shape[1]
        out = np.empty((len(ctx), H, len(QUANTILES)))
        for i, y in enumerate(ctx):
            resid = y[168:] - y[:-168]
            q_resid = np.quantile(resid, QUANTILES)
            # target hour h (0-based from origin) is at position T + h; one week earlier is T + h - 168
            point = y[T + np.arange(H) - 168]
            out[i] = point[:, None] + q_resid[None, :]
        return out


class Timestral:
    """Timestral zero-shot. Covariates are passed as known-future rows (oracle values over the horizon)."""

    def __init__(self, size="209M", covariates=(), batch_size=25):
        from timestral import TimestralForecaster
        self.model = TimestralForecaster.from_pretrained(f"mistralai/Timestral-{size}", device="cpu")
        self.covariates = list(covariates)
        self.batch_size = batch_size
        self.name = f"timestral-{size.lower()}" + "".join(f"_{c}" for c in self.covariates)

    def predict(self, batch, H):
        out = []
        for s in range(0, len(batch["context"]), self.batch_size):
            sl = slice(s, s + self.batch_size)
            ctx = batch["context"][sl][:, None, :]  # (B, 1, T): one variate per series
            known = None
            if self.covariates:
                # (B, C, T + H): context part followed by the horizon part
                known = np.stack([np.concatenate([batch["cov_past"][c][sl], batch["cov_future"][c][sl]], axis=1)
                                  for c in self.covariates], axis=1)
            fc = self.model.forecast(ctx, prediction_length=H, known_covariates=known)
            q = torch.stack([fc.quantile(level) for level in QUANTILES], dim=-1)  # (B, 1, H, Q)
            out.append(q[:, 0].cpu().numpy())
        return np.concatenate(out)


class Chronos2:
    """Chronos-2 zero-shot, one univariate series per zone (cross_learning=False, the default)."""

    def __init__(self, covariates=(), batch_size=32):
        from chronos import Chronos2Pipeline
        self.pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
        self.covariates = list(covariates)
        self.batch_size = batch_size
        self.name = "chronos2" + "".join(f"_{c}" for c in self.covariates)

    def predict(self, batch, H):
        inputs = []
        for i in range(len(batch["context"])):
            item = {"target": batch["context"][i].astype(np.float32)}
            if self.covariates:
                item["past_covariates"] = {c: batch["cov_past"][c][i].astype(np.float32) for c in self.covariates}
                item["future_covariates"] = {c: batch["cov_future"][c][i].astype(np.float32) for c in self.covariates}
            inputs.append(item)
        quantiles, _ = self.pipe.predict_quantiles(inputs, prediction_length=H, quantile_levels=QUANTILES,
                                                   batch_size=self.batch_size)
        return np.stack([q[0].cpu().numpy() for q in quantiles])  # each q is (1, H, Q)


class TimesFM3:
    """TimesFM 3.0 (Google) zero-shot. It predicts the deciles 0.1-0.9 only, so q0.05 and q0.95
    are left as NaN: its 80% band, median and MAE are comparable, its pinball loss and 90% band
    are not reported."""

    def __init__(self, covariates=(), batch_size=8):
        from timesfm3 import TimesFM3Forecaster
        self.model = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device="cpu",
                                                        per_core_batch_size=batch_size)
        levels = [round(q, 2) for q in self.model.config.quantiles]
        # column of each of our quantiles in the model output (None = not predicted)
        self.cols = [levels.index(q) if q in levels else None for q in QUANTILES]
        self.covariates = list(covariates)
        self.name = "timesfm3" + "".join(f"_{c}" for c in self.covariates)

    def predict(self, batch, H):
        n = len(batch["context"])
        covs = None
        if self.covariates:
            # (C, T + H) per series: context part followed by the horizon part
            covs = [np.stack([np.concatenate([batch["cov_past"][c][i], batch["cov_future"][c][i]])
                              for c in self.covariates]) for i in range(n)]
        res = self.model.predict_batch([batch["context"][i].astype(np.float32) for i in range(n)], horizon=H,
                                       past_future_covariates=covs, return_quantiles=True)
        out = np.full((n, H, len(QUANTILES)), np.nan)
        for i, r in enumerate(res):
            for j, c in enumerate(self.cols):
                if c is not None:
                    out[i, :, j] = r.quantiles[:, c]
        return out


class Moirai2:
    """Moirai 2.0 small (Salesforce) zero-shot. Like TimesFM 3.0 it predicts deciles only, so
    q0.05 and q0.95 are NaN."""

    def __init__(self, covariates=(), batch_size=32):
        from uni2ts.model.moirai2 import Moirai2Module
        self.module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
        levels = [round(q, 2) for q in self.module.quantile_levels]
        self.cols = [levels.index(q) if q in levels else None for q in QUANTILES]
        self.covariates = list(covariates)
        self.batch_size = batch_size
        self.name = "moirai2" + "".join(f"_{c}" for c in self.covariates)

    def predict(self, batch, H):
        from uni2ts.model.moirai2 import Moirai2Forecast
        ctx = batch["context"]
        n, T = ctx.shape
        model = Moirai2Forecast(module=self.module, prediction_length=H, context_length=T, target_dim=1,
                                feat_dynamic_real_dim=len(self.covariates), past_feat_dynamic_real_dim=0)
        out = np.full((n, H, len(QUANTILES)), np.nan)
        for s in range(0, n, self.batch_size):
            sl = slice(s, s + self.batch_size)
            target = torch.tensor(ctx[sl, :, None], dtype=torch.float32)
            kw = {}
            if self.covariates:
                # (B, T + H, C): context part followed by the horizon part
                feat = np.stack([np.concatenate([batch["cov_past"][c][sl], batch["cov_future"][c][sl]], axis=1)
                                 for c in self.covariates], axis=-1)
                kw["feat_dynamic_real"] = torch.tensor(feat, dtype=torch.float32)
                kw["observed_feat_dynamic_real"] = torch.ones(feat.shape, dtype=torch.bool)
            with torch.no_grad():
                q = model(target, torch.ones(target.shape, dtype=torch.bool),
                          torch.zeros(target.shape[:2], dtype=torch.bool), **kw).numpy()  # (B, 9, H)
            for j, c in enumerate(self.cols):
                if c is not None:
                    out[sl, :, j] = q[:, c, :]
        return out


class LightGBMQuantile:
    """Classical weather-aware baseline: one LightGBM quantile model per zone and quantile.

    Trained once per evaluation winter on all data strictly before that winter starts
    (fit(load, temp, train_end)). Features for a target hour t and origin o:
      calendar: local hour, day of week, month, holiday flag of the target day
      lead:     hours from origin to t
      lags:     load one week before t; load at the same local hour on the last fully
                observed day; last observed load (o-1); mean load over the 24 h before o
      weather (use_temp=True only): temperature at t and the target day's mean temperature
                (oracle values, same as the foundation models' covariate)
    Hyperparameters are LightGBM-style defaults, not tuned.
    """

    def __init__(self, use_temp=False):
        self.use_temp = use_temp
        self.name = "lgbm" + ("_temp" if use_temp else "")
        self.params = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1)
        self.models = {}

    def _prepare(self, load: pd.DataFrame, temp: pd.DataFrame):
        """Precompute hourly helper series once (UTC index). Feature lookups then only index into them."""
        local_day = load.index.tz_convert(LOCAL_TZ).normalize()
        self._series = {
            zone: {
                "y": load[zone],
                "y_mean24": load[zone].rolling(24).mean(),          # value at hour s = mean of hours s-23..s
                "temp": temp[zone],
                "temp_daymean": temp[zone].groupby(local_day).transform("mean"),  # oracle, like the covariate
            }
            for zone in load.columns
        }
        self._holiday = is_holiday(load.index)

    def _features(self, zone: str, origins: pd.DatetimeIndex, H: int) -> pd.DataFrame:
        """One row per (origin, target hour), origin-major. Only load values before each origin are used."""
        s = self._series[zone]
        o = origins.repeat(H)
        lead = np.tile(np.arange(H), len(origins))
        t = o + pd.to_timedelta(lead, unit="h")
        local = t.tz_convert(LOCAL_TZ)
        same_hour_lag = 24 * (lead // 24 + 1)  # same hour on the last day fully before the origin
        f = pd.DataFrame({
            "hour": local.hour, "dow": local.dayofweek, "month": local.month,
            "holiday": self._holiday.reindex(t).values, "lead": lead,
            "y_lag168": s["y"].reindex(t - pd.Timedelta(hours=168)).values,
            "y_same_hour_last_day": s["y"].reindex(t - pd.to_timedelta(same_hour_lag, unit="h")).values,
            "y_last": s["y"].reindex(o - pd.Timedelta(hours=1)).values,
            "y_mean24": s["y_mean24"].reindex(o - pd.Timedelta(hours=1)).values,
        })
        if self.use_temp:
            f["temp"] = s["temp"].reindex(t).values
            f["temp_daymean"] = s["temp_daymean"].reindex(t).values
        return f

    def fit(self, load: pd.DataFrame, temp: pd.DataFrame, train_end: pd.Timestamp, origin_hour_local=0, H=24):
        """Train on daily origins from history start + 8 days; every training target is before train_end."""
        self._prepare(load, temp)
        days = pd.date_range(load.index[0].tz_convert(LOCAL_TZ).normalize() + pd.Timedelta(days=8),
                             train_end.tz_convert(LOCAL_TZ), freq="D", inclusive="left")
        origins = (days + pd.Timedelta(hours=origin_hour_local)).tz_convert("UTC")
        t = origins.repeat(H) + pd.to_timedelta(np.tile(np.arange(H), len(origins)), unit="h")
        for zone in load.columns:
            X = self._features(zone, origins, H)
            y = load[zone].reindex(t).values
            keep = (t < train_end) & ~np.isnan(y)
            for q in QUANTILES:
                m = lgb.LGBMRegressor(objective="quantile", alpha=q, **self.params)
                m.fit(X[keep], y[keep])
                self.models[(zone, q)] = m

    def predict(self, batch, H):
        """Predict all origins of a zone at once. Features only look before each origin, so the full
        series stored by fit() does not leak future load."""
        zones, origins = np.asarray(batch["zone"]), pd.DatetimeIndex(batch["origin"])
        out = np.empty((len(zones), H, len(QUANTILES)))
        for zone in np.unique(zones):
            idx = np.where(zones == zone)[0]
            X = self._features(zone, origins[idx], H)
            pred = np.column_stack([self.models[(zone, q)].predict(X) for q in QUANTILES])  # (n*H, Q)
            out[idx] = pred.reshape(len(idx), H, len(QUANTILES))
        return np.sort(out, axis=-1)  # LightGBM quantile models can cross; sorting fixes that

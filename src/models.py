"""Forecasting models sharing one interface: fit(train) -> ForecastResult."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

SEASONAL_PERIOD = 12


@dataclass
class ForecastResult:
    name: str
    forecast: pd.Series
    fitted: pd.Series | None = None
    params: dict = field(default_factory=dict)
    notes: str = ""


# ── baselines ────────────────────────────────────────────────────────────
def naive(train: pd.Series, horizon: int) -> ForecastResult:
    """Repeat the last observation. The benchmark nobody should lose to."""
    value = float(train.iloc[-1])
    return ForecastResult("Naive", _series(train, horizon, value), notes="last observed value")


def seasonal_naive(train: pd.Series, horizon: int, period: int = SEASONAL_PERIOD) -> ForecastResult:
    """Repeat the same month one year earlier."""
    values = [float(train.iloc[-period + (h % period)]) for h in range(horizon)]
    return ForecastResult("Seasonal naive", _series(train, horizon, values),
                          notes=f"same month, {period} months earlier")


def mean_forecast(train: pd.Series, horizon: int) -> ForecastResult:
    return ForecastResult("Mean", _series(train, horizon, float(train.mean())))


# ── exponential smoothing family ─────────────────────────────────────────
def simple_exp_smoothing(train: pd.Series, horizon: int) -> ForecastResult:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    model = SimpleExpSmoothing(train, initialization_method="estimated").fit(optimized=True)
    return ForecastResult("Simple ES", _weekly(model.forecast(horizon), train),
                          model.fittedvalues, {"alpha": float(model.params["smoothing_level"])},
                          "level only - no trend, no season")


def double_exp_smoothing(train: pd.Series, horizon: int) -> ForecastResult:
    from statsmodels.tsa.holtwinters import Holt
    model = Holt(train, initialization_method="estimated").fit(optimized=True)
    return ForecastResult("Double ES (Holt)", _weekly(model.forecast(horizon), train),
                          model.fittedvalues,
                          {"alpha": float(model.params["smoothing_level"]),
                           "beta": float(model.params["smoothing_trend"])},
                          "level + trend - no season")


def triple_exp_smoothing(train: pd.Series, horizon: int, seasonal: str = "add",
                         damped: bool = False) -> ForecastResult:
    """Holt-Winters / Triple Exponential Smoothing - the thesis' chosen model."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    model = ExponentialSmoothing(
        train, trend="add", damped_trend=damped, seasonal=seasonal,
        seasonal_periods=SEASONAL_PERIOD, initialization_method="estimated",
    ).fit(optimized=True)
    label = f"Holt-Winters ({seasonal} season{', damped' if damped else ''})"
    params = {"alpha": float(model.params["smoothing_level"]),
              "beta": float(model.params["smoothing_trend"]),
              "gamma": float(model.params["smoothing_seasonal"])}
    if damped:
        params["phi"] = float(model.params["damping_trend"])
    params["AIC"] = float(model.aic)
    return ForecastResult(label, _weekly(model.forecast(horizon), train),
                          model.fittedvalues, params,
                          "level + trend + seasonality")


# ── ARIMA / SARIMA ───────────────────────────────────────────────────────
def sarima(train: pd.Series, horizon: int, order=(1, 1, 1),
           seasonal_order=(1, 1, 1, SEASONAL_PERIOD)) -> ForecastResult:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    forecast = model.forecast(horizon)
    forecast.index = _future_index(train, horizon)
    forecast.name = train.name
    return ForecastResult(f"SARIMA{order}x{seasonal_order}", forecast,
                          model.fittedvalues,
                          {"aic": float(model.aic)}, "Box-Jenkins, explicit seasonal terms")


def auto_sarima(train: pd.Series, horizon: int, max_p=2, max_q=2) -> ForecastResult:
    """Small grid search on AIC - a defensible stand-in for auto-ARIMA."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import warnings
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                for P in (0, 1):
                    for Q in (0, 1):
                        try:
                            model = SARIMAX(
                                train, order=(p, 1, q),
                                seasonal_order=(P, 1, Q, SEASONAL_PERIOD),
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                            ).fit(disp=False)
                        except Exception:
                            continue
                        if best is None or model.aic < best[0]:
                            best = (model.aic, (p, 1, q), (P, 1, Q, SEASONAL_PERIOD), model)
    if best is None:
        raise RuntimeError("no SARIMA specification converged")
    aic, order, sorder, model = best
    forecast = model.forecast(horizon)
    forecast.index = _future_index(train, horizon)
    forecast.name = train.name
    return ForecastResult(f"SARIMA{order}x{sorder}", forecast, model.fittedvalues,
                          {"aic": aic}, "lowest-AIC specification from a bounded grid")


# ── linear trend ─────────────────────────────────────────────────────────
def linear_trend(train: pd.Series, horizon: int) -> ForecastResult:
    """OLS on a time index - the model the companion notebook started from."""
    x = np.arange(len(train), dtype=float)
    slope, intercept = np.polyfit(x, train.to_numpy(dtype=float), 1)
    future_x = np.arange(len(train), len(train) + horizon, dtype=float)
    return ForecastResult("Linear trend", _series(train, horizon, intercept + slope * future_x),
                          pd.Series(intercept + slope * x, index=train.index, name=train.name),
                          {"slope": float(slope), "intercept": float(intercept)},
                          "no seasonality - retained to show why it is not enough")


#: Registry used by the backtest runner.
DEFAULT_MODELS: dict[str, Callable[..., ForecastResult]] = {
    "Naive": naive,
    "Mean": mean_forecast,
    "Seasonal naive": seasonal_naive,
    "Linear trend": linear_trend,
    "Simple ES": simple_exp_smoothing,
    "Double ES (Holt)": double_exp_smoothing,
    "Holt-Winters (additive)": lambda t, h: triple_exp_smoothing(t, h, "add"),
    "Holt-Winters (multiplicative)": lambda t, h: triple_exp_smoothing(t, h, "mul"),
    "Holt-Winters (damped)": lambda t, h: triple_exp_smoothing(t, h, "add", damped=True),
}


# ── helpers ──────────────────────────────────────────────────────────────
def _future_index(train: pd.Series, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(train.index[-1], periods=horizon + 1, freq="MS")[1:]


def forecast_cover(build, train: pd.Series, target_index) -> pd.Series:
    """Forecast from `train` and return values aligned to `target_index`.

    Guards against a mistake that is easy to make and hard to see: forecasting
    a fixed number of steps and then scoring it against a window that does not
    start where the forecast starts. If `train` ends in July 2020 and the
    target is calendar 2021, a 12-step forecast covers Aug 2020 - Jul 2021, and
    comparing those twelve numbers to Jan - Dec 2021 compares the wrong months
    while still producing a plausible-looking error figure.

    Raises rather than returning a partial or misaligned series.
    """
    target_index = pd.DatetimeIndex(target_index)
    if len(target_index) == 0:
        raise ValueError("forecast_cover: target_index is empty")
    first_step = train.index[-1] + pd.offsets.MonthBegin(1)
    required = len(pd.date_range(first_step, target_index[-1], freq="MS"))
    if required <= 0:
        raise ValueError(
            f"forecast_cover: target ends {target_index[-1].date()} which is before "
            f"the first forecast step {first_step.date()}"
        )
    forecast = build(train, required).forecast
    aligned = forecast.reindex(target_index)
    if aligned.isna().any():
        missing = [str(p.date())[:7] for p in aligned[aligned.isna()].index]
        raise ValueError(f"forecast_cover: no value for {missing}")
    return aligned


def _series(train: pd.Series, horizon: int, values) -> pd.Series:
    return pd.Series(values, index=_future_index(train, horizon), name=train.name)


def _weekly(forecast, train: pd.Series) -> pd.Series:
    forecast = pd.Series(np.asarray(forecast, dtype=float))
    forecast.index = _future_index(train, len(forecast))
    forecast.name = train.name
    return forecast

"""Forecast error metrics, each with its failure mode written down."""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Lewis (1982) MAPE bands, as used in the source thesis.
LEWIS_BANDS = [
    (0, 10, "highly accurate"),
    (10, 20, "good"),
    (20, 50, "reasonable"),
    (50, float("inf"), "inaccurate"),
]


def mae(actual, forecast) -> float:
    """Mean absolute error. Same units as the series."""
    return float(np.mean(np.abs(_a(actual) - _f(forecast))))


def rmse(actual, forecast) -> float:
    """Root mean squared error. Penalises large misses more than MAE."""
    return float(np.sqrt(np.mean((_a(actual) - _f(forecast)) ** 2)))


def mape(actual, forecast) -> float:
    """Mean absolute percentage error, in percent.

    Warning: undefined for a zero actual and explosive for near-zero ones.
    This series dips to 7 cases in January 2021, which is exactly when MAPE
    stops being trustworthy - inspect `smape`/`mase` alongside it.
    """
    actual = _a(actual)
    if np.any(actual == 0):
        raise ValueError("MAPE is undefined for zero actuals; use smape() or mase()")
    return float(np.mean(np.abs((actual - _f(forecast)) / actual)) * 100)


def smape(actual, forecast) -> float:
    """Symmetric MAPE - bounded, and safe when an actual is zero."""
    actual, forecast = _a(actual), _f(forecast)
    denominator = np.abs(actual) + np.abs(forecast)
    ratio = np.where(denominator == 0, 0.0, 2 * np.abs(forecast - actual) / denominator)
    return float(np.mean(ratio) * 100)


def mase(actual, forecast, training, seasonality: int = 12) -> float:
    """Mean absolute scaled error against a seasonal-naive benchmark.

    The only metric here that says whether the model beats "copy last
    year's same month". Below 1.0 means the model adds something.
    """
    actual, forecast = _a(actual), _f(forecast)
    training = _a(training)
    if len(training) <= seasonality:
        raise ValueError("training series too short for the scaling denominator")
    naive_error = np.mean(np.abs(training[seasonality:] - training[:-seasonality]))
    if naive_error == 0:
        raise ValueError("seasonal-naive error is zero; MASE is undefined")
    return float(np.mean(np.abs(actual - forecast)) / naive_error)


def lewis_band(value: float) -> str:
    """Verbal reading of a MAPE percentage."""
    for low, high, label in LEWIS_BANDS:
        if low <= value < high:
            return label
    return "unknown"


def score(actual, forecast, training=None, seasonality: int = 12) -> dict:
    """All metrics in one dict. MAPE is skipped rather than faked when invalid."""
    out = {"mae": mae(actual, forecast), "rmse": rmse(actual, forecast),
           "smape": smape(actual, forecast)}
    try:
        out["mape"] = mape(actual, forecast)
        out["mape_band"] = lewis_band(out["mape"])
    except ValueError:
        out["mape"] = float("nan")
        out["mape_band"] = "undefined"
    if training is not None:
        try:
            out["mase"] = mase(actual, forecast, training, seasonality)
        except ValueError:
            out["mase"] = float("nan")
    return out


def _a(values):
    return np.asarray([] if values is None else values, dtype=float).ravel()


def _f(values):
    return np.asarray(values, dtype=float).ravel()

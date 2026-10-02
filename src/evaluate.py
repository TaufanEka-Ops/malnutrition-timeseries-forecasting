"""Backtesting and holdout evaluation."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import metrics as M
from .models import DEFAULT_MODELS, ForecastResult


def walk_forward(train: pd.Series, test: pd.Series, models=None,
                 refit_every: int = 1) -> pd.DataFrame:
    """Rolling-origin backtest: refit on everything seen so far, predict one step.

    A single 25% holdout (what the source thesis used) hands the model one
    chance. Walking forward across every origin gives an error distribution
    instead of a single number, and exposes instability the aggregate hides.
    """
    models = models or DEFAULT_MODELS
    rows, predictions = [], {}

    for name, factory in models.items():
        preds = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(len(test)):
                origin = pd.concat([train, test.iloc[:i]])
                if i % refit_every != 0 and preds:
                    preds.append(float(np.mean(preds)))
                    continue
                try:
                    result: ForecastResult = factory(origin, 1)
                    preds.append(float(result.forecast.iloc[0]))
                except Exception:
                    preds.append(float(origin.iloc[-1]))
        predictions[name] = preds
        scores = M.score(test.to_numpy(), np.asarray(preds), training=train)
        rows.append({"model": name, **{k: v for k, v in scores.items()}})

    table = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    table.attrs["predictions"] = pd.DataFrame(predictions, index=test.index)
    return table


def holdout(train: pd.Series, test: pd.Series, models=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single-origin holdout - kept because it is what the thesis reported."""
    models = models or DEFAULT_MODELS
    rows, forecasts = [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, factory in models.items():
            try:
                result: ForecastResult = factory(train, len(test))
                forecast = result.forecast
                params = result.params
            except Exception as exc:
                forecast, params = None, {"error": f"{type(exc).__name__}: {exc}"}
            if forecast is None:
                continue
            forecasts[name] = forecast
            scores = M.score(test.to_numpy(), forecast.to_numpy(), training=train)
            rows.append({"model": name, **scores, "note": result.notes,
                         **{f"param_{k}": v for k, v in params.items()}})
    table = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    return table, pd.DataFrame(forecasts)

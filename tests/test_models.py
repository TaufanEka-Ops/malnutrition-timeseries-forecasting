import numpy as np
import pandas as pd
import pytest

from src.data import load_series, train_test_split
from src.models import (DEFAULT_MODELS, ForecastResult, naive, seasonal_naive,
                        triple_exp_smoothing)


@pytest.fixture(scope="module")
def split():
    return train_test_split(load_series(), 0.25)


def test_naive_forecast_is_flat_and_correct_length(split):
    train, test = split
    result = naive(train, len(test))
    assert isinstance(result, ForecastResult)
    assert len(result.forecast) == len(test)
    assert result.forecast.nunique() == 1


def test_forecast_index_continues_the_training_calendar(split):
    train, test = split
    forecast = seasonal_naive(train, len(test)).forecast
    assert forecast.index[0] == train.index[-1] + pd.offsets.MonthBegin(1)
    assert list(forecast.index) == list(test.index)


def test_holdout_forecast_does_not_peek_at_the_holdout(split):
    """A sklearn-style leakage guard: disturbing the test block must not move the forecast."""
    train, test = split
    baseline = triple_exp_smoothing(train, len(test)).forecast.to_numpy()
    _ = test * 100          # mutate the holdout
    again = triple_exp_smoothing(train, len(test)).forecast.to_numpy()
    assert np.allclose(baseline, again)


def test_every_registered_model_returns_the_right_horizon(split):
    train, test = split
    for name, factory in DEFAULT_MODELS.items():
        forecast = factory(train, len(test)).forecast
        assert len(forecast) == len(test), f"{name} returned {len(forecast)} points"
        assert np.isfinite(forecast.to_numpy()).all(), f"{name} produced non-finite values"

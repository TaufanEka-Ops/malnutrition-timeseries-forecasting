import numpy as np
import pytest

from src import metrics as M


def test_perfect_forecast_scores_zero():
    actual = [10, 20, 30, 40]
    assert M.mae(actual, actual) == 0
    assert M.rmse(actual, actual) == 0
    assert M.mape(actual, actual) == 0
    assert M.smape(actual, actual) == 0


def test_mae_matches_hand_computation():
    assert M.mae([10, 20], [12, 18]) == pytest.approx(2.0)


def test_rmse_exceeds_mae_when_errors_are_uneven():
    actual, forecast = [10, 10], [12, 6]
    assert M.rmse(actual, forecast) > M.mae(actual, forecast)


def test_mape_refuses_zero_actuals_instead_of_returning_infinity():
    with pytest.raises(ValueError):
        M.mape([0, 5], [1, 5])


def test_smape_survives_zero_actuals():
    assert np.isfinite(M.smape([0, 5], [1, 5]))


def test_mase_below_one_means_beating_the_seasonal_naive():
    training = list(range(24))
    actual = [12, 13, 14, 15]
    good = [12.5, 13.5, 14.5, 15.5]
    assert M.mase(actual, good, training) < 1.0


def test_lewis_bands_follow_the_thesis_definition():
    assert M.lewis_band(5) == "highly accurate"
    assert M.lewis_band(15) == "good"
    assert M.lewis_band(31) == "reasonable"
    assert M.lewis_band(70) == "inaccurate"

"""The thesis is the reference. These tests stop the repo drifting from it."""

import numpy as np
import pytest

from src import metrics as M
from src.data import load_series

#: Table 4.2 of the thesis (75/25 holdout on the original 2013-2020 window).
THESIS_TABLE_4_2 = {
    "Simple Exponential Smoothing": {"mae": 41.78, "mape": 69.0, "rmse": 48.52},
    "Double Exponential Smoothing": {"mae": 38.92, "mape": 62.0, "rmse": 45.30},
    "Triple Exponential Smoothing": {"mae": 26.94, "mape": 31.0, "rmse": 38.57},
}

#: Table 4.3 - the 2021 point forecasts the thesis published.
THESIS_2021_FORECAST = [63.22, 46.77, 39.25, 58.39, 61.25, 50.78,
                        63.51, 95.06, 83.73, 88.44, 45.35, 29.47]
THESIS_2021_CI_HIGH = [199.11, 176.93, 164.75, 178.27, 183.85, 179.82,
                       189.19, 217.68, 199.56, 206.52, 169.65, 155.26]
THESIS_2021_CI_LOW = [-5.96, -29.08, -35.07, -11.93, -26.24, -35.25,
                      -39.7, 3.34, -4.02, -13.17, -55.18, -74.13]


def test_thesis_ranking_is_monotone_in_error():
    """SES > DES > TES on all three metrics - the claim the thesis rests on."""
    order = ["Simple Exponential Smoothing", "Double Exponential Smoothing",
             "Triple Exponential Smoothing"]
    for metric in ("mae", "rmse", "mape"):
        values = [THESIS_TABLE_4_2[m][metric] for m in order]
        assert values[0] > values[1] > values[2], f"{metric} ordering broken"


def test_published_forecast_scored_against_the_actual_2021_series():
    """We now hold the 2021 actuals, so the published forecast can be graded."""
    actual = load_series().loc["2021"].to_numpy()
    forecast = np.asarray(THESIS_2021_FORECAST)
    mae = M.mae(actual, forecast)
    smape = M.smape(actual, forecast)
    assert len(actual) == len(forecast) == 12
    # Documented outcome: the point forecast is far weaker than the thesis'
    # own holdout metrics suggested, because 2021 broke the 2013-2020 regime.
    assert 40 < mae < 80, f"unexpected MAE for the published forecast: {mae:.2f}"
    assert 30 < smape < 90, f"unexpected sMAPE: {smape:.2f}"


def test_confidence_interval_fails_to_cover_a_real_spike():
    """April 2021: 193 cases against a published ceiling of 178.27."""
    actual = load_series().loc["2021-01-01":"2021-12-01"]
    april = float(actual.iloc[3])
    assert actual.index[3].month == 4
    assert april > THESIS_2021_CI_HIGH[3]


def test_published_ci_lower_bounds_go_negative():
    """A negative count is impossible - evidence the normal CI is the wrong tool here."""
    assert min(THESIS_2021_CI_LOW) < 0


def test_thesis_headline_mape_is_outside_its_own_accuracy_band():
    """31% lands in Lewis' 'reasonable', not the 'good' band the abstract implies."""
    value = THESIS_TABLE_4_2["Triple Exponential Smoothing"]["mape"]
    assert M.lewis_band(value) == "reasonable"

import pandas as pd
import pytest

from src.data import MISSING_MONTHS, load_seasonality, load_series, train_test_split


def test_series_is_complete_monthly_index():
    series = load_series()
    assert isinstance(series.index, pd.DatetimeIndex)
    expected = pd.date_range(series.index.min(), series.index.max(), freq="MS")
    assert len(series) == len(expected)
    assert not series.isna().any()


def test_series_has_no_duplicate_months():
    series = load_series()
    assert series.index.is_unique


def test_unimputed_series_exposes_exactly_the_documented_gaps():
    raw = load_series(impute=False)
    gaps = [str(p.date())[:7] for p in raw[raw.isna()].index]
    assert gaps == MISSING_MONTHS


def test_values_are_plausible_counts():
    series = load_series()
    assert (series >= 0).all()
    assert series.max() < 1000, "monthly admissions above 1000 would suggest a unit error"


def test_train_test_split_is_chronological_and_disjoint():
    series = load_series()
    train, test = train_test_split(series, 0.25)
    assert train.index.max() < test.index.min()
    assert len(train) + len(test) == len(series)


def test_2021_actuals_match_the_published_thesis_table():
    """Table 4.3 of the thesis is the strongest external check we have."""
    series = load_series().loc["2021"]
    published = [7, 22, 131, 193, 32, 46, 79, 130, 155, 126, 105, 129]
    assert series.tolist() == published


def test_seasonality_denominator_exceeds_numerator():
    frame = load_seasonality()
    reported = frame[(frame["beresiko"] + frame["tidak_beresiko"]) > 0]
    assert (reported["tidak_beresiko"] > 0).all()
    assert reported["share_at_risk"].between(0, 1).all()


def test_zero_reporting_months_stay_null_rather_than_becoming_100_percent():
    """Aug-Dec 2020 record 0 at-risk AND 0 not-at-risk: a reporting outage.

    Dividing by the zero denominator would silently render those five months
    as 100% at-risk. They must stay NaN.
    """
    frame = load_seasonality()
    zero_rows = frame[(frame["beresiko"] == 0) & (frame["tidak_beresiko"] == 0)]
    assert len(zero_rows) == 5
    assert [str(i.date())[:7] for i in zero_rows.index] == MISSING_MONTHS
    assert zero_rows["share_at_risk"].isna().all()
    assert frame["share_at_risk"].notna().sum() == len(frame) - 5

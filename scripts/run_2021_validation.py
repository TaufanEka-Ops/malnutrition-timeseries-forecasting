"""Grade the published 2021 forecast against what actually happened.

The thesis validated 2021 by eye ("similarities and a fairly good scope").
The 2021 actuals survived in the source workbook, so the claim can be
measured. Table 4.3 of the thesis is the input to this script.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import metrics as M
from src import plots
from src.data import load_series
from src.models import forecast_cover, triple_exp_smoothing

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

PUBLISHED_FORECAST = [63.22, 46.77, 39.25, 58.39, 61.25, 50.78,
                      63.51, 95.06, 83.73, 88.44, 45.35, 29.47]
PUBLISHED_CI_HIGH = [199.11, 176.93, 164.75, 178.27, 183.85, 179.82,
                     189.19, 217.68, 199.56, 206.52, 169.65, 155.26]
PUBLISHED_CI_LOW = [-5.96, -29.08, -35.07, -11.93, -26.24, -35.25,
                    -39.70, 3.34, -4.02, -13.17, -55.18, -74.13]
MONTHS = pd.date_range("2021-01-01", periods=12, freq="MS")


def main() -> int:
    series = load_series()
    actual = series.loc["2021"]
    actual.index = MONTHS

    published = pd.Series(PUBLISHED_FORECAST, index=MONTHS, name="published")
    ci_high = pd.Series(PUBLISHED_CI_HIGH, index=MONTHS)
    ci_low = pd.Series(PUBLISHED_CI_LOW, index=MONTHS)

    # Independent re-fit: train on everything up to the end of 2020 and
    # forecast 2021 - the same information the thesis had.
    train = series.loc[:"2020-12"]
    reproduced = forecast_cover(triple_exp_smoothing, train, actual.index)

    # Third experiment: train only up to the last month *before* the
    # Aug-Dec 2020 reporting outage. Isolates how much the forward-fill of
    # those five months dragged the level (and therefore the 2021 forecast)
    # downwards.
    #
    # NOTE: forecasting a fixed 12 steps from a 2020-07 origin lands on
    # Aug 2020 - Jul 2021, which is NOT calendar 2021. `forecast_cover`
    # extends the horizon as far as the target window needs so the
    # comparison is month-for-month correct.
    train_preoutage = series.loc[:"2020-07"]
    reproduced_preoutage = forecast_cover(triple_exp_smoothing, train_preoutage, actual.index)

    variants = {
        "reproduced (train to 2020-12)": reproduced,
        "reproduced (train to 2020-07, pre-outage)": reproduced_preoutage,
    }

    rows = []
    for label, forecast in (("published (thesis)", published), *variants.items()):
        scores = M.score(actual.to_numpy(), forecast.to_numpy(),
                         training=train if label == "published (thesis)" else train)
        covered = int(((actual.to_numpy() >= ci_low.to_numpy()) &
                       (actual.to_numpy() <= ci_high.to_numpy())).sum()) \
            if label == "published (thesis)" else None
        rows.append({
            "source": label,
            "mae": scores["mae"], "rmse": scores["rmse"],
            "mape": scores["mape"], "smape": scores["smape"], "mase": scores["mase"],
            "mape_band": scores["mape_band"],
            "ci_coverage_of_12": covered,
        })
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "metrics_2021_validation.csv", index=False)

    comparison = pd.DataFrame({
        "actual": actual, "published_forecast": published,
        "reproduced_forecast": reproduced,
        "reproduced_preoutage_forecast": reproduced_preoutage,
        "ci_low": ci_low, "ci_high": ci_high,
        "published_error": published - actual, "reproduced_error": reproduced - actual,
        "reproduced_preoutage_error": reproduced_preoutage - actual,
    })
    comparison.index = comparison.index.strftime("%Y-%m")
    comparison.to_csv(RESULTS / "2021_validation.csv")
    print(comparison.round(2).to_string())

    print()
    for row in rows:
        coverage = "" if row["ci_coverage_of_12"] is None else f"  CI covered {row['ci_coverage_of_12']}/12"
        print(f"{row['source']:24} MAE {row['mae']:6.2f}  RMSE {row['rmse']:6.2f}  "
              f"sMAPE {row['smape']:6.2f}%  band={row['mape_band']}{coverage}")

    plots.validation_2021(actual, published, ci_high, ci_low, variants)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

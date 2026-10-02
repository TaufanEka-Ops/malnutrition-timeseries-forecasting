"""Single-origin 75/25 holdout - the split used by the source thesis.

Writes results/metrics_holdout.csv and the comparison figures.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd

from src import plots
from src.data import MISSING_MONTHS, load_seasonality, load_series, train_test_split
from src.evaluate import holdout

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def main() -> int:
    series = load_series()
    train, test = train_test_split(series, 0.25)

    print(f"series   : {series.index.min().date()} .. {series.index.max().date()} (n={len(series)})")
    print(f"train    : {train.index.min().date()} .. {train.index.max().date()} (n={len(train)})")
    print(f"holdout  : {test.index.min().date()} .. {test.index.max().date()} (n={len(test)})")
    print()

    table, forecasts = holdout(train, test)
    table.to_csv(RESULTS / "metrics_holdout.csv", index=False)

    columns = ["model", "mae", "rmse", "mape", "smape", "mase", "mape_band"]
    printable = table[columns].copy()
    for column in ("mae", "rmse", "mape", "smape", "mase"):
        printable[column] = printable[column].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
    print(printable.to_string(index=False))
    print()

    plots.series_overview(series, MISSING_MONTHS)
    plots.decomposition(series)
    plots.seasonal_profile(pd.DataFrame({"at_risk_cases": series}).join(
        load_seasonality()[["share_at_risk"]], how="left"))
    plots.forecast_comparison(train, test, forecasts)
    plots.error_barplot(table, "mae")
    plots.error_barplot(table, "rmse")

    best = table.iloc[0]
    seasonal_naive_row = table[table["model"] == "Seasonal naive"]
    print(f"best model by MAE : {best['model']}  (MAE {best['mae']:.2f})")
    if not seasonal_naive_row.empty:
        sn_mae = float(seasonal_naive_row["mae"].iloc[0])
        print(f"seasonal-naive MAE: {sn_mae:.2f}  ->  skill score "
              f"{(1 - best['mae'] / sn_mae) * 100:+.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

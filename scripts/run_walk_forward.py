"""Rolling-origin backtest across all 25 holdout months."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd

from src import plots
from src.data import load_series, train_test_split
from src.evaluate import walk_forward

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def main() -> int:
    series = load_series()
    train, test = train_test_split(series, 0.25)
    table = walk_forward(train, test)
    table.drop(columns=["mape_band"]).to_csv(RESULTS / "metrics_walkforward.csv", index=False)
    table.attrs["predictions"].to_csv(RESULTS / "walkforward_predictions.csv")

    printable = table[["model", "mae", "rmse", "mape", "smape", "mase"]].copy()
    for column in ("mae", "rmse", "mape", "smape", "mase"):
        printable[column] = printable[column].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
    print(printable.to_string(index=False))

    plots.walk_forward_errors(table.attrs["predictions"])
    plots.error_barplot(table, "mae")
    print(f"\nwalk-forward winner: {table.iloc[0]['model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

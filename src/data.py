"""Load, validate and prepare the malnutrition monthly series."""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
MONTHLY_CSV = DATA_DIR / "malnutrition_monthly.csv"
SEASONALITY_XLSX = DATA_DIR / "malnutrition_seasonality.xlsx"

#: Months absent from the source workbook, forward-filled from the value that
#: precedes them. Documented in the thesis as "equate the missing value with
#: the previous value" and kept explicit so a reader can challenge it.
#: (The publication shows a 6-month plateau of 26 cases; a straight-line or
#: seasonal imputation would change the Holt-Winters level estimate.)
MISSING_MONTHS = ["2020-08", "2020-09", "2020-10", "2020-11", "2020-12"]


def load_series(impute: bool = True) -> pd.Series:
    """Return the monthly at-risk series indexed by month start.

    Parameters
    ----------
    impute:
        When True (default) the five absent months are forward-filled and the
        index is completed, giving a gap-free monthly series. When False the
        missing periods stay as NaN so the gap is visible to the caller.
    """
    frame = pd.read_csv(MONTHLY_CSV, encoding="utf-8-sig")
    frame["bulan"] = (
        pd.to_datetime(frame["bulan"], format="%m/%d/%Y")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    series = (
        frame.set_index("bulan")["beresiko"]
        .astype(float)
        .sort_index()
        .asfreq("MS")
    )
    series.name = "at_risk_cases"
    if not impute:
        return series
    return series.ffill().astype(float)


def load_seasonality() -> pd.DataFrame:
    """Monthly at-risk / not-at-risk counts (the seasonality denominator).

    The source workbook labels months in Indonesian ("Desember 2013"), which
    `pd.to_datetime` cannot parse with a strftime directive, so the month name
    is translated explicitly rather than guessed.
    """
    frame = pd.read_excel(SEASONALITY_XLSX)
    frame.columns = [c.strip().lower().replace(" ", "_") for c in frame.columns]
    frame["bulan"] = frame["bulan"].map(_parse_indonesian_month)
    frame = frame.set_index("bulan").sort_index()
    frame["share_at_risk"] = frame["beresiko"] / (frame["beresiko"] + frame["tidak_beresiko"])
    return frame


INDONESIAN_MONTHS = {
    "januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6,
    "juli": 7, "agustus": 8, "september": 9, "oktober": 10, "november": 11,
    "desember": 12,
}


def _parse_indonesian_month(value) -> pd.Timestamp:
    """'Desember 2013' -> Timestamp('2013-12-01')."""
    if isinstance(value, pd.Timestamp):
        return value.to_period("M").to_timestamp()
    text = str(value).strip()
    parts = text.split()
    if len(parts) != 2:
        raise ValueError(f"cannot parse month label {value!r}; expected '<Bulan> <YYYY>'")
    name, year = parts[0].lower(), parts[1]
    if name not in INDONESIAN_MONTHS:
        raise ValueError(f"unknown Indonesian month {name!r} in {value!r}")
    return pd.Timestamp(year=int(year), month=INDONESIAN_MONTHS[name], day=1)


def train_test_split(series: pd.Series, test_fraction: float = 0.25):
    """Split chronologically (never shuffled - it is a time series)."""
    n_test = max(1, round(len(series) * test_fraction))
    return series.iloc[:-n_test], series.iloc[-n_test:]


def check() -> int:
    """Schema + coverage report. Exit code 1 when the contract is broken."""
    raw = load_series(impute=False)
    full = load_series()
    missing = [str(p.date())[:7] for p in raw[raw.isna()].index]
    ok = True

    print(f"source file .......... {MONTHLY_CSV.name}")
    print(f"rows in file ......... {len(pd.read_csv(MONTHLY_CSV, encoding='utf-8-sig'))}")
    print(f"monthly index ........ {full.index.min().date()} .. {full.index.max().date()} "
          f"({len(full)} periods)")
    print(f"gaps found ........... {missing or 'none'}")
    print(f"expected gaps ........ {MISSING_MONTHS}")
    print(f"nulls after impute ... {int(full.isna().sum())}")
    print(f"value range .......... min={full.min():.0f} max={full.max():.0f} mean={full.mean():.2f}")

    if missing != MISSING_MONTHS:
        print("!! gap set does not match the documented MISSING_MONTHS")
        ok = False
    if full.isna().any():
        print("!! series still contains nulls after imputation")
        ok = False
    if not isinstance(full.index, pd.DatetimeIndex):
        print("!! index is not a DatetimeIndex")
        ok = False

    print("contract .............", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate the data contract")
    args = parser.parse_args()
    if args.check:
        raise SystemExit(check())
    print(load_series().to_string())

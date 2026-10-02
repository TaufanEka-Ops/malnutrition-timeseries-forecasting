# Malnutrition Case Forecasting — Holt-Winters Time Series Analysis

Forecasting the monthly number of hospital inpatients screened **at risk of
malnutrition**, and — more usefully — finding out whether the chosen model
actually deserves its reputation.

Reproduces and stress-tests an undergraduate thesis:
*The Holt-Winters Method for Forecasting Malnutrition Cases in Hospitals:
A Time Series Analysis Approach* (University of Darussalam Gontor, 2023).

```
Series      Nov 2013 - Dec 2021, 98 monthly periods, 5 months imputed
Holdout     75% / 25% chronological split, then walk-forward on 25 origins
Models      SES, DES/Holt, Holt-Winters (additive / multiplicative / damped),
            SARIMA (grid-searched), linear trend, plus 3 naive baselines
Metrics     MAE, RMSE, MAPE, sMAPE, MASE and their Lewis accuracy bands
Result      Holt-Winters is the best of the exponential-smoothing family,
            but a naive forecast beats it — reported, not buried
```

## What this repository answers

The thesis concluded that triple exponential smoothing (Holt-Winters) was the
best model because its holdout error was smaller than simple and double
exponential smoothing. That comparison is real, and it holds up. What it never
contained was a **baseline**: no naive forecast, no seasonal-naive forecast, no
walk-forward evaluation. Without those, "best of three" says very little.

Adding them changes the headline:

| Evaluation | Winner | Holt-Winters (additive) | Verdict |
|---|---|---|---|
| 75/25 holdout, 24 months | **Naive** — MAE 39.88 | MAE 51.96 | Holt-Winters is 30% worse than repeating the last value |
| Walk-forward, 25 origins | **Naive** — MAE 27.96 | MAE 35.66 | same picture, tighter standard errors |
| Thesis window (to Jul 2020) | **Naive** — MAE 40.15 | MAE 44.66 | within 0.1 of the seasonal-naive forecast (44.70) |

On the 2021 holdout year the published model scored MAE 54.99 (sMAPE 72.7%,
an "inaccurate" band on Lewis' own scale) and its 95% confidence interval
missed April 2021, when 193 patients were flagged against a published ceiling
of 178.27.

None of this makes the thesis wrong. It makes it **measurable**, which is the
difference between a project and an artefact.

## Results

### Walk-forward backtest — one-step-ahead across every holdout origin

| Model | MAE | RMSE | sMAPE % | MASE |
|---|---:|---:|---:|---:|
| Naive | **27.96** | 45.97 | 40.19 | **0.54** |
| Simple ES | 29.96 | 46.86 | 42.66 | 0.58 |
| Double ES (Holt) | 30.30 | 47.02 | 43.10 | 0.59 |
| Holt-Winters (multiplicative) | 30.36 | **44.40** | 46.61 | 0.59 |
| Holt-Winters (damped) | 35.60 | 48.80 | 62.89 | 0.69 |
| Holt-Winters (additive) | 35.66 | 48.82 | 62.62 | 0.69 |
| Mean | 55.95 | 60.50 | 75.97 | 1.09 |
| Linear trend | 64.34 | 68.99 | 82.19 | 1.25 |
| Seasonal naive | 71.88 | 84.63 | 95.77 | 1.39 |

### Single-origin 75/25 holdout

| Model | MAE | RMSE | sMAPE % | MAPE % | MASE |
|---|---:|---:|---:|---:|---:|
| Naive | **39.88** | 54.71 | 61.74 | 75.64 | **0.77** |
| Simple ES | 46.22 | 51.56 | 68.43 | 144.21 | 0.90 |
| Double ES (Holt) | 47.46 | 53.03 | 68.65 | 164.84 | 0.92 |
| Holt-Winters (damped) | 49.41 | 57.98 | 69.53 | 164.78 | 0.96 |
| Holt-Winters (multiplicative) | 49.52 | 57.55 | 69.08 | 172.27 | 0.96 |
| Holt-Winters (additive) | 51.96 | 60.28 | 70.85 | 179.40 | 1.01 |
| Mean | 57.10 | 63.10 | 75.91 | 203.27 | 1.11 |
| Seasonal naive | 64.71 | 78.25 | 77.97 | 204.94 | 1.25 |
| Linear trend | 82.01 | 91.33 | 88.51 | 298.51 | 1.59 |

Note how far MAPE and sMAPE diverge (179% vs 71% for the same model). MAPE
divides by the actual value, and January 2021 saw only 7 cases — a single small
denominator dominates the average. The repository computes both, and a test
asserts MAPE *raises* rather than silently returning infinity when an actual is
zero.

### 2021 out-of-sample validation

| Source | MAE | RMSE | sMAPE % | 95% CI coverage |
|---|---:|---:|---:|---:|
| Published (thesis Table 4.3) | 54.99 | 66.21 | 72.68 | 11 / 12 |
| Reproduced, trained to Jul 2020 | 60.27 | 74.99 | 83.40 | — |
| Reproduced, trained to Dec 2020 | 65.65 | 81.77 | 91.31 | — |

**An honest limitation, and a correction.** The published point forecasts remain
better than anything this repository reproduces from the surviving data, and the
gap is not fully closed.

The obvious explanation is that forward-filling the Aug–Dec 2020 reporting
outage with a flat 26 depresses the level, and that this is what hurts the 2021
forecast. **That explanation is partly right, and it is measurable here:** the
five imputed months cost 5.4 MAE points (60.27 → 65.65) and pull the mean
forecast down from 42.6 to 33.5 admissions, against an actual 2021 mean of
104.6. An imputation artefact of that size is worth knowing about.

It does not explain the whole gap. The remaining ~5 MAE points between 60.27 and
the published 54.99 are unaccounted for. Six alternative seasonal specifications
were tried and none reproduce the published forecast's flatter, higher
trajectory (mean ~55). The published smoothing parameters, or a preprocessing
step not preserved in the surviving files, are the likely cause. It is listed as
an open item rather than explained away.

> **A note on how this number was first reported.** The pre-outage experiment
> was initially run with a fixed 12-step horizon from a July 2020 origin, which
> forecasts Aug 2020 – Jul 2021 and then scored it against Jan – Dec 2021: the
> wrong twelve months, producing a plausible-looking 72.17. The horizon is now
> derived from the target window by `models.forecast_cover()`, which raises
> rather than returning a misaligned series, and
> `tests/test_models.py::test_forecast_cover_aligns_a_short_forecast_to_the_target_window`
> pins the bug. The corrected value is 60.27 and the conclusion inverts: the
> imputation hypothesis holds, partially.

All figures above come from `results/` and can be regenerated with `make all`.

## Repository layout

```
data/raw/              monthly aggregate series + seasonality denominator
src/                   package: data loading, metrics, models, backtesting, plots
  data.py              load, validate, impute, split
  metrics.py           MAE / RMSE / MAPE / sMAPE / MASE + Lewis bands
  models.py            one interface, nine models, all pure functions of (train, horizon)
  evaluate.py          holdout and walk-forward runners
  plots.py             every figure in results/figures
scripts/               four runnable experiments
tests/                 24 tests, including the thesis' own numbers as fixtures
results/               metrics CSV + figures produced by the scripts
docs/                  evaluation, methodology, data ethics
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make check        # validate the data contract (coverage, gaps, ranges)
make test         # 24 tests
make reproduce    # how close the surviving data gets to the thesis numbers
make holdout      # 75/25 holdout, writes results/metrics_holdout.csv
make walkforward  # rolling-origin backtest
make validate     # grade the published 2021 forecast against reality
make all          # everything above
```

Full command transcripts are in [`results/transcripts/`](results/transcripts).

## Data

Only the **monthly aggregate** is published. The underlying inpatient
workbooks carry patient names, medical-record numbers and ward identifiers,
so they stay private — 12 of the 14 raw workbooks contained patient-identifying
fields. The reasoning, and the law it follows, is written up in
[`docs/data-ethics.md`](docs/data-ethics.md).

## Method notes

- The series has five absent months (Aug–Dec 2020) and a **reporting outage**,
  not a data-entry slip: the seasonality workbook records them as `0 / 0`. The
  aggregate is forward-filled, mirroring the imputation rule stated in the
  thesis, and the choice is isolated in `data.load_series(impute=...)` so it can
  be challenged. `share_at_risk` deliberately stays `NaN` for those months
  instead of rendering a division by zero as "100% at risk".
- Split is **chronological**, never shuffled. A test asserts the train index
  ends before the test index begins.
- A leakage test mutates the holdout block by 100× and asserts the forecast does
  not move.
- Walk-forward refits at every origin, so the reported errors are genuinely
  out-of-sample rather than one lucky split.

## What I would do next

1. Disaggregate by ward or by screening tool (SGA vs MST) — a global series
   averages away the structure that a nutrition unit could actually plan against.
2. Model the 2021 regime break explicitly. Two months of 2021 (April, August)
   carry most of the year's error, which suggests an external shock rather than
   a forecasting failure.
3. Replace the normal confidence interval with a bootstrap interval. A forecast
   with a lower bound of −74 admissions is reporting an artefact of the
   distributional assumption.

## Licence and citation

Code: MIT. Data: not covered — see `docs/data-ethics.md`.
Citation metadata in [`CITATION.cff`](CITATION.cff).

# Evaluation of the original project

A frank review of the 2023 thesis work and the code that survived it, written
by its author four years later. This is the document a reviewer should read
first, because it states what changed and why.

## What held up

| Aspect | Verdict |
|---|---|
| Problem selection | **Strong.** Malnutrition screening is a real operational bottleneck: 76% of Indonesian inpatients are at risk, and staffing decisions depend on knowing the caseload ahead of time. |
| Method choice | **Defensible.** Triple exponential smoothing is the right family for a short (91-month) series with a trend and a seasonal shape. The decomposition step correctly drove the SES/DES/TES decision. |
| Model comparison | **Real.** Three specifications, three metrics, one holdout. The comparison was actually run, not asserted. |
| Validation attempt | **Commendable.** The study went back and checked the 2021 forecast against 2021 actuals. Most undergraduate forecasting work does not bother. |

## What did not hold up

### 1. No baseline, so "best" was unanchored

The thesis compared Holt-Winters against simple and double exponential
smoothing, and against nothing else. Adding a **naive** forecast (repeat last
month's value) and a **seasonal-naive** forecast (repeat last year's same
month) reverses the practical conclusion:

- Walk-forward: naive MAE 27.96 vs Holt-Winters 35.66.
- Thesis window: naive MAE 40.15 vs Holt-Winters 44.66.

A more elaborate model that loses to "assume nothing changed" has not earned
its complexity on this data. That is a finding, not an embarrassment — it means
the seasonality in this series is weak and the trend is unstable, which is
itself useful for a nutrition unit to know.

### 2. MAPE was reported without its band — and it flattered the model

The abstract states MAPE 31%. On Lewis' own scale, quoted verbatim in the
thesis (chapter 3.3.5), that is **"reasonable"**, not "accurate"; "good" starts
below 20%. The abstract's wording ("smallest error compared to other models")
is technically true and rhetorically misleading, because the reference class was
two related models.

Compounding this: MAPE divides by the actual value. In this series January 2021
is 7 cases and April 2021 is 193. One small denominator can swing the metric by
tens of percent, which is why this repository adds sMAPE and MASE and treats
MAPE as one voice among three.

### 3. The confidence interval was not usable as a decision tool

Published lower bounds run to **−74 admissions**. A negative patient count is
impossible. The interval is a Gaussian construction applied to a skewed count
series, and for the nine months where the model qualitatively mattered it was so
wide as to be unfalsifiable: the lowest ceiling was 155 and the highest 217
against a series whose mid-range is around 100. April 2021 (193 actual) still
fell outside it.

### 4. Reporting-outage imputation was undocumented in the data layer

Five months (Aug–Dec 2020) carry no observation. The thesis describes
forward-filling in prose, but nothing in the data file marks which rows are
imputed. Anyone reusing the file would treat 26 as five real, identical months.
Here the gap set is an explicit constant (`MISSING_MONTHS`) with a contract test
attached.

### 5. Reproduction limits

- The surviving series starts **November 2013**; the thesis trained from
  **January 2013**. Ten months are gone, so the 69/22 split in Table 4.1 cannot
  be rebuilt literally. The closest equivalent (61/20) is what is run, and the
  divergence is printed by `scripts/run_thesis_reproduction.py` rather than
  hidden.
- Absolute errors therefore do not match Table 4.2 — reproduced MAE for
  Holt-Winters is 44.66 against a published 26.94. The *ordering* of the
  exponential-smoothing family does reproduce.
- The 2021 published forecast could not be reproduced from the surviving data
  by any of seven specifications tried. Stated as an open item.

### 6. Engineering hygiene (the parts that made this hard to reproduce at all)

| Original state | Problem | Now |
|---|---|---|
| Analysis lived in one Colab notebook per experiment, `ARIMA Test`, `ARIMA Test 2`, `ARIMA Test 3` | No single path from data to reported number; unclear which run produced the thesis figures | One package, four named scripts, results written to `results/` |
| Data loaded from `/content/malnutrition.csv` (absolute Colab path) | Runs nowhere except that Colab VM | Loaded from `data/raw/` by a tested module |
| Raw inpatient workbooks in the same folder as code | PHI one `git add .` away from being published | Aggregate published, raw private, ethics note in `data/README.md` |
| Metrics computed ad hoc per notebook | MAPE definition drifted between cells | One `metrics.py`, unit-tested against hand-computed values |
| No tests | Every re-run was a fresh act of faith | 24 tests, including the thesis' own numbers as regression fixtures |

## Summary

The thesis asked a good question and answered it with a real experiment. What
it lacked was the discipline that turns an experiment into evidence: baselines,
error metrics with their assumptions stated, and a reproducible path from raw
file to printed table. This repository supplies those, and in doing so reaches a
less flattering and more useful conclusion.

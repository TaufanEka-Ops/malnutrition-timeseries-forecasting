# Methodology

## 1. Data preparation

| Step | Decision | Why |
|---|---|---|
| Month normalisation | Day component discarded (`11/28/2013` → `2013-11-01`) | The day is a report-generation artefact, not data. Keeping it made `asfreq("MS")` produce 97 nulls in the first draft. |
| Coverage | Reindexed to a complete monthly range | A missing month must be visible as `NaN`, never as an absent row that silently shortens the series. |
| Imputation | Forward-fill, 5 months | Mirrors the rule the thesis states. Isolated behind `impute=` so a reader can turn it off and see the gap. |
| Denominators | `share_at_risk` left `NaN` for 0/0 months | Dividing by zero would render a reporting outage as "100% of patients at risk". |
| Split | Chronological 75/25 | Time series. Shuffling would leak the future into the training set. |

## 2. Model interface

Every model is a pure function of `(train, horizon)` returning a
`ForecastResult` with a forecast series indexed to the future months, fitted
values, parameters and a plain-language note. Nothing touches the holdout.

```
naive                 repeat the last observed value
mean                  repeat the training mean
seasonal_naive        repeat the same month one year earlier
linear_trend          OLS on a time index
simple_exp_smoothing  level only
double_exp_smoothing  level + trend
triple_exp_smoothing  level + trend + season (additive / multiplicative / damped)
sarima                Box-Jenkins with explicit seasonal terms
auto_sarima           bounded AIC grid search over (p,d,q)x(P,D,Q,12)
```

## 3. Evaluation design

Two schemes, because one is not enough:

- **Single-origin 75/25 holdout** — what the thesis used, retained for
  comparability. One number per model, high variance.
- **Walk-forward** — refit at each of the 25 holdout origins and predict one
  step ahead. Gives an error distribution instead of a point, and exposes
  instability the single split hides. This is what changes the conclusion.

## 4. Metrics and their failure modes

| Metric | Unit | Fails when | Handled by |
|---|---|---|---|
| MAE | cases | Hides that some months matter more | Reported alongside RMSE |
| RMSE | cases | Dominated by the single worst month | Reported alongside MAE |
| MAPE | % | Actual is zero (undefined) or tiny (explodes) | Raises on zero; paired with sMAPE |
| sMAPE | % | Symmetric, bounded, no division by zero | Always computed |
| MASE | ratio | Needs a full season of training data | Uses seasonal-naive scaling |

MASE below 1 means the model beats a seasonal-naive forecast on the same
series. It is the only metric here that answers "did this model earn its
complexity", which is why the README leads with it.

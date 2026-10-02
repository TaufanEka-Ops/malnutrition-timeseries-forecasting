# Data

## `raw/malnutrition_monthly.csv`

Monthly count of inpatients screened **at risk of malnutrition** (moderate +
severe) at a private hospital in East Java, Indonesia.

| Column | Type | Description |
|---|---|---|
| `bulan` | date (`M/D/YYYY`) | Month of the nutrition-screening report. Day component is a reporting artefact. |
| `beresiko` | int | Number of inpatients flagged at risk by the ward nutrition screening (SGA / MST). |

- **Coverage:** November 2013 – December 2021 (98 monthly periods).
- **Gaps:** August–December 2020 are absent from the source workbook. They are
  forward-filled at load time (value 26), which reproduces the imputation rule
  documented in the thesis (*"missing value equated with the previous value"*).
  This is flagged in code via `data.load_series(impute=...)`.
- **Data dictionary:** the monthly series is a derived aggregate. It contains no
  patient identifiers — no names, no medical-record numbers, no ward-level rows.

## `raw/malnutrition_seasonality.xlsx`

Same monthly aggregation with a *denominator*: `beresiko` (at risk) and
`tidak beresiko` (not at risk) per month, used for the seasonality and
share-of-caseload analysis.

**All five absent months appear here as `0 / 0`** — zero patients screened at
risk *and* zero not at risk. That is the signature of a reporting outage, not
of a missing row: the nutrition unit recorded nothing for August–December
2020. `share_at_risk` is therefore left as `NaN` for those months rather than
dividing by zero and rendering them as 100% at-risk, which is what a naive
implementation would show. `tests/test_data.py` pins this behaviour.

## Provenance and ethics

The source material is the nutrition-unit reporting workbook of a single
hospital. The thesis anonymises it as *"hospital X, East Java"*.

This repository publishes **only the derived monthly aggregate** — the same
figures that already appear in the publicly available thesis. It deliberately
does **not** publish:

- the underlying inpatient workbooks (they carry `Nama Pasien` /
  `Identitas Pasien` columns, medical-record numbers, and identifiable
  patient names), or
- ward-level or per-dietitian breakdowns.

Those raw files are patient health information and are retained privately.
See `docs/data-ethics.md`.

## Reproducing

```bash
python -m src.data --check     # validates schema, prints coverage + gaps
```

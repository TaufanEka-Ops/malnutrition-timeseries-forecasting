# Results

Everything here is generated. Do not edit by hand.

| File | Produced by | Contents |
|---|---|---|
| `thesis_reproduction.csv` | `scripts/run_thesis_reproduction.py` | Reproduction on the thesis' data window, against Table 4.2 |
| `metrics_holdout.csv` | `scripts/run_holdout.py` | 75/25 holdout, 9 models, 6 metrics |
| `metrics_walkforward.csv` | `scripts/run_walk_forward.py` | Rolling-origin backtest across 25 origins |
| `walkforward_predictions.csv` | `scripts/run_walk_forward.py` | The per-origin predictions behind the metrics |
| `metrics_2021_validation.csv` | `scripts/run_2021_validation.py` | Published vs reproduced 2021 forecast |
| `2021_validation.csv` | `scripts/run_2021_validation.py` | Month-by-month actual / forecast / error / CI |
| `figures/*.png` | `src/plots.py` | Seven figures, all cited in the README |
| `transcripts/*.log` | `make all` | Raw stdout of each run, unedited |

```bash
make all     # regenerates everything above
```

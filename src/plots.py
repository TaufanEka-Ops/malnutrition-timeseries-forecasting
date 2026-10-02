"""Figures. Every function returns the path it wrote."""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIGURES = pathlib.Path(__file__).resolve().parent.parent / "results" / "figures"
sns.set_theme(style="whitegrid", context="talk")
PALETTE = sns.color_palette("deep")


def _save(fig, name: str) -> pathlib.Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  figure ->", path.relative_to(FIGURES.parent.parent))
    return path


def series_overview(series: pd.Series, missing: list[str] | None = None) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(series.index, series.values, marker="o", ms=3, lw=1.4,
            color=PALETTE[0], label="At-risk admissions")
    ax.plot(series.index, series.rolling(12, min_periods=1).mean(),
            lw=2.2, color=PALETTE[3], label="12-month rolling mean")
    if missing:
        for label in missing:
            ax.axvspan(pd.Timestamp(label + "-01"), pd.Timestamp(label + "-01") + pd.offsets.MonthEnd(1),
                       color="crimson", alpha=0.15)
        ax.axvspan(np.nan, np.nan, color="crimson", alpha=0.15, label="Imputed months")
    ax.set_title("Monthly at-risk malnutrition admissions, 2013-2021")
    ax.set_xlabel("Month"); ax.set_ylabel("Patients flagged at risk")
    ax.legend(loc="upper left", fontsize=10)
    return _save(fig, "01_series_overview.png")


def decomposition(series: pd.Series, period: int = 12) -> pathlib.Path:
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="period")
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    for ax, (part, label) in zip(axes, [(series, "Observed"), (result.trend, "Trend"),
                                        (result.seasonal, "Seasonal"),
                                        (result.resid, "Residual")]):
        ax.plot(part.index, part.values, color=PALETTE[0], lw=1.3)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Month")
    fig.suptitle("Additive decomposition - the step that decides SES vs DES vs TES", y=1.01)
    return _save(fig, "02_decomposition.png")


def seasonal_profile(frame: pd.DataFrame) -> pathlib.Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    by_month = frame.groupby(frame.index.month)["at_risk_cases"].mean()
    axes[0].bar(month_names, by_month.reindex(range(1, 13)).values, color=PALETTE[1])
    axes[0].set_title("Average caseload by calendar month")
    axes[0].set_ylabel("Mean at-risk admissions")
    if "share_at_risk" in frame:
        share = frame.groupby(frame.index.month)["share_at_risk"].mean() * 100
        axes[1].bar(month_names, share.reindex(range(1, 13)).values, color=PALETTE[2])
        axes[1].set_title("Mean share of screened patients at risk")
        axes[1].set_ylabel("Percent of screened patients")
    return _save(fig, "03_seasonal_profile.png")


def forecast_comparison(train, test, forecasts: pd.DataFrame) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(train.index[-36:], train.values[-36:], color="black", lw=1.6, label="Train (tail)")
    ax.plot(test.index, test.values, color="black", lw=2.6, label="Actual")
    for i, column in enumerate(forecasts.columns):
        ax.plot(forecasts.index, forecasts[column], ls="--", lw=1.5,
                color=PALETTE[i % len(PALETTE)], label=column)
    ax.set_title("Holdout forecasts against actual admissions")
    ax.set_xlabel("Month"); ax.set_ylabel("At-risk admissions")
    ax.legend(fontsize=9, ncol=2)
    return _save(fig, "04_holdout_forecasts.png")


def error_barplot(table: pd.DataFrame, metric: str = "mae") -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(11, 6))
    ordered = table.sort_values(metric)
    colors = ["crimson" if m.startswith(("Naive", "Seasonal", "Mean")) else PALETTE[0]
              for m in ordered["model"]]
    ax.barh(ordered["model"], ordered[metric], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric.upper()}")
    ax.set_title(f"{metric.upper()} by model (red = naive benchmarks)")
    return _save(fig, f"05_{metric}_by_model.png")


def walk_forward_errors(predictions: pd.DataFrame) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, column in enumerate(predictions.columns):
        ax.plot(predictions.index, predictions[column], lw=1.4, alpha=0.9,
                color=PALETTE[i % len(PALETTE)], label=column)
    ax.set_title("Walk-forward one-step predictions across every holdout origin")
    ax.set_xlabel("Month"); ax.set_ylabel("Predicted at-risk admissions")
    ax.legend(fontsize=9, ncol=2)
    return _save(fig, "06_walk_forward.png")


def validation_2021(actual: pd.Series, published: pd.Series, ci_high: pd.Series,
                    ci_low: pd.Series, models: dict) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.fill_between(published.index, ci_low.values, ci_high.values,
                    color=PALETTE[2], alpha=0.25, label="Published 95% CI")
    ax.plot(actual.index, actual.values, "o-", color="black", lw=2.4, label="Actual 2021")
    ax.plot(published.index, published.values, "s--", color=PALETTE[3], lw=1.8,
            label="Published forecast (thesis)")
    for i, (label, series) in enumerate(models.items()):
        ax.plot(series.index, series.values, "^:", color=PALETTE[i], lw=1.8, label=label)
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_title("2021 out-of-sample validation - published forecast vs reality")
    ax.set_xlabel("Month"); ax.set_ylabel("At-risk admissions")
    ax.legend(fontsize=10)
    return _save(fig, "07_2021_validation.png")

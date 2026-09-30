from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_eda(loans: pd.DataFrame, reports_dir: Path, max_plot_rows: int = 120_000) -> dict:
    """Create a compact, reproducible EDA pack from the cleaned loan-level data.

    Grouped statistics use the full cleaned dataset; plots use a deterministic sample
    to keep rendering responsive on a student laptop.
    """
    reports_dir = Path(reports_dir)
    plots = reports_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    d = loans.copy()

    summary = {
        "rows": int(len(d)),
        "columns": int(d.shape[1]),
        "default_rate": float(d["target_default"].mean()),
        "issue_year_min": int(d["issue_year"].min()),
        "issue_year_max": int(d["issue_year"].max()),
        "years": sorted(d["issue_year"].dropna().astype(int).unique().tolist()),
    }

    # 1) Target distribution
    target_counts = d["target_default"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Good", "Default"], [int(target_counts.get(0, 0)), int(target_counts.get(1, 0))])
    ax.set_title("Target distribution")
    ax.set_ylabel("Loans")
    for x, y in enumerate([int(target_counts.get(0, 0)), int(target_counts.get(1, 0))]):
        ax.text(x, y, f"{y:,}", ha="center", va="bottom")
    _save(fig, plots / "eda_target_distribution.png")

    # 2) Default rate by year
    yearly = d.groupby("issue_year", observed=True)["target_default"].agg(["mean", "count"]).reset_index()
    yearly.to_csv(reports_dir / "eda_default_rate_by_year.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(yearly["issue_year"], yearly["mean"], marker="o")
    ax.set_title("Default rate by issue year")
    ax.set_xlabel("Issue year")
    ax.set_ylabel("Default rate")
    ax.set_ylim(0, max(0.35, float(yearly["mean"].max()) * 1.15))
    _save(fig, plots / "eda_default_rate_by_year.png")

    # Plot sample only for distributions
    if len(d) > max_plot_rows:
        plot_df = d.sample(n=max_plot_rows, random_state=SEED)
    else:
        plot_df = d

    # 3) Default rate by loan amount bins
    amount_bins = np.quantile(plot_df["loan_amnt"].dropna(), np.linspace(0, 1, 7))
    amount_bins = np.unique(amount_bins)
    if len(amount_bins) >= 3:
        plot_df = plot_df.copy()
        plot_df["loan_amount_bin"] = pd.cut(plot_df["loan_amnt"], bins=amount_bins, include_lowest=True, duplicates="drop")
        by_amount = plot_df.groupby("loan_amount_bin", observed=False)["target_default"].agg(["mean", "count"]).reset_index()
        by_amount["bin_label"] = by_amount["loan_amount_bin"].astype(str)
        by_amount.to_csv(reports_dir / "eda_default_rate_by_loan_amount.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(range(len(by_amount)), by_amount["mean"])
        ax.set_xticks(range(len(by_amount)), by_amount["bin_label"], rotation=30, ha="right")
        ax.set_title("Default rate by loan amount")
        ax.set_ylabel("Default rate")
        _save(fig, plots / "eda_default_rate_by_loan_amount.png")

    # 4) Default rate by DTI bins
    dti = plot_df.dropna(subset=["dti"]).copy()
    if not dti.empty:
        dti_bins = [-np.inf, 10, 20, 30, 40, 50, np.inf]
        dti["dti_bin"] = pd.cut(dti["dti"], bins=dti_bins, right=False)
        by_dti = dti.groupby("dti_bin", observed=False)["target_default"].agg(["mean", "count"]).reset_index()
        by_dti["bin_label"] = by_dti["dti_bin"].astype(str)
        by_dti.to_csv(reports_dir / "eda_default_rate_by_dti.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(by_dti)), by_dti["mean"])
        ax.set_xticks(range(len(by_dti)), by_dti["bin_label"], rotation=25, ha="right")
        ax.set_title("Default rate by DTI")
        ax.set_ylabel("Default rate")
        _save(fig, plots / "eda_default_rate_by_dti.png")

    # 5) Default rate by purpose (top 12 by volume)
    purpose = d.groupby("purpose", dropna=False)["target_default"].agg(["mean", "count"]).sort_values("count", ascending=False).head(12).reset_index()
    purpose["purpose"] = purpose["purpose"].astype("string")
    purpose.to_csv(reports_dir / "eda_default_rate_by_purpose.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(purpose)), purpose["mean"])
    ax.set_xticks(range(len(purpose)), purpose["purpose"].astype(str), rotation=35, ha="right")
    ax.set_title("Default rate by loan purpose (top 12 by volume)")
    ax.set_ylabel("Default rate")
    _save(fig, plots / "eda_default_rate_by_purpose.png")

    # 6) Default rate by term
    term = d.groupby("term_months", dropna=False)["target_default"].agg(["mean", "count"]).reset_index().sort_values("term_months")
    term.to_csv(reports_dir / "eda_default_rate_by_term.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    valid = term.dropna(subset=["term_months"])
    ax.bar(valid["term_months"].astype(str), valid["mean"])
    ax.set_title("Default rate by loan term")
    ax.set_xlabel("Term (months)")
    ax.set_ylabel("Default rate")
    _save(fig, plots / "eda_default_rate_by_term.png")

    insights = [
        f"Cleaned dataset contains {len(d):,} completed loans used for modeling.",
        f"Overall observed default rate is {d['target_default'].mean():.1%}.",
        f"Issue-year coverage spans {int(d['issue_year'].min())}–{int(d['issue_year'].max())}.",
    ]
    if not yearly.empty:
        max_year = yearly.loc[yearly["mean"].idxmax()]
        min_year = yearly.loc[yearly["mean"].idxmin()]
        insights.append(f"Highest yearly default rate: {int(max_year['issue_year'])} ({max_year['mean']:.1%}).")
        insights.append(f"Lowest yearly default rate: {int(min_year['issue_year'])} ({min_year['mean']:.1%}).")
    if not purpose.empty:
        max_p = purpose.loc[purpose["mean"].idxmax()]
        insights.append(f"Among the highest-volume purposes, '{max_p['purpose']}' has the highest observed default rate ({max_p['mean']:.1%}).")

    (reports_dir / "eda_insights.md").write_text("# EDA Insights\n\n" + "\n".join(f"- {x}" for x in insights) + "\n", encoding="utf-8")
    summary["insights"] = insights
    (reports_dir / "eda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

"""
src/data/eda.py
─────────────────────────────────────────────────────────────────────────────
STEP 3 — EXPLORATORY DATA ANALYSIS

Generates all key visualisations and statistical summaries needed before
feature engineering and modelling.

Key questions this EDA answers:
  1. What is the resistance rate per organism-drug pair?
  2. Is there significant class imbalance?
  3. What clinical features correlate with resistance?
  4. Are there temporal trends (seasonal patterns, increasing resistance)?
  5. Which hospitals / wards show highest resistance burden?
  6. Are there missing data patterns that need addressing?
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from loguru import logger

# ── Global plot style ─────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})
OUTPUT_DIR = Path("data/eda_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Overview stats ─────────────────────────────────────────────────────

def overview(df: pd.DataFrame) -> None:
    """Print dataset dimensions, dtypes, and null summary."""
    print("\n── DATASET OVERVIEW ────────────────────────────────")
    print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    print(f"\n  Dtypes:\n{df.dtypes.value_counts()}")
    null_pct = df.isnull().mean().mul(100).round(1)
    null_pct = null_pct[null_pct > 0].sort_values(ascending=False)
    if len(null_pct):
        print(f"\n  Missing values (%):\n{null_pct}")
    else:
        print("\n  No missing values ✓")
    print("─" * 50)


# ── 2. Resistance rates ───────────────────────────────────────────────────

def plot_resistance_rates(df: pd.DataFrame,
                           top_n: int = 15,
                           save: bool = True) -> pd.DataFrame:
    """
    Bar chart of resistance rates per organism-drug combination.
    Returns a summary DataFrame sorted by resistance rate.
    """
    summary = (
        df.groupby(["organism", "antibiotic"])["resistant"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "resistance_rate", "count": "n_isolates"})
        .reset_index()
        .sort_values("resistance_rate", ascending=False)
        .head(top_n)
    )
    summary["combo"] = (summary["organism"].str.split().str[-1]
                        + " / " + summary["antibiotic"])

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(summary["combo"], summary["resistance_rate"],
                   color=sns.color_palette("Reds_r", len(summary)))
    ax.set_xlabel("Resistance Rate")
    ax.set_title(f"Top {top_n} Organism-Drug Resistance Rates")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.invert_yaxis()

    # Annotate with sample size
    for bar, n in zip(bars, summary["n_isolates"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"n={n:,}", va="center", fontsize=8, color="#555")

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "resistance_rates.png")
        logger.info("Saved resistance_rates.png")
    plt.show()
    return summary


# ── 3. Class imbalance ────────────────────────────────────────────────────

def plot_class_balance(df: pd.DataFrame, save: bool = True) -> None:
    """Pie + bar chart of overall and per-organism class balance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overall
    counts = df["resistant"].value_counts()
    axes[0].pie(counts, labels=["Susceptible/I", "Resistant"],
                autopct="%1.1f%%", colors=["#4CAF93", "#E45C3A"],
                startangle=140)
    axes[0].set_title("Overall Class Balance")

    # Per organism
    org_balance = (
        df.groupby("organism")["resistant"].mean()
        .sort_values(ascending=True)
    )
    org_balance.plot(kind="barh", ax=axes[1],
                     color=sns.color_palette("RdYlGn_r", len(org_balance)))
    axes[1].set_xlabel("Resistance Rate")
    axes[1].set_title("Resistance Rate by Organism")
    axes[1].xaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "class_balance.png")
    plt.show()


# ── 4. Temporal trends ────────────────────────────────────────────────────

def plot_temporal_trends(df: pd.DataFrame, save: bool = True) -> None:
    """
    Line chart of resistance rate over time.
    Reveals whether AMR is increasing (the critical surveillance question).
    """
    if "date" not in df.columns:
        logger.warning("No date column — skipping temporal analysis")
        return

    monthly = (
        df.set_index("date")
        .groupby([pd.Grouper(freq="ME"), "organism"])["resistant"]
        .mean()
        .reset_index()
        .rename(columns={"date": "month", "resistant": "resistance_rate"})
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    for org, grp in monthly.groupby("organism"):
        ax.plot(grp["month"], grp["resistance_rate"],
                label=org.split()[-1], linewidth=1.5, alpha=0.8)

    ax.set_title("Monthly Resistance Rate by Organism")
    ax.set_ylabel("Resistance Rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.axhline(df["resistant"].mean(), ls="--", color="gray",
               alpha=0.6, label="Overall mean")

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "temporal_trends.png")
    plt.show()


# ── 5. Risk factor analysis ───────────────────────────────────────────────

def plot_risk_factors(df: pd.DataFrame, save: bool = True) -> None:
    """
    Compare resistance rates across clinical risk factors.
    This is a key EDA step — it validates clinical hypotheses before
    formalising features.
    """
    risk_factors = [
        ("hiv_status",            "HIV Positive"),
        ("prior_antibiotic_90d",  "Prior Antibiotic (90d)"),
        ("prior_hospitalisation_1y", "Prior Hospitalisation"),
        ("diabetes",              "Diabetes"),
        ("catheter_present",      "Catheter"),
        ("sex",                   "Sex"),
        ("age_group",             "Age Group"),
        ("ward",                  "Ward Type"),
        ("specimen_type",         "Specimen Type"),
        ("hospital",              "Hospital"),
    ]

    available = [(col, label) for col, label in risk_factors
                 if col in df.columns]

    fig, axes = plt.subplots(
        nrows=(len(available) + 1) // 2, ncols=2,
        figsize=(14, 3.5 * ((len(available) + 1) // 2))
    )
    axes = axes.flatten()

    for idx, (col, label) in enumerate(available):
        rates = (
            df.groupby(col)["resistant"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "rate", "count": "n"})
            .sort_values("rate", ascending=False)
        )
        bars = axes[idx].bar(rates.index.astype(str), rates["rate"],
                             color=sns.color_palette("muted", len(rates)))
        axes[idx].set_title(label)
        axes[idx].set_ylabel("Resistance Rate")
        axes[idx].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        axes[idx].tick_params(axis="x", rotation=30)

        for bar, (_, row) in zip(bars, rates.iterrows()):
            axes[idx].text(bar.get_x() + bar.get_width() / 2,
                           bar.get_height() + 0.01,
                           f"n={row['n']:,}", ha="center", fontsize=7)

    # Hide unused axes
    for idx in range(len(available), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Resistance Rate by Clinical Risk Factor", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "risk_factors.png", bbox_inches="tight")
    plt.show()


# ── 6. Correlation heatmap ────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True) -> None:
    """
    Correlation matrix for numeric and boolean features vs target.
    Highlights multicollinearity and feature relevance.
    """
    bool_cols = [c for c in df.select_dtypes("bool").columns]
    num_cols  = list(df.select_dtypes("number").columns)
    feature_cols = list(set(num_cols + bool_cols) - {"resistant", "week"})

    if not feature_cols:
        logger.warning("No numeric/boolean features for correlation heatmap")
        return

    corr_df = df[feature_cols + ["resistant"]].copy()
    for col in bool_cols:
        corr_df[col] = corr_df[col].astype(int)

    corr = corr_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix")

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "correlation_heatmap.png")
    plt.show()


# ── 7. Heatmap: resistance by organism × antibiotic ──────────────────────

def plot_resistance_heatmap(df: pd.DataFrame, save: bool = True) -> None:
    """
    Pivot table heatmap showing resistance rate for every organism-drug pair.
    The core AMR surveillance visualisation.
    """
    pivot = (
        df.groupby(["organism", "antibiotic"])["resistant"]
        .mean()
        .unstack(fill_value=np.nan)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".0%", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Resistance Rate"})
    ax.set_title("AMR Resistance Rates: Organism × Antibiotic")
    ax.set_xlabel("Antibiotic")
    ax.set_ylabel("Organism")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "resistance_heatmap.png")
    plt.show()


# ── Full EDA run ──────────────────────────────────────────────────────────

def run_full_eda(df: pd.DataFrame) -> None:
    """Run the complete EDA suite."""
    logger.info("Starting full EDA ...")
    overview(df)
    plot_resistance_rates(df)
    plot_class_balance(df)
    plot_temporal_trends(df)
    plot_risk_factors(df)
    plot_correlation_heatmap(df)
    plot_resistance_heatmap(df)
    logger.success(f"EDA complete — plots saved to {OUTPUT_DIR}/")


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.load_data import generate_synthetic_whonet
    from src.data.clean_data import clean

    df = clean(generate_synthetic_whonet(5000))
    run_full_eda(df)

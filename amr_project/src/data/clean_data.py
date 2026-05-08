"""
src/data/clean_data.py
─────────────────────────────────────────────────────────────────────────────
STEP 2 — DATA CLEANING

Handles all data quality issues common in WHONET / hospital lab exports:
  • Duplicate isolates (same patient, same organism, same week)
  • Invalid / inconsistent SIR results
  • Outlier ages and missing demographics
  • Rare organism-drug combinations (too few records to train on)
  • Date parsing and temporal feature extraction
  • Binary target variable creation (Resistant vs Not-Resistant)

Clinical note on I (Intermediate):
  EUCAST 2019 replaced "Intermediate" with "Increased Exposure" (I→S+).
  For binary classification we treat I as Non-Resistant (0) — conservative
  approach that avoids over-predicting resistance. This can be revisited.
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────────────────

MIN_ISOLATES_PER_COMBO = 30   # Drop organism-drug combos below this

VALID_SIR = {"S", "I", "R"}

VALID_ORGANISMS = [
    "Escherichia coli", "Klebsiella pneumoniae", "Staphylococcus aureus",
    "Salmonella typhi", "Streptococcus pneumoniae", "Pseudomonas aeruginosa",
    "Acinetobacter baumannii", "Enterococcus faecalis", "Proteus mirabilis",
]

VALID_SPECIMENS = [
    "Blood", "Urine", "Sputum", "Wound swab", "CSF", "Stool",
    "Pus", "Endotracheal aspirate",
]


# ── Main cleaning pipeline ────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline. Returns cleaned DataFrame with target column added.

    Pipeline steps:
        1. Column standardisation
        2. Date parsing & temporal features
        3. SIR result validation
        4. Duplicate removal
        5. Age outlier handling
        6. Rare combo filtering
        7. Target variable creation
        8. Final report
    """
    logger.info(f"Starting cleaning — input shape: {df.shape}")
    original_len = len(df)

    df = _standardise_columns(df)
    df = _parse_dates(df)
    df = _validate_sir(df)
    df = _remove_duplicates(df)
    df = _clean_age(df)
    df = _clean_demographics(df)
    df = _filter_rare_combos(df)
    df = _create_target(df)

    logger.success(
        f"Cleaning complete — {len(df):,} records remain "
        f"({original_len - len(df):,} removed, "
        f"{(original_len - len(df)) / original_len:.1%} drop rate)"
    )
    _cleaning_report(df)
    return df


# ── Step functions ────────────────────────────────────────────────────────

def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names, strip whitespace from string columns."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Strip string columns
    str_cols = df.select_dtypes("object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Standardise SIR to uppercase
    if "sir_result" in df.columns:
        df["sir_result"] = df["sir_result"].str.upper()

    # Standardise organism capitalisation
    if "organism" in df.columns:
        df["organism"] = df["organism"].str.title()

    # Standardise antibiotic to lowercase
    if "antibiotic" in df.columns:
        df["antibiotic"] = df["antibiotic"].str.lower()

    logger.debug("Columns standardised")
    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date column and extract temporal features."""
    if "date" not in df.columns:
        logger.warning("No 'date' column found — skipping temporal features")
        return df

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop rows where date could not be parsed
    bad_dates = df["date"].isna().sum()
    if bad_dates > 0:
        logger.warning(f"Dropped {bad_dates:,} rows with unparseable dates")
    df = df[df["date"].notna()]

    # Extract temporal features useful for seasonality modelling
    df["year"]    = df["date"].dt.year
    df["month"]   = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["week"]    = df["date"].dt.isocalendar().week.astype(int)

    logger.debug(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df


def _validate_sir(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with invalid SIR values."""
    df = df.copy()
    before = len(df)
    df = df[df["sir_result"].isin(VALID_SIR)]
    removed = before - len(df)
    if removed:
        logger.info(f"Removed {removed:,} rows with invalid SIR values")
    return df


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate isolates.

    Clinical logic: if the SAME patient has the SAME organism tested against
    the SAME drug within a 7-day window, keep only the first record.
    This prevents data leakage from repeat testing of the same infection.
    """
    df = df.copy()
    before = len(df)

    # Exact duplicates first
    df = df.drop_duplicates()

    # Near-duplicate: same isolate_id + antibiotic
    if "isolate_id" in df.columns:
        df = df.sort_values("date").drop_duplicates(
            subset=["isolate_id", "antibiotic"], keep="first"
        )

    removed = before - len(df)
    logger.info(f"Removed {removed:,} duplicate records")
    return df


def _clean_age(df: pd.DataFrame) -> pd.DataFrame:
    """Handle age outliers and create age group feature."""
    if "age_years" not in df.columns:
        return df

    df = df.copy()

    # Clip extreme ages (data entry errors)
    df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")
    outliers = ((df["age_years"] < 0) | (df["age_years"] > 110)).sum()
    if outliers:
        logger.warning(f"Clipping {outliers:,} age outliers to [0, 110]")
    df["age_years"] = df["age_years"].clip(0, 110)

    # Median impute missing ages
    median_age = df["age_years"].median()
    missing = df["age_years"].isna().sum()
    if missing:
        logger.info(f"Imputing {missing:,} missing ages with median ({median_age:.0f})")
    df["age_years"] = df["age_years"].fillna(median_age)

    # Age group (clinically meaningful bins)
    df["age_group"] = pd.cut(
        df["age_years"],
        bins=[0, 5, 18, 40, 60, 110],
        labels=["infant", "paediatric", "young_adult", "adult", "elderly"],
        right=True
    )

    return df


def _clean_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise and impute demographic fields."""
    df = df.copy()

    # Sex: standardise to M/F, fill unknown as 'Unknown'
    if "sex" in df.columns:
        df["sex"] = df["sex"].str.upper().str[0]   # Take first char
        df["sex"] = df["sex"].where(df["sex"].isin(["M", "F"]), "Unknown")

    # Ward: fill missing
    if "ward" in df.columns:
        df["ward"] = df["ward"].fillna("Unknown")

    # Specimen: fill missing
    if "specimen_type" in df.columns:
        df["specimen_type"] = df["specimen_type"].fillna("Unknown")

    # Boolean clinical flags: fill missing as False (absent until stated)
    bool_cols = ["prior_antibiotic_90d", "prior_hospitalisation_1y",
                 "hiv_status", "diabetes", "catheter_present"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    return df


def _filter_rare_combos(df: pd.DataFrame,
                         min_count: int = MIN_ISOLATES_PER_COMBO) -> pd.DataFrame:
    """
    Remove organism-antibiotic combinations with fewer than min_count records.
    These combos cannot support reliable model training.
    """
    df = df.copy()
    before = len(df)

    counts = df.groupby(["organism", "antibiotic"]).size()
    rare   = counts[counts < min_count].index
    mask   = df.set_index(["organism", "antibiotic"]).index.isin(rare)
    df     = df[~mask.values]

    removed = before - len(df)
    logger.info(
        f"Removed {removed:,} records from {len(rare)} rare organism-drug combos "
        f"(< {min_count} isolates)"
    )

    remaining_combos = df.groupby(["organism", "antibiotic"]).size()
    logger.info(f"Retained {len(remaining_combos)} organism-drug combinations")
    return df


def _create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target variable.
      R  → 1 (Resistant)
      S  → 0 (Susceptible)
      I  → 0 (Intermediate treated as Non-Resistant — conservative)
    """
    df = df.copy()
    df["resistant"] = (df["sir_result"] == "R").astype(int)

    resistance_rate = df["resistant"].mean()
    logger.info(f"Overall resistance rate: {resistance_rate:.1%} "
                f"({df['resistant'].sum():,} resistant / {len(df):,} total)")
    return df


# ── Cleaning report ───────────────────────────────────────────────────────

def _cleaning_report(df: pd.DataFrame) -> None:
    """Print a summary of the cleaned dataset."""
    print("\n" + "=" * 60)
    print("  CLEANING REPORT")
    print("=" * 60)
    print(f"  Total records:      {len(df):>10,}")
    print(f"  Unique isolates:    {df['isolate_id'].nunique():>10,}"
          if "isolate_id" in df.columns else "  Unique isolates:  N/A")
    print(f"  Unique organisms:   {df['organism'].nunique():>10,}")
    print(f"  Unique antibiotics: {df['antibiotic'].nunique():>10,}")
    print(f"  Date range:         {df['date'].min().date()} → "
          f"{df['date'].max().date()}" if "date" in df.columns else "")
    print(f"  Resistance rate:    {df['resistant'].mean():>10.1%}")
    print(f"  Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print("=" * 60 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from load_data import generate_synthetic_whonet

    raw = generate_synthetic_whonet(n_isolates=5000)
    cleaned = clean(raw)

    out = Path("data/processed/amr_clean.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out, index=False)
    logger.info(f"Saved cleaned data to {out}")

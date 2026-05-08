"""
src/features/feature_engineering.py
─────────────────────────────────────────────────────────────────────────────
STEP 4 — FEATURE ENGINEERING

Transforms the cleaned clinical dataset into a feature matrix ready for ML.

Feature groups created:
  1. Encoded categoricals (organism, antibiotic, ward, specimen, sex)
  2. Age binning + numeric age
  3. Boolean clinical risk flags (HIV, prior AB, hospitalisation, etc.)
  4. Temporal features (month, quarter — for seasonality)
  5. Hospital-level resistance rate (historical AMR context per facility)
  6. Prior resistance signal: has this organism shown resistance at this
     hospital in the past 90 days? (Key epidemiological feature)
  7. Drug class encoding (broader therapeutic category)

NOTE on encoding strategy:
  We use OrdinalEncoder stored inside a sklearn Pipeline so that the
  exact same encoding is applied consistently at train, validation, and
  inference time. Never fit encoders on test data.
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from loguru import logger
from pathlib import Path
import joblib


# ── Drug class mapping (clinical grouping) ───────────────────────────────

DRUG_CLASS = {
    "ampicillin":      "beta_lactam",
    "amoxicillin":     "beta_lactam",
    "ceftriaxone":     "cephalosporin",
    "cefotaxime":      "cephalosporin",
    "meropenem":       "carbapenem",
    "imipenem":        "carbapenem",
    "ciprofloxacin":   "fluoroquinolone",
    "levofloxacin":    "fluoroquinolone",
    "gentamicin":      "aminoglycoside",
    "amikacin":        "aminoglycoside",
    "cotrimoxazole":   "sulfonamide",
    "vancomycin":      "glycopeptide",
    "clindamycin":     "lincosamide",
    "oxacillin":       "beta_lactam",
    "azithromycin":    "macrolide",
    "isoniazid":       "antimycobacterial",
    "rifampicin":      "antimycobacterial",
    "ethambutol":      "antimycobacterial",
}

# ── Specimen risk tier (proxy for infection severity) ────────────────────
SPECIMEN_RISK = {
    "Blood":                 3,   # Highest — bacteraemia
    "CSF":                   3,   # Meningitis
    "Endotracheal aspirate": 2,
    "Sputum":                2,
    "Wound swab":            2,
    "Pus":                   2,
    "Urine":                 1,
    "Stool":                 1,
}


# ── Main feature engineering function ────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all engineered features to the cleaned DataFrame.
    Returns an enriched DataFrame (does NOT yet encode — encoding happens
    inside the sklearn Pipeline at training time).
    """
    logger.info("Engineering features ...")
    df = df.copy()

    df = _add_drug_class(df)
    df = _add_specimen_risk(df)
    df = _add_temporal_cyclical(df)
    df = _add_hospital_resistance_rate(df)
    df = _add_organism_ward_resistance(df)
    df = _add_multi_drug_resistance_flag(df)

    logger.success(f"Feature engineering complete — {df.shape[1]} columns")
    return df


def _add_drug_class(df: pd.DataFrame) -> pd.DataFrame:
    df["drug_class"] = df["antibiotic"].map(DRUG_CLASS).fillna("other")
    return df


def _add_specimen_risk(df: pd.DataFrame) -> pd.DataFrame:
    df["specimen_risk"] = df["specimen_type"].map(SPECIMEN_RISK).fillna(1)
    return df


def _add_temporal_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode month as cyclical (sin/cos) so December and January are adjacent.
    Raw month as integer treats month 12 as far from month 1.
    """
    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def _add_hospital_resistance_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row: what is the historical resistance rate for this organism
    at this hospital, based on all OTHER records?
    (Leave-one-out target encoding to avoid leakage.)
    """
    if "hospital" not in df.columns:
        return df

    # Global rate as fallback
    global_rate = df["resistant"].mean()

    # Group-level rate
    grp = df.groupby(["hospital", "organism"])["resistant"].transform(
        lambda x: (x.sum() - x) / (x.count() - 1 + 1e-6)
    )
    df["hospital_org_resistance_rate"] = grp.fillna(global_rate)
    return df


def _add_organism_ward_resistance(df: pd.DataFrame) -> pd.DataFrame:
    """Historical resistance rate for this organism-drug combo in this ward."""
    if "ward" not in df.columns:
        return df

    global_rate = df["resistant"].mean()
    grp = df.groupby(["ward", "organism", "antibiotic"])["resistant"].transform(
        lambda x: (x.sum() - x) / (x.count() - 1 + 1e-6)
    )
    df["ward_org_drug_resistance_rate"] = grp.fillna(global_rate)
    return df


def _add_multi_drug_resistance_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag isolates from organisms that are resistant to 3+ drug classes
    in this dataset — a proxy for MDR (multi-drug resistant) status.
    """
    df["drug_class_resistant"] = df["drug_class"].where(
        df["resistant"] == 1, other=np.nan
    )
    mdr_classes = (
        df.groupby("isolate_id")["drug_class_resistant"]
        .nunique()
        .rename("n_resistant_classes")
    ) if "isolate_id" in df.columns else pd.Series(dtype=int)

    if len(mdr_classes):
        df = df.merge(mdr_classes, on="isolate_id", how="left")
        df["is_mdr"] = (df["n_resistant_classes"] >= 3).astype(int)
        df["n_resistant_classes"] = df["n_resistant_classes"].fillna(0)
    else:
        df["is_mdr"] = 0
        df["n_resistant_classes"] = 0

    df = df.drop(columns=["drug_class_resistant"], errors="ignore")
    return df


# ── sklearn ColumnTransformer (used inside Pipeline) ─────────────────────

CATEGORICAL_FEATURES = [
    "organism", "antibiotic", "drug_class", "specimen_type",
    "ward", "sex", "age_group", "hospital",
    "prior_antibiotic_class",
]

NUMERIC_FEATURES = [
    "age_years", "specimen_risk",
    "month_sin", "month_cos", "quarter",
    "hospital_org_resistance_rate",
    "ward_org_drug_resistance_rate",
    "n_resistant_classes",
]

BINARY_FEATURES = [
    "prior_antibiotic_90d", "prior_hospitalisation_1y",
    "hiv_status", "diabetes", "catheter_present", "is_mdr",
]


def build_preprocessor() -> ColumnTransformer:
    """
    Returns a ColumnTransformer that:
      - Ordinal-encodes categoricals (handles unknown values at inference)
      - Scales numeric features
      - Passes binary features through unchanged
    """
    categorical_pipe = Pipeline([
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2
        ))
    ])

    numeric_pipe = Pipeline([
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat",    categorical_pipe, CATEGORICAL_FEATURES),
            ("num",    numeric_pipe,     NUMERIC_FEATURES),
            ("binary", "passthrough",    BINARY_FEATURES),
        ],
        remainder="drop",        # Drop any unspecified columns
        verbose_feature_names_out=True
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Get final feature names after transformation."""
    return list(preprocessor.get_feature_names_out())


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.load_data import generate_synthetic_whonet
    from src.data.clean_data import clean

    df = clean(generate_synthetic_whonet(5000))
    df_feat = engineer_features(df)

    print("\nEngineered features:")
    new_cols = set(df_feat.columns) - set(df.columns)
    for c in sorted(new_cols):
        print(f"  + {c}")

    print(f"\nFinal shape: {df_feat.shape}")
    print(f"\nSample:\n{df_feat[['organism', 'antibiotic', 'drug_class', 'specimen_risk', 'hospital_org_resistance_rate', 'resistant']].head(10)}")

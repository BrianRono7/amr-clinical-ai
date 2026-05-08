"""
src/data/split_data.py
─────────────────────────────────────────────────────────────────────────────
STEP 5 — DATASET SPLITTING

AMR splitting has specific clinical requirements beyond random splitting:

  PROBLEM with naive random split:
    If isolate ISO_001 appears in both train and test (tested for 3 drugs),
    the model sees the patient in training → data leakage.

  SOLUTION — Isolate-level split:
    Split on unique isolate_id so ALL records from one patient episode
    appear in exactly ONE split (train / val / test).

  ADDITIONAL STRATEGY — Temporal split (recommended for deployment):
    Train on older data, test on newer data. This simulates real deployment
    where the model must generalise to future, unseen patients.
    Use: split_temporal()

  STRATIFICATION:
    Stratify by organism to ensure all splits have representative
    organism distributions (important for rare organisms like TB).
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from loguru import logger
from pathlib import Path


# ── A. Isolate-level stratified split (primary method) ───────────────────

def split_by_isolate(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data at the ISOLATE level to prevent data leakage.

    Args:
        df:           Cleaned + feature-engineered DataFrame
        test_size:    Fraction for test set (held-out, touch only at end)
        val_size:     Fraction for validation set (used for model selection)
        random_state: Reproducibility seed

    Returns:
        (train_df, val_df, test_df)
    """
    if "isolate_id" not in df.columns:
        logger.warning("No isolate_id column — falling back to row-level split")
        return _row_level_split(df, test_size, val_size, random_state)

    # Get unique isolates with their dominant organism (for stratification)
    isolate_meta = (
        df.groupby("isolate_id")["organism"]
        .agg(lambda x: x.mode()[0])   # Most common organism for this isolate
        .reset_index()
        .rename(columns={"organism": "strat_label"})
    )

    # Remove strat labels with too few samples for stratification
    label_counts = isolate_meta["strat_label"].value_counts()
    rare_labels  = label_counts[label_counts < 10].index
    isolate_meta["strat_label"] = isolate_meta["strat_label"].where(
        ~isolate_meta["strat_label"].isin(rare_labels), other="other"
    )

    # ── First split: train+val vs test ────────────────────────────────────
    train_val_ids, test_ids = train_test_split(
        isolate_meta["isolate_id"],
        test_size=test_size,
        stratify=isolate_meta["strat_label"],
        random_state=random_state,
    )

    # ── Second split: train vs val ────────────────────────────────────────
    val_fraction = val_size / (1 - test_size)   # Relative to train+val
    train_val_meta = isolate_meta[isolate_meta["isolate_id"].isin(train_val_ids)]

    train_ids, val_ids = train_test_split(
        train_val_meta["isolate_id"],
        test_size=val_fraction,
        stratify=train_val_meta["strat_label"],
        random_state=random_state,
    )

    train_df = df[df["isolate_id"].isin(train_ids)].copy()
    val_df   = df[df["isolate_id"].isin(val_ids)].copy()
    test_df  = df[df["isolate_id"].isin(test_ids)].copy()

    _split_report(train_df, val_df, test_df)
    return train_df, val_df, test_df


# ── B. Temporal split (for deployment realism) ────────────────────────────

def split_temporal(
    df: pd.DataFrame,
    test_months: int = 6,
    val_months: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Temporal split: train on older data, validate and test on newer data.

    Rationale: a deployed model will always predict on FUTURE patients.
    Random splits are optimistic — temporal splits are more honest.

    Args:
        df:          DataFrame with a 'date' column
        test_months: Most recent N months held out as test set
        val_months:  Next N months (before test) as validation

    Returns:
        (train_df, val_df, test_df)
    """
    if "date" not in df.columns:
        raise ValueError("Temporal split requires a 'date' column")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    max_date = df["date"].max()

    test_cutoff = max_date - pd.DateOffset(months=test_months)
    val_cutoff  = test_cutoff - pd.DateOffset(months=val_months)

    test_df  = df[df["date"] > test_cutoff].copy()
    val_df   = df[(df["date"] > val_cutoff) & (df["date"] <= test_cutoff)].copy()
    train_df = df[df["date"] <= val_cutoff].copy()

    logger.info(f"Temporal split cutoffs:")
    logger.info(f"  Train:  up to {val_cutoff.date()}")
    logger.info(f"  Val:    {val_cutoff.date()} → {test_cutoff.date()}")
    logger.info(f"  Test:   after {test_cutoff.date()}")

    _split_report(train_df, val_df, test_df)
    return train_df, val_df, test_df


# ── C. Row-level fallback ─────────────────────────────────────────────────

def _row_level_split(df, test_size, val_size, random_state):
    """Fallback when no isolate_id — less ideal but functional."""
    train_val, test_df = train_test_split(
        df, test_size=test_size, stratify=df["resistant"],
        random_state=random_state
    )
    val_fraction = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val, test_size=val_fraction, stratify=train_val["resistant"],
        random_state=random_state
    )
    _split_report(train_df, val_df, test_df)
    return train_df, val_df, test_df


# ── Report ────────────────────────────────────────────────────────────────

def _split_report(train, val, test):
    total = len(train) + len(val) + len(test)
    print("\n" + "=" * 55)
    print("  DATASET SPLIT SUMMARY")
    print("=" * 55)
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        res_rate = split["resistant"].mean()
        print(f"  {name:<6}: {len(split):>7,} records "
              f"({len(split)/total:.1%}) | "
              f"Resistance rate: {res_rate:.1%}")
    print("=" * 55 + "\n")

    # Warn if resistance rates diverge significantly across splits
    rates = [s["resistant"].mean() for s in [train, val, test]]
    if max(rates) - min(rates) > 0.05:
        logger.warning(
            "Resistance rate varies >5% across splits — "
            "consider more aggressive stratification"
        )


# ── Prepare X / y arrays ──────────────────────────────────────────────────

def get_X_y(df: pd.DataFrame,
             feature_cols: list[str],
             target: str = "resistant"):
    """
    Extract feature matrix X and target vector y from a split DataFrame.

    Args:
        df:           Split DataFrame (train / val / test)
        feature_cols: List of column names to use as features
        target:       Target column name

    Returns:
        (X: pd.DataFrame, y: pd.Series)
    """
    available = [c for c in feature_cols if c in df.columns]
    missing   = set(feature_cols) - set(available)
    if missing:
        logger.warning(f"Missing feature columns: {missing}")

    X = df[available].copy()
    y = df[target].copy()
    return X, y


# ── Save splits ───────────────────────────────────────────────────────────

def save_splits(train, val, test,
                out_dir: str = "data/processed") -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    train.to_csv(f"{out_dir}/train.csv", index=False)
    val.to_csv(f"{out_dir}/val.csv",   index=False)
    test.to_csv(f"{out_dir}/test.csv", index=False)
    logger.success(f"Splits saved to {out_dir}/")


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.load_data import generate_synthetic_whonet
    from src.data.clean_data import clean
    from src.features.feature_engineering import engineer_features

    df = engineer_features(clean(generate_synthetic_whonet(5000)))
    train, val, test = split_by_isolate(df)
    save_splits(train, val, test)
    print(f"Train: {train.shape} | Val: {val.shape} | Test: {test.shape}")

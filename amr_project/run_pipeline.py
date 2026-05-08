"""
run_pipeline.py
─────────────────────────────────────────────────────────────────────────────
MASTER PIPELINE SCRIPT

Runs the complete AMR prediction pipeline from data loading to model saving.
Execute this to train and evaluate the full system.

Usage:
    # Full pipeline with synthetic data (development):
    python run_pipeline.py

    # Full pipeline with real WHONET data:
    python run_pipeline.py --data data/raw/whonet_export.csv

    # Skip training, evaluate existing saved model:
    python run_pipeline.py --eval-only

    # Train only, skip evaluation:
    python run_pipeline.py --train-only
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import sys
import time
from pathlib import Path
from loguru import logger

# Configure loguru for clean output
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | "
           "<level>{level:<8}</level> | <cyan>{message}</cyan>",
           colorize=True, level="INFO")
logger.add("data/pipeline.log", rotation="10 MB", level="DEBUG")


# ── Step runner helper ────────────────────────────────────────────────────

def step(num: int, name: str):
    """Print a clearly visible step header."""
    print(f"\n{'═' * 65}")
    print(f"  STEP {num}  ▶  {name}")
    print(f"{'═' * 65}")


# ── Main pipeline ─────────────────────────────────────────────────────────

def run(data_path: str = None,
        eval_only: bool = False,
        train_only: bool = False,
        n_synthetic: int = 8000,
        use_temporal_split: bool = False) -> None:

    t_start = time.time()

    # ── STEP 1: Load data ─────────────────────────────────────────────────
    step(1, "DATA LOADING")
    from src.data.load_data import load_whonet, generate_synthetic_whonet

    if data_path and Path(data_path).exists():
        logger.info(f"Loading real WHONET data from: {data_path}")
        df_raw = load_whonet(data_path)
    else:
        logger.info(f"No real data path provided — generating {n_synthetic:,} "
                    "synthetic isolates for development")
        df_raw = generate_synthetic_whonet(n_isolates=n_synthetic, seed=42)

    logger.info(f"Raw data shape: {df_raw.shape}")

    # ── STEP 2: Clean data ────────────────────────────────────────────────
    step(2, "DATA CLEANING")
    from src.data.clean_data import clean
    df_clean = clean(df_raw)

    out_clean = Path("data/processed/amr_clean.csv")
    out_clean.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_clean, index=False)
    logger.info(f"Saved cleaned data → {out_clean}")

    # ── STEP 3: EDA ───────────────────────────────────────────────────────
    step(3, "EXPLORATORY DATA ANALYSIS")
    try:
        from src.data.eda import run_full_eda
        run_full_eda(df_clean)
    except Exception as e:
        logger.warning(f"EDA skipped (display issue in headless env): {e}")

    # ── STEP 4: Feature engineering ───────────────────────────────────────
    step(4, "FEATURE ENGINEERING")
    from src.features.feature_engineering import engineer_features
    df_feat = engineer_features(df_clean)
    logger.info(f"Feature matrix shape: {df_feat.shape}")

    new_features = set(df_feat.columns) - set(df_clean.columns)
    logger.info(f"New features added ({len(new_features)}): "
                f"{sorted(new_features)}")

    # ── STEP 5: Split data ────────────────────────────────────────────────
    step(5, "DATASET SPLITTING")
    from src.data.split_data import (
        split_by_isolate, split_temporal, save_splits
    )

    if use_temporal_split and "date" in df_feat.columns:
        logger.info("Using TEMPORAL split (recommended for deployment)")
        train_df, val_df, test_df = split_temporal(
            df_feat, test_months=6, val_months=3
        )
    else:
        logger.info("Using ISOLATE-LEVEL stratified split")
        train_df, val_df, test_df = split_by_isolate(
            df_feat, test_size=0.15, val_size=0.15
        )

    save_splits(train_df, val_df, test_df)

    if eval_only:
        logger.info("--eval-only flag set — loading existing model")
    else:
        # ── STEP 6: Train models ──────────────────────────────────────────
        step(6, "MODEL TRAINING & PIPELINE")
        from src.models.train import train_all
        results = train_all(train_df, val_df)

        logger.success("All models trained and saved to data/models/")
        for name, bundle in results.items():
            auroc = bundle["metrics"]["val_auroc"]
            rec   = bundle["metrics"]["val_recall"]
            thresh = bundle["threshold"]
            logger.info(f"  {name:<25} AUROC={auroc:.4f}  "
                        f"Recall={rec:.4f}  Threshold={thresh:.2f}")

    if not train_only:
        # ── STEP 7: Evaluate ──────────────────────────────────────────────
        step(7, "MODEL EVALUATION (TEST SET)")
        from src.evaluation.evaluate import evaluate_on_test
        try:
            metrics = evaluate_on_test(test_df)
            logger.success(
                f"Final test AUROC: {metrics['auroc']:.4f}  "
                f"AUPRC: {metrics['auprc']:.4f}"
            )

            # Clinical threshold check
            target_auroc = 0.80
            if metrics["auroc"] >= target_auroc:
                logger.success(
                    f"✓ AUROC {metrics['auroc']:.4f} meets research target "
                    f"(≥ {target_auroc})"
                )
            else:
                logger.warning(
                    f"✗ AUROC {metrics['auroc']:.4f} below target {target_auroc}. "
                    "Consider: more data, feature engineering, or hyperparameter tuning."
                )
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")

    # ── Done ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'═' * 65}")
    print(f"  PIPELINE COMPLETE  ✓  ({elapsed:.1f}s)")
    print(f"{'═' * 65}")
    print(f"""
  Outputs:
    Cleaned data    → data/processed/amr_clean.csv
    Train / Val / Test → data/processed/{{train,val,test}}.csv
    Models          → data/models/
    Best model      → data/models/best_model.joblib
    EDA plots       → data/eda_plots/
    Eval plots      → data/evaluation_plots/
    MLflow runs     → mlruns/

  Next steps:
    Launch Streamlit app:
      streamlit run src/app/streamlit_app.py

    Launch Flask API:
      python src/app/flask_api.py

    View MLflow dashboard:
      mlflow ui --backend-store-uri mlruns/
      (open http://localhost:5000)
    """)


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AMR Prediction Pipeline — East Africa"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to WHONET CSV export. Omit to use synthetic data."
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training, evaluate existing saved model only."
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Train models only, skip final test set evaluation."
    )
    parser.add_argument(
        "--n-synthetic", type=int, default=8000,
        help="Number of synthetic isolates to generate (default: 8000)."
    )
    parser.add_argument(
        "--temporal-split", action="store_true",
        help="Use temporal (date-based) split instead of random isolate split."
    )

    args = parser.parse_args()
    run(
        data_path=args.data,
        eval_only=args.eval_only,
        train_only=args.train_only,
        n_synthetic=args.n_synthetic,
        use_temporal_split=args.temporal_split,
    )

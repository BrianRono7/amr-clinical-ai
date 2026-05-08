"""
src/models/train.py
─────────────────────────────────────────────────────────────────────────────
STEP 6 — PIPELINE CONSTRUCTION & MODEL TRAINING

Builds sklearn Pipelines that chain:
  preprocessor (ColumnTransformer) → SMOTE (class balance) → classifier

Four models are trained and compared:
  1. Logistic Regression      — interpretable baseline
  2. Random Forest            — robust non-linear baseline
  3. XGBoost                  — primary model (strong tabular performer)
  4. LightGBM                 — primary model (fast, memory-efficient)

All experiments are tracked with MLflow.
Best model is saved to disk for deployment.

Clinical requirement:
  Sensitivity (recall for Resistant class) > 0.85
  We prefer to over-predict resistance (safer) than miss it.
  → We tune decision threshold on validation set post-training.
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from loguru import logger

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline   # supports SMOTE in pipeline
import xgboost as xgb
import lightgbm as lgb

import sys
sys.path.insert(0, ".")
from src.features.feature_engineering import (
    build_preprocessor, engineer_features,
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, BINARY_FEATURES,
)
from src.data.split_data import get_X_y


# ── Constants ─────────────────────────────────────────────────────────────

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_URI = "mlruns"
EXPERIMENT  = "AMR_Resistance_Prediction"
RANDOM_STATE = 42
TARGET = "resistant"

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BINARY_FEATURES


# ── Model definitions ─────────────────────────────────────────────────────

def get_classifiers() -> dict:
    """
    Return dict of {name: classifier} to train and compare.
    Hyperparameters are set conservatively — use train_with_cv() for tuning.
    """
    pos_weight = 1.0   # Recomputed per dataset in train_all()

    return {
        "logistic_regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="auc",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=400,
            num_leaves=63,
            learning_rate=0.05,
            min_child_samples=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
    }


# ── Build full pipeline ────────────────────────────────────────────────────

def build_pipeline(classifier, use_smote: bool = True) -> ImbPipeline:
    """
    Assemble: preprocessor → [SMOTE] → classifier

    SMOTE is applied INSIDE the pipeline so it only resamples training folds
    during cross-validation — never leaks into validation folds.

    Args:
        classifier: An unfitted sklearn-compatible classifier
        use_smote:  Whether to include SMOTE resampling step

    Returns:
        Fitted-ready ImbPipeline
    """
    preprocessor = build_preprocessor()

    steps = [("preprocessor", preprocessor)]

    if use_smote:
        steps.append(("smote", SMOTE(
            sampling_strategy="minority",
            k_neighbors=5,
            random_state=RANDOM_STATE,
        )))

    steps.append(("classifier", classifier))

    return ImbPipeline(steps=steps)


# ── Cross-validation ──────────────────────────────────────────────────────

def cross_validate_model(pipeline: ImbPipeline,
                          X: pd.DataFrame,
                          y: pd.Series,
                          n_folds: int = 5) -> dict:
    """
    5-fold stratified cross-validation on training set.
    Returns mean ± std of key metrics.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                         random_state=RANDOM_STATE)

    scores = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=["roc_auc", "average_precision",
                 "f1", "recall", "precision"],
        return_train_score=True,
        n_jobs=-1,
    )

    results = {}
    for metric in ["roc_auc", "average_precision", "f1", "recall", "precision"]:
        test_scores = scores[f"test_{metric}"]
        results[metric] = {
            "mean": float(test_scores.mean()),
            "std":  float(test_scores.std()),
        }
        logger.info(f"  CV {metric:<22}: "
                    f"{test_scores.mean():.4f} ± {test_scores.std():.4f}")

    return results


# ── Train a single model ──────────────────────────────────────────────────

def train_model(name: str,
                classifier,
                X_train: pd.DataFrame,
                y_train: pd.Series,
                X_val: pd.DataFrame,
                y_val: pd.Series,
                use_smote: bool = True) -> tuple:
    """
    Train one pipeline, evaluate on validation set, log to MLflow.

    Returns:
        (fitted_pipeline, val_metrics, val_proba)
    """
    logger.info(f"\n{'='*55}")
    logger.info(f"Training: {name.upper()}")
    logger.info(f"{'='*55}")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=name):
        # ── Build & fit pipeline ──────────────────────────────────────────
        pipeline = build_pipeline(classifier, use_smote=use_smote)

        # XGBoost early stopping needs eval set — handle separately
        if name == "xgboost":
            # Fit preprocessor on train, transform val for eval
            pipeline_no_clf = ImbPipeline(
                steps=pipeline.steps[:-1]   # All steps except classifier
            )
            X_train_proc = pipeline_no_clf.fit_resample(X_train, y_train)[0]
            X_val_proc   = pipeline_no_clf.named_steps["preprocessor"].transform(X_val)

            classifier.fit(
                X_train_proc, pipeline_no_clf.steps[-1][1].fit_resample(
                    pipeline_no_clf.named_steps["preprocessor"].transform(X_train),
                    y_train
                )[1] if use_smote else y_train,
                eval_set=[(X_val_proc, y_val)],
                verbose=50,
            )
            # Refit full pipeline without early stopping for clean deployment
            pipeline = build_pipeline(classifier, use_smote=use_smote)
            pipeline.fit(X_train, y_train)
        else:
            pipeline.fit(X_train, y_train)

        # ── Evaluate on validation set ────────────────────────────────────
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        val_pred  = (val_proba >= 0.50).astype(int)

        metrics = {
            "val_auroc":   roc_auc_score(y_val, val_proba),
            "val_auprc":   average_precision_score(y_val, val_proba),
            "val_f1":      _f1(y_val, val_pred),
            "val_recall":  _recall(y_val, val_pred),
            "val_specificity": _specificity(y_val, val_pred),
        }

        for k, v in metrics.items():
            logger.info(f"  {k:<22}: {v:.4f}")
            mlflow.log_metric(k, v)

        # Log model params
        mlflow.log_params({
            "model_type": name,
            "use_smote":  use_smote,
            "n_train":    len(X_train),
            "n_val":      len(X_val),
            "pos_rate":   float(y_train.mean()),
        })

        mlflow.sklearn.log_model(pipeline, artifact_path=name)

        # ── Detailed classification report ────────────────────────────────
        print("\nClassification Report (Val Set):")
        print(classification_report(y_val, val_pred,
                                    target_names=["Susceptible", "Resistant"]))
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, val_pred))

    return pipeline, metrics, val_proba


# ── Threshold tuning ──────────────────────────────────────────────────────

def tune_threshold(y_true: pd.Series,
                   y_proba: np.ndarray,
                   min_sensitivity: float = 0.85) -> float:
    """
    Find the lowest decision threshold that achieves the target sensitivity.
    Clinical rationale: missing a resistant case is more dangerous than
    over-predicting resistance.

    Args:
        y_true:          True binary labels
        y_proba:         Predicted probabilities for Resistant class
        min_sensitivity: Minimum recall for Resistant class (default 0.85)

    Returns:
        Optimal threshold float in [0, 1]
    """
    thresholds = np.arange(0.10, 0.90, 0.01)
    best_threshold = 0.50
    best_specificity = 0.0

    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        sens = _recall(y_true, pred)
        spec = _specificity(y_true, pred)
        if sens >= min_sensitivity and spec > best_specificity:
            best_specificity = spec
            best_threshold = t

    logger.info(f"Optimal threshold: {best_threshold:.2f} "
                f"(specificity={best_specificity:.3f} at "
                f"sensitivity≥{min_sensitivity:.0%})")
    return best_threshold


# ── Train all models ──────────────────────────────────────────────────────

def train_all(train_df: pd.DataFrame,
              val_df: pd.DataFrame) -> dict:
    """
    Train all models, compare on validation set, save best model.

    Returns:
        dict of {model_name: {"pipeline": ..., "metrics": ..., "threshold": ...}}
    """
    X_train, y_train = get_X_y(train_df, ALL_FEATURES, TARGET)
    X_val,   y_val   = get_X_y(val_df,   ALL_FEATURES, TARGET)

    # Recompute scale_pos_weight for XGBoost from actual training data
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    classifiers = get_classifiers()
    classifiers["xgboost"].scale_pos_weight = pos_weight

    results = {}
    leaderboard = []

    for name, clf in classifiers.items():
        pipeline, metrics, val_proba = train_model(
            name, clf, X_train, y_train, X_val, y_val
        )
        threshold = tune_threshold(y_val, val_proba)

        results[name] = {
            "pipeline":  pipeline,
            "metrics":   metrics,
            "threshold": threshold,
        }
        leaderboard.append({
            "model":    name,
            "AUROC":    metrics["val_auroc"],
            "AUPRC":    metrics["val_auprc"],
            "Recall":   metrics["val_recall"],
            "F1":       metrics["val_f1"],
        })

        # Save each model
        model_path = MODEL_DIR / f"{name}_pipeline.joblib"
        joblib.dump({"pipeline": pipeline, "threshold": threshold}, model_path)
        logger.info(f"Saved {name} → {model_path}")

    # ── Leaderboard ───────────────────────────────────────────────────────
    lb_df = pd.DataFrame(leaderboard).sort_values("AUROC", ascending=False)
    print("\n" + "=" * 60)
    print("  MODEL LEADERBOARD (Validation Set)")
    print("=" * 60)
    print(lb_df.to_string(index=False, float_format="{:.4f}".format))
    print("=" * 60)

    # ── Save best model ───────────────────────────────────────────────────
    best_name = lb_df.iloc[0]["model"]
    best_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(results[best_name], best_path)
    logger.success(f"Best model: {best_name} (AUROC={lb_df.iloc[0]['AUROC']:.4f})")
    logger.success(f"Saved best model → {best_path}")

    return results


# ── Metric helpers ────────────────────────────────────────────────────────

def _recall(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return tp / (tp + fn + 1e-9)

def _specificity(y_true, y_pred):
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    return tn / (tn + fp + 1e-9)

def _f1(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    return 2 * prec * rec / (prec + rec + 1e-9)


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data.load_data import generate_synthetic_whonet
    from src.data.clean_data import clean
    from src.features.feature_engineering import engineer_features
    from src.data.split_data import split_by_isolate

    df    = engineer_features(clean(generate_synthetic_whonet(8000)))
    train, val, test = split_by_isolate(df)
    results = train_all(train, val)

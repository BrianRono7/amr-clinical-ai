"""
src/evaluation/evaluate.py
─────────────────────────────────────────────────────────────────────────────
STEP 7 — MODEL EVALUATION

Comprehensive evaluation on the HELD-OUT TEST SET.
Only run this ONCE — after all model selection decisions are made.
Running evaluation repeatedly on test data inflates reported performance.

Evaluation suite:
  1.  ROC curve + AUC
  2.  Precision-Recall curve + AUPRC
  3.  Confusion matrix (at tuned threshold)
  4.  Classification report (sensitivity, specificity, PPV, NPV)
  5.  Calibration curve (is the model's confidence trustworthy?)
  6.  Feature importance (SHAP values for XGBoost/LightGBM, coef for LR)
  7.  Per-organism-drug performance breakdown
  8.  Clinical utility curve (net benefit analysis)
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
from pathlib import Path
from loguru import logger

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report,
    calibration_curve, brier_score_loss,
)

import sys
sys.path.insert(0, ".")
from src.features.feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, BINARY_FEATURES
from src.data.split_data import get_X_y

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BINARY_FEATURES
TARGET       = "resistant"
OUTPUT_DIR   = Path("data/evaluation_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Load model ────────────────────────────────────────────────────────────

def load_best_model(model_path: str = "data/models/best_model.joblib") -> dict:
    model_bundle = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    return model_bundle   # {"pipeline": ..., "threshold": ...}


# ── 1 & 2. ROC + PR curves ────────────────────────────────────────────────

def plot_roc_pr(y_true, y_proba, model_name: str = "Best Model",
                save: bool = True) -> dict:
    """Plot ROC and Precision-Recall curves side by side."""
    auroc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    prec, rec, _ = precision_recall_curve(y_true, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC
    axes[0].plot(fpr, tpr, color="#1D9E75", lw=2,
                 label=f"AUC = {auroc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")
    axes[0].fill_between(fpr, tpr, alpha=0.1, color="#1D9E75")
    axes[0].set(xlabel="False Positive Rate (1 - Specificity)",
                ylabel="True Positive Rate (Sensitivity)",
                title=f"ROC Curve — {model_name}")
    axes[0].legend(loc="lower right")
    axes[0].axhline(0.85, ls="--", color="#E45C3A", alpha=0.6,
                    label="Target sensitivity")

    # PR
    baseline = y_true.mean()
    axes[1].plot(rec, prec, color="#378ADD", lw=2,
                 label=f"AUPRC = {auprc:.3f}")
    axes[1].axhline(baseline, ls="--", color="gray", alpha=0.5,
                    label=f"Baseline ({baseline:.2f})")
    axes[1].fill_between(rec, prec, alpha=0.1, color="#378ADD")
    axes[1].set(xlabel="Recall (Sensitivity)",
                ylabel="Precision (PPV)",
                title=f"Precision-Recall Curve — {model_name}")
    axes[1].legend()

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "roc_pr_curves.png", dpi=150)
    plt.show()

    logger.info(f"AUROC: {auroc:.4f}  |  AUPRC: {auprc:.4f}")
    return {"auroc": auroc, "auprc": auprc}


# ── 3. Confusion matrix ───────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, threshold: float,
                           save: bool = True) -> None:
    """Annotated confusion matrix with clinical label names."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity  = tp / (tp + fn + 1e-9)
    specificity  = tn / (tn + fp + 1e-9)
    ppv          = tp / (tp + fp + 1e-9)
    npv          = tn / (tn + fn + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw={"width_ratios": [1.2, 1]})

    # Heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred: Susceptible", "Pred: Resistant"],
                yticklabels=["True: Susceptible", "True: Resistant"],
                ax=axes[0], cbar=False, linewidths=0.5)
    axes[0].set_title(f"Confusion Matrix (threshold={threshold:.2f})")

    # Clinical metrics table
    metrics_data = [
        ["Sensitivity (Recall)",    f"{sensitivity:.3f}",  "≥ 0.85 target"],
        ["Specificity",             f"{specificity:.3f}",  ""],
        ["PPV (Precision)",         f"{ppv:.3f}",          ""],
        ["NPV",                     f"{npv:.3f}",          ""],
        ["True Positives",          str(tp),               "Correctly flagged resistant"],
        ["False Negatives",         str(fn),               "Missed resistant — DANGER"],
        ["False Positives",         str(fp),               "Over-treated"],
        ["True Negatives",          str(tn),               ""],
    ]
    axes[1].axis("off")
    tbl = axes[1].table(
        cellText=metrics_data,
        colLabels=["Metric", "Value", "Note"],
        cellLoc="left", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width([0, 1, 2])
    axes[1].set_title("Clinical Performance Metrics")

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150,
                    bbox_inches="tight")
    plt.show()

    print(f"\n  Sensitivity: {sensitivity:.3f}  "
          f"(target ≥ 0.85 → {'✓ PASS' if sensitivity >= 0.85 else '✗ FAIL'})")
    print(f"  Specificity: {specificity:.3f}")
    print(f"  PPV:         {ppv:.3f}")
    print(f"  NPV:         {npv:.3f}")


# ── 4. Per-organism-drug breakdown ────────────────────────────────────────

def evaluate_per_combo(df_test: pd.DataFrame,
                        y_proba: np.ndarray,
                        threshold: float,
                        min_n: int = 20,
                        save: bool = True) -> pd.DataFrame:
    """
    AUROC and sensitivity for every organism-drug combination in test set.
    This is the key table for a research paper — it shows where the model
    works well and where it struggles.
    """
    df = df_test.copy()
    df["y_proba"] = y_proba
    df["y_pred"]  = (y_proba >= threshold).astype(int)

    rows = []
    for (org, drug), grp in df.groupby(["organism", "antibiotic"]):
        if len(grp) < min_n:
            continue
        y_t = grp[TARGET]
        if y_t.nunique() < 2:
            continue   # Can't compute AUROC with only one class

        auroc = roc_auc_score(y_t, grp["y_proba"])
        sens  = _recall(y_t, grp["y_pred"])
        spec  = _specificity(y_t, grp["y_pred"])
        res_rate = y_t.mean()

        rows.append({
            "Organism":        org,
            "Antibiotic":      drug,
            "N":               len(grp),
            "Resistance Rate": res_rate,
            "AUROC":           auroc,
            "Sensitivity":     sens,
            "Specificity":     spec,
            "Pass (≥0.80)":    "✓" if auroc >= 0.80 else "✗",
        })

    combo_df = pd.DataFrame(rows).sort_values("AUROC", ascending=False)

    print("\n── PER ORGANISM-DRUG PERFORMANCE (Test Set) ─────────────────")
    print(combo_df.to_string(index=False, float_format="{:.3f}".format))

    if save:
        combo_df.to_csv(OUTPUT_DIR / "per_combo_performance.csv", index=False)
        logger.info("Saved per_combo_performance.csv")

    # Heatmap
    pivot = combo_df.pivot(index="Organism", columns="Antibiotic", values="AUROC")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=0.5, vmax=1.0, ax=ax, linewidths=0.5,
                cbar_kws={"label": "AUROC"})
    ax.set_title("AUROC per Organism-Drug Pair (Test Set)")
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "per_combo_heatmap.png", dpi=150)
    plt.show()

    return combo_df


# ── 5. Calibration curve ──────────────────────────────────────────────────

def plot_calibration(y_true, y_proba, model_name: str = "Model",
                     save: bool = True) -> None:
    """
    Calibration plot: do predicted probabilities match actual frequencies?
    A well-calibrated model can be trusted when it says "70% resistant".
    """
    prob_true, prob_pred = calibration_curve(y_true, y_proba,
                                              n_bins=10, strategy="uniform")
    brier = brier_score_loss(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(prob_pred, prob_true, "s-", color="#1D9E75", lw=2,
            label=f"{model_name} (Brier={brier:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.fill_between(prob_pred, prob_pred, prob_true,
                    alpha=0.1, color="#1D9E75")
    ax.set(xlabel="Mean Predicted Probability",
           ylabel="Fraction of Positives (Actual Resistance Rate)",
           title="Calibration Curve")
    ax.legend()

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "calibration_curve.png", dpi=150)
    plt.show()

    logger.info(f"Brier Score: {brier:.4f} (lower is better; 0 = perfect)")


# ── 6. Feature importance ─────────────────────────────────────────────────

def plot_feature_importance(pipeline, feature_names: list,
                             top_n: int = 20, save: bool = True) -> None:
    """
    Plot feature importances from the classifier.
    Works for tree-based models (XGBoost, LightGBM, RF) and LR coefficients.
    """
    clf = pipeline.named_steps["classifier"]

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        title = "Feature Importance (Gain)"
        color = "#378ADD"
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
        title = "Feature Importance (|Coefficient|)"
        color = "#7F77DD"
    else:
        logger.warning("Classifier has no interpretable feature importances")
        return

    # Match to transformed feature names
    n_features = min(len(importances), len(feature_names))
    imp_df = pd.DataFrame({
        "feature":    feature_names[:n_features],
        "importance": importances[:n_features],
    }).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color=color)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Features — {title}")
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.show()

    logger.info(f"Top 5 features: {imp_df['feature'].head(5).tolist()}")


# ── Full evaluation run ───────────────────────────────────────────────────

def evaluate_on_test(test_df: pd.DataFrame,
                     model_path: str = "data/models/best_model.joblib") -> dict:
    """
    Run the full evaluation suite on the test set.
    Call this ONCE, after all modelling decisions are finalized.
    """
    bundle    = load_best_model(model_path)
    pipeline  = bundle["pipeline"]
    threshold = bundle["threshold"]

    X_test, y_test = get_X_y(test_df, ALL_FEATURES, TARGET)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    logger.info(f"Evaluating on {len(X_test):,} test records "
                f"(threshold={threshold:.2f})")

    metrics = plot_roc_pr(y_test, y_proba)
    plot_confusion_matrix(y_test, y_pred, threshold)
    plot_calibration(y_test, y_proba)
    combo_df = evaluate_per_combo(test_df, y_proba, threshold)

    # Feature importance
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        feat_names   = list(preprocessor.get_feature_names_out())
        plot_feature_importance(pipeline, feat_names)
    except Exception as e:
        logger.warning(f"Feature importance plot skipped: {e}")

    return {**metrics, "per_combo": combo_df}


# ── Metric helpers ────────────────────────────────────────────────────────

def _recall(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp / (tp + fn + 1e-9)

def _specificity(y_true, y_pred):
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return tn / (tn + fp + 1e-9)


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data.load_data import generate_synthetic_whonet
    from src.data.clean_data import clean
    from src.features.feature_engineering import engineer_features
    from src.data.split_data import split_by_isolate
    from src.models.train import train_all

    df = engineer_features(clean(generate_synthetic_whonet(8000)))
    train, val, test = split_by_isolate(df)
    train_all(train, val)               # Train first
    evaluate_on_test(test)              # Then evaluate

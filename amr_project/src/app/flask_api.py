"""
src/app/flask_api.py
─────────────────────────────────────────────────────────────────────────────
STEP 8B — FLASK REST API

Production-grade REST API wrapper around the trained model.
Use this when integrating AMR prediction into:
  - Hospital HIS (Hospital Information Systems)
  - DHIS2 (Kenya's national health data platform)
  - Mobile apps for field clinicians
  - Other microservices

Endpoints:
  POST /predict        — Single isolate prediction
  POST /predict/batch  — Batch CSV prediction
  GET  /health         — Health check
  GET  /model/info     — Model metadata
  GET  /docs           — API documentation

Run with:
    python src/app/flask_api.py

Or with gunicorn (production):
    gunicorn -w 4 -b 0.0.0.0:8000 "src.app.flask_api:create_app()"
─────────────────────────────────────────────────────────────────────────────
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import traceback
import io
import csv
from pathlib import Path
from datetime import datetime
from loguru import logger

# ── Configuration ─────────────────────────────────────────────────────────

MODEL_PATH  = "data/models/best_model.joblib"
API_VERSION = "v1"
BASE_URL    = f"/api/{API_VERSION}"

# Required fields for a valid prediction request
REQUIRED_FIELDS = ["organism", "antibiotic", "age_years", "sex",
                   "ward", "specimen_type"]

# Optional fields with defaults
OPTIONAL_FIELDS = {
    "hospital":                  "Unknown",
    "hiv_status":                False,
    "prior_antibiotic_90d":      False,
    "prior_antibiotic_class":    "None",
    "prior_hospitalisation_1y":  False,
    "diabetes":                  False,
    "catheter_present":          False,
    "date":                      None,
}

DRUG_CLASS = {
    "ampicillin": "beta_lactam", "amoxicillin": "beta_lactam",
    "ceftriaxone": "cephalosporin", "cefotaxime": "cephalosporin",
    "meropenem": "carbapenem", "imipenem": "carbapenem",
    "ciprofloxacin": "fluoroquinolone", "levofloxacin": "fluoroquinolone",
    "gentamicin": "aminoglycoside", "amikacin": "aminoglycoside",
    "cotrimoxazole": "sulfonamide", "vancomycin": "glycopeptide",
    "clindamycin": "lincosamide", "oxacillin": "beta_lactam",
    "azithromycin": "macrolide", "isoniazid": "antimycobacterial",
    "rifampicin": "antimycobacterial", "ethambutol": "antimycobacterial",
}

SPECIMEN_RISK = {
    "Blood": 3, "CSF": 3, "Endotracheal aspirate": 2,
    "Sputum": 2, "Wound swab": 2, "Pus": 2,
    "Urine": 1, "Stool": 1,
}


# ── Model loader ──────────────────────────────────────────────────────────

_model_bundle = None   # Singleton — loaded once at startup

def get_model():
    global _model_bundle
    if _model_bundle is None:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Train the model first: python -m src.models.train"
            )
        _model_bundle = joblib.load(MODEL_PATH)
        logger.success(f"Model loaded from {MODEL_PATH}")
    return _model_bundle


# ── Input validation ──────────────────────────────────────────────────────

def validate_input(data: dict) -> tuple[bool, str]:
    """Returns (is_valid, error_message)."""
    for field in REQUIRED_FIELDS:
        if field not in data or data[field] is None or data[field] == "":
            return False, f"Missing required field: '{field}'"

    if not isinstance(data.get("age_years"), (int, float)):
        try:
            float(data["age_years"])
        except (TypeError, ValueError):
            return False, "age_years must be a number"

    if float(data.get("age_years", 0)) < 0 or float(data.get("age_years", 0)) > 120:
        return False, "age_years must be between 0 and 120"

    return True, ""


# ── Feature builder ───────────────────────────────────────────────────────

def build_feature_row(data: dict) -> pd.DataFrame:
    """Convert API request dict into a feature DataFrame."""
    # Fill optional fields
    for field, default in OPTIONAL_FIELDS.items():
        if field not in data:
            data[field] = default

    # Date handling
    date = pd.Timestamp(data["date"]) if data.get("date") else pd.Timestamp.now()
    month   = date.month
    quarter = (date.month - 1) // 3 + 1

    antibiotic   = str(data["antibiotic"]).lower()
    specimen     = str(data.get("specimen_type", "Unknown"))
    age          = float(data["age_years"])

    def age_group(a):
        if a <= 5:   return "infant"
        if a <= 18:  return "paediatric"
        if a <= 40:  return "young_adult"
        if a <= 60:  return "adult"
        return "elderly"

    row = {
        "isolate_id":                      "API_PRED",
        "date":                            date,
        "organism":                        data["organism"],
        "antibiotic":                      antibiotic,
        "drug_class":                      DRUG_CLASS.get(antibiotic, "other"),
        "specimen_type":                   specimen,
        "ward":                            data.get("ward", "Unknown"),
        "age_years":                       age,
        "age_group":                       age_group(age),
        "sex":                             data.get("sex", "Unknown"),
        "hospital":                        data.get("hospital", "Unknown"),
        "prior_antibiotic_90d":            bool(data.get("prior_antibiotic_90d", False)),
        "prior_antibiotic_class":          data.get("prior_antibiotic_class", "None"),
        "prior_hospitalisation_1y":        bool(data.get("prior_hospitalisation_1y", False)),
        "hiv_status":                      bool(data.get("hiv_status", False)),
        "diabetes":                        bool(data.get("diabetes", False)),
        "catheter_present":                bool(data.get("catheter_present", False)),
        "month":                           month,
        "quarter":                         quarter,
        "year":                            date.year,
        "specimen_risk":                   SPECIMEN_RISK.get(specimen, 1),
        "month_sin":                       np.sin(2 * np.pi * month / 12),
        "month_cos":                       np.cos(2 * np.pi * month / 12),
        "hospital_org_resistance_rate":    0.45,
        "ward_org_drug_resistance_rate":   0.45,
        "n_resistant_classes":             0,
        "is_mdr":                          0,
        "sir_result":                      "R",
        "resistant":                       0,
    }
    return pd.DataFrame([row])


def run_prediction(data: dict) -> dict:
    """Core prediction logic — shared by single and batch endpoints."""
    bundle    = get_model()
    pipeline  = bundle["pipeline"]
    threshold = bundle["threshold"]

    import sys
    sys.path.insert(0, ".")
    from src.features.feature_engineering import (
        CATEGORICAL_FEATURES, NUMERIC_FEATURES, BINARY_FEATURES
    )
    all_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BINARY_FEATURES

    df_row = build_feature_row(data)
    X      = df_row[[c for c in all_features if c in df_row.columns]]

    proba = float(pipeline.predict_proba(X)[0, 1])
    pred  = int(proba >= threshold)

    if proba >= 0.70:
        risk_level = "HIGH"
    elif proba >= 0.40:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    return {
        "organism":               data["organism"],
        "antibiotic":             data["antibiotic"],
        "resistance_probability": round(proba, 4),
        "prediction":             "RESISTANT" if pred else "SUSCEPTIBLE",
        "risk_level":             risk_level,
        "decision_threshold":     round(threshold, 2),
        "interpretation": (
            f"This isolate has a {proba:.1%} predicted probability of being "
            f"resistant to {data['antibiotic']}. "
            + ("Consider alternative therapy." if pred else
               "Empiric treatment may be appropriate — confirm with culture.")
        )
    }


# ── Flask app factory ─────────────────────────────────────────────────────

def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)    # Allow cross-origin requests (for frontend integration)

    # ── Health check ──────────────────────────────────────────────────────
    @app.route(f"{BASE_URL}/health", methods=["GET"])
    def health():
        model_ok = Path(MODEL_PATH).exists()
        return jsonify({
            "status":     "healthy" if model_ok else "degraded",
            "model":      "loaded" if model_ok else "missing",
            "timestamp":  datetime.utcnow().isoformat(),
            "version":    API_VERSION,
        }), 200 if model_ok else 503

    # ── Model info ────────────────────────────────────────────────────────
    @app.route(f"{BASE_URL}/model/info", methods=["GET"])
    def model_info():
        try:
            bundle = get_model()
            clf    = bundle["pipeline"].named_steps["classifier"]
            return jsonify({
                "model_type":        type(clf).__name__,
                "decision_threshold": round(bundle["threshold"], 4),
                "required_fields":   REQUIRED_FIELDS,
                "optional_fields":   list(OPTIONAL_FIELDS.keys()),
                "version":           API_VERSION,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 503

    # ── Single prediction ─────────────────────────────────────────────────
    @app.route(f"{BASE_URL}/predict", methods=["POST"])
    def predict():
        """
        Predict AMR resistance for a single isolate.

        Request body (JSON):
        {
            "organism":             "Escherichia coli",
            "antibiotic":           "ciprofloxacin",
            "age_years":            42,
            "sex":                  "F",
            "ward":                 "General Medical",
            "specimen_type":        "Urine",
            "hiv_status":           false,
            "prior_antibiotic_90d": true,
            "prior_antibiotic_class": "Fluoroquinolone",
            "hospital":             "Kenyatta National Hospital"
        }

        Response:
        {
            "organism": "Escherichia coli",
            "antibiotic": "ciprofloxacin",
            "resistance_probability": 0.7234,
            "prediction": "RESISTANT",
            "risk_level": "HIGH",
            "decision_threshold": 0.42,
            "interpretation": "..."
        }
        """
        try:
            data = request.get_json(force=True)
            if not data:
                return jsonify({"error": "Request body must be JSON"}), 400

            valid, err = validate_input(data)
            if not valid:
                return jsonify({"error": err}), 422

            result = run_prediction(data)
            logger.info(
                f"Prediction: {data['organism']} / {data['antibiotic']} "
                f"→ {result['prediction']} ({result['resistance_probability']:.2%})"
            )
            return jsonify(result), 200

        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 503
        except Exception as e:
            logger.error(traceback.format_exc())
            return jsonify({"error": "Internal server error",
                            "detail": str(e)}), 500

    # ── Batch prediction ──────────────────────────────────────────────────
    @app.route(f"{BASE_URL}/predict/batch", methods=["POST"])
    def predict_batch():
        """
        Predict resistance for multiple isolates from a CSV upload.

        Form data: file=<csv_file>
        CSV must contain columns matching required_fields.

        Returns: CSV with original columns + resistance_probability,
                 prediction, risk_level columns appended.
        """
        try:
            if "file" not in request.files:
                return jsonify({"error": "No file provided. "
                                "Send CSV as form-data field 'file'"}), 400

            file = request.files["file"]
            df   = pd.read_csv(file)

            # Validate required columns
            missing_cols = [c for c in REQUIRED_FIELDS if c not in df.columns]
            if missing_cols:
                return jsonify({
                    "error": f"CSV missing required columns: {missing_cols}"
                }), 422

            results = []
            errors  = []

            for idx, row in df.iterrows():
                try:
                    res = run_prediction(row.to_dict())
                    results.append(res)
                except Exception as e:
                    results.append({
                        "resistance_probability": None,
                        "prediction":             "ERROR",
                        "risk_level":             "UNKNOWN",
                        "interpretation":         str(e),
                    })
                    errors.append({"row": idx, "error": str(e)})

            df["resistance_probability"] = [r["resistance_probability"] for r in results]
            df["prediction"]             = [r["prediction"]             for r in results]
            df["risk_level"]             = [r["risk_level"]             for r in results]
            df["interpretation"]         = [r["interpretation"]         for r in results]

            # Return as CSV
            out = io.StringIO()
            df.to_csv(out, index=False)
            out.seek(0)

            return Response(
                out.getvalue(),
                mimetype="text/csv",
                headers={"Content-Disposition":
                         "attachment; filename=amr_predictions.csv"}
            )

        except Exception as e:
            logger.error(traceback.format_exc())
            return jsonify({"error": str(e)}), 500

    # ── API documentation ─────────────────────────────────────────────────
    @app.route(f"{BASE_URL}/docs", methods=["GET"])
    def docs():
        return jsonify({
            "title":       "AMR Prediction API — East Africa",
            "version":     API_VERSION,
            "description": (
                "Predicts antimicrobial resistance probability from routine "
                "clinical data. No genomic sequencing required."
            ),
            "endpoints": {
                f"GET  {BASE_URL}/health":        "Health check",
                f"GET  {BASE_URL}/model/info":    "Model metadata",
                f"POST {BASE_URL}/predict":       "Single isolate prediction (JSON body)",
                f"POST {BASE_URL}/predict/batch": "Batch prediction (CSV upload)",
                f"GET  {BASE_URL}/docs":          "This documentation",
            },
            "required_fields": REQUIRED_FIELDS,
            "optional_fields": OPTIONAL_FIELDS,
            "example_request": {
                "organism":               "Escherichia coli",
                "antibiotic":             "ciprofloxacin",
                "age_years":              42,
                "sex":                    "F",
                "ward":                   "General Medical",
                "specimen_type":          "Urine",
                "hiv_status":             False,
                "prior_antibiotic_90d":   True,
                "prior_antibiotic_class": "Fluoroquinolone",
                "hospital":               "Kenyatta National Hospital",
            },
            "example_response": {
                "organism":               "Escherichia coli",
                "antibiotic":             "ciprofloxacin",
                "resistance_probability": 0.7234,
                "prediction":             "RESISTANT",
                "risk_level":             "HIGH",
                "decision_threshold":     0.42,
                "interpretation":         "Consider alternative therapy.",
            }
        })

    return app


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = create_app()
    logger.info("Starting AMR Prediction API on http://0.0.0.0:8000")
    logger.info(f"Docs: http://localhost:8000{BASE_URL}/docs")
    app.run(host="0.0.0.0", port=8000, debug=False)

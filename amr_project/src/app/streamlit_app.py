"""
src/app/streamlit_app.py
─────────────────────────────────────────────────────────────────────────────
STEP 8A — STREAMLIT DEPLOYMENT

Clinical decision-support tool for AMR resistance prediction.
Designed for hospital lab technicians and clinicians.

Run with:
    streamlit run src/app/streamlit_app.py

Features:
  • Single patient prediction with resistance probability
  • Batch prediction from CSV upload
  • Model performance dashboard
  • Resistance surveillance heatmap
  • Downloadable reports
─────────────────────────────────────────────────────────────────────────────
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
from pathlib import Path
from datetime import date

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AMR Predictor — East Africa",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────
MODEL_PATH = "data/models/best_model.joblib"

ORGANISMS = [
    "Escherichia coli", "Klebsiella pneumoniae", "Staphylococcus aureus",
    "Salmonella typhi", "Streptococcus pneumoniae",
    "Pseudomonas aeruginosa", "Acinetobacter baumannii",
]
ANTIBIOTICS = [
    "ampicillin", "ciprofloxacin", "ceftriaxone", "meropenem",
    "cotrimoxazole", "gentamicin", "amikacin", "oxacillin",
    "clindamycin", "vancomycin", "azithromycin",
]
WARDS      = ["ICU", "General Medical", "Surgical", "Maternity",
              "Paediatric", "Outpatient"]
SPECIMENS  = ["Blood", "Urine", "Sputum", "Wound swab", "CSF", "Stool"]
HOSPITALS  = ["Kenyatta National Hospital", "Coast General Hospital",
              "Moi Teaching Hospital", "Other"]
AB_CLASSES = ["None", "Beta-lactam", "Fluoroquinolone", "Aminoglycoside",
              "Macrolide", "Sulfonamide", "Other"]

# Risk colour thresholds
HIGH_RISK   = 0.70
MEDIUM_RISK = 0.40


# ── Model loader (cached) ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not Path(MODEL_PATH).exists():
        return None
    return joblib.load(MODEL_PATH)


# ── Feature builder ───────────────────────────────────────────────────────
def build_input_row(organism, antibiotic, age, sex, ward, specimen,
                    hospital, hiv, prior_ab, prior_ab_class,
                    prior_hosp, diabetes, catheter,
                    collection_date) -> pd.DataFrame:
    """Convert form inputs into a single-row DataFrame matching training schema."""
    from src.features.feature_engineering import (
        DRUG_CLASS, SPECIMEN_RISK, engineer_features
    )
    drug_class    = DRUG_CLASS.get(antibiotic, "other")
    specimen_risk = SPECIMEN_RISK.get(specimen, 1)

    month   = collection_date.month
    quarter = (collection_date.month - 1) // 3 + 1
    year    = collection_date.year

    row = {
        "isolate_id":                "SINGLE_PRED",
        "date":                      pd.Timestamp(collection_date),
        "organism":                  organism,
        "antibiotic":                antibiotic,
        "drug_class":                drug_class,
        "specimen_type":             specimen,
        "ward":                      ward,
        "age_years":                 float(age),
        "sex":                       sex,
        "hospital":                  hospital,
        "prior_antibiotic_90d":      prior_ab,
        "prior_antibiotic_class":    prior_ab_class,
        "prior_hospitalisation_1y":  prior_hosp,
        "hiv_status":                hiv,
        "diabetes":                  diabetes,
        "catheter_present":          catheter,
        "month":                     month,
        "quarter":                   quarter,
        "year":                      year,
        "specimen_risk":             specimen_risk,
        "month_sin":                 np.sin(2 * np.pi * month / 12),
        "month_cos":                 np.cos(2 * np.pi * month / 12),
        "hospital_org_resistance_rate":   0.45,  # Prior — updated with real data
        "ward_org_drug_resistance_rate":  0.45,
        "n_resistant_classes":       0,
        "is_mdr":                    0,
        "age_group":                 _age_group(age),
        "sir_result":                "R",   # Placeholder — not used in prediction
        "resistant":                 0,     # Placeholder
    }
    return pd.DataFrame([row])


def _age_group(age):
    if age <= 5:   return "infant"
    if age <= 18:  return "paediatric"
    if age <= 40:  return "young_adult"
    if age <= 60:  return "adult"
    return "elderly"


# ── Risk badge ────────────────────────────────────────────────────────────
def risk_badge(prob: float) -> tuple[str, str]:
    """Returns (label, colour) based on probability."""
    if prob >= HIGH_RISK:
        return "🔴 HIGH RESISTANCE RISK", "#FF4B4B"
    if prob >= MEDIUM_RISK:
        return "🟡 MODERATE RESISTANCE RISK", "#FFA500"
    return "🟢 LOW RESISTANCE RISK", "#21C55D"


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/WHO_logo.svg/320px-WHO_logo.svg.png",
             width=120)
    st.markdown("## 🦠 AMR Predictor")
    st.markdown("**East Africa Clinical Decision Support**")
    st.divider()
    page = st.radio(
        "Navigation",
        ["🔬 Single Prediction", "📋 Batch Prediction",
         "📊 Surveillance Dashboard", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Model: XGBoost / LightGBM ensemble")
    st.caption("Data: WHONET + KEMRI")
    st.caption("Version: 1.0.0 — Research Prototype")
    st.warning("⚠️ For research use only. Not a substitute for clinical judgement.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════════════
if page == "🔬 Single Prediction":
    st.title("🔬 AMR Resistance Prediction")
    st.markdown(
        "Enter patient and isolate details below to receive a resistance "
        "probability estimate for the selected organism-antibiotic pair."
    )

    bundle = load_model()
    if bundle is None:
        st.error(
            "⚠️ Model not found at `data/models/best_model.joblib`. "
            "Run `python -m src.models.train` first to train the model."
        )
        st.stop()

    pipeline  = bundle["pipeline"]
    threshold = bundle["threshold"]

    # ── Input form ────────────────────────────────────────────────────────
    with st.form("prediction_form"):
        st.subheader("🧫 Microbiological Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            organism   = st.selectbox("Organism *", ORGANISMS)
        with col2:
            antibiotic = st.selectbox("Antibiotic *", ANTIBIOTICS)
        with col3:
            specimen   = st.selectbox("Specimen Type", SPECIMENS)

        st.subheader("👤 Patient Demographics")
        col4, col5, col6, col7 = st.columns(4)
        with col4:
            age  = st.number_input("Age (years)", min_value=0,
                                   max_value=110, value=35)
        with col5:
            sex  = st.selectbox("Sex", ["M", "F", "Unknown"])
        with col6:
            ward = st.selectbox("Ward", WARDS)
        with col7:
            hospital = st.selectbox("Hospital", HOSPITALS)

        st.subheader("🏥 Clinical Risk Factors")
        col8, col9 = st.columns(2)
        with col8:
            hiv       = st.checkbox("HIV Positive")
            prior_ab  = st.checkbox("Antibiotic use in prior 90 days")
            prior_ab_class = st.selectbox(
                "Prior antibiotic class (if yes)", AB_CLASSES,
                disabled=not prior_ab
            )
        with col9:
            prior_hosp = st.checkbox("Hospitalised in prior 12 months")
            diabetes   = st.checkbox("Diabetes")
            catheter   = st.checkbox("Catheter present (urinary/IV)")

        collection_date = st.date_input(
            "Culture collection date", value=date.today()
        )

        submitted = st.form_submit_button(
            "🔍 Predict Resistance", use_container_width=True, type="primary"
        )

    # ── Prediction ────────────────────────────────────────────────────────
    if submitted:
        input_df = build_input_row(
            organism, antibiotic, age, sex, ward, specimen, hospital,
            hiv, prior_ab, prior_ab_class, prior_hosp, diabetes, catheter,
            collection_date
        )

        try:
            from src.features.feature_engineering import (
                CATEGORICAL_FEATURES, NUMERIC_FEATURES, BINARY_FEATURES
            )
            all_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BINARY_FEATURES
            X = input_df[[c for c in all_features if c in input_df.columns]]

            proba     = pipeline.predict_proba(X)[0, 1]
            pred      = int(proba >= threshold)
            label, colour = risk_badge(proba)

            st.divider()
            st.subheader("📋 Prediction Result")

            # Main result card
            st.markdown(
                f"""
                <div style="
                    background:{colour}20; border:2px solid {colour};
                    border-radius:12px; padding:20px; text-align:center;
                ">
                    <h2 style="color:{colour}; margin:0">{label}</h2>
                    <h1 style="font-size:3rem; margin:10px 0">
                        {proba:.1%}
                    </h1>
                    <p style="color:#555; margin:0">
                        Probability of <b>{organism}</b> being resistant to
                        <b>{antibiotic}</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("---")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Resistance Probability", f"{proba:.1%}")
            col_b.metric("Decision Threshold",     f"{threshold:.2f}")
            col_c.metric("Prediction",
                         "RESISTANT" if pred else "SUSCEPTIBLE")

            # Gauge chart
            fig, ax = plt.subplots(figsize=(6, 1.5))
            ax.barh(["Risk"], [proba], color=colour, height=0.4)
            ax.barh(["Risk"], [1 - proba], left=[proba],
                    color="#EEEEEE", height=0.4)
            ax.axvline(threshold, color="#333", lw=1.5,
                       ls="--", label=f"Threshold ({threshold:.2f})")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Resistance Probability")
            ax.legend(loc="upper right", fontsize=8)
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:.0%}")
            )
            fig.patch.set_alpha(0)
            st.pyplot(fig, use_container_width=True)

            # Clinical interpretation
            st.subheader("💊 Clinical Interpretation")
            if pred == 1:
                st.error(
                    f"**Likely Resistant** — Consider alternative antibiotics. "
                    f"Await culture sensitivity results before committing to "
                    f"{antibiotic}. Consult antimicrobial stewardship team if available."
                )
            else:
                st.success(
                    f"**Likely Susceptible** — {antibiotic.capitalize()} may be "
                    f"appropriate empirically. Continue to confirm with culture results."
                )

            # Key risk factors driving prediction
            st.subheader("⚠️ Active Risk Factors")
            risks = []
            if hiv:         risks.append("HIV-positive status (1.25x resistance risk)")
            if prior_ab:    risks.append(f"Prior antibiotic use: {prior_ab_class}")
            if ward == "ICU": risks.append("ICU admission (1.20x resistance risk)")
            if prior_hosp:  risks.append("Prior hospitalisation (1.10x resistance risk)")
            if catheter:    risks.append("Catheter present (device-associated infection risk)")
            if diabetes:    risks.append("Diabetes (immunosuppression)")

            if risks:
                for r in risks:
                    st.warning(f"• {r}")
            else:
                st.info("No high-risk clinical flags identified.")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════
elif page == "📋 Batch Prediction":
    st.title("📋 Batch Prediction")
    st.markdown(
        "Upload a CSV file of isolate records to generate resistance "
        "predictions for all rows simultaneously."
    )

    bundle = load_model()
    if bundle is None:
        st.error("Model not found. Train the model first.")
        st.stop()

    pipeline  = bundle["pipeline"]
    threshold = bundle["threshold"]

    # Template download
    template_cols = [
        "organism", "antibiotic", "age_years", "sex", "ward",
        "specimen_type", "hospital", "hiv_status",
        "prior_antibiotic_90d", "prior_antibiotic_class",
        "prior_hospitalisation_1y", "diabetes", "catheter_present", "date"
    ]
    template_df = pd.DataFrame(columns=template_cols)
    csv_template = template_df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download CSV Template",
        csv_template,
        "amr_batch_template.csv",
        "text/csv",
    )

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            st.write(f"**Loaded {len(df_up):,} records**")
            st.dataframe(df_up.head(5), use_container_width=True)

            if st.button("▶ Run Batch Prediction", type="primary"):
                import sys
                sys.path.insert(0, ".")
                from src.data.clean_data import clean
                from src.features.feature_engineering import engineer_features
                from src.features.feature_engineering import (
                    CATEGORICAL_FEATURES, NUMERIC_FEATURES, BINARY_FEATURES
                )

                # Add placeholder columns if missing
                for col in ["sir_result", "resistant"]:
                    if col not in df_up.columns:
                        df_up[col] = "S" if col == "sir_result" else 0

                df_proc = engineer_features(clean(df_up))
                all_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BINARY_FEATURES
                X = df_proc[[c for c in all_features if c in df_proc.columns]]

                probas = pipeline.predict_proba(X)[:, 1]
                preds  = (probas >= threshold).astype(int)

                df_up["resistance_probability"] = probas.round(4)
                df_up["prediction"] = np.where(preds == 1, "RESISTANT", "SUSCEPTIBLE")
                df_up["risk_level"] = pd.cut(
                    probas,
                    bins=[0, MEDIUM_RISK, HIGH_RISK, 1.0],
                    labels=["Low", "Moderate", "High"],
                    include_lowest=True
                )

                st.success(f"✅ Predictions complete for {len(df_up):,} records")

                # Summary stats
                col1, col2, col3 = st.columns(3)
                col1.metric("High Risk",     int((preds == 1).sum()))
                col2.metric("Low Risk",      int((preds == 0).sum()))
                col3.metric("Mean Risk Prob", f"{probas.mean():.1%}")

                st.dataframe(
                    df_up[["organism", "antibiotic", "resistance_probability",
                            "prediction", "risk_level"]],
                    use_container_width=True
                )

                # Download results
                csv_out = df_up.to_csv(index=False).encode()
                st.download_button(
                    "⬇️ Download Results CSV",
                    csv_out, "amr_predictions.csv", "text/csv",
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.exception(e)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — SURVEILLANCE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
elif page == "📊 Surveillance Dashboard":
    st.title("📊 AMR Surveillance Dashboard")
    st.markdown(
        "Real-time resistance trends from your hospital dataset. "
        "Upload your WHONET processed data to populate this dashboard."
    )

    uploaded = st.file_uploader(
        "Upload processed AMR data (amr_clean.csv)", type=["csv"]
    )

    if uploaded:
        df = pd.read_csv(uploaded, parse_dates=["date"])
        st.success(f"Loaded {len(df):,} records")

        # ── Filters ───────────────────────────────────────────────────────
        with st.expander("🔧 Filters"):
            col1, col2, col3 = st.columns(3)
            orgs_sel = col1.multiselect(
                "Organisms", df["organism"].unique(),
                default=list(df["organism"].unique()[:3])
            )
            drugs_sel = col2.multiselect(
                "Antibiotics", df["antibiotic"].unique(),
                default=list(df["antibiotic"].unique()[:3])
            )
            hosp_sel = col3.multiselect(
                "Hospitals", df["hospital"].unique(),
                default=list(df["hospital"].unique())
            ) if "hospital" in df.columns else df["hospital"].unique()

        filt = df[
            df["organism"].isin(orgs_sel) &
            df["antibiotic"].isin(drugs_sel)
        ]
        if "hospital" in df.columns and len(hosp_sel):
            filt = filt[filt["hospital"].isin(hosp_sel)]

        # ── KPI row ───────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Isolates",  f"{len(filt):,}")
        k2.metric("Overall Resistance", f"{filt['resistant'].mean():.1%}")
        k3.metric("Organisms",       filt["organism"].nunique())
        k4.metric("Antibiotics",     filt["antibiotic"].nunique())

        st.divider()
        col_l, col_r = st.columns(2)

        # Heatmap
        with col_l:
            st.subheader("Resistance Heatmap")
            pivot = filt.groupby(["organism", "antibiotic"])["resistant"].mean().unstack()
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(pivot, annot=True, fmt=".0%",
                        cmap="YlOrRd", ax=ax, linewidths=0.5)
            ax.set_title("Resistance Rate by Organism × Drug")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        # Trend
        with col_r:
            st.subheader("Resistance Trend Over Time")
            if "date" in filt.columns:
                monthly = (
                    filt.set_index("date")
                    .groupby([pd.Grouper(freq="ME"), "organism"])["resistant"]
                    .mean()
                    .reset_index()
                )
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                for org, grp in monthly.groupby("organism"):
                    ax2.plot(grp["date"], grp["resistant"],
                             label=org.split()[-1], lw=1.8)
                ax2.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f"{x:.0%}")
                )
                ax2.set_ylabel("Resistance Rate")
                ax2.legend(fontsize=7)
                ax2.set_title("Monthly Resistance Rate")
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)

        # Ward breakdown
        st.subheader("Resistance by Ward")
        if "ward" in filt.columns:
            ward_data = filt.groupby("ward")["resistant"].mean().sort_values(ascending=False)
            fig3, ax3 = plt.subplots(figsize=(10, 3))
            ward_data.plot(kind="bar", ax=ax3,
                           color=sns.color_palette("RdYlGn_r", len(ward_data)))
            ax3.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:.0%}")
            )
            ax3.set_title("Resistance Rate by Ward")
            ax3.set_xlabel("")
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)

    else:
        st.info(
            "Upload `data/processed/amr_clean.csv` (generated after the "
            "cleaning step) to see your surveillance dashboard."
        )


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Tool")
    st.markdown("""
    ### AI-Powered AMR Prediction for East Africa

    This tool predicts antimicrobial resistance profiles from **routine clinical
    data** — no whole-genome sequencing required.

    ---
    #### How it works
    1. Clinical data (organism, patient demographics, risk factors) is collected
    2. Features are engineered to capture clinical context
    3. A trained XGBoost/LightGBM ensemble estimates the probability of resistance
    4. A calibrated decision threshold (tuned for sensitivity ≥ 85%) gives the prediction

    ---
    #### Key datasets used
    | Database | URL |
    |----------|-----|
    | BV-BRC / PATRIC | https://www.bv-brc.org |
    | NCBI NDARO | https://www.ncbi.nlm.nih.gov/pathogens/antimicrobial-resistance/ |
    | WHONET | https://whonet.org |
    | CARD | https://card.mcmaster.ca |
    | KEMRI | https://www.kemri.go.ke |

    ---
    #### Limitations
    - This is a **research prototype** — not validated for clinical use
    - Model performance varies by organism-drug pair (see evaluation report)
    - Predictions should always be interpreted alongside culture results
    - Training data from East African hospitals may not generalise globally

    ---
    #### Citation
    > [Your Name] (2025). AI-Powered AMR Prediction from Routine Clinical Data
    > in East Africa. [Institution], Kenya.

    ---
    ⚠️ **Disclaimer**: This tool does not replace clinical judgement, culture
    results, or antimicrobial stewardship guidance.
    """)

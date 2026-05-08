"""
src/data/load_data.py
─────────────────────────────────────────────────────────────────────────────
STEP 1 — DATA LOADING

Handles two scenarios:
  A) Real WHONET CSV export from hospital lab software
  B) Synthetic data generator for development & testing (when real data is
     not yet available under ethics approval)

WHONET column reference:
  https://whonet.org/software.html
  Typical export columns: Identification, Organism, Antibiotic, SIR, MIC,
  Specimen, Date, Ward, Age, Sex, etc.
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger


# ── A. Load real WHONET export ────────────────────────────────────────────

def load_whonet(filepath: str) -> pd.DataFrame:
    """
    Load a WHONET CSV export and apply minimal structural normalisation.

    WHONET exports resistance as 'S', 'I', 'R' in separate antibiotic columns
    (wide format). This function melts it into long format:
        one row = one organism + one antibiotic tested.

    Args:
        filepath: Path to the WHONET .csv export file

    Returns:
        Long-format DataFrame with columns:
        [isolate_id, date, organism, specimen_type, ward, age_years, sex,
         antibiotic, sir_result, mic_value, hospital]
    """
    logger.info(f"Loading WHONET data from {filepath}")
    df = pd.read_csv(filepath, low_memory=False, encoding="latin-1")

    # WHONET uses uppercase column names — normalise
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # ── Identify antibiotic result columns ────────────────────────────────
    # WHONET antibiotic columns follow pattern: DRUG_SIR (e.g. AMP_SIR, CIP_SIR)
    sir_cols = [c for c in df.columns if c.endswith("_sir")]
    mic_cols = [c for c in df.columns if c.endswith("_mic")]

    logger.info(f"Found {len(sir_cols)} antibiotic result columns")

    # ── Melt SIR results to long format ───────────────────────────────────
    id_vars = ["isolate_id", "date", "organism", "specimen_type",
               "ward", "age_years", "sex", "hospital"]
    # Keep only id_vars that actually exist in this export
    id_vars = [v for v in id_vars if v in df.columns]

    df_long = df[id_vars + sir_cols].melt(
        id_vars=id_vars,
        value_vars=sir_cols,
        var_name="antibiotic_code",
        value_name="sir_result"
    )

    # Map antibiotic codes to full names (partial — extend as needed)
    code_to_name = {
        "amp_sir": "ampicillin",
        "cip_sir": "ciprofloxacin",
        "ctx_sir": "ceftriaxone",
        "mem_sir": "meropenem",
        "sxt_sir": "cotrimoxazole",
        "gen_sir": "gentamicin",
        "ami_sir": "amikacin",
        "oxn_sir": "oxacillin",
        "cli_sir": "clindamycin",
        "van_sir": "vancomycin",
        "azi_sir": "azithromycin",
        "inh_sir": "isoniazid",
        "rif_sir": "rifampicin",
        "emb_sir": "ethambutol",
    }
    df_long["antibiotic"] = df_long["antibiotic_code"].map(code_to_name)
    df_long = df_long.drop(columns=["antibiotic_code"])

    # Drop rows where SIR was not tested (blank / NaN)
    df_long = df_long[df_long["sir_result"].notna()]
    df_long = df_long[df_long["sir_result"].isin(["S", "I", "R"])]

    logger.success(f"Loaded {len(df_long):,} isolate-antibiotic records")
    return df_long


# ── B. Synthetic data generator (development only) ───────────────────────

def generate_synthetic_whonet(n_isolates: int = 5000,
                               seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic AMR dataset for development and pipeline
    testing BEFORE real hospital data is available.

    Resistance probabilities are calibrated to approximate published East
    African AMR surveillance rates (GLASS 2022, Kenya KQMH data).

    Args:
        n_isolates: Number of unique patient isolates to simulate
        seed:       Random seed for reproducibility

    Returns:
        Long-format DataFrame (same schema as load_whonet output)
    """
    rng = np.random.default_rng(seed)
    logger.info(f"Generating {n_isolates:,} synthetic isolates (seed={seed})")

    # ── Organism distribution ─────────────────────────────────────────────
    organisms = ["Escherichia coli", "Klebsiella pneumoniae",
                 "Staphylococcus aureus", "Salmonella typhi",
                 "Streptococcus pneumoniae", "Pseudomonas aeruginosa",
                 "Acinetobacter baumannii"]
    org_probs = [0.30, 0.22, 0.18, 0.12, 0.08, 0.06, 0.04]

    # ── Resistance probability by organism-drug pair ──────────────────────
    # Source: GLASS 2022 Africa data + Kenya KQMH surveillance
    resistance_probs = {
        ("Escherichia coli",       "ampicillin"):     0.82,
        ("Escherichia coli",       "ciprofloxacin"):  0.48,
        ("Escherichia coli",       "ceftriaxone"):    0.55,
        ("Escherichia coli",       "meropenem"):      0.08,
        ("Escherichia coli",       "cotrimoxazole"):  0.70,
        ("Klebsiella pneumoniae",  "ceftriaxone"):    0.62,
        ("Klebsiella pneumoniae",  "meropenem"):      0.12,
        ("Klebsiella pneumoniae",  "gentamicin"):     0.44,
        ("Klebsiella pneumoniae",  "amikacin"):       0.18,
        ("Staphylococcus aureus",  "oxacillin"):      0.38,   # MRSA rate
        ("Staphylococcus aureus",  "clindamycin"):    0.32,
        ("Staphylococcus aureus",  "vancomycin"):     0.02,
        ("Salmonella typhi",       "ciprofloxacin"):  0.22,
        ("Salmonella typhi",       "azithromycin"):   0.08,
        ("Salmonella typhi",       "ceftriaxone"):    0.15,
    }

    # ── Build isolate-level records ───────────────────────────────────────
    records = []
    hospitals = ["Kenyatta National Hospital", "Coast General Hospital",
                 "Moi Teaching Hospital"]
    wards = ["ICU", "General Medical", "Surgical", "Maternity",
             "Paediatric", "Outpatient"]
    specimens = ["Blood", "Urine", "Sputum", "Wound swab", "CSF", "Stool"]
    antibiotic_classes = ["Beta-lactam", "Fluoroquinolone", "Aminoglycoside",
                          "Macrolide", "None"]

    for i in range(n_isolates):
        organism = rng.choice(organisms, p=org_probs)
        age      = int(rng.integers(1, 85))
        sex      = rng.choice(["M", "F"])
        ward     = rng.choice(wards)
        specimen = rng.choice(specimens)
        hospital = rng.choice(hospitals)
        date     = pd.Timestamp("2019-01-01") + pd.Timedelta(
                       days=int(rng.integers(0, 5 * 365)))

        # Clinical risk factors — influence resistance probability
        hiv      = rng.random() < (0.06 if age > 15 else 0.005)
        prior_ab = rng.random() < 0.45
        prior_ab_class = rng.choice(antibiotic_classes) if prior_ab else "None"
        prior_hosp = rng.random() < 0.30
        diabetes = rng.random() < (0.08 if age > 40 else 0.01)
        catheter = ward in ["ICU", "Surgical"] and rng.random() < 0.55

        # Determine which antibiotics were tested for this organism
        drug_list = [pair[1] for pair in resistance_probs if pair[0] == organism]
        if not drug_list:
            continue

        for drug in drug_list:
            base_prob = resistance_probs.get((organism, drug), 0.20)

            # Adjust probability based on clinical risk factors
            prob = base_prob
            if hiv:
                prob = min(prob * 1.25, 0.99)
            if prior_ab:
                prob = min(prob * 1.15, 0.99)
            if ward == "ICU":
                prob = min(prob * 1.20, 0.99)
            if prior_hosp:
                prob = min(prob * 1.10, 0.99)

            sir = "R" if rng.random() < prob else (
                  "I" if rng.random() < 0.10 else "S")

            records.append({
                "isolate_id":           f"ISO_{i:06d}",
                "date":                 date,
                "organism":             organism,
                "specimen_type":        specimen,
                "ward":                 ward,
                "age_years":            age,
                "sex":                  sex,
                "hospital":             hospital,
                "antibiotic":           drug,
                "sir_result":           sir,
                "mic_value":            None,
                "prior_antibiotic_90d": prior_ab,
                "prior_antibiotic_class": prior_ab_class,
                "prior_hospitalisation_1y": prior_hosp,
                "hiv_status":           hiv,
                "diabetes":             diabetes,
                "catheter_present":     catheter,
            })

    df = pd.DataFrame(records)
    logger.success(f"Generated {len(df):,} isolate-antibiotic records "
                   f"from {n_isolates:,} isolates")
    return df


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Generate synthetic data and save for pipeline testing
    df = generate_synthetic_whonet(n_isolates=5000)
    out = Path("data/raw/whonet_synthetic.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Saved to {out}")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nOrganisms:\n{df['organism'].value_counts()}")
    print(f"\nAntibiotics:\n{df['antibiotic'].value_counts()}")
    print(f"\nResistance rate: {(df['sir_result']=='R').mean():.1%}")

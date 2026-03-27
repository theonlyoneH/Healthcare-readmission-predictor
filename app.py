import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Healthcare Readmission Dashboard",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — card-style containers, spacing, typography
# ============================================================
st.markdown("""
<style>
    /* ── Global font ── */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }

    /* ── Main area background ── */
    .main { background-color: #f4f6f9; }

    /* ── Card component ── */
    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 18px;
        border: 1px solid #e0e4ea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    /* ── App header banner ── */
    .app-header {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        color: white;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .app-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .app-header p  { color: #d0e4ff; margin: 6px 0 0; font-size: 0.95rem; }

    /* ── Risk badges ── */
    .risk-high {
        background: #fde8e8;
        border-left: 5px solid #e53935;
        border-radius: 8px;
        padding: 16px 20px;
        font-size: 1.05rem;
        font-weight: 600;
        color: #b71c1c;
    }
    .risk-low {
        background: #e8f5e9;
        border-left: 5px solid #43a047;
        border-radius: 8px;
        padding: 16px 20px;
        font-size: 1.05rem;
        font-weight: 600;
        color: #1b5e20;
    }

    /* ── Sidebar section headers ── */
    .sidebar-section {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280;
        margin: 16px 0 8px;
        padding-bottom: 4px;
        border-bottom: 1px solid #e5e7eb;
    }

    /* ── Metric value emphasis ── */
    div[data-testid="metric-container"] {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 12px 18px;
    }

    /* ── Tab styling ── */
    .stTabs [role="tab"] { font-size: 0.95rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL  (cached so it only loads once per session)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("final_model.pkl")

model_package = load_model()
model     = model_package["model"]
threshold = model_package["threshold"]
features  = model_package["features"]

# ============================================================
# APP HEADER
# ============================================================
st.markdown("""
<div class="app-header">
  <h1>🏥 Healthcare Readmission Prediction</h1>
  <p>A clinical decision-support tool — enter patient details in the sidebar and review the risk assessment below.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LABEL MAPS — human-readable text for encoded values
# ============================================================
RACE_MAP = {
    0: "Caucasian",
    1: "AfricanAmerican",
    2: "Hispanic",
    3: "Asian",
    4: "Other",
    5: "Unknown",
}
GENDER_MAP  = {0: "Male", 1: "Female", 2: "Unknown/Other"}
YES_NO_MAP  = {0: "No",  1: "Yes"}
GLU_MAP     = {0: "Not Measured", 1: "Normal (≤200)", 2: "High (>200)", 3: "Very High (>300)"}
A1C_MAP     = {0: "Not Measured", 1: "Normal (<8%)", 2: "Elevated (≥8%)", 3: "Not Available"}
INSULIN_MAP = {0: "None", 1: "Steady Dose", 2: "Dose Increased", 3: "Dose Decreased"}
DIAG_MAP    = {
    0: "Circulatory",
    1: "Respiratory",
    2: "Digestive",
    3: "Diabetes",
    4: "Other / Unknown",
}
ADM_MAP  = {0: "Emergency", 1: "Urgent", 2: "Elective / Other"}
DISC_MAP = {0: "Discharged Home", 1: "Other / Transfer"}


def _sel(label, mapping, default=0, help_text=None):
    """Selectbox that displays human-readable labels but stores the raw int."""
    options = list(mapping.keys())
    idx = options.index(default) if default in options else 0
    return st.sidebar.selectbox(
        label,
        options=options,
        index=idx,
        format_func=lambda x: mapping[x],
        help=help_text,
    )


# ============================================================
# SIDEBAR — STRUCTURED INPUT FORM
# ============================================================
st.sidebar.markdown("## 📋 Patient Details")
st.sidebar.caption("Fill in patient information to generate a readmission risk assessment.")

# ── SECTION 1: Patient Information ──────────────────────────
st.sidebar.markdown('<p class="sidebar-section">👤 Patient Information</p>', unsafe_allow_html=True)

age_val   = st.sidebar.select_slider(
    "Age Group",
    options=[0, 1, 2],
    format_func=lambda x: ["Under 30", "30 – 60", "Over 60"][x],
    value=1,
    help="Decade-bucketed age group from the dataset.",
)
race_val   = _sel("Race / Ethnicity", RACE_MAP, default=2)
gender_val = _sel("Gender",           GENDER_MAP, default=0)
medicare_val = _sel("Medicare Coverage",  YES_NO_MAP, default=0)
medicaid_val = _sel("Medicaid Coverage",  YES_NO_MAP, default=0)

# ── SECTION 2: Clinical Data ─────────────────────────────────
st.sidebar.markdown('<p class="sidebar-section">🩺 Clinical Data</p>', unsafe_allow_html=True)

diag_val      = _sel("Primary Diagnosis Category", DIAG_MAP, default=4)
n_diag        = st.sidebar.slider("Number of Diagnoses",     min_value=1,  max_value=16,  value=5)
n_meds        = st.sidebar.slider("Number of Medications",   min_value=1,  max_value=81,  value=15)
n_lab         = st.sidebar.slider("Lab Procedures Count",    min_value=1,  max_value=132, value=43)
n_proc        = st.sidebar.slider("Procedures Performed",    min_value=0,  max_value=6,   value=1)
glu_val       = _sel("Max Glucose Serum Test",  GLU_MAP,     default=0)
a1c_val       = _sel("HbA1c (A1C) Result",      A1C_MAP,     default=0)
insulin_val   = _sel("Insulin Administration",  INSULIN_MAP, default=0)
change_val    = _sel("Medication Change During Stay", YES_NO_MAP, default=0,
                     help_text="Was the patient's medication regimen changed before discharge?")
diab_med_val  = _sel("Diabetes Medication Prescribed", YES_NO_MAP, default=0)

# ── SECTION 3: Hospital Utilisation & History ─────────────
st.sidebar.markdown('<p class="sidebar-section">🏥 Hospital Utilisation & History</p>', unsafe_allow_html=True)

time_hosp     = st.sidebar.slider("Days in Hospital",         min_value=1,   max_value=14,  value=3)
total_act     = st.sidebar.slider("Total Clinical Activities", min_value=1,  max_value=200, value=50,
                                  help="Sum of all lab, procedure, and medication events.")
adm_src_val   = _sel("Admission Source",        ADM_MAP,  default=0)
disc_disp_val = _sel("Discharge Disposition",   DISC_MAP, default=0)
had_emg_val   = _sel("Emergency Visit in Past Year",    YES_NO_MAP, default=0)
had_inp_val   = _sel("Inpatient Days in Past Year",     YES_NO_MAP, default=0)
had_out_val   = _sel("Outpatient Visit in Past Year",   YES_NO_MAP, default=0)

# ============================================================
# BUILD INPUT DATAFRAME & COMPUTE DERIVED FEATURES
# ============================================================
raw = {
    "race":                    race_val,
    "gender":                  gender_val,
    "age":                     age_val,
    "discharge_disposition_id": disc_disp_val,
    "admission_source_id":     adm_src_val,
    "time_in_hospital":        time_hosp,
    "num_lab_procedures":      n_lab,
    "num_procedures":          n_proc,
    "num_medications":         n_meds,
    "primary_diagnosis":       diag_val,
    "number_diagnoses":        n_diag,
    "max_glu_serum":           glu_val,
    "A1Cresult":               a1c_val,
    "insulin":                 insulin_val,
    "change":                  change_val,
    "diabetesMed":             diab_med_val,
    "medicare":                medicare_val,
    "medicaid":                medicaid_val,
    "had_emergency":           had_emg_val,
    "had_inpatient_days":      had_inp_val,
    "had_outpatient_days":     had_out_val,
}

df_input = pd.DataFrame([raw])

# Derived / engineered features (same logic as 3_modeling.ipynb)
df_input["medication_ratio"]  = df_input["num_medications"] / (df_input["number_diagnoses"] + 1)
df_input["lab_ratio"]         = df_input["num_lab_procedures"] / (df_input["time_in_hospital"] + 1)
df_input["procedure_ratio"]   = df_input["num_procedures"]    / (df_input["time_in_hospital"] + 1)
df_input["total_activity"]    = total_act  # user supplied

# Ensure every model feature is present; fill any gap with 0
for feat in features:
    if feat not in df_input.columns:
        df_input[feat] = 0

input_df = df_input[features]

# ============================================================
# PREDICTION
# ============================================================
prob       = model.predict_proba(input_df)[0][1]
prediction = int(prob > threshold)

# ============================================================
# EXPLAINER  (shared across tabs)
# ============================================================
@st.cache_resource(show_spinner=False)
def build_explainer(_model):
    return shap.TreeExplainer(_model)

explainer    = build_explainer(model)
shap_values  = explainer(input_df)

# ============================================================
# GLOBAL SHAP  (cached — computed once on sampled training data)
# ============================================================
@st.cache_data(show_spinner=False)
def compute_global_shap():
    df = pd.read_csv("cleaned_data.csv")
    df["medication_ratio"] = df["num_medications"] / (df["number_diagnoses"] + 1)
    df["lab_ratio"]        = df["num_lab_procedures"] / (df["time_in_hospital"] + 1)
    df["procedure_ratio"]  = df["num_procedures"]     / (df["time_in_hospital"] + 1)
    # total_activity already in cleaned_data
    for feat in features:
        if feat not in df.columns:
            df[feat] = 0
    X_sample = df[features].sample(min(500, len(df)), random_state=42)
    _expl = shap.TreeExplainer(model)
    return _expl(X_sample)

global_shap = compute_global_shap()

# ============================================================
# TABBED LAYOUT
# ============================================================
tab_pred, tab_shap, tab_global = st.tabs([
    "📊  Prediction",
    "🧠  AI Explanation (SHAP)",
    "📈  Global Feature Importance",
])

# ─────────────────────────────────────────────────────────────
# TAB 1 — PREDICTION DASHBOARD
# ─────────────────────────────────────────────────────────────
with tab_pred:
    st.markdown("### Patient Readmission Risk Assessment")

    col_metric, col_summary = st.columns([1, 2], gap="large")

    # ── Left: metric card ──
    with col_metric:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric(
            label="Readmission Probability",
            value=f"{prob * 100:.1f} %",
            delta=f"Threshold: {threshold * 100:.0f}%",
            delta_color="off",
        )
        st.progress(float(prob))
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Right: risk status + guidance ──
    with col_summary:
        if prediction == 1:
            st.markdown(
                '<div class="risk-high">🔴 &nbsp; High Risk of Readmission<br>'
                '<span style="font-weight:400;font-size:0.88rem;">'
                'This patient has an elevated probability of being readmitted within 30 days. '
                'Consider scheduling an early follow-up, reviewing discharge medications, '
                'and coordinating with care management.</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="risk-low">🟢 &nbsp; Low Risk of Readmission<br>'
                '<span style="font-weight:400;font-size:0.88rem;">'
                'This patient has a low probability of being readmitted within 30 days. '
                'Standard discharge protocols and routine follow-up are appropriate.</span></div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Input summary table ──
    st.markdown("#### 📝 Patient Profile Summary")
    summary = {
        "Age Group":              ["Under 30", "30–60", "Over 60"][age_val],
        "Gender":                 GENDER_MAP[gender_val],
        "Primary Diagnosis":      DIAG_MAP[diag_val],
        "Days in Hospital":       time_hosp,
        "Number of Medications":  n_meds,
        "Lab Procedures":         n_lab,
        "Procedures":             n_proc,
        "Medicare":               YES_NO_MAP[medicare_val],
        "Medicaid":               YES_NO_MAP[medicaid_val],
        "Medication Change":      YES_NO_MAP[change_val],
        "Diabetes Medication":    YES_NO_MAP[diab_med_val],
        "Past Emergency Visit":   YES_NO_MAP[had_emg_val],
    }
    st.dataframe(
        pd.DataFrame(summary.items(), columns=["Feature", "Value"]),
        use_container_width=True,
        hide_index=True,
    )

# ─────────────────────────────────────────────────────────────
# TAB 2 — INDIVIDUAL SHAP EXPLANATION
# ─────────────────────────────────────────────────────────────
with tab_shap:
    st.markdown("### Why did the model make this prediction?")
    st.markdown(
        "The chart below shows **which features pushed the prediction up (🔴) or down (🔵)** "
        "for this specific patient.  Longer bars = stronger influence on the prediction."
    )

    fig_shap, _ = plt.subplots(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False, max_display=15)
    plt.tight_layout()
    st.pyplot(fig_shap, use_container_width=True)
    plt.close(fig_shap)

    st.caption(
        "💡 Tip: Features shown in red increase readmission risk; features in blue reduce it. "
        "Values are SHAP (SHapley Additive exPlanations) scores."
    )

# ─────────────────────────────────────────────────────────────
# TAB 3 — GLOBAL FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
with tab_global:
    st.markdown("### Overall Model Feature Importance")
    st.markdown(
        "This chart shows which features **most often influence predictions** across a random "
        "sample of 500 historical patients — giving a population-level view of the model's "
        "decision drivers."
    )

    fig_global, _ = plt.subplots(figsize=(10, 6))
    shap.plots.bar(global_shap, show=False, max_display=15)
    plt.tight_layout()
    st.pyplot(fig_global, use_container_width=True)
    plt.close(fig_global)

    st.caption("📊 Based on a random sample of 500 records from the training dataset.")
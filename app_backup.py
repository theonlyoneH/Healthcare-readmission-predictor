import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Healthcare Readmission Predictor", layout="wide")

# ============================================================
# LOAD MODEL
# ============================================================
model_package = joblib.load("final_model.pkl")

model = model_package["model"]
threshold = model_package["threshold"]
features = model_package["features"]

# ============================================================
# TITLE
# ============================================================
st.title("🏥 Healthcare Readmission Prediction System")
st.markdown("Predict patient readmission risk with Explainable AI (SHAP)")

# ============================================================
# USER INPUT
# ============================================================
st.sidebar.header("Patient Inputs")

def user_input():
    data = {}

    for feature in features:
        data[feature] = st.sidebar.number_input(feature, value=0.0)

    return pd.DataFrame([data])

input_df = user_input()

# ============================================================
# PREDICTION
# ============================================================
st.subheader("📊 Prediction")

prob = model.predict_proba(input_df)[0][1]
prediction = int(prob > threshold)

st.write(f"**Readmission Probability:** {prob:.2f}")

if prediction == 1:
    st.error("⚠️ High Risk of Readmission")
else:
    st.success("✅ Low Risk of Readmission")

# ============================================================
# SHAP EXPLANATION
# ============================================================
st.subheader("🧠 Explainability (SHAP)")

explainer = shap.TreeExplainer(model)
shap_values = explainer(input_df)

fig, ax = plt.subplots()
shap.plots.bar(shap_values, show=False)
st.pyplot(fig)

# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================
st.subheader("📈 Global Feature Importance")

df = pd.read_csv("cleaned_data.csv")

# Recreate features (IMPORTANT)
df['meds_x_procedures'] = df['num_medications'] * df['num_procedures']
df['med_per_diagnosis'] = df['num_medications'] / (df['number_diagnoses'] + 1)
df['procedures_per_day'] = df['num_procedures'] / (df['time_in_hospital'] + 1)
df['lab_ratio'] = df['num_lab_procedures'] / (df['time_in_hospital'] + 1)

X = df.drop(columns=['target'])
X = X[features]

shap_values_full = explainer(X.sample(500, random_state=42))

fig2, ax2 = plt.subplots()
shap.plots.bar(shap_values_full, show=False)
st.pyplot(fig2)
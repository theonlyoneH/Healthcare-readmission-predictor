# 🏥 Explainable Healthcare Readmission Prediction & Analytics System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-Model-FF6600?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SHAP-Explainability-7C3AED?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Power_BI-Dashboard-F2C811?style=for-the-badge&logo=powerbi&logoColor=black"/>
</p>

<p align="center">
  <b>🔗 Live Demo → <a href="https://healthcare-readmission-predictor.streamlit.app/">healthcare-readmission-predictor.streamlit.app</a></b>
</p>

---



> A clinical decision-support system that predicts patient readmission risk and explains **why** — built for transparency, trust, and actionable insight.

---

## 🧠 Problem Statement

Hospitals struggle to identify high-risk readmission patients, leading to increased costs and reduced quality of care. Traditional ML models predict readmissions but lack transparency — making it hard for clinicians to trust and act on them.

This project builds an **end-to-end** <mark>**healthcare analytics**</mark> system combining:
- Predictive ML modeling
- <mark>**Exploratory Data Analysis**</mark> and clinical trend mining
- Explainable AI (SHAP) for model transparency
- An interactive Power BI + Streamlit reporting layer

---

## 🎯 Key Objectives

| # | Objective |
|---|-----------|
| 1 | Predict 30-day patient readmission risk with interpretable ML |
| 2 | Perform <mark>**clinical data analysis**</mark> to surface trends (age, diagnosis, medication) |
| 3 | Explain model decisions using SHAP — globally and per patient |
| 4 | Deliver <mark>**actionable analytics**</mark> to clinical and operational teams via dashboard |

---

## 🏗️ System Architecture

```
Raw Healthcare Dataset (Diabetes Hospital Data)
             │
             ▼
    ┌─────────────────────┐
    │   Data Cleaning      │  ← Missing values, encoding, deduplication
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │ Feature Engineering  │  ← Age groups, risk indicators, interaction features
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │    Clean Dataset     │
    └────┬────────────┬───┘
         │            │
         ▼            ▼
  ┌────────────┐  ┌────────────────────┐
  │  ML Models  │  │  Analytics Engine   │◄─────────┐
  │ LR·RF·XGB  │  │ Trends·Segmentation │          │
  └─────┬──────┘  └──────────┬─────────┘          │
        │                    │         feedback ───┘
        ▼
  ┌──────────────────┐
  │ Model Evaluation  │  ← AUC-ROC, F1, class imbalance
  └──────┬───────────┘
         ▼
  ┌──────────────────┐
  │  Best Model      │  ← XGBoost (benchmark winner)
  └──────┬───────────┘
         ▼
  ┌──────────────────┐
  │ Model Persistence │  ← joblib serialization
  └──────┬───────────┘
         │            │
         ▼            ▼ (analytics insights)
  ┌──────────────────────────────┐
  │  Predictions + SHAP Layer    │  ← Global overview · per-patient drill-down
  └──────────────┬───────────────┘
                 ▼
  ┌──────────────────────────────┐
  │  Power BI + Streamlit        │  ← Reports · alerts · clinical drill-down
  └──────────────────────────────┘
```

---

## 🔬 Dataset

**Source:** [UCI Diabetes 130-US Hospitals Dataset](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)

| Property | Value |
|----------|-------|
| Records | ~100,000 patient encounters |
| Features | 50 clinical + administrative |
| Target | Readmission within 30 days |
| Class balance | ~11% positive (imbalanced) |

**Key features used:**
`age`, `race`, `gender`, `time_in_hospital`, `num_lab_procedures`, `num_medications`, `primary_diagnosis`, `A1Cresult`, `insulin`, `medicare`, `medicaid`

---

## 🤖 ML Pipeline

### Models Trained
| Model | Role |
|-------|------|
| Logistic Regression | Baseline benchmark |
| Random Forest | Ensemble comparison |
| **XGBoost** | **Final selected model** |

### Evaluation Metrics
- **AUC-ROC** (primary — handles class imbalance)
- F1-Score (precision-recall balance)
- Confusion matrix + threshold tuning (default threshold: 35%)

### Class Imbalance Handling
- SMOTE oversampling on training set
- `scale_pos_weight` tuning in XGBoost
- Threshold adjustment from default 50% → 35%

---

## 🔍 Explainable AI — SHAP

<mark>**Explainability**</mark> is the core differentiator of this project. SHAP (SHapley Additive exPlanations) is used at two levels:

### Global Feature Importance
Shows which features drive readmission risk across the entire patient population — useful for <mark>**healthcare analysts**</mark> and hospital policy teams.

### Per-Patient Local Explanation
For each individual prediction, SHAP shows exactly which factors pushed the risk score up or down. Clinicians see *why* a patient is flagged, not just *that* they are.

```python
import shap
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Global
shap.summary_plot(shap_values, X_test)

# Local (single patient)
shap.force_plot(explainer.expected_value, shap_values[i], X_test.iloc[i])
```

---

## 📊 <mark>Analytics</mark> & Insights Layer

The <mark>**data analytics**</mark> engine runs in parallel to the ML pipeline and its findings feed back into feature engineering:

- **Readmission by age group** — which demographics are highest risk
- **Medication impact analysis** — insulin, metformin, and readmission correlation
- **Diagnosis segmentation** — top ICD codes linked to repeated admissions
- **Length of stay vs. readmission** — is discharge timing a predictor?
- **High-risk patient clustering** — unsupervised segmentation for targeted intervention

> These insights are surfaced in both the Streamlit app (AI Explanation tab) and the Power BI dashboard.

---

## 📈 Power BI Dashboard

The Power BI layer consumes a structured output (CSV with predictions + SHAP values) exported by the Python pipeline:

| Report Page | Content |
|-------------|---------|
| Overview | Population-level readmission KPIs |
| Risk Distribution | Score histogram, threshold line |
| <mark>**Clinical Analytics**</mark> | Trends by diagnosis, age, medication |
| Patient Drill-down | Per-patient SHAP waterfall |
| Model Monitoring | Score drift over time |

**Integration pattern:**
```
Python pipeline → predictions_with_shap.csv → Power BI Desktop → Published report
```

---

## 🖥️ Streamlit App

The Streamlit app provides a real-time clinical decision-support interface:

**Tabs:**
- **Prediction** — Enter patient details, get risk score + confidence bar
- **AI Explanation (SHAP)** — Force plot showing per-patient feature contributions
- **Global Feature Importance** — Summary plot across all patients

**Live:** [https://healthcare-readmission-predictor.streamlit.app/](https://healthcare-readmission-predictor.streamlit.app/)

---

## 📁 Project Structure

```
healthcare-readmission/
│
├── data/
│   ├── raw/                    # Original UCI dataset
│   └── processed/              # Cleaned, engineered dataset
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda_analytics.ipynb  # ← Core data analysis work
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_shap_explainability.ipynb
│
├── models/
│   └── xgboost_best_model.pkl  # joblib serialized model
│
├── outputs/
│   └── predictions_with_shap.csv  # Power BI input
│
├── powerbi/
│   └── readmission_dashboard.pbix
│
├── app.py                      # Streamlit app entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/healthcare-readmission.git
cd healthcare-readmission

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```

**requirements.txt (core):**
```
pandas==2.0.3
numpy==1.24.4
scikit-learn==1.3.0
xgboost==1.7.6
shap==0.42.1
streamlit==1.28.0
imbalanced-learn==0.11.0
joblib==1.3.2
matplotlib==3.7.2
seaborn==0.12.2
```

---

## 🛠️ Skills Demonstrated

| Skill | Application |
|-------|-------------|
| <mark>**Data Analysis**</mark> | EDA, trend mining, clinical segmentation |
| <mark>**Data Cleaning**</mark> | Null handling, encoding, deduplication |
| <mark>**Feature Engineering**</mark> | Age grouping, risk composites, interaction terms |
| ML Modeling | LR, RF, XGBoost with benchmarking |
| <mark>**Analytics Reporting**</mark> | Power BI dashboard with SHAP output |
| Explainable AI | SHAP global + local interpretations |
| Python | pandas, scikit-learn, XGBoost, SHAP, Streamlit |
| SQL | <mark>**Data querying**</mark> and aggregation for analytics layer |

---

## 📌 Interview Summary

> *"I built an end-to-end <mark>**healthcare analytics**</mark> system that predicts 30-day patient readmission risk using XGBoost, applies SHAP-based Explainable AI to make predictions transparent at both population and patient level, and delivers findings through a Power BI dashboard and live Streamlit app — covering the full <mark>**data analysis**</mark> lifecycle from raw clinical data to actionable insight."*

---

## 📄 License

MIT License — see [LICENSE](./LICENSE) for details.

---

<p align="center">Built for the Abbott GDSA Internship | Mumbai · 2025</p>

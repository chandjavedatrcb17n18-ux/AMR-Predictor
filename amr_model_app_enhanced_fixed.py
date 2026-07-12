import io
import json
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Load models and encoders
model = joblib.load('prediction_model.pkl')
encoders = joblib.load('feature_encoders.pkl')

def prepare_features(organism, antibiotic, specimen, gender, age):
    input_dict = {
        'organism': organism,
        'antibiotic': antibiotic,
        'specimen': specimen,
        'gender': gender,
        'age': age
    }
    input_df = pd.DataFrame([input_dict])
    input_df['organism'] = encoders['organism'].transform(input_df['organism'])
    input_df['antibiotic'] = encoders['antibiotic'].transform(input_df['antibiotic'])
    input_df['specimen'] = encoders['specimen'].transform(input_df['specimen'])
    input_df['gender'] = encoders['gender'].transform(input_df['gender'])
    input_df = input_df[['age', 'gender', 'specimen', 'organism', 'antibiotic']]
    return input_df

ORGANISMS = ['E. coli', 'K. pneumoniae', 'S. aureus', 'P. aeruginosa']
ANTIBIOTICS = ['Ciprofloxacin', 'Ceftriaxone', 'Meropenem', 'Levofloxacin', 'Amikacin']
SPECIMENS = ['urine', 'blood', 'sputum']
GENDERS = ['M', 'F']

st.set_page_config(page_title="AMR Predictor — Research Edition", page_icon="🦠", layout="wide")

st.markdown("""
<style>
:root {
    --clr-bg-app: #F1F5F9;
    --clr-bg-card: #FFFFFF;
    --clr-text-main: #1E293B;
    --clr-text-muted: #64748B;
    --clr-accent: #0EA5E9;
}

/* Ensure all app text is clearly visible */
.stApp { background-color: var(--clr-bg-app); color: var(--clr-text-main); }
h1, h2, h3, p, label, div { color: var(--clr-text-main) !important; }

/* Interactive Card Style */
.card {
    background: var(--clr-bg-card);
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    transition: all 0.3s ease;
    margin-bottom: 1rem;
}
.card:hover {
    box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
    border-color: var(--clr-accent);
}

/* Sidebar Styling */
section[data-testid="stSidebar"] { background-color: #FFFFFF; }

/* Tab hover/active styles */
.stTabs [data-baseweb="tab"] { transition: 0.3s; }
.stTabs [data-baseweb="tab"]:hover { color: var(--clr-accent) !important; }

/* Badge Row */
.badge {
    padding: 0.4rem 0.8rem;
    border-radius: 20px;
    background: #E0F2FE;
    color: var(--clr-accent);
    font-weight: 600;
    font-size: 0.85rem;
    transition: transform 0.2s;
}
.badge:hover { transform: scale(1.05); cursor: default; }
</style>
""", unsafe_allow_html=True)

# ... [Keep your existing gauge_svg, _load_json, risk_category, 
# clinical_interpretation, get_explainer, local_shap_contributions, 
# and build_pdf_report functions here] ...

# UI Layout
st.markdown('<div class="app-header"><h1>🦠 AI-Powered Antibiotic Resistance Predictor</h1></div>', unsafe_allow_html=True)

# Main Application Logic
# ... [Use the st.tabs and logic as per your original file] ...
# Note: Wrap your content inside div class="card" to apply the hover effects.

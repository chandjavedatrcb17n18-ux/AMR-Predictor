
import io
import json
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import joblib
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
st.set_page_config(
    page_title="AMR Predictor — Research Edition",
    page_icon="🦠",
    layout="wide",
)
def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None
metrics = _load_json("metrics.json")
model_card = _load_json("model_card.json")
shap_importance = _load_json("shap_global_importance.json")
if "history" not in st.session_state:
    st.session_state.history = []  
def risk_category(prob_percent: float) -> tuple[str, str]:
    """Returns (label, color) for a 3-tier risk banding — a simple, defensible
    clinical-communication convention rather than an arbitrary >50% cutoff."""
    if prob_percent < 30:
        return "Low Risk", "green"
    elif prob_percent < 60:
        return "Moderate Risk", "orange"
    else:
        return "High Risk", "red"
def clinical_interpretation(organism, antibiotic, prob_percent, alt_probs):
    label, _ = risk_category(prob_percent)
    best_alt = min(alt_probs, key=lambda x: x[1])
    text = (
        f"The model estimates a **{prob_percent:.1f}%** probability that this "
        f"**{organism}** isolate is resistant to **{antibiotic}** "
        f"({label.lower()}). "
    )
    if best_alt[0] != antibiotic and best_alt[1] < prob_percent - 5:
        text += (
            f"Among the antibiotics considered, **{best_alt[0]}** shows the lowest "
            f"predicted resistance probability ({best_alt[1]:.1f}%) for this organism, "
            "and may warrant discussion as an alternative — pending culture and "
            "susceptibility confirmation."
        )
    else:
        text += "No clearly lower-risk alternative was identified among the antibiotics considered."
    return text
@st.cache_resource(show_spinner=False)
def get_explainer(_model):
    import shap
    return shap.TreeExplainer(_model)
def local_shap_contributions(features_df):
    """Returns a small DataFrame of per-feature SHAP contributions for one
    prediction — this is what makes the tool 'explainable' rather than a
    black box, which is the single feature reviewers most consistently ask
    healthcare-AI projects for."""
    try:
        explainer = get_explainer(model)
        sv = explainer.shap_values(features_df)
        sv = sv[1] if isinstance(sv, list) else sv
        sv = np.array(sv).reshape(-1)
        out = pd.DataFrame({
            "Feature": features_df.columns,
            "Value": features_df.iloc[0].values,
            "SHAP contribution": sv,
        }).sort_values("SHAP contribution", key=abs, ascending=False)
        return out
    except Exception as e:
        return None
def build_pdf_report(patient, prob_percent, label, interpretation) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 2 * cm
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, y, "AMR Resistance Prediction Report (Research Prototype)")
    y -= 1 * cm
    c.setFont("Helvetica", 9)
    c.drawString(2 * cm, y, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 0.8 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Input Summary")
    y -= 0.6 * cm
    c.setFont("Helvetica", 10)
    for k, v in patient.items():
        c.drawString(2.2 * cm, y, f"{k}: {v}")
        y -= 0.5 * cm
    y -= 0.3 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Prediction")
    y -= 0.6 * cm
    c.setFont("Helvetica", 10)
    c.drawString(2.2 * cm, y, f"Predicted resistance probability: {prob_percent:.1f}%  ({label})")
    y -= 0.8 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Clinical Interpretation")
    y -= 0.6 * cm
    c.setFont("Helvetica", 9)
    import textwrap
    for line in textwrap.wrap(interpretation.replace("**", ""), 95):
        c.drawString(2.2 * cm, y, line)
        y -= 0.45 * cm
    y -= 0.8 * cm
    c.setFont("Helvetica-Oblique", 8)
    for line in [
        "DISCLAIMER: This output is generated by a machine learning prototype trained on",
        "synthetic data for educational and research purposes only. It is NOT a substitute",
        "for clinical judgement, culture and susceptibility testing, or institutional",
        "antimicrobial stewardship guidance.",
    ]:
        c.drawString(2 * cm, y, line)
        y -= 0.4 * cm
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()
st.title("🦠 AI-Powered Antibiotic Resistance Predictor")
st.caption("Research prototype for antimicrobial stewardship decision support · Educational use only")
with st.expander("ℹ️ About this tool, its data, and its limitations", expanded=False):
    if model_card:
        st.markdown(f"**Model type:** `{model_card['model_type']}`  ·  **Version:** {model_card['version']}")
        st.markdown(f"**Intended use:** {model_card['intended_use']}")
        st.markdown("**Limitations:**")
        for l in model_card["limitations"]:
            st.markdown(f"- {l}")
    else:
        st.markdown(
            "This tool predicts the probability of antimicrobial resistance from "
            "patient/specimen/organism/antibiotic features, trained on a synthetic "
            "antibiogram-style dataset. **Not for clinical use.**"
        )
st.sidebar.header("Patient & Sample Information")
if st.sidebar.button("🎲 Load Random Sample Patient"):
    rng = np.random.default_rng()
    st.session_state["_sample"] = {
        "organism": rng.choice(ORGANISMS),
        "antibiotic": rng.choice(ANTIBIOTICS),
        "specimen": rng.choice(SPECIMENS),
        "gender": rng.choice(GENDERS),
        "age": int(rng.integers(1, 91)),
    }
sample = st.session_state.get("_sample", {})
organism = st.sidebar.selectbox("Organism (bacteria)", ORGANISMS,
                                 index=ORGANISMS.index(sample["organism"]) if sample.get("organism") in ORGANISMS else 0)
antibiotic = st.sidebar.selectbox("Antibiotic", ANTIBIOTICS,
                                   index=ANTIBIOTICS.index(sample["antibiotic"]) if sample.get("antibiotic") in ANTIBIOTICS else 0)
specimen = st.sidebar.selectbox("Specimen type", SPECIMENS,
                                 index=SPECIMENS.index(sample["specimen"]) if sample.get("specimen") in SPECIMENS else 0)
gender = st.sidebar.selectbox("Gender", GENDERS,
                               index=GENDERS.index(sample["gender"]) if sample.get("gender") in GENDERS else 0)
age = st.sidebar.number_input("Age (years)", min_value=1, max_value=90, value=int(sample.get("age", 30)))
predict_clicked = st.sidebar.button("Predict Resistance", type="primary")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Educational/research prototype only. Not for clinical use. "
    "See 'About & Model Card' tab for full disclaimers."
)
tab_predict, tab_perf, tab_shap, tab_batch, tab_history, tab_about = st.tabs(
    [" Predict", " Model Performance", " Explainability", "Batch Prediction", " History", " About & Model Card"]
)
with tab_predict:
    if predict_clicked:
        features = prepare_features(organism, antibiotic, specimen, gender, age)
        proba = model.predict_proba(features)[0]
        resistant_prob = proba[1]
        prob_percent = resistant_prob * 100
        label, color = risk_category(prob_percent)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Prediction Result")
            st.metric("Probability of Resistance", f"{prob_percent:.1f}%")
            st.progress(min(int(prob_percent), 100))
            st.markdown(f"**Risk category:** :{color}[{label}]")
        with col2:
            st.subheader("Resistance probability — other antibiotics (same organism)")
            probs = []
            for ab in ANTIBIOTICS:
                feat = prepare_features(organism, ab, specimen, gender, age)
                p = model.predict_proba(feat)[0][1] * 100
                probs.append((ab, p))
            chart_data = pd.DataFrame(probs, columns=["Antibiotic", "Resistance (%)"]).set_index("Antibiotic")
            st.bar_chart(chart_data)
        st.markdown("---")
        st.subheader("🩺 Clinical Interpretation")
        interpretation = clinical_interpretation(organism, antibiotic, prob_percent, probs)
        st.info(interpretation)
        st.subheader(" Why did the model predict this? (local SHAP explanation)")
        contrib = local_shap_contributions(features)
        if contrib is not None:
            st.dataframe(
                contrib.style.format({"SHAP contribution": "{:+.3f}"}),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "Positive values push the prediction toward 'resistant'; negative values "
                "push toward 'sensitive'. Computed with SHAP TreeExplainer on the exact "
                "deployed model — this is a real explanation, not a static importance chart."
            )
        else:
            st.caption("SHAP explanation unavailable (install `shap` to enable).")
        st.session_state.history.append({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "age": age, "gender": gender, "specimen": specimen,
            "organism": organism, "antibiotic": antibiotic,
            "resistance_probability_%": round(prob_percent, 1),
            "risk_category": label,
        })
        st.markdown("---")
        st.subheader("📥 Export this prediction")
        patient = {"Age": age, "Gender": gender, "Specimen": specimen,
                   "Organism": organism, "Antibiotic": antibiotic}
        csv_bytes = pd.DataFrame([{**patient, "Resistance_Probability_%": round(prob_percent, 1),
                                    "Risk_Category": label}]).to_csv(index=False).encode()
        colA, colB = st.columns(2)
        with colA:
            st.download_button("⬇️ Download as CSV", csv_bytes, file_name="amr_prediction.csv", mime="text/csv")
        with colB:
            try:
                pdf_bytes = build_pdf_report(patient, prob_percent, label, interpretation)
                st.download_button("⬇️ Download PDF Report", pdf_bytes, file_name="amr_prediction_report.pdf", mime="application/pdf")
            except Exception as e:
                st.caption(f"PDF export unavailable ({e}). Install `reportlab` to enable.")
        st.caption("Prediction based on a synthetic-data research prototype. Not for clinical use.")
    else:
        st.info("Set patient/sample details in the sidebar and click **Predict Resistance** to begin.")
with tab_perf:
    st.subheader("Held-out Test Set Performance")
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        c2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        c3.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
        c4.metric("F1-score", f"{metrics['f1_score']:.3f}")
        st.caption(f"Model: {metrics['model_type']}  ·  Train/Test split: {metrics['n_train']}/{metrics['n_test']}")
        colA, colB, colC = st.columns(3)
        for col, img, cap in [
            (colA, "roc_curve.png", "ROC Curve"),
            (colB, "pr_curve.png", "Precision–Recall Curve"),
            (colC, "confusion_matrix.png", "Confusion Matrix"),
        ]:
            try:
                col.image(img, caption=cap, use_container_width=True)
            except Exception:
                col.caption(f"{cap} not found — run generate_research_artifacts.py")

        with st.expander("Full classification report"):
            st.json(metrics["classification_report"])
    else:
        st.warning(
            "metrics.json not found. Run `python generate_research_artifacts.py` "
            "once (does not modify your model) to generate ROC/PR/confusion-matrix "
            "evidence for this tab."
        )
with tab_shap:
    st.subheader("Global Feature Importance (SHAP)")
    st.markdown(
        "Unlike a single static feature-importance bar chart, SHAP values quantify "
        "**each feature's direction and magnitude of contribution to individual "
        "predictions**, aggregated here across the full test set — this is the "
        "standard for explainable AI in clinical ML literature."
    )
    colA, colB = st.columns(2)
    try:
        colA.image("shap_bar.png", caption="Mean |SHAP value| per feature", use_container_width=True)
    except Exception:
        colA.caption("shap_bar.png not found — run generate_research_artifacts.py")
    try:
        colB.image("shap_summary.png", caption="SHAP summary (distribution & direction)", use_container_width=True)
    except Exception:
        colB.caption("shap_summary.png not found — run generate_research_artifacts.py")

    if shap_importance:
        st.markdown("**Ranked global importance:**")
        imp_df = pd.DataFrame(shap_importance.items(), columns=["Feature", "Mean |SHAP|"]).sort_values(
            "Mean |SHAP|", ascending=False)
        st.dataframe(imp_df, use_container_width=True, hide_index=True)
with tab_batch:
    st.subheader("Batch / Multi-Patient Prediction")
    st.markdown(
        "Upload a CSV with columns `age, gender, specimen, organism, antibiotic` "
        "to score many patients at once — useful for cohort-level stewardship review "
        "or demonstrating throughput in a portfolio."
    )
    st.download_button(
        "⬇️ Download sample input template",
        pd.DataFrame([{"age": 45, "gender": "F", "specimen": "urine", "organism": "E. coli", "antibiotic": "Ciprofloxacin"}]).to_csv(index=False).encode(),
        file_name="batch_template.csv", mime="text/csv",
    )
    uploaded = st.file_uploader("Upload patient batch (CSV)", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            required_cols = {"age", "gender", "specimen", "organism", "antibiotic"}
            missing = required_cols - set(batch_df.columns)
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                results = []
                for _, row in batch_df.iterrows():
                    feat = prepare_features(row["organism"], row["antibiotic"], row["specimen"], row["gender"], row["age"])
                    p = model.predict_proba(feat)[0][1] * 100
                    label, _ = risk_category(p)
                    results.append({**row.to_dict(), "resistance_probability_%": round(p, 1), "risk_category": label})
                results_df = pd.DataFrame(results)
                st.success(f"Scored {len(results_df)} patients.")
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download results", results_df.to_csv(index=False).encode(),
                    file_name="batch_predictions.csv", mime="text/csv",
                )
        except Exception as e:
            st.error(f"Could not process file: {e}")
with tab_history:
    st.subheader("Prediction History (this session)")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        colA, colB = st.columns(2)
        with colA:
            st.download_button("⬇️ Download history", hist_df.to_csv(index=False).encode(),
                                file_name="prediction_history.csv", mime="text/csv")
        with colB:
            if st.button("🗑️ Clear history"):
                st.session_state.history = []
                st.rerun()
    else:
        st.caption("No predictions made yet this session." 
with tab_about:
    st.subheader("Model Card")
    if model_card:
        st.json(model_card)
    else:
        st.caption("model_card.json not found — run generate_research_artifacts.py")

    st.subheader("Technology Stack")
    st.markdown(
        "- **ML:** scikit-learn (RandomForestClassifier), SHAP for explainability\n"
        "- **App:** Streamlit\n"
        "- **Reporting:** ReportLab (PDF), pandas (CSV)\n"
        "- **Data:** synthetic antibiogram-style dataset (n=500)"
    )
    st.subheader("Ethical AI & Clinical Disclaimer")
    st.warning(
        "This tool is a research/educational prototype. It must not be used to make "
        "unsupervised clinical treatment decisions. All outputs should be reviewed by a "
        "qualified clinician or pharmacist alongside microbiological culture and "
        "susceptibility testing and local antibiogram data."
    )
    st.subheader("Research References")
    st.markdown(
        "- WHO. *Global Antimicrobial Resistance and Use Surveillance System (GLASS) Report.*\n"
        "- CDC. *Antibiotic Resistance Threats in the United States.*\n"
        "- Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions* (SHAP).\n"
        "- Add your own conference/publication citations here."
    )
    st.subheader("Citation")
    st.code(
        "Javed, C. (2026). AI-Powered Antibiotic Resistance Predictor: "
        "A Machine Learning Prototype for Antimicrobial Stewardship [Software].",
        language="text",
    )
    st.markdown("---")
    st.markdown(
        "- **Project:** AMR Prediction Tool (Research Prototype)\n"
        "- **Developer:** Chand Javed\n"
        "- **GitHub:** [Link to your repository](https://github.com/chandjavedatrcb17n18-ux/AMR-Predictor)\n"
        "- **Contact:** chandjavedatrcb17n18@gmail.com"
    )

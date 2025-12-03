import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# --- Load model artifacts ---
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# --- Streamlit page settings ---
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

# --- Header ---
st.markdown("""
# 🩺 Smart Diabetes Risk Predictor  
Estimate your diabetes risk based on your clinical data.Powered by Machine Learning & SHAP for transparent decision-making.
""")

# --- Input form ---
st.markdown("### 📝 Enter Your Health Information")

with st.form("prediction_form"):
    inputs = []
    cols = st.columns(2)

    for i, name in enumerate(feature_names):
        minv, maxv, defv = 0, 200, 100
        if name == "Pregnancies":
            minv, maxv, defv = 0, 10, 1
        elif name == "BMI":
            minv, maxv, defv = 10.0, 60.0, 25.0
        elif name == "DiabetesPedigreeFunction":
            minv, maxv, defv = 0.0, 2.5, 0.5
        elif name == "Age":
            minv, maxv, defv = 10, 100, 30

        col = cols[i % 2]
        val = col.number_input(
            f"{name}:", min_value=minv, max_value=maxv, value=defv,
            step=0.1 if isinstance(defv, float) else 1,
            format="%.1f" if isinstance(defv, float) else "%d"
        )
        inputs.append(val)

    submit = st.form_submit_button("🚀 Predict Risk")

# --- Predict and explain ---
if submit:
    st.markdown("## 🧠 Prediction Result")

    user_input = np.array(inputs).reshape(1, -1)
    scaled_input = scaler.transform(user_input)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction:
        st.error(f"⚠️ **High risk of diabetes**\n\n🩸 **Probability:** `{probability:.2%}`")
    else:
        st.success(f"✅ **Low risk of diabetes**\n\n🩺 **Probability:** `{probability:.2%}`")

    # --- SHAP Explanation ---
    st.markdown("### 🔍 Feature Importance (via SHAP)")
    try:
        explainer = shap.Explainer(model, masker=scaler.transform(np.zeros((1, len(inputs)))))
        shap_values = explainer(scaled_input)

        # Bar plot
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)

        # Verbal explanation
        st.markdown("### 🧬 Top Contributing Features")
        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_values[0].values,
            "Your Value": user_input[0]
        }).sort_values(by="SHAP Value", key=abs, ascending=False)

        for i in range(3):
            row = shap_df.iloc[i]
            f, v, s = row["Feature"], row["Your Value"], row["SHAP Value"]
            direction = "increased" if s > 0 else "decreased"
            st.markdown(f"🔹 **{f} = {v}** → This feature **{direction}** your diabetes risk.")

            if f == "Glucose":
                st.markdown("🩸 _Glucose is the strongest indicator for diabetes risk._")
            elif f == "BMI":
                st.markdown("⚖️ _Higher BMI correlates with increased insulin resistance._")
            elif f == "Age":
                st.markdown("👵 _Older individuals generally have higher risk._")
            elif f == "Insulin":
                st.markdown("💉 _Abnormal insulin levels may signal diabetes._")

    except Exception as e:
        st.warning("⚠️ SHAP explanation failed.")
        st.error(str(e))

    st.info("💡 _This result is an estimate. Please consult a medical professional for diagnosis._")

# --- Footer ---
st.markdown("""
---
#### 📌 Disclaimer  
This tool is a machine learning demo made for educational purposes only.  
It is **not** a substitute for professional medical diagnosis or advice.  

_Built by **Yash Bharvada** (CSE) with ❤️ using Streamlit, Scikit-learn, and SHAP._
""")

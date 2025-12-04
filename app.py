import json
import numpy as np
import pandas as pd
import streamlit as st

from diabetes_app import HealthInput, get_predictor
from diabetes_app.analytics import emit_event
from diabetes_app.i18n import get_translator
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from diabetes_app.model import load_artifacts

_predict = get_predictor()
model, scaler, feature_names = None, None, None
try:
    # Access feature names by calling once
    meta = _predict([0] * 8)
    feature_names = meta["feature_names"]
except Exception:
    pass

df = pd.read_csv("pima.csv")
zero_columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_columns] = df[zero_columns].replace(0, np.nan)
df[zero_columns] = df[zero_columns].fillna(df[zero_columns].median())
feature_defaults = df.drop("Outcome", axis=1).median().to_dict()
MAIN_FEATURES = [f for f in [
    "Glucose",
    "Insulin",
    "BloodPressure",
    "SkinThickness",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Pregnancies",
] if f in feature_names]

# --- Streamlit page settings ---
st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")

# --- Header ---
st.markdown("""
<div style="text-align:center;">
  <h1 style="margin-bottom:0;">Diabetes Risk Predictor</h1>
  <div style="opacity:0.8;">Interactive health assessment tool for early diabetes detection</div>
</div>
""", unsafe_allow_html=True)

t = get_translator("en")

# Determine current view and theme from query params
try:
    q = st.query_params
    view = q.get("view", "input")
    theme_param = q.get("theme", "Dark")
except Exception:
    q = st.experimental_get_query_params()
    view = q.get("view", ["input"])[0]
    theme_param = q.get("theme", ["Dark"])[0]

with st.sidebar:
    theme = st.radio("Theme", ["Light", "Dark"], index=(0 if theme_param == "Light" else 1), horizontal=True)
    lang = st.selectbox("Language", ["en", "hi"], index=(0 if q.get("lang", ["en"]) [0] == "en" else 1))
    st.markdown("### Settings")
    # Persist theme/lang into URL for future sessions
    try:
        st.query_params["theme"] = theme
        st.query_params["lang"] = lang
    except Exception:
        st.experimental_set_query_params(view=view, theme=theme, lang=lang)

is_dark = theme == "Dark"
primary_bg = "#0E1117" if is_dark else "#FFFFFF"
primary_text = "#FAFAFA" if is_dark else "#0E1117"
accent = "#7c3aed"  # unified accent for both themes
panel_bg = "#1F2937" if is_dark else "#FFFFFF"
panel_border = "rgba(255,255,255,0.15)" if is_dark else "rgba(0,0,0,0.1)"
pill_bg = "#2f3b52" if is_dark else "#eef2ff"
pill_border = "#3b4a66" if is_dark else "#dde3ff"
pill_text = "#e5e7eb" if is_dark else "#111827"
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {primary_bg};
        color: {primary_text};
    }}
    .block-container {{
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1200px;
    }}
    .stButton>button {{
        background-color: {accent} !important;
        color: #ffffff !important;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        border: none;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2);
        font-weight: 600;
    }}
    .stButton button,
    form button,
    div[data-testid="baseButton-primary"] > button,
    div[data-testid="stFormSubmitButton"] > button {{
        background-color: {accent} !important;
        color: #ffffff !important;
        font-weight: 600;
    }}
    .stButton>button:focus {{
        outline: 2px solid #ffffff;
        box-shadow: 0 0 0 3px rgba(124,58,237,0.4);
    }}
    .stNumberInput input {{
        font-size: 1rem;
        padding: 0.4rem;
    }}
    /* Ensure widget labels are readable in dark mode */
    .stSlider > label,
    .stNumberInput > label,
    div[data-testid="stWidgetLabel"] > label {{
        color: {primary_text} !important;
        font-weight: 600;
    }}
    h1, h2, h3 {{
        color: {primary_text};
    }}
    .card {{
        border: 1px solid {panel_border};
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: {panel_bg};
    }}
    .panel {{
        border: 1px solid {panel_border};
        border-radius: 12px;
        padding: 1rem;
        background: {panel_bg};
        color: {primary_text};
    }}
    .title {{
        font-weight: 600;
        margin-bottom: 0.25rem;
    }}
    .subtitle {{
        font-size: 0.9rem;
        opacity: 0.85;
        margin-bottom: 0.5rem;
    }}
    .pill {{
        display:inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 999px;
        font-size: 0.8rem;
        background:{pill_bg};
        border:1px solid {pill_border};
        color:{pill_text};
    }}
    .circle {{
        width: 120px;
        height: 120px;
        border-radius: 50%;
        display:flex;
        align-items:center;
        justify-content:center;
        border: 8px solid {accent};
        margin:auto;
        font-weight:700;
        color: {primary_text};
    }}
    div[data-testid="stProgressBar"] > div > div {{
        background-color: {accent} !important;
    }}
    div[role="alert"] {{
        color: {primary_text};
    }}
    @media (max-width: 768px) {{
        .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100%;
        }}
        .circle {{ width: 100px; height: 100px; border-width: 6px; }}
    }}
    </style>
""", unsafe_allow_html=True)

t = get_translator(lang)

# Fade-in animation on load
components.html(
    """
    <script>
    document.documentElement.style.opacity = 0;
    document.documentElement.style.transition = 'opacity 300ms';
    window.addEventListener('load', ()=>{document.documentElement.style.opacity = 1;});
    </script>
    """,
    height=0,
)

if view == "input":
    with st.form("prediction_form"):
        user_inputs = {}
        left, right = st.columns(2)
        defaults = st.session_state.get("user_inputs", {})

        with left:
            st.markdown(f"<div class='title' aria-label='{t('glucose_insulin')}'>{t('glucose_insulin')}</div><div class='subtitle'>Blood sugar and insulin levels</div>", unsafe_allow_html=True)
            g = st.slider(t("glucose"), 0, 300, int(defaults.get("Glucose", feature_defaults.get("Glucose", 110))))
            st.caption("Normal: 70-100 mg/dL (fasting)")
            i = st.slider(t("insulin"), 0, 300, int(min(defaults.get("Insulin", feature_defaults.get("Insulin", 80)), 300)))
            user_inputs["Glucose"] = g
            user_inputs["Insulin"] = i

        with right:
            st.markdown(f"<div class='title' aria-label='{t('cardio')}'>{t('cardio')}</div><div class='subtitle'>Blood pressure measurements</div>", unsafe_allow_html=True)
            bp = st.slider(t("bp"), 40, 140, int(max(40, min(defaults.get("BloodPressure", feature_defaults.get("BloodPressure", 80)), 140))))
            st.caption("Normal: 60-80 mmHg (diastolic)")
            st_val = st.slider(t("skin"), 0, 80, int(min(defaults.get("SkinThickness", feature_defaults.get("SkinThickness", 20)), 80)))
            user_inputs["BloodPressure"] = bp
            user_inputs["SkinThickness"] = st_val

        left2, right2 = st.columns(2)
        with left2:
            st.markdown(f"<div class='title' aria-label='{t('body_metrics')}'>{t('body_metrics')}</div><div class='subtitle'>Physical measurements</div>", unsafe_allow_html=True)
            bmi = st.slider(t("bmi"), 10.0, 60.0, float(defaults.get("BMI", feature_defaults.get("BMI", 25.0))), step=0.1)
            st.caption("Normal: 18.5-24.9")
            dpf = st.slider(t("dpf"), 0.0, 2.5, float(defaults.get("DiabetesPedigreeFunction", feature_defaults.get("DiabetesPedigreeFunction", 0.5))), step=0.01)
            user_inputs["BMI"] = bmi
            user_inputs["DiabetesPedigreeFunction"] = dpf

        with right2:
            st.markdown(f"<div class='title' aria-label='{t('demographics')}'>{t('demographics')}</div><div class='subtitle'>Personal information</div>", unsafe_allow_html=True)
            age = st.slider(t("age"), 10, 100, int(defaults.get("Age", feature_defaults.get("Age", 35))))
            preg = st.slider(t("preg"), 0, 15, int(defaults.get("Pregnancies", feature_defaults.get("Pregnancies", 0))))
            user_inputs["Age"] = age
            user_inputs["Pregnancies"] = preg

        submit = st.form_submit_button(t("calc_button"))

# --- Predict and explain ---
if view == "input" and submit:
    full_values = []
    # Use validation model if feature_names known
    if feature_names:
        # Construct validated input
        validated = HealthInput(**{fn: user_inputs.get(fn, int(feature_defaults.get(fn, 0))) for fn in feature_names})
        full_values = validated.to_array(feature_names)
    else:
        for name in user_inputs:
            full_values.append(user_inputs[name])
    with st.spinner("Calculating..."):
        result = _predict(full_values)
    st.session_state["prediction"] = result["prediction"]
    st.session_state["probability"] = result["probability"]
    st.session_state["user_inputs"] = user_inputs
    emit_event("predict", {"probability": result["probability"], "prediction": result["prediction"]})
    st.session_state["calc_ts"] = pd.Timestamp.utcnow().isoformat()
    # Persist inputs & timestamp to localStorage, then navigate to results with animation
    components.html(
        f"""
        <script>
        try {{
            localStorage.setItem('diabetes_inputs', '{json.dumps(user_inputs)}');
            localStorage.setItem('calc_ts', '{st.session_state['calc_ts']}');
        }} catch(e) {{}}
        </script>
        """,
        height=0,
    )
    try:
        st.query_params["view"] = "results"
    except Exception:
        st.experimental_set_query_params(view="results")
    st.rerun()

if view == "results":
    if "probability" not in st.session_state:
        st.info("Provide input data and click Calculate Risk Score.")
    else:
        prob = st.session_state["probability"]
        pred = st.session_state["prediction"]
        ui = st.session_state["user_inputs"]

        st.markdown(f"<div class='title' aria-label='{t('results_title')}'>{t('results_title')}</div><div class='subtitle'>{t('results_sub')}</div>", unsafe_allow_html=True)
        cols_top = st.columns([1,3])
        with cols_top[0]:
            st.markdown(f"<div class='circle'>{int(prob*100)}%</div>", unsafe_allow_html=True)
        with cols_top[1]:
            label = t("high_risk") if pred == 1 else t("low_risk")
            st.subheader(label)
            st.progress(int(prob*100))
            if pred == 1:
                st.error("Your risk factors exceed healthy ranges. Consider medical guidance.")
            else:
                st.success("Your risk factors are within healthy ranges. Continue a healthy lifestyle.")
            st.caption(f"Calculated at {st.session_state.get('calc_ts','')} UTC")

        # Risk gauge using actual probability
        theme_is_dark = (theme == "Dark")
        paper = "#0E1117" if theme_is_dark else "#FFFFFF"
        fontc = "#FAFAFA" if theme_is_dark else "#0E1117"
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={"valueformat": ".0%", "font": {"color": fontc}},
            gauge={
                "axis": {"range": [0, 1], "tickformat": ".0%", "tickcolor": fontc},
                "bar": {"color": accent},
                "steps": [
                    {"range": [0, 0.2], "color": "#16a34a"},
                    {"range": [0.2, 0.5], "color": "#f59e0b"},
                    {"range": [0.5, 1.0], "color": "#ef4444"},
                ],
            },
            domain={"x": [0, 1], "y": [0, 1]},
        ))
        gauge.update_layout(title={"text": "Overall Risk Level", "font": {"color": fontc}}, paper_bgcolor=paper, plot_bgcolor=paper)
        st.plotly_chart(gauge, width='stretch')

        st.markdown(f"<div class='title'>{t('breakdown')}</div><div class='subtitle'>{t('breakdown_sub')}</div>", unsafe_allow_html=True)
        rcols = st.columns(2)
        def badge(val):
            return f"<span class='pill'>{val}</span>"
        def risk_tag(feature, val):
            if feature == "Glucose":
                if val < 100:
                    return "LOW RISK"
                elif val < 126:
                    return "MEDIUM RISK"
                else:
                    return "HIGH RISK"
            if feature == "BMI":
                if val < 25:
                    return "LOW RISK"
                elif val < 30:
                    return "MEDIUM RISK"
                else:
                    return "HIGH RISK"
            if feature == "BloodPressure":
                if 60 <= val <= 80:
                    return "LOW RISK"
                elif 80 < val <= 90 or 50 <= val < 60:
                    return "MEDIUM RISK"
                else:
                    return "HIGH RISK"
            if feature == "Age":
                if val < 40:
                    return "LOW RISK"
                elif val < 60:
                    return "MEDIUM RISK"
                else:
                    return "HIGH RISK"
            return "—"
        with rcols[0]:
            st.markdown(f"<div class='panel'>Glucose Level {badge(ui.get('Glucose',''))}<br><small>{risk_tag('Glucose', ui.get('Glucose',0))}</small></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='panel'>Blood Pressure {badge(ui.get('BloodPressure',''))}<br><small>{risk_tag('BloodPressure', ui.get('BloodPressure',0))}</small></div>", unsafe_allow_html=True)
        with rcols[1]:
            st.markdown(f"<div class='panel'>BMI {badge(ui.get('BMI',''))}<br><small>{risk_tag('BMI', ui.get('BMI',0))}</small></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='panel'>Age {badge(ui.get('Age',''))}<br><small>{risk_tag('Age', ui.get('Age',0))}</small></div>", unsafe_allow_html=True)

        st.markdown(f"<div class='title'>{t('recommendations')}</div>", unsafe_allow_html=True)
        def recommendations(probability, values):
            recs = []
            if probability < 0.2:
                recs += [
                    "Maintain balanced diet with whole grains and lean proteins.",
                    "Continue regular physical activity (150 min/week).",
                    "Annual screening for blood glucose.",
                ]
            elif probability < 0.5:
                recs += [
                    "Reduce refined sugars; increase fiber intake.",
                    "Add 2–3 strength sessions weekly alongside cardio.",
                    "Quarterly monitoring of fasting glucose.",
                ]
            else:
                recs += [
                    "Consult a healthcare provider for diagnostic testing.",
                    "Adopt a medically supervised nutrition plan.",
                    "Track glucose more frequently (weekly).",
                ]
            # Personalize by feature values
            if values.get("BMI", 0) >= 30:
                recs.append("Target 5–10% weight reduction over 6 months.")
            if values.get("Glucose", 0) >= 126:
                recs.append("Prioritize fasting glucose control and reduce high-glycemic foods.")
            if values.get("BloodPressure", 0) >= 90:
                recs.append("Limit sodium intake; monitor blood pressure weekly.")
            return recs

        for item in recommendations(prob, ui):
            st.write(f"- {item}")

        # Validation messages
        st.caption("Recommendations adapt to your risk and specific metrics (e.g., BMI, Glucose, BP).")

        vis_tabs = st.tabs(["Feature Importance", "Correlation Matrix", "Interactive Explorer", "Risk Contribution"])
        with vis_tabs[0]:
            try:
                m, s, fn = load_artifacts()
                if hasattr(m, "feature_importances_"):
                    imp = m.feature_importances_
                elif hasattr(m, "coef_"):
                    imp = abs(m.coef_[0])
                else:
                    imp = None
                if imp is not None:
                    df_imp = pd.DataFrame({"Feature": fn, "Importance": imp}).sort_values("Importance", ascending=False)
                    fig = px.bar(df_imp, x="Feature", y="Importance", title="Feature Importance Ranking", color="Importance", color_continuous_scale=[[0, accent],[1, accent]])
                    fig.update_layout(xaxis_title="Feature", yaxis_title="Importance", legend_title="")
                    st.plotly_chart(fig, width='stretch')
                    top = df_imp.iloc[0]
                    st.write(f"Top contributor: {top['Feature']}.")
                else:
                    st.info("Importance not available for this model.")
            except Exception as e:
                st.warning("Could not render importance.")
                st.write(str(e))
        with vis_tabs[1]:
            try:
                corr = df[[c for c in feature_names if c in df.columns]].corr()
                fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale="RdBu", zmid=0))
                fig.update_layout(title="Feature Correlation Matrix", xaxis_title="Feature", yaxis_title="Feature")
                st.plotly_chart(fig, width='stretch')
                st.write("Positive values indicate direct relationships; negative values indicate inverse relationships.")
            except Exception as e:
                st.warning("Could not render correlation matrix.")
                st.write(str(e))
        with vis_tabs[2]:
            cols_int = st.columns(2)
            x_choice = cols_int[0].selectbox("X Axis", [c for c in feature_names if c in df.columns], index=0)
            y_choice = cols_int[1].selectbox("Y Axis", [c for c in feature_names if c in df.columns], index=1)
            fig = px.scatter(df, x=x_choice, y=y_choice, color=df["Outcome"].astype(str), title="Interactive Feature Explorer", labels={"color": "Outcome"})
            fig.update_layout(xaxis_title=x_choice, yaxis_title=y_choice)
            st.plotly_chart(fig, width='stretch')
            st.write("Use this plot to explore relationships between features.")
        with vis_tabs[3]:
            try:
                # Risk contribution for current input using model coefficients (logit domain)
                m, s, fn = load_artifacts()
                if hasattr(m, "coef_") and hasattr(m, "intercept_"):
                    X_df = pd.DataFrame([ui.get(f, 0) for f in fn]).T
                    X_df.columns = fn
                    X_scaled = s.transform(X_df)
                    contrib = (m.coef_[0] * X_scaled[0]).tolist()
                    base = float(m.intercept_[0])
                    df_wf = pd.DataFrame({"Feature": fn, "Contribution": contrib})
                    df_wf = df_wf.sort_values("Contribution", key=abs, ascending=False)
                    fig = go.Figure(go.Waterfall(
                        measure=["relative"] * len(df_wf) + ["total"],
                        x=list(df_wf["Feature"]) + ["Total"],
                        text=[f"{v:+.2f}" for v in df_wf["Contribution"]] + [""],
                        y=list(df_wf["Contribution"]) + [sum(df_wf["Contribution"]) + base],
                    ))
                    fig.update_layout(title="Risk Contribution (logit)", yaxis_title="Contribution", paper_bgcolor=paper, plot_bgcolor=paper, font={"color": fontc})
                    st.plotly_chart(fig, width='stretch')
                    st.caption("Bars show how your values shift the model's logit; total maps to predicted probability via sigmoid.")
                else:
                    st.info("Risk contribution view available for linear models with coefficients.")
            except Exception as e:
                st.warning("Could not render risk contribution.")
                st.write(str(e))

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Modify inputs"):
                st.toast("Navigating to input…")
                components.html(
                    f"""
                    <script>
                    try {{
                        localStorage.setItem('diabetes_inputs', '{json.dumps(ui)}');
                        localStorage.setItem('return_to','results');
                    }} catch(e) {{}}
                    </script>
                    """,
                    height=0,
                )
                try:
                    st.query_params["view"] = "input"
                except Exception:
                    st.experimental_set_query_params(view="input")
                st.rerun()
        with b2:
            if st.button("Recalculate"):
                st.toast("Modify inputs and recalculate")
                components.html(
                    f"""
                    <script>
                    try {{
                        localStorage.setItem('diabetes_inputs', '{json.dumps(ui)}');
                        localStorage.setItem('return_to','results');
                    }} catch(e) {{}}
                    </script>
                    """,
                    height=0,
                )
                try:
                    st.query_params["view"] = "input"
                except Exception:
                    st.experimental_set_query_params(view="input")
                st.rerun()

    st.info("💡 This result is an estimate. Please consult a medical professional.")

# --- Footer ---
st.markdown("""
---
Disclaimer: This tool is for educational purposes only and should not replace professional medical advice. Always consult with a qualified healthcare provider for proper diagnosis and treatment.
""")

"""
app.py  —  Streamlit interactive dashboard
Run: streamlit run app.py
"""
import json
import numpy as np
import streamlit as st
import joblib

st.set_page_config(
    page_title="Market Direction Predictor",
    page_icon="📈",
    layout="centered",
)

# ── Load model (cached so it only reads disk once per session) ─────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("models/custom_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    with open("models/feature_names.json") as f:
        features = json.load(f)
    return model, scaler, features

model, scaler, FEATURES = load_artifacts()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📈 Market Direction Predictor")
st.markdown("Custom Logistic Regression trained on synthetic Nifty-style data.")
st.divider()

# ── Reload button (busts Streamlit RAM cache after retraining) ─────────────────
if st.button("🔄 Reload Model from Disk"):
    st.cache_resource.clear()
    st.rerun()

st.subheader("Input Features")

col1, col2, col3 = st.columns(3)

with col1:
    ema_crossover = st.slider(
        "EMA Crossover (9 - 20)",
        min_value=-50.0,
        max_value=50.0,
        value=0.0,
        step=0.5,
        help="Positive = 9-EMA above 20-EMA (bullish), Negative = bearish"
    )

with col2:
    volume_pct = st.slider(
        "Volume % Change",
        min_value=-1.0,
        max_value=5.0,
        value=0.0,
        step=0.05,
        help="% change in volume vs previous period, output is a decimal fraction — 0.5 means 50% up"

    )

with col3:
    ema_momentum = st.slider(
        "EMA Momentum",
        min_value=-5.0,
        max_value=5.0,
        value=0.0,
        step=0.1,
        help="Rate of change of the EMA Crossover spread"
    )

st.divider()

# ── Prediction ─────────────────────────────────────────────────────────────────
raw_input    = np.array([[ema_crossover, volume_pct, ema_momentum]])
scaled_input = scaler.transform(raw_input)
prob_up      = model.predict_proba(scaled_input)[0]
prob_down    = 1 - prob_up
prediction   = "📈 MARKET UP" if prob_up >= 0.5 else "📉 MARKET DOWN"
confidence   = max(prob_up, prob_down) * 100

st.subheader("Prediction")
if prob_up >= 0.5:
    st.success(f"### {prediction}")
else:
    st.error(f"### {prediction}")

col_a, col_b, col_c = st.columns(3)
col_a.metric("Confidence",    f"{confidence:.1f}%")
col_b.metric("P(UP)",         f"{prob_up * 100:.1f}%")
col_c.metric("P(DOWN)",       f"{prob_down * 100:.1f}%")

# ── Debug panel ────────────────────────────────────────────────────────────────
with st.expander("🔍 Debug — Scaled Input Values"):
    st.write("Raw input fed to model (after StandardScaler):")
    for name, val in zip(FEATURES, scaled_input[0]):
        st.write(f"  **{name}**: `{val:.4f}`")
    st.write(f"Model weights: `{model.weights}`")
    st.write(f"Model bias:    `{model.bias:.4f}`")

st.caption("Built with a custom NumPy Logistic Regression | Synthetic training data")

# Market Direction Predictor

> A fully containerized Machine Learning web application that predicts short-term market direction using a **custom Logistic Regression engine** built from scratch with NumPy — no sklearn black-box for the core model.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Baseline-F7931E?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧠 What This Project Does

This app ingests three market-derived technical indicators and outputs the **real-time probability of an upward or downward price move** for the next session.

The prediction engine is a **Logistic Regression classifier implemented from scratch** — sigmoid activation, binary cross-entropy loss, and vectorized gradient descent — validated side-by-side against scikit-learn's production implementation. Both converge to identical accuracy, confirming the underlying calculus and matrix operations are correct.

---

## 🏗️ Architecture

```
ml_market_app/
├── model.py              # Custom Logistic Regression (pure NumPy)
├── train.py              # Data pipeline, feature engineering, model export
├── app.py                # Streamlit interactive dashboard
├── models/               # Auto-generated at build time
│   ├── custom_model.pkl
│   ├── scaler.pkl
│   └── feature_names.json
├── Dockerfile            # Multi-stage build (builder + runtime)
├── docker-compose.yml
└── requirements.txt
```

---

## ⚙️ Core Features

- **Custom ML Engine** — Logistic Regression via Gradient Descent, coded in pure NumPy with no library abstractions
- **Stationary Feature Engineering** — EMA Crossover spread and Volume % Change keep inputs scale-invariant for stable gradient descent
- **Regime-Switching Synthetic Data** — Alternating bull/bear price blocks create detectable EMA signals for the model to learn
- **Stratified Train/Test Split** — Preserves class balance across both sets, eliminating label leakage from data imbalance
- **Interactive Streamlit UI** — Real-time probability output with live sliders and confidence metrics
- **Cache-Bust Button** — Reload updated model from disk without restarting the Streamlit server
- **Debug Panel** — Inspect scaled inputs and model weights live in the browser
- **Multi-Stage Docker Build** — Builder stage installs dependencies; runtime stage ships a lean image with model pre-trained inside

---

## 🚀 Quick Start

### Option A — Docker (Recommended)

```bash
# Clone the repo
git clone https://github.com/JoshiSwastik/market-direction-predictor.git
cd market-direction-predictor

# Build image and run (model trains automatically during Docker build)
docker compose up --build
```

Open **http://localhost:8501** in your browser. To stop: `docker compose down`

### Option B — Local Python

```bash
# 1. Create and activate virtual environment
python -m venv ai_env
source ai_env/bin/activate        # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model — generates models/*.pkl
python train.py

# 4. Launch the Streamlit app
streamlit run app.py
```

---

## 📊 Input Features

| Feature | Description | Bullish Signal |
|---|---|---|
| **EMA Crossover** | 9-EMA minus 20-EMA price spread | Positive (9-EMA above 20-EMA) |
| **Volume % Change** | % change in volume vs prior period | Spike above 0 |
| **EMA Momentum** | Rate of change of the EMA spread | Expanding (positive) |

---

## 🔬 Model Results

Trained on 1,000 days of regime-switching synthetic data with ~52% UP / 48% DOWN class balance:

| Metric | Custom NumPy Model | Scikit-Learn Baseline |
|---|---|---|
| Accuracy | ~55% | ~55% |
| Final Cost | ~0.685 | — |
| Class Balance | Stratified | Stratified |

> 55% accuracy is the realistic ceiling for a 3-feature linear model on synthetic price data. In production, performance improves with real OHLCV data, additional features (RSI, VWAP deviation, order flow), and non-linear models.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Core | NumPy (custom implementation) |
| Baseline Validation | scikit-learn |
| Data Pipeline | Pandas |
| Model Serialization | joblib |
| Web UI | Streamlit |
| Containerization | Docker + Docker Compose |
| Language | Python 3.11 |

---

## Key Engineering Decisions

**Why implement Logistic Regression from scratch?**
To demonstrate working knowledge of the underlying mathematics — sigmoid activation, binary cross-entropy loss, and vectorized gradient descent — rather than treating ML as a black box.

**Why EMA Crossover spread instead of raw EMAs?**
Raw EMA values are non-stationary and scale indefinitely with price. The spread `9_EMA - 20_EMA` stays in a consistent range regardless of price level, giving gradient descent stable inputs to converge on.

**Why regime-switching synthetic data?**
Pure random walks produce labels with no learnable signal — the model trains on noise. Alternating bull/bear regimes create genuine EMA crossover patterns the model can detect.

**Why multi-stage Docker build?**
Separating build and runtime stages keeps the final image lean. `RUN python train.py` bakes the trained model directly into the image — zero startup delay, deploy-anywhere portability.

---

## 👤 Author

**Swastik Joshi** — Infrastructure & Network Engineer | Cloud computing and Machine learning enthusiast 
- GitHub: [@JoshiSwastik](https://github.com/JoshiSwastik)

---

*Built as a full ML engineering sprint: blank folder → custom algorithm → stationary feature engineering → interactive UI → containerized deployment.*

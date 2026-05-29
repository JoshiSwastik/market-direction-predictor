"""
train.py  —  Fixed synthetic data pipeline + model export
Run once to generate models/custom_model.pkl and models/scaler.pkl
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

from model import LogisticRegressionFromScratch

np.random.seed(42)
NUM_DAYS = 1000  # more data → more stable learning

# ── 1. Realistic synthetic price series ──────────────────────────────────────
# Regime-switching returns: alternating bull/bear blocks create detectable EMA signals
regime = np.repeat(np.random.choice([-1, 1], size=NUM_DAYS // 20), 20)[:NUM_DAYS]
daily_returns = regime * np.random.normal(0.003, 0.012, NUM_DAYS)  # ~0.3% mean per regime
close_prices = 150 * np.exp(np.cumsum(daily_returns))

# Volume correlated to absolute price move (like real markets)
abs_move = np.abs(daily_returns)
volumes = (abs_move * 3_000_000 + np.random.normal(500_000, 80_000, NUM_DAYS)).clip(50_000).astype(int)

df = pd.DataFrame({"Close": close_prices, "Volume": volumes})

# ── 2. Feature Engineering ───────────────────────────────────────────────────
df["9_EMA"]  = df["Close"].ewm(span=9,  adjust=False).mean()
df["20_EMA"] = df["Close"].ewm(span=20, adjust=False).mean()
df["EMA_Crossover"]      = df["9_EMA"] - df["20_EMA"]       # stationary spread
df["Volume_Pct_Change"]  = df["Volume"].pct_change()          # stationary volume

# EMA momentum: is the spread expanding or contracting?
df["EMA_Momentum"] = df["EMA_Crossover"].diff()

# Target: did the next close go UP?
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df.dropna(inplace=True)

# ── 3. Class balance diagnostic ───────────────────────────────────────────────
up_pct   = df["Target"].mean() * 100
down_pct = 100 - up_pct
print(f"\nClass Distribution → UP: {up_pct:.1f}%  DOWN: {down_pct:.1f}%")
if abs(up_pct - 50) > 10:
    print("WARNING: imbalanced labels — consider resampling or adjusting regime params")

# ── 4. Train / Test split ─────────────────────────────────────────────────────
FEATURES = ["EMA_Crossover", "Volume_Pct_Change", "EMA_Momentum"]
X = df[FEATURES].values
y = df["Target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y   # stratify preserves class ratio
)

# ── 5. Feature Scaling ────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 6. Train Custom Model ─────────────────────────────────────────────────────
print("\n--- Training Custom Model From Scratch ---\n")
custom_model = LogisticRegressionFromScratch(learning_rate=0.05, iterations=3000)
custom_model.fit(X_train_scaled, y_train)

custom_preds    = custom_model.predict(X_test_scaled)
custom_accuracy = accuracy_score(y_test, custom_preds)
print(f"\nCustom Model Accuracy: {custom_accuracy * 100:.2f}%")
print(classification_report(y_test, custom_preds, target_names=["DOWN", "UP"]))

# ── 7. Train Sklearn Baseline ─────────────────────────────────────────────────
print("--- Training Production Scikit-Learn Model ---")
sklearn_model = LogisticRegression(penalty=None, max_iter=3000)
sklearn_model.fit(X_train_scaled, y_train)
sklearn_preds    = sklearn_model.predict(X_test_scaled)
sklearn_accuracy = accuracy_score(y_test, sklearn_preds)
print(f"Scikit-Learn Accuracy: {sklearn_accuracy * 100:.2f}%")

print("\n==================================")
print(f"Custom Model Accuracy  : {custom_accuracy * 100:.2f}%")
print(f"Scikit-Learn Accuracy  : {sklearn_accuracy * 100:.2f}%")
print("==================================")

# ── 8. Export Artifacts ───────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(custom_model, "models/custom_model.pkl")
joblib.dump(scaler,       "models/scaler.pkl")

# Save feature names so app.py always uses the right order
import json
with open("models/feature_names.json", "w") as f:
    json.dump(FEATURES, f)

print("\nSaved → models/custom_model.pkl")
print("Saved → models/scaler.pkl")
print("Saved → models/feature_names.json")

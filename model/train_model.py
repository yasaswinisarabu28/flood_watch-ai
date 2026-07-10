"""
train_model.py
---------------
Trains Decision Tree, Random Forest, KNN, and XGBoost classifiers on the
flood dataset, compares their performance, and saves the best-performing
model (plus the scaler) to disk for use by the Flask app.

Run:
    python train_model.py

Requires (see requirements.txt):
    pandas, numpy, scikit-learn, xgboost, joblib
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] xgboost is not installed. Run: pip install xgboost")
    print("          Continuing with the other 3 models only.\n")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "flood_data.csv")
FEATURES = ["Annual_Rainfall", "Seasonal_Rainfall", "Cloud_Visibility",
            "Humidity", "Temperature", "Wind_Speed"]
TARGET = "Flood_Occurred"

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (helps KNN in particular; harmless for tree models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 2. Define models
# ---------------------------------------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
}

if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )

# ---------------------------------------------------------------
# 3. Train + evaluate
# ---------------------------------------------------------------
results = {}
trained_models = {}

for name, model in models.items():
    # KNN benefits from scaled data; tree models are scale-invariant but
    # using the same scaled input keeps the pipeline consistent/simple.
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[name] = round(acc * 100, 2)
    trained_models[name] = model

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, preds, target_names=["No Flood", "Flood"]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

# ---------------------------------------------------------------
# 4. Pick best model and save
# ---------------------------------------------------------------
best_name = max(results, key=results.get)
best_model = trained_models[best_name]

print("\n" + "=" * 40)
print("Model comparison (test accuracy):")
for name, acc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {name:<15} {acc}%")
print(f"\nBest model: {best_name} ({results[best_name]}%)")
print("=" * 40)

joblib.dump(best_model, os.path.join(os.path.dirname(__file__), "flood_model.pkl"))
joblib.dump(scaler, os.path.join(os.path.dirname(__file__), "scaler.pkl"))

with open(os.path.join(os.path.dirname(__file__), "model_info.json"), "w") as f:
    json.dump({
        "best_model": best_name,
        "accuracy": results[best_name],
        "all_results": results,
        "features": FEATURES,
    }, f, indent=2)

print("\nSaved: flood_model.pkl, scaler.pkl, model_info.json")

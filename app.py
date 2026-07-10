
import os
import json
import joblib
import warnings
import datetime
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "flood_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
INFO_PATH = os.path.join(BASE_DIR, "model", "model_info.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "flood_data.csv")
LOG_PATH = os.path.join(BASE_DIR, "data", "predictions_log.json")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(INFO_PATH) as f:
    model_info = json.load(f)

FEATURES = model_info["features"]
XGBOOST_AVAILABLE = "XGBoost" in model_info["all_results"]

DISTRICTS = [
    "Krishna (Andhra Pradesh)",
    "East Godavari (Andhra Pradesh)",
    "Guntur (Andhra Pradesh)",
    "Patna (Bihar)",
    "Guwahati (Assam)"
]
DISTRICT_COORDS = {
    "Krishna (Andhra Pradesh)": (16.5062, 80.6480),
    "East Godavari (Andhra Pradesh)": (17.0005, 81.8040),
    "Guntur (Andhra Pradesh)": (16.3067, 80.4365),
    "Patna (Bihar)": (25.5941, 85.1376),
    "Guwahati (Assam)": (26.1445, 91.7362),
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


SETTINGS = {
    "name": "Disaster Management Analyst",
    "role": "Regional Coordinator",
    "active_model": model_info["best_model"],
    "retrain_schedule": "Monthly",
    "sms": True,
    "email": True,
    "priority": "High",
    "cloud_deployed": False,
    "scalability": 20,
    "weather_api": True,
    "local_feeds": False,
}



def _seed_log():
    """Create a handful of historical entries so pages aren't empty on first run."""
    rng = np.random.default_rng(7)
    rows = []
    start = datetime.date(2025, 2, 1)
    for i in range(8):
        d = start + datetime.timedelta(days=int(rng.integers(5, 40)) * (i + 1) // 3)
        district = DISTRICTS[i % len(DISTRICTS)]
        rainfall = float(rng.uniform(900, 3400))
        prob = float(rng.uniform(5, 97))
        tier, cls = _risk_tier(prob)
        rows.append({
            "date": d.isoformat(),
            "district": district,
            "rainfall": round(rainfall, 1),
            "prediction": "Flood" if prob >= 50 else "No Flood",
            "risk_level": tier,
            "risk_class": cls,
            "probability": round(prob, 1),
        })
    with open(LOG_PATH, "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def _load_log():
    if not os.path.exists(LOG_PATH):
        return _seed_log()
    with open(LOG_PATH) as f:
        return json.load(f)


def _append_log(entry):
    rows = _load_log()
    rows.append(entry)
    with open(LOG_PATH, "w") as f:
        json.dump(rows, f, indent=2)


@app.route("/clear-history", methods=["POST"])
def clear_history():
    with open(LOG_PATH, "w") as f:
        json.dump([], f, indent=4)

    return redirect(url_for("history"))

def _risk_tier(probability):
    """Map a 0-100 flood probability to a (label, css-class) risk tier."""
    if probability >= 66:
        return "High", "risk-high"
    if probability >= 33:
        return "Medium", "risk-medium"
    return "Low", "risk-low"



def _compute_scatter_points(max_points=150):
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["Flood_Occurred"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_scaled = scaler.transform(X_test)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        proba = model.predict(X_test_scaled).astype(float)

    preds = model.predict(X_test_scaled)
    error = (preds != y_test.values).astype(int)

    points = [{"x": round(float(p), 3), "y": int(e)} for p, e in zip(proba, error)]
    if len(points) > max_points:
        idx = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        points = [points[i] for i in idx]
    return points


SCATTER_POINTS = _compute_scatter_points()


MONTHLY_RAINFALL = [85, 95, 130, 190, 330, 540, 615, 575, 405, 215, 110, 88]



HEATMAP = {
    "Krishna (Andhra Pradesh)": [
        5, 6, 8, 10, 15, 22, 28, 26, 18, 10, 6, 4
    ],

    "East Godavari (Andhra Pradesh)": [
        6, 7, 9, 12, 18, 26, 34, 31, 22, 13, 8, 5
    ],

    "Guntur (Andhra Pradesh)": [
        3, 4, 6, 8, 12, 18, 24, 22, 15, 9, 5, 3
    ],

    "Patna (Bihar)": [
        2, 3, 5, 7, 13, 21, 29, 27, 19, 10, 5, 2
    ],

    "Guwahati (Assam)": [
        8, 10, 14, 18, 25, 34, 42, 40, 31, 20, 12, 8
    ]
}




@app.route("/")
def dashboard():
    log_rows = _load_log()
    return render_template(
        "dashboard.html",
        active="dashboard",
        model_name=model_info["best_model"],
        model_accuracy=model_info["accuracy"],
        region_count=len(DISTRICTS),
        prediction_count=len(log_rows),
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template(
            "predict.html",
            active="predict",
            districts=DISTRICTS,
            prediction=None,
            form_values=None,
        )

    try:
        values = [float(request.form[f]) for f in FEATURES]
    except (KeyError, ValueError):
        return render_template(
            "predict.html",
            active="predict",
            districts=DISTRICTS,
            prediction=None,
            form_values=request.form,
            error="Please enter valid numeric values for every field.",
        )

    X = pd.DataFrame([values], columns=FEATURES)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    if hasattr(model, "predict_proba"):
        proba = round(float(model.predict_proba(X_scaled)[0][1]) * 100, 1)
    else:
        proba = 100.0 if pred == 1 else 0.0

    tier, risk_class = _risk_tier(proba)
    district = request.form.get("District", DISTRICTS[0])

    _append_log({
        "date": datetime.date.today().isoformat(),
        "district": district,
        "rainfall": float(request.form["Annual_Rainfall"]),
        "prediction": "Flood" if pred == 1 else "No Flood",
        "risk_level": tier,
        "risk_class": risk_class,
        "probability": proba,
    })

    return render_template(
        "predict.html",
        active="predict",
        districts=DISTRICTS,
        prediction="Flood" if pred == 1 else "No Flood",
        probability=proba,
        risk_class=risk_class,
        risk_tier=tier,
        form_values=request.form,
    )


@app.route("/history")
def history():
    rows = _load_log()
    rows_sorted = sorted(rows, key=lambda r: r["date"], reverse=True)

    counts = {"High": 0, "Medium": 0, "Low": 0}
    for r in rows:
        counts[r["risk_level"]] = counts.get(r["risk_level"], 0) + 1

    
    rng = np.random.default_rng(11)
    predicted_series = [int(v) for v in np.clip(np.array(MONTHLY_RAINFALL) / 60 + rng.normal(0, 1, 12), 0, None)]
    actual_series = [int(max(0, v + rng.integers(-2, 3))) for v in predicted_series]

    return render_template(
        "history.html",
        active="history",
        log_rows=rows_sorted,
        months=MONTHS,
        predicted_series=predicted_series,
        actual_series=actual_series,
        risk_counts=[counts["High"], counts["Medium"], counts["Low"]],
    )


@app.route("/map")
def flood_map():
    rows = _load_log()
    latest_by_district = {}
    for r in rows:
        latest_by_district[r["district"]] = r  

    region_status = []
    for d in DISTRICTS:
        lat, lng = DISTRICT_COORDS[d]
        latest = latest_by_district.get(d)
        if latest:
            region_status.append({
                "district": d, "lat": lat, "lng": lng,
                "risk_level": latest["risk_level"], "risk_class": latest["risk_class"],
                "probability": latest["probability"],
            })
        else:
            region_status.append({
                "district": d, "lat": lat, "lng": lng,
                "risk_level": "Low", "risk_class": "risk-low", "probability": 10.0,
            })

    region_status.sort(key=lambda r: -r["probability"])

    return render_template(
        "map.html",
        active="map",
        region_status=region_status,
    )


@app.route("/analytics")
def analytics():
    model_names = list(model_info["all_results"].keys())
    model_accuracies = list(model_info["all_results"].values())

    heatmap_max = max(v for row in HEATMAP.values() for v in row) or 1

    return render_template(
        "analytics.html",
        active="analytics",
        model_names=model_names,
        model_accuracies=model_accuracies,
        xgboost_available=XGBOOST_AVAILABLE,
        months=MONTHS,
        monthly_rainfall=MONTHLY_RAINFALL,
        heatmap=HEATMAP,
        heatmap_max=heatmap_max,
        scatter_points=SCATTER_POINTS,
    )


@app.route("/settings", methods=["GET", "POST"])
def settings():
    saved = False
    if request.method == "POST":
        section = request.form.get("section")
        if section == "profile":
            SETTINGS["name"] = request.form.get("name", SETTINGS["name"])
            SETTINGS["role"] = request.form.get("role", SETTINGS["role"])
        elif section == "model":
            SETTINGS["active_model"] = request.form.get("active_model", SETTINGS["active_model"])
            SETTINGS["retrain_schedule"] = request.form.get("retrain_schedule", SETTINGS["retrain_schedule"])
        elif section == "alerts":
            SETTINGS["sms"] = "sms" in request.form
            SETTINGS["email"] = "email" in request.form
            SETTINGS["priority"] = request.form.get("priority", SETTINGS["priority"])
        elif section == "cloud":
            SETTINGS["cloud_deployed"] = "cloud_deployed" in request.form
            SETTINGS["scalability"] = int(request.form.get("scalability", SETTINGS["scalability"]))
        elif section == "sources":
            SETTINGS["weather_api"] = "weather_api" in request.form
            SETTINGS["local_feeds"] = "local_feeds" in request.form
        saved = True

    model_names = list(model_info["all_results"].keys())
    return render_template(
        "settings.html",
        active="settings",
        settings=SETTINGS,
        model_names=model_names,
        saved=saved,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

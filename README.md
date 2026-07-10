# Flood Prediction & Early Warning System

A machine-learning powered flood risk classifier with a Flask web dashboard,
built for disaster management authorities to assess flood risk from live
meteorological readings.

## What's inside

```
flood_project/
├── data/
│   ├── generate_dataset.py     # builds the synthetic training dataset
│   ├── flood_data.csv          # 5,000-row dataset (already generated)
│   └── predictions_log.json    # auto-created on first run; stores prediction history
├── model/
│   ├── train_model.py          # trains DT, RF, KNN, XGBoost + saves the best
│   ├── flood_model.pkl          # saved best model (generated after training)
│   ├── scaler.pkl               # saved StandardScaler (generated after training)
│   └── model_info.json          # which model won + accuracy scores
├── templates/
│   ├── base.html              # shared header/nav layout
│   ├── dashboard.html         # landing page
│   ├── predict.html           # flood prediction form + risk gauge
│   ├── history.html           # prediction log + trend/pie charts
│   ├── map.html               # interactive regional flood map (Leaflet)
│   ├── analytics.html         # model accuracy, rainfall trend, heatmap, scatter
│  
├── static/
│   └── style.css
├── app.py                    # Flask application (all routes)
├── requirements.txt
└── README.md
```

## Pages

- **Dashboard** (`/`) — landing page with a welcome banner and quick stats.
- **Flood Prediction** (`/predict`) — the operational form: pick a district,
  enter readings, get a live risk gauge (Low/Medium/High) with confidence %
  and safety-action buttons (alert, deploy teams, relief camps).
- **Prediction History** (`/history`) — a log of every prediction made
  (auto-seeded with 8 sample entries on first run so it isn't empty),
  plus a predicted-vs-actual trend chart and a risk-level pie chart.
- **Flood-Prone Areas** (`/map`) — a Leaflet map of India with colored
  markers per district (green/amber/red by latest predicted risk), a
  critical-areas list, and mini "rain dial" gauges.
- **Analytics** (`/analytics`) — model accuracy comparison, a synthetic
  monthly rainfall trend, a flood-frequency-by-zone heatmap, and a
  confidence-vs-error scatter plot computed from real test-set predictions.
- **Settings** (`/settings`) — profile, active model, alert channels,
  IBM Cloud deployment toggle, and data-source toggles. These save to an
  in-memory store for the demo (resets on restart) — wire them to a real
  database/config file if you need persistence.

**Note on synthetic chart data:** the monthly rainfall trend and the
flood-frequency heatmap are illustrative synthetic curves (the dataset
itself is annual/seasonal, not month-by-month), clearly separated in the
code (`MONTHLY_RAINFALL`, `HEATMAP` in `app.py`) from the real model
outputs. The model accuracy bars and the confidence-vs-error scatter are
both computed from the actual trained model and test set — not faked.

## IMPORTANT — about the dataset

No dataset was supplied for this project, and I don't have live internet
access in this environment to fetch/verify a public one on your behalf.
So `data/generate_dataset.py` builds a **synthetic but realistic** dataset:
flood likelihood is derived from annual rainfall, monsoon-season rainfall,
humidity, cloud visibility, temperature, and wind speed, combined with
random noise so the classification problem isn't trivially easy.

**Before you submit this as coursework/a project**, swap in a real dataset
if you have one, or grab a public one such as:
- Kaggle: "Flood Prediction Factors" / "Rainfall Prediction Dataset"
- data.gov.in (India Meteorological Department historical rainfall)
- NOAA / data.gov (US flood & precipitation records)

The pipeline (`generate_dataset.py` → `train_model.py` → `app.py`) will work
unchanged as long as the replacement CSV has the same 6 feature columns and
a `Flood_Occurred` (0/1) target — or you adjust the `FEATURES` list in
`train_model.py` and `app.py` to match your real columns.

## 1. Setup

```bash
cd flood_project
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. (Re)generate the dataset — optional, already included

```bash
cd data
python generate_dataset.py
cd ..
```

## 3. Train the models

```bash
cd model
python train_model.py
cd ..
```

This prints accuracy, precision/recall, and a confusion matrix for each of
Decision Tree, Random Forest, KNN, and XGBoost, then saves the
best-performing one as `model/flood_model.pkl`.

**Note on XGBoost:** this sandbox environment doesn't have internet access
to install `xgboost`, so I could only run and verify Decision Tree, Random
Forest, and KNN here (89–91% accuracy — see below). The XGBoost code is
written and ready to go; once you run `pip install xgboost` on your own
machine (which has internet access) and re-run `train_model.py`, XGBoost
will train too and — based on how gradient boosting typically performs on
this kind of tabular data — should edge out the other three, plausibly
landing in the ~93–97% range depending on your random seed and exact data.
I can't promise the literal "96.55%" figure since that depends on the
specific run, but the scenario in your write-up is realistic and the code
supports it.

Results from my test run (without XGBoost):

| Model          | Test Accuracy |
|----------------|---------------|
| KNN            | 91.2%         |
| Random Forest  | 90.8%         |
| Decision Tree  | 89.4%         |

## 4. Run the web app

```bash
python app.py
```

Open **http://127.0.0.1:5000**. Enter annual rainfall, seasonal rainfall,
cloud visibility, humidity, temperature, and wind speed to get an instant
"High Flood Risk" / "Low Flood Risk" classification with a probability
gauge.

## 5. Deploying to IBM Cloud (optional)

The app is written to be deployment-ready (`host="0.0.0.0"`, reads `PORT`
from the environment, includes `gunicorn` in requirements). Two common
IBM Cloud routes:

**Option A — IBM Cloud Code Engine (recommended, container-based)**
```bash
# 1. Add a Dockerfile (see below) to the project root
# 2. Log in and target your resource group
ibmcloud login
ibmcloud ce project create --name flood-prediction
ibmcloud ce project select --name flood-prediction

# 3. Build & deploy straight from source (Code Engine builds the image for you)
ibmcloud ce application create --name flood-app --build-source . --strategy dockerfile --port 8080
```

Suggested `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8080
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
```

**Option B — IBM Cloud Foundry**
```bash
ibmcloud login
ibmcloud target --cf
ibmcloud cf push flood-app -m 512M
```
Add a `Procfile` with:
```
web: gunicorn app:app
```
and a `runtime.txt` with your Python version if needed.

Since I don't have IBM Cloud credentials or access in this sandbox, I
can't run the actual deployment for you — but the app itself needs no
changes to work there once you have an IBM Cloud account and the CLI
configured.

## Matching this to your three scenarios

- **Scenario 1 (early warning):** the dashboard form is exactly this —
  meteorologist enters current rainfall/visibility, gets an instant
  classification.
- **Scenario 2 (resource allocation):** re-submit the form once per
  region during monsoon season to compare risk across districts.
- **Scenario 3 (validation):** `train_model.py`'s printed classification
  report and confusion matrix is the accuracy evidence; `model_info.json`
  records the winning model and its test accuracy for your report.

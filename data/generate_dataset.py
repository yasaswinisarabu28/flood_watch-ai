"""
generate_dataset.py
--------------------
Generates a synthetic but realistic meteorological dataset for flood
prediction, since a verified public dataset was not supplied.

Features:
    Annual_Rainfall     (mm)   - total yearly rainfall for the district
    Seasonal_Rainfall   (mm)   - monsoon-season (Jun-Sep) rainfall
    Cloud_Visibility    (km)   - visibility distance (low = heavy cloud cover)
    Humidity            (%)    - average relative humidity
    Temperature         (C)    - average temperature
    Wind_Speed          (km/h) - average wind speed

Target:
    Flood_Occurred (0 = No Flood, 1 = Flood)

The relationships below are built from well-known flood-risk drivers
(high rainfall + high humidity + low visibility + saturated soil signal
via seasonal rainfall ratio) with random noise, so classifiers have a
realistic, non-trivial decision boundary to learn (not a hand-drawn
rule that would make every model hit 100%).

Run:
    python generate_dataset.py
Produces:
    flood_data.csv  (5000 rows)
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000

# ---- Base meteorological features ----
annual_rainfall = np.random.normal(2200, 700, N).clip(400, 4800)

# seasonal (monsoon) rainfall is a noisy fraction of annual rainfall
seasonal_ratio = np.random.normal(0.55, 0.12, N).clip(0.2, 0.9)
seasonal_rainfall = (annual_rainfall * seasonal_ratio).clip(50, 3500)

# humidity rises with seasonal rainfall
humidity = (40 + (seasonal_rainfall / 3500) * 55 + np.random.normal(0, 6, N)).clip(20, 100)

# cloud visibility falls as humidity/rainfall rise (fog, heavy cloud cover)
cloud_visibility = (10 - (humidity / 100) * 7 - (seasonal_rainfall / 3500) * 2
                     + np.random.normal(0, 0.8, N)).clip(0.3, 10)

# temperature drops slightly during high-rainfall periods
temperature = (32 - (seasonal_rainfall / 3500) * 8 + np.random.normal(0, 3, N)).clip(10, 45)

# wind speed - mostly independent, mild positive link to rainfall
wind_speed = (8 + (seasonal_rainfall / 3500) * 20 + np.random.normal(0, 6, N)).clip(0, 70)

# ---- Build flood probability via a logistic combination ----
z = (
    3.2 * (annual_rainfall - 2200) / 700
    + 3.8 * (seasonal_rainfall - 1200) / 700
    + 2.0 * (humidity - 70) / 20
    - 2.5 * (cloud_visibility - 5) / 3
    + 0.8 * (wind_speed - 20) / 15
    + np.random.normal(0, 1.6, N)   # noise so it isn't perfectly separable
)
flood_prob = 1 / (1 + np.exp(-z / 3))
flood_occurred = (flood_prob > 0.5).astype(int)

df = pd.DataFrame({
    "Annual_Rainfall": annual_rainfall.round(1),
    "Seasonal_Rainfall": seasonal_rainfall.round(1),
    "Cloud_Visibility": cloud_visibility.round(2),
    "Humidity": humidity.round(1),
    "Temperature": temperature.round(1),
    "Wind_Speed": wind_speed.round(1),
    "Flood_Occurred": flood_occurred,
})

df.to_csv("flood_data.csv", index=False)
print("Saved flood_data.csv with shape:", df.shape)
print(df["Flood_Occurred"].value_counts(normalize=True))

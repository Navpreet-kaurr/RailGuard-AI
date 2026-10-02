import pandas as pd
import joblib


data = pd.read_csv(
    "data/processed/railguard_accidents.csv"
)

# Exclude 2024-25 because it is a partial year
completed_data = data.iloc[:-1].copy()

# Use the most recent 3 completed years
recent_years = completed_data.tail(3)

baseline_prediction = recent_years[
    "Consequential Train Accidents"
].mean()

print("\nRecent completed years:")
print(
    recent_years[
        ["Year", "Consequential Train Accidents"]
    ]
)

print(
    "\nBaseline predicted next-period accidents:",
    round(baseline_prediction, 2)
)

# Save the baseline value
prediction_info = {
    "method": "3-year historical average",
    "prediction": round(baseline_prediction, 2)
}

joblib.dump(
    prediction_info,
    "models/baseline_forecast.pkl"
)

print("\nPrediction information saved successfully!")
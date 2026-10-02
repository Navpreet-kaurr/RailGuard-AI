import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest


def detect_anomalies(data):

    features = data[
        [
            "Consequential Train Accidents",
            "Accident_Change",
            "Percentage_Change"
        ]
    ].copy()

    model = IsolationForest(
        contamination=0.2,
        random_state=42
    )

    model.fit(features)

    data = data.copy()

    data["Anomaly_Score"] = model.decision_function(features)
    data["Anomaly"] = model.predict(features)

    data["Anomaly_Label"] = data["Anomaly"].map({
        -1: "Anomaly",
        1: "Normal"
    })

    return data, model


if __name__ == "__main__":

    data = pd.read_csv(
        "data/processed/railguard_accidents.csv"
    )

    # Remove 2024-25 because it is a partial year
    data = data[data["Year"] != "2024-25"].copy()

    data["Accident_Change"] = (
        data["Consequential Train Accidents"].diff()
    )

    data["Percentage_Change"] = (
        data["Consequential Train Accidents"]
        .pct_change() * 100
    )

    data = data.dropna()

    result, model = detect_anomalies(data)

    joblib.dump(
        model,
        "models/isolation_forest.pkl"
    )

    print("\nIsolation Forest model saved successfully!")

    print("\nAnomaly Detection Results:")

    print(
        result[
            [
                "Year",
                "Consequential Train Accidents",
                "Accident_Change",
                "Percentage_Change",
                "Anomaly_Score",
                "Anomaly_Label"
            ]
        ]
    )
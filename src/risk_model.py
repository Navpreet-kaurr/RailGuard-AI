import pandas as pd


def calculate_risk(data):

    data = data.copy()

    # Exclude incomplete 2024-25 data
    data = data[
        data["Year"] != "2024-25"
    ].copy()

    # Calculate year-to-year change
    data["Accident_Change"] = (
        data["Consequential Train Accidents"].diff()
    )

    # Calculate percentage change
    data["Percentage_Change"] = (
        data["Consequential Train Accidents"]
        .pct_change() * 100
    )

    # Calculate historical average
    average_accidents = (
        data["Consequential Train Accidents"].mean()
    )

    # Risk classification
    def classify_risk(accidents):

        if accidents >= average_accidents * 1.25:
            return "High"

        elif accidents >= average_accidents * 0.90:
            return "Medium"

        else:
            return "Low"

    data["Risk_Level"] = (
        data["Consequential Train Accidents"]
        .apply(classify_risk)
    )

    return data


if __name__ == "__main__":

    data = pd.read_csv(
        "data/processed/railguard_accidents.csv"
    )

    result = calculate_risk(data)

    print("\nHistorical Risk Analysis:")
    print(
        result[
            [
                "Year",
                "Consequential Train Accidents",
                "Risk_Level"
            ]
        ]
    )
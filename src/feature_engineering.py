import pandas as pd


def create_features(data):
    data = data.copy()

    # Previous year's accident count
    data["Previous_Year_Accidents"] = (
        data["Consequential Train Accidents"].shift(1)
    )

    # Change from previous year
    data["Accident_Change"] = (
        data["Consequential Train Accidents"].diff()
    )

    # Percentage change from previous year
    data["Percentage_Change"] = (
        data["Consequential Train Accidents"]
        .pct_change() * 100
    )

    # Two-year rolling average
    data["Two_Year_Average"] = (
        data["Consequential Train Accidents"]
        .rolling(window=2)
        .mean()
    )

    return data


if __name__ == "__main__":

    data = pd.read_csv(
        "data/processed/railguard_accidents.csv"
    )

    data = create_features(data)

    print("\nFeature Engineered Data:")
    print(data)

    print("\nColumns:")
    print(data.columns.tolist())
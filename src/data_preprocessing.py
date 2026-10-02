import pandas as pd


def load_accident_data():
    data = pd.read_csv("data/raw/accidents.csv")

    # Convert accident count to numeric
    data["Consequential Train Accidents"] = pd.to_numeric(
        data["Consequential Train Accidents"],
        errors="coerce"
    )

    # Create a numeric year for analysis
    data["Year_Start"] = data["Year"].str[:4].astype(int)

    return data


def save_processed_data(data):
    data.to_csv(
        "data/processed/railguard_accidents.csv",
        index=False
    )


if __name__ == "__main__":
    accidents = load_accident_data()

    print("Processed Accident Data:")
    print(accidents)

    print("\nData Types:")
    print(accidents.dtypes)

    print("\nMissing Values:")
    print(accidents.isnull().sum())

    save_processed_data(accidents)

    print("\nProcessed file saved successfully!")
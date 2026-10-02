import pandas as pd

print("Reading accidents.csv...")

accidents = pd.read_csv("data/raw/accidents.csv")

print("\nACCIDENTS DATA")
print(accidents)

print("\nAccident Columns:")
print(accidents.columns.tolist())


print("\nReading station_funds.csv...")

station_funds = pd.read_csv("data/raw/station_funds.csv")

print("\nSTATION FUNDS DATA")
print(station_funds)

print("\nStation Funds Columns:")
print(station_funds.columns.tolist())
import pandas as pd
import numpy as np

# GameSense AI - Data Preprocessing

INPUT_FILE = "data/raw/gamesense_champions.csv"
OUTPUT_FILE = "data/processed/gamesense_processed.csv"

# Load raw dataset

df = pd.read_csv(INPUT_FILE)

print("Raw dataset loaded")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# Convert timestamp

df["utc_timestamp"] = pd.to_datetime(
    df["utc_timestamp"],
    errors="coerce"
)

# Convert numeric columns

numeric_columns = [
    "rating_all",
    "acs_all",
    "adr_all",
    "kills_all",
    "deaths_all",
    "assists_all",
    "fk_all",
    "fd_all",
    "kast_all",
    "hs_pct_all",
    "two_k",
    "three_k",
    "four_k",
    "five_k",
    "clutch_1v1",
    "clutch_1v2",
    "clutch_1v3",
    "clutch_1v4",
    "clutch_1v5",
    "econ",
    "plants",
    "defuses"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Sort chronologically

df = df.sort_values(
    [
        "player_id",
        "utc_timestamp",
        "match_id",
        "game_id"
    ]
).reset_index(drop=True)

# Calculate K/D ratio

df["kd_ratio"] = np.where(
    df["deaths_all"] > 0,
    df["kills_all"] / df["deaths_all"],
    df["kills_all"]
)

# Calculate total multikills

df["multikills"] = (
    df["two_k"].fillna(0)
    + df["three_k"].fillna(0)
    + df["four_k"].fillna(0)
    + df["five_k"].fillna(0)
)

# Remove rows without essential performance data

before = len(df)

df = df.dropna(
    subset=[
        "rating_all",
        "acs_all",
        "adr_all",
        "kills_all",
        "deaths_all"
    ]
)

after = len(df)

print("\nRows removed due to missing essential statistics:", before - after)

# Remove duplicate player-map records

before = len(df)

df = df.drop_duplicates(
    subset=[
        "match_id",
        "game_id",
        "player_id"
    ]
)

after = len(df)

print("Duplicate rows removed:", before - after)

# Final dataset information

print("\nPREPROCESSING COMPLETE")

print("Final rows:", len(df))
print("Final columns:", len(df.columns))
print("Unique players:", df["player_id"].nunique())
print("Unique matches:", df["match_id"].nunique())
print("Missing values:", df.isna().sum().sum())

# Save processed dataset

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)
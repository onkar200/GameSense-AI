import pandas as pd
import numpy as np

# GameSense AI - GameSense Tilt Index

INPUT_FILE = "data/features/gamesense_features.csv"
OUTPUT_FILE = "data/features/gamesense_gti.csv"

# Load feature dataset

df = pd.read_csv(INPUT_FILE)

print("Feature dataset loaded")
print("Rows:", len(df))

# Convert timestamp

df["utc_timestamp"] = pd.to_datetime(
    df["utc_timestamp"],
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

# Calculate multikill total

df["multikills"] = (
    df["two_k"].fillna(0)
    + df["three_k"].fillna(0)
    + df["four_k"].fillna(0)
    + df["five_k"].fillna(0)
)

# Calculate total rounds on the map

df["total_rounds"] = (
    df["map_score_team0"] +
    df["map_score_team1"]
)

# Calculate first-death rate

df["first_death_rate"] = np.where(
    df["total_rounds"] > 0,
    df["fd_all"] / df["total_rounds"],
    np.nan
)

# Create player groups

group = df.groupby("player_id", group_keys=False)

# Calculate previous 5-map baselines

baseline_columns = [
    "acs_all",
    "adr_all",
    "kd_ratio",
    "kast_all",
    "deaths_all",
    "first_death_rate",
    "multikills"
]

for column in baseline_columns:

    df[f"baseline_{column}"] = (
        group[column]
        .transform(
            lambda x: x.shift(1).rolling(
                window=5,
                min_periods=1
            ).mean()
        )
    )

# Calculate percentage change

def percentage_change(baseline, current):

    return np.where(
        baseline.notna()
        & (baseline != 0)
        & current.notna(),
        ((current - baseline) / baseline) * 100,
        np.nan
    )

# ACS decline

df["acs_decline"] = (
    percentage_change(
        df["baseline_acs_all"],
        df["acs_all"]
    ) * -1
)

# K/D decline

df["kd_decline"] = (
    percentage_change(
        df["baseline_kd_ratio"],
        df["kd_ratio"]
    ) * -1
)

# ADR decline

df["adr_decline"] = (
    percentage_change(
        df["baseline_adr_all"],
        df["adr_all"]
    ) * -1
)

# KAST decline

df["kast_decline"] = (
    df["baseline_kast_all"] -
    df["kast_all"]
)

# Death increase

df["death_increase"] = percentage_change(
    df["baseline_deaths_all"],
    df["deaths_all"]
)

# First-death rate increase

df["fd_rate_increase"] = percentage_change(
    df["baseline_first_death_rate"],
    df["first_death_rate"]
)

# Multikill decline

df["multikill_decline"] = (
    percentage_change(
        df["baseline_multikills"],
        df["multikills"]
    ) * -1
)

# Calculate previous map loss streak

def calculate_loss_streak(results):

    streaks = []
    streak = 0

    for result in results:

        if result == "Loss":
            streak += 1
        else:
            streak = 0

        streaks.append(streak)

    return pd.Series(
        streaks,
        index=results.index
    )

df["loss_streak"] = (
    df.groupby("player_id")["map_result"]
    .transform(calculate_loss_streak)
)

# Initialize GTI score

df["gti_score"] = 0

# ACS decline >= 20%

df.loc[
    df["acs_decline"] >= 20,
    "gti_score"
] += 20

# K/D decline >= 20%

df.loc[
    df["kd_decline"] >= 20,
    "gti_score"
] += 20

# ADR decline >= 15%

df.loc[
    df["adr_decline"] >= 15,
    "gti_score"
] += 15

# KAST decline >= 10 percentage points

df.loc[
    df["kast_decline"] >= 10,
    "gti_score"
] += 15

# Death increase >= 25%

df.loc[
    df["death_increase"] >= 25,
    "gti_score"
] += 10

# First-death rate increase >= 25%

df.loc[
    df["fd_rate_increase"] >= 25,
    "gti_score"
] += 10

# Multikill decline >= 25%

df.loc[
    df["multikill_decline"] >= 25,
    "gti_score"
] += 10

# Loss streak >= 2

df.loc[
    df["loss_streak"] >= 2,
    "gti_score"
] += 10

# Cap score at 100

df["gti_score"] = df["gti_score"].clip(
    upper=100
)

# Classify GTI

def classify_gti(score):

    if score <= 25:
        return "Stable"

    elif score <= 50:
        return "Mild Signs"

    elif score <= 75:
        return "Elevated"

    else:
        return "Strong Potential Tilt"

df["gti_level"] = df["gti_score"].apply(
    classify_gti
)

# Save GTI dataset

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# Display results

print("\nGTI CALCULATION COMPLETE")

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Players:", df["player_id"].nunique())

print("\nGTI distribution:")

print(
    df["gti_level"]
    .value_counts()
)

print("\nAverage GTI:")

print(
    round(
        df["gti_score"].mean(),
        2
    )
)

print("\nHighest GTI scores:")

display_columns = [
    "player_name",
    "team_name",
    "event",
    "map_name",
    "map_number",
    "map_result",
    "gti_score",
    "gti_level"
]

print(
    df[
        display_columns
    ]
    .sort_values(
        "gti_score",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)

print("\nSaved:")
print(OUTPUT_FILE)
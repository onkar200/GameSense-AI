import pandas as pd
import numpy as np

# GameSense AI - Death Contribution Analysis

INPUT_FILE = "data/features/gamesense_role_features.csv"
OUTPUT_FILE = "data/features/gamesense_death_contribution.csv"

# Load role-enhanced dataset

df = pd.read_csv(INPUT_FILE)

print("Role-enhanced dataset loaded")
print("Rows:", len(df))

# Convert numeric columns

numeric_columns = [
    "kills_all",
    "deaths_all",
    "assists_all",
    "fk_all",
    "fd_all",
    "kast_all",
    "rating_all",
    "map_score_team0",
    "map_score_team1"
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

# Calculate total rounds played on the map

df["total_rounds"] = (
    df["map_score_team0"]
    + df["map_score_team1"]
)

# Calculate assist rate

df["assist_rate"] = np.where(
    df["total_rounds"] > 0,
    df["assists_all"] / df["total_rounds"],
    np.nan
)

# Calculate first-death rate

df["first_death_rate"] = np.where(
    df["total_rounds"] > 0,
    df["fd_all"] / df["total_rounds"],
    np.nan
)

# Calculate first-death ratio

df["first_death_ratio"] = np.where(
    df["deaths_all"] > 0,
    df["fd_all"] / df["deaths_all"],
    np.nan
)

# Calculate first-kill / first-death ratio

df["fk_fd_ratio"] = np.where(
    df["fd_all"] > 0,
    df["fk_all"] / df["fd_all"],
    np.nan
)

# Calculate K/D ratio

df["kd_ratio"] = np.where(
    df["deaths_all"] > 0,
    df["kills_all"] / df["deaths_all"],
    df["kills_all"]
)

# Normalize contribution components

def min_max_normalize(series):

    minimum = series.min()
    maximum = series.max()

    if pd.isna(minimum) or pd.isna(maximum):

        return pd.Series(
            np.nan,
            index=series.index
        )

    if maximum == minimum:

        return pd.Series(
            0.5,
            index=series.index
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    )


df["assist_component"] = min_max_normalize(
    df["assist_rate"]
)

df["kast_component"] = min_max_normalize(
    df["kast_all"]
)

df["kd_component"] = min_max_normalize(
    df["kd_ratio"]
)

df["fk_fd_component"] = min_max_normalize(
    df["fk_fd_ratio"].replace(
        [np.inf, -np.inf],
        np.nan
    )
)

# Lower first-death ratio indicates better contribution

fd_component = min_max_normalize(
    df["first_death_ratio"]
)

df["first_death_component"] = (
    1 - fd_component
)

# Calculate Death Contribution Score

df["death_contribution_score"] = (
    df["assist_component"] * 25
    + df["kast_component"] * 20
    + df["kd_component"] * 20
    + df["fk_fd_component"] * 15
    + df["first_death_component"] * 20
)

# Check whether required contribution data is available

contribution_columns = [
    "assists_all",
    "kast_all",
    "kills_all",
    "deaths_all",
    "fk_all",
    "fd_all"
]

df["death_contribution_data_available"] = (
    df[contribution_columns]
    .notna()
    .all(axis=1)
).astype(int)

# Classify contribution level

def classify_contribution(score):

    if pd.isna(score):

        return "Insufficient Data"

    if score >= 67:

        return "High Contribution"

    elif score >= 34:

        return "Moderate Contribution"

    else:

        return "Low Contribution"


df["death_contribution_level"] = (
    df["death_contribution_score"]
    .apply(classify_contribution)
)

# Role-level contribution summary

role_summary = (
    df.groupby("player_role")
    .agg(
        players=("player_id", "nunique"),
        maps=("game_id", "nunique"),
        avg_assists=("assists_all", "mean"),
        avg_kast=("kast_all", "mean"),
        avg_first_death_rate=("first_death_rate", "mean"),
        avg_kd=("kd_ratio", "mean"),
        avg_fk_fd=("fk_fd_ratio", "mean"),
        avg_contribution_score=(
            "death_contribution_score",
            "mean"
        )
    )
    .reset_index()
)

print("\nROLE DEATH CONTRIBUTION SUMMARY")

print(
    role_summary
    .round(3)
    .to_string(index=False)
)

# Contribution distribution

print("\nDEATH CONTRIBUTION DISTRIBUTION")

print(
    df["death_contribution_level"]
    .value_counts()
)

# Display highest contribution examples

display_columns = [
    "player_name",
    "team_name",
    "map_name",
    "primary_agent",
    "player_role",
    "assists_all",
    "deaths_all",
    "fd_all",
    "fk_all",
    "kast_all",
    "kd_ratio",
    "death_contribution_score",
    "death_contribution_level"
]

print("\nTOP CONTRIBUTION EXAMPLES")

print(
    df[
        display_columns
    ]
    .sort_values(
        "death_contribution_score",
        ascending=False
    )
    .head(15)
    .to_string(index=False)
)

# Save death contribution dataset

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDEATH CONTRIBUTION ANALYSIS COMPLETE")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nSaved:")
print(OUTPUT_FILE)
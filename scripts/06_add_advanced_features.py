import pandas as pd
import numpy as np
import duckdb

# GameSense AI - Advanced Feature Engineering

INPUT_FILE = "data/features/gamesense_death_contribution.csv"
OUTPUT_FILE = "data/features/gamesense_advanced_features.csv"
DATABASE_FILE = "vct.duckdb"

# Load death contribution dataset

df = pd.read_csv(INPUT_FILE)

print("Death contribution dataset loaded")
print("Rows:", len(df))

# Convert numeric columns

numeric_columns = [
    "rating_all",
    "rating_atk",
    "rating_def",
    "acs_all",
    "acs_atk",
    "acs_def",
    "adr_all",
    "adr_atk",
    "adr_def",
    "kills_all",
    "kills_atk",
    "kills_def",
    "deaths_all",
    "deaths_atk",
    "deaths_def",
    "assists_all",
    "assists_atk",
    "assists_def",
    "fk_all",
    "fk_atk",
    "fd_all",
    "fd_atk",
    "two_k",
    "three_k",
    "four_k",
    "five_k",
    "clutch_1v1",
    "clutch_1v2",
    "clutch_1v3",
    "clutch_1v4",
    "clutch_1v5",
    "map_score_team0",
    "map_score_team1"
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

# Calculate total rounds

df["total_rounds"] = (
    df["map_score_team0"]
    + df["map_score_team1"]
)

# Calculate map score margin

df["score_margin"] = (
    df["map_score_team0"]
    - df["map_score_team1"]
).abs()

# Calculate player's team score

df["player_team_score"] = np.where(
    df["team_idx"] == 0,
    df["map_score_team0"],
    df["map_score_team1"]
)

# Calculate opponent score

df["opponent_team_score"] = np.where(
    df["team_idx"] == 0,
    df["map_score_team1"],
    df["map_score_team0"]
)

# Calculate player score margin

df["player_score_margin"] = (
    df["player_team_score"]
    - df["opponent_team_score"]
)

# Classify loss severity

def classify_loss_impact(row):

    if row["map_result"] != "Loss":
        return "No Loss"

    margin = row["score_margin"]

    if margin <= 2:
        return "Low"

    elif margin <= 4:
        return "Moderate"

    elif margin <= 8:
        return "High"

    else:
        return "Very High"


df["loss_impact"] = df.apply(
    classify_loss_impact,
    axis=1
)

# Calculate clutch total

clutch_columns = [
    "clutch_1v1",
    "clutch_1v2",
    "clutch_1v3",
    "clutch_1v4",
    "clutch_1v5"
]

df["clutch_total"] = (
    df[clutch_columns]
    .fillna(0)
    .sum(axis=1)
)

# Calculate multikill total

multikill_columns = [
    "two_k",
    "three_k",
    "four_k",
    "five_k"
]

df["multikill_total"] = (
    df[multikill_columns]
    .fillna(0)
    .sum(axis=1)
)

# Calculate attack-defense rating gap

df["rating_side_gap"] = (
    df["rating_atk"]
    - df["rating_def"]
)

# Calculate attack-defense ACS gap

df["acs_side_gap"] = (
    df["acs_atk"]
    - df["acs_def"]
)

# Calculate attack-defense ADR gap

df["adr_side_gap"] = (
    df["adr_atk"]
    - df["adr_def"]
)

# Calculate attack-defense kill gap

df["kills_side_gap"] = (
    df["kills_atk"]
    - df["kills_def"]
)

# Calculate attack-defense death gap

df["deaths_side_gap"] = (
    df["deaths_atk"]
    - df["deaths_def"]
)

# Calculate attack-defense performance balance

df["rating_side_balance"] = np.where(
    df["rating_atk"].notna()
    & df["rating_def"].notna()
    & ((df["rating_atk"] + df["rating_def"]) > 0),
    (
        df["rating_atk"]
        / (
            df["rating_atk"]
            + df["rating_def"]
        )
    ),
    np.nan
)

# Calculate role-relative performance

role_rating_mean = (
    df.groupby("player_role")["rating_all"]
    .transform("mean")
)

role_acs_mean = (
    df.groupby("player_role")["acs_all"]
    .transform("mean")
)

role_adr_mean = (
    df.groupby("player_role")["adr_all"]
    .transform("mean")
)

df["role_relative_rating"] = (
    df["rating_all"]
    - role_rating_mean
)

df["role_relative_acs"] = (
    df["acs_all"]
    - role_acs_mean
)

df["role_relative_adr"] = (
    df["adr_all"]
    - role_adr_mean
)

# Calculate role-relative percentage performance

df["role_relative_rating_pct"] = np.where(
    role_rating_mean != 0,
    (
        (
            df["rating_all"]
            - role_rating_mean
        )
        / role_rating_mean
    ) * 100,
    np.nan
)

df["role_relative_acs_pct"] = np.where(
    role_acs_mean != 0,
    (
        (
            df["acs_all"]
            - role_acs_mean
        )
        / role_acs_mean
    ) * 100,
    np.nan
)

df["role_relative_adr_pct"] = np.where(
    role_adr_mean != 0,
    (
        (
            df["adr_all"]
            - role_adr_mean
        )
        / role_adr_mean
    ) * 100,
    np.nan
)

# Data availability flags

df["clutch_data_available"] = (
    df[clutch_columns]
    .notna()
    .any(axis=1)
).astype(int)

df["multikill_data_available"] = (
    df[multikill_columns]
    .notna()
    .any(axis=1)
).astype(int)

df["side_performance_available"] = (
    df[
        [
            "rating_atk",
            "rating_def",
            "acs_atk",
            "acs_def",
            "adr_atk",
            "adr_def"
        ]
    ]
    .notna()
    .all(axis=1)
).astype(int)

# Connect to VCT database

print("\nConnecting to VCT database...")

con = duckdb.connect(
    DATABASE_FILE,
    read_only=True
)

# Get player matchup statistics

matchup_query = """
SELECT
    km.match_id,
    km.game_id,
    km.row_player_id AS player_id,
    SUM(km.kills) AS matchup_kills,
    SUM(km.deaths) AS matchup_deaths,
    SUM(km.op_kills) AS matchup_operator_kills,
    SUM(km.op_deaths) AS matchup_operator_deaths,
    SUM(km.fk_kills) AS matchup_first_kills,
    SUM(km.fk_deaths) AS matchup_first_deaths
FROM kill_matrix km
GROUP BY
    km.match_id,
    km.game_id,
    km.row_player_id
"""

matchup_df = con.execute(
    matchup_query
).fetchdf()

con.close()

print(
    "Matchup records loaded:",
    len(matchup_df)
)

# Convert join keys to strings

df["match_id"] = df["match_id"].astype(str)
df["game_id"] = df["game_id"].astype(str)

matchup_df["match_id"] = (
    matchup_df["match_id"]
    .astype(str)
)

matchup_df["game_id"] = (
    matchup_df["game_id"]
    .astype(str)
)

# Merge matchup statistics

df = df.merge(
    matchup_df,
    on=[
        "match_id",
        "game_id",
        "player_id"
    ],
    how="left"
)

# Calculate matchup K/D

df["matchup_kd"] = np.where(
    df["matchup_deaths"] > 0,
    df["matchup_kills"]
    / df["matchup_deaths"],
    df["matchup_kills"]
)

# Matchup data availability

df["matchup_data_available"] = (
    df["matchup_kills"]
    .notna()
).astype(int)

# Calculate loss streak

df = df.sort_values(
    [
        "player_id",
        "utc_timestamp",
        "match_id",
        "game_id"
    ]
).reset_index(drop=True)

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
    .transform(
        calculate_loss_streak
    )
)

# Previous loss streak

df["previous_loss_streak"] = (
    df.groupby("player_id")["loss_streak"]
    .shift(1)
    .fillna(0)
)

# Series context

df["series_stage_available"] = (
    df["series_stage"]
    .notna()
).astype(int)

df["series_round_available"] = (
    df["series_round"]
    .notna()
).astype(int)

# Overall advanced feature availability

df["advanced_feature_data_available"] = (
    (
        df["side_performance_available"] == 1
    )
    | (
        df["clutch_data_available"] == 1
    )
    | (
        df["matchup_data_available"] == 1
    )
).astype(int)

# Print advanced feature summary

print("\nADVANCED FEATURE SUMMARY")

summary_columns = [
    "rating_all",
    "clutch_total",
    "multikill_total",
    "score_margin",
    "loss_streak",
    "role_relative_rating",
    "matchup_kills",
    "matchup_deaths"
]

print(
    df[summary_columns]
    .describe()
    .round(2)
    .to_string()
)

# Print loss impact distribution

print("\nLOSS IMPACT DISTRIBUTION")

print(
    df["loss_impact"]
    .value_counts()
)

# Print side performance availability

print("\nSIDE PERFORMANCE DATA")

print(
    df["side_performance_available"]
    .value_counts()
)

# Print matchup availability

print("\nMATCHUP DATA")

print(
    df["matchup_data_available"]
    .value_counts()
)

# Save advanced feature dataset

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nADVANCED FEATURE ENGINEERING COMPLETE")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nSaved:")
print(OUTPUT_FILE)
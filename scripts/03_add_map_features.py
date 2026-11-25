import pandas as pd
import numpy as np
import duckdb

INPUT_FILE = "data/processed/gamesense_processed.csv"
OUTPUT_FILE = "data/features/gamesense_features.csv"
DATABASE_FILE = "vct.duckdb"

EVENT_IDS = [1657, 2097, 2283]

print("Loading processed dataset...")

df = pd.read_csv(INPUT_FILE)

print("Processed dataset loaded")
print("Rows:", len(df))

df["match_id"] = df["match_id"].astype(str)
df["game_id"] = df["game_id"].astype(str)
df["player_id"] = df["player_id"].astype(str)

print("\nConnecting to VCT database...")

con = duckdb.connect(DATABASE_FILE, read_only=True)

query = f"""
SELECT
    mp.match_id,
    mp.game_id,
    mp.player_id,

    mp.team_idx,

    mp.rating_atk,
    mp.rating_def,
    mp.acs_atk,
    mp.acs_def,
    mp.adr_atk,
    mp.adr_def,

    mp.kills_atk,
    mp.kills_def,
    mp.deaths_atk,
    mp.deaths_def,

    mp.assists_atk,
    mp.assists_def,

    mp.fk_atk,
    mp.fk_def,
    mp.fd_atk,
    mp.fd_def,

    mp.kast_atk,
    mp.kast_def,
    mp.hs_pct_atk,
    mp.hs_pct_def,

    mp.econ,

    m.map_name,
    m.duration,
    m.score0 AS map_score_team0,
    m.score1 AS map_score_team1,
    m.performance_available,
    m.economy_available,

    mt.event,
    mt.event_id,
    mt.series_stage,
    mt.series_round,
    mt.format,
    mt.patch,
    mt.status,
    mt.listing_status,
    mt.is_international,
    mt.region,
    mt.is_showmatch,
    mt.team0_id,
    mt.team1_id

FROM player_map mp

JOIN maps m
    ON mp.match_id = m.match_id
    AND mp.game_id = m.game_id

JOIN matches mt
    ON mp.match_id = mt.match_id

WHERE mt.event_id IN ({",".join(map(str, EVENT_IDS))})
"""

map_df = con.execute(query).fetchdf()

con.close()

print("Map/player records loaded:", len(map_df))

map_df["match_id"] = map_df["match_id"].astype(str)
map_df["game_id"] = map_df["game_id"].astype(str)
map_df["player_id"] = map_df["player_id"].astype(str)

key_columns = [
    "match_id",
    "game_id",
    "player_id"
]

duplicate_count = map_df.duplicated(
    subset=key_columns
).sum()

print("Duplicate map/player records:", duplicate_count)

if duplicate_count > 0:
    map_df = map_df.drop_duplicates(
        subset=key_columns,
        keep="first"
    )

feature_columns = [
    "match_id",
    "game_id",
    "player_id",
    "team_idx",

    "rating_atk",
    "rating_def",
    "acs_atk",
    "acs_def",
    "adr_atk",
    "adr_def",

    "kills_atk",
    "kills_def",
    "deaths_atk",
    "deaths_def",

    "assists_atk",
    "assists_def",

    "fk_atk",
    "fk_def",
    "fd_atk",
    "fd_def",

    "kast_atk",
    "kast_def",
    "hs_pct_atk",
    "hs_pct_def",

    "econ",

    "map_name",
    "duration",
    "map_score_team0",
    "map_score_team1",

    "performance_available",
    "economy_available",

    "event",
    "event_id",
    "series_stage",
    "series_round",
    "format",
    "patch",
    "status",
    "listing_status",
    "is_international",
    "region",
    "is_showmatch",
    "team0_id",
    "team1_id"
]

map_df = map_df[feature_columns]

df = df.merge(
    map_df,
    on=key_columns,
    how="left",
    suffixes=("", "_map")
)

if "team_idx_map" in df.columns:
    df["team_idx"] = df["team_idx_map"]
    df = df.drop(columns=["team_idx_map"])

df["map_result"] = np.where(
    df["team_idx"] == 0,
    np.where(
        df["map_score_team0"] > df["map_score_team1"],
        "Win",
        "Loss"
    ),
    np.where(
        df["map_score_team1"] > df["map_score_team0"],
        "Win",
        "Loss"
    )
)

df["map_number"] = (
    df.sort_values(
        ["match_id", "utc_timestamp", "game_id"]
    )
    .groupby("match_id")
    .cumcount()
    + 1
)

df["map_win"] = (
    df["map_result"] == "Win"
).astype(int)

df["total_rounds"] = (
    df["map_score_team0"] +
    df["map_score_team1"]
)

df["player_team_score"] = np.where(
    df["team_idx"] == 0,
    df["map_score_team0"],
    df["map_score_team1"]
)

df["opponent_team_score"] = np.where(
    df["team_idx"] == 0,
    df["map_score_team1"],
    df["map_score_team0"]
)

df["score_margin"] = (
    df["player_team_score"] -
    df["opponent_team_score"]
)

df["absolute_score_margin"] = (
    df["score_margin"].abs()
)

df["performance_data_available"] = (
    df["performance_available"]
    .fillna(False)
    .astype(int)
)

df["economy_data_available"] = (
    df["economy_available"]
    .fillna(False)
    .astype(int)
)

side_columns = [
    "rating_atk",
    "rating_def",
    "acs_atk",
    "acs_def",
    "adr_atk",
    "adr_def",
    "kills_atk",
    "kills_def",
    "deaths_atk",
    "deaths_def"
]

df["side_performance_available"] = (
    df[side_columns]
    .notna()
    .all(axis=1)
    .astype(int)
)

print("\nMAP RESULTS")
print(df["map_result"].value_counts())

print("\nMAP DISTRIBUTION")
print(
    df["map_number"]
    .value_counts()
    .sort_index()
)

print("\nMAP NAMES")
print(
    df["map_name"]
    .value_counts()
)

print("\nMissing map names:", df["map_name"].isna().sum())

print("\nATTACK/DEFENSE DATA")
print(
    df["side_performance_available"]
    .value_counts()
)

print("\nSERIES STAGE")
print(
    df["series_stage"]
    .value_counts(dropna=False)
)

print("\nSERIES ROUND")
print(
    df["series_round"]
    .value_counts(dropna=False)
)

print("\nPATCH")
print(
    df["patch"]
    .value_counts(dropna=False)
)

print("\nMAP FEATURE ENGINEERING COMPLETE")
print("Rows:", len(df))
print("Columns:", len(df.columns))

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)
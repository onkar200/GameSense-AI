
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = "data/features/gamesense_advanced_features.csv"
OUTPUT_FILE = "data/features/gamesense_gti.csv"

BASELINE_WINDOW = 5
MINIMUM_HISTORY = 2

METRICS = {
    "acs_all": {
        "weight": 20,
        "direction": "decline",
        "threshold": 0.25
    },
    "kd_ratio": {
        "weight": 20,
        "direction": "decline",
        "threshold": 0.30
    },
    "adr_all": {
        "weight": 15,
        "direction": "decline",
        "threshold": 0.25
    },
    "kast_all": {
        "weight": 15,
        "direction": "decline",
        "threshold": 0.15
    },
    "deaths_all": {
        "weight": 10,
        "direction": "increase",
        "threshold": 0.40
    },
    "first_death_rate": {
        "weight": 10,
        "direction": "increase",
        "threshold": 0.50
    },
    "multikill_total": {
        "weight": 10,
        "direction": "decline",
        "threshold": 0.50
    }
}

print("Loading advanced feature dataset...")

df = pd.read_csv(INPUT_FILE)

print("Rows:", len(df))
print("Columns:", len(df.columns))

required_columns = [
    "player_id",
    "utc_timestamp",
    "match_id",
    "game_id",
    "map_result"
]

required_columns += list(METRICS.keys())

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df["utc_timestamp"] = pd.to_datetime(
    df["utc_timestamp"],
    errors="coerce",
    utc=True
)

for column in METRICS:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df["original_order"] = np.arange(len(df))

df = df.sort_values(
    [
        "player_id",
        "utc_timestamp",
        "match_id",
        "game_id",
        "original_order"
    ],
    na_position="last"
).reset_index(drop=True)

print("\nCalculating previous-map baselines...")

grouped = df.groupby("player_id", sort=False)

df["previous_maps_available"] = grouped.cumcount()

for metric in METRICS:
    df[f"{metric}_baseline"] = (
        grouped[metric]
        .transform(
            lambda series: series.shift(1).rolling(
                window=BASELINE_WINDOW,
                min_periods=MINIMUM_HISTORY
            ).mean()
        )
    )

print("Baselines use previous maps only.")

print("\nCalculating GTI components...")

available_components = []
component_columns = []

for metric, config in METRICS.items():
    baseline_column = f"{metric}_baseline"
    component_column = f"gti_{metric}_points"

    current = df[metric]
    baseline = df[baseline_column]

    valid = (
        current.notna()
        & baseline.notna()
        & (baseline > 0)
    )

    if config["direction"] == "decline":
        deterioration = (
            (baseline - current) / baseline
        )
    else:
        deterioration = (
            (current - baseline) / baseline
        )

    deterioration = deterioration.clip(lower=0)

    component = (
        deterioration / config["threshold"]
    ).clip(lower=0, upper=1) * config["weight"]

    df[component_column] = component.where(valid)

    available_components.append(valid.astype(int))
    component_columns.append(component_column)

df["gti_components_available"] = pd.concat(
    available_components,
    axis=1
).sum(axis=1)

df["gti_available_weight"] = pd.concat(
    [
        pd.Series(
            valid.astype(int).to_numpy() * config["weight"],
            index=df.index
        )
        for valid, config in zip(
            available_components,
            METRICS.values()
        )
    ],
    axis=1
).sum(axis=1)

raw_points = df[component_columns].sum(
    axis=1,
    min_count=1
)

df["gti_score"] = (
    raw_points / df["gti_available_weight"] * 100
).where(df["gti_available_weight"] > 0)

enough_history = (
    df["previous_maps_available"] >= MINIMUM_HISTORY
)

enough_components = (
    df["gti_components_available"] >= 4
)

df["gti_data_available"] = (
    enough_history & enough_components
).astype(int)

df.loc[
    df["gti_data_available"] == 0,
    "gti_score"
] = np.nan

df["gti_score"] = df["gti_score"].clip(0, 100)

df["gti_confidence"] = np.select(
    [
        (df["gti_data_available"] == 1)
        & (df["gti_components_available"] >= 6),

        (df["gti_data_available"] == 1)
        & (df["gti_components_available"] >= 4)
    ],
    [
        "High",
        "Moderate"
    ],
    default="Insufficient Data"
)

df["gti_level"] = np.select(
    [
        df["gti_score"].notna()
        & (df["gti_score"] <= 25),

        df["gti_score"].notna()
        & (df["gti_score"] <= 50),

        df["gti_score"].notna()
        & (df["gti_score"] <= 75),

        df["gti_score"].notna()
        & (df["gti_score"] > 75)
    ],
    [
        "Stable",
        "Mild Signs",
        "Elevated",
        "Strong Potential Tilt"
    ],
    default="Insufficient Data"
)

df["gti_interpretation"] = np.select(
    [
        df["gti_level"] == "Stable",
        df["gti_level"] == "Mild Signs",
        df["gti_level"] == "Elevated",
        df["gti_level"] == "Strong Potential Tilt"
    ],
    [
        "Limited deterioration across measured indicators",
        "Some performance indicators have deteriorated",
        "Multiple or substantial deterioration signals",
        "Strong deterioration signals; review before next map"
    ],
    default="Insufficient historical baseline"
)

print("\nGTI SUMMARY")

valid_scores = df.loc[
    df["gti_score"].notna(),
    "gti_score"
]

print("Records with valid GTI:", len(valid_scores))
print("Records needing more history:", df["gti_score"].isna().sum())

if not valid_scores.empty:
    print("Average GTI:", round(valid_scores.mean(), 2))
    print("Median GTI:", round(valid_scores.median(), 2))
    print("Minimum GTI:", round(valid_scores.min(), 2))
    print("Maximum GTI:", round(valid_scores.max(), 2))

print("\nGTI LEVEL DISTRIBUTION")
print(df["gti_level"].value_counts())

print("\nGTI CONFIDENCE")
print(df["gti_confidence"].value_counts())

print("\nGTI BY MAP RESULT")
print(
    df.groupby("map_result", dropna=False)["gti_score"]
    .agg(["count", "mean", "median"])
    .round(2)
)

print("\nHIGHEST GTI RECORDS")

display_columns = [
    "player_name",
    "team_name",
    "map_name",
    "map_number",
    "map_result",
    "previous_maps_available",
    "gti_score",
    "gti_level",
    "gti_confidence"
]

display_columns = [
    column for column in display_columns
    if column in df.columns
]

print(
    df.loc[df["gti_score"].notna()]
    .nlargest(15, "gti_score")[display_columns]
    .to_string(index=False)
)

Path(OUTPUT_FILE).parent.mkdir(
    parents=True,
    exist_ok=True
)

df = df.drop(columns=["original_order"])

df.to_csv(OUTPUT_FILE, index=False)

print("\nGTI CALCULATION COMPLETE")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Saved:", OUTPUT_FILE)

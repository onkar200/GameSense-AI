
import pandas as pd
import numpy as np
from pathlib import Path

# GameSense AI - PABER Scoring
# Player-Adaptive Baseline and Early-warning Risk

INPUT_FILE = "data/features/gamesense_gti.csv"
PABER_OUTPUT = "data/features/gamesense_paber.csv"
VALIDATION_INPUT = "data/features/gamesense_validation.csv"
VALIDATION_OUTPUT = "data/features/gamesense_validation_paber.csv"

WINDOW = 5
MIN_HISTORY = 3
MIN_COMPONENTS = 4

# Same seven indicators and weights as the original GTI.
# direction = -1: lower value indicates deterioration
# direction = +1: higher value indicates deterioration

METRICS = {
    "acs_all": {"weight": 20, "direction": -1},
    "kd_ratio": {"weight": 20, "direction": -1},
    "adr_all": {"weight": 15, "direction": -1},
    "kast_all": {"weight": 15, "direction": -1},
    "deaths_all": {"weight": 10, "direction": 1},
    "first_death_rate": {"weight": 10, "direction": 1},
    "multikill_total": {"weight": 10, "direction": -1},
}

print("Loading existing GTI dataset...")

df = pd.read_csv(INPUT_FILE)

required = [
    "player_id",
    "match_id",
    "game_id",
    "utc_timestamp"
] + list(METRICS.keys())

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Preserve identifiers and ensure consistent sorting.
df["player_id"] = df["player_id"].astype(str)
df["match_id"] = df["match_id"].astype(str)
df["game_id"] = df["game_id"].astype(str)

df["utc_timestamp"] = pd.to_datetime(
    df["utc_timestamp"],
    errors="coerce",
    utc=True
)

df["game_id_numeric"] = pd.to_numeric(
    df["game_id"],
    errors="coerce"
)

df = df.sort_values(
    [
        "player_id",
        "utc_timestamp",
        "match_id",
        "game_id_numeric"
    ],
    kind="stable"
).reset_index(drop=True)

print("Rows loaded:", len(df))

# Convert metrics to numeric.
for metric in METRICS:
    df[metric] = pd.to_numeric(
        df[metric],
        errors="coerce"
    )

# Previous-map count.
df["paber_previous_maps"] = (
    df.groupby("player_id").cumcount()
)

# Prepare output columns.
component_columns = []

for metric in METRICS:
    component_columns.append(f"paber_{metric}_points")

    df[f"paber_{metric}_baseline"] = np.nan
    df[f"paber_{metric}_mad"] = np.nan
    df[f"paber_{metric}_points"] = np.nan

df["paber_components_available"] = 0
df["paber_raw_score"] = np.nan
df["paber_persistence"] = np.nan
df["paber_score"] = np.nan
df["paber_data_available"] = 0

print("\nCalculating player-adaptive baselines...")

# Process each player's history independently.
for player_id, group in df.groupby("player_id", sort=False):

    idx = group.index

    for metric, settings in METRICS.items():

        values = group[metric]

        # Historical median: previous five maps only.
        baseline = (
            values.shift(1)
            .rolling(
                window=WINDOW,
                min_periods=MIN_HISTORY
            )
            .median()
        )

        # Historical MAD: median absolute deviation.
        mad = (
            values.shift(1)
            .rolling(
                window=WINDOW,
                min_periods=MIN_HISTORY
            )
            .apply(
                lambda x: np.median(
                    np.abs(x - np.median(x))
                ),
                raw=True
            )
        )

        # Robust scale estimate.
        robust_scale = 1.4826 * mad

        # Fallback when MAD is zero.
        fallback_scale = np.maximum(
            baseline.abs() * 0.10,
            1e-6
        )

        robust_scale = robust_scale.where(
            robust_scale > 1e-9,
            fallback_scale
        )

        # Store historical reference values.
        df.loc[idx, f"paber_{metric}_baseline"] = (
            baseline.to_numpy()
        )

        df.loc[idx, f"paber_{metric}_mad"] = (
            mad.to_numpy()
        )

        # Positive values represent deterioration.
        deterioration = (
    settings["direction"]
    * (values - baseline)
        )

        robust_deviation = (
            deterioration / robust_scale
        )

        # Ignore improvements and cap extreme deterioration.
        robust_deviation = (
            robust_deviation
            .clip(lower=0, upper=3)
        )

        # Convert to weighted component score.
        points = (
            robust_deviation / 3
        ) * settings["weight"]

        points = points.where(
            baseline.notna() & values.notna()
        )

        df.loc[idx, f"paber_{metric}_points"] = (
            points.to_numpy()
        )

# Calculate available components.
df["paber_components_available"] = (
    df[component_columns]
    .notna()
    .sum(axis=1)
)

# Normalize available metric weights to 100.
available_weights = pd.Series(
    0.0,
    index=df.index
)

weighted_points = pd.Series(
    0.0,
    index=df.index
)

for metric, settings in METRICS.items():

    points_column = f"paber_{metric}_points"

    available = df[points_column].notna()

    available_weights += (
        available.astype(float)
        * settings["weight"]
    )

    weighted_points += (
        df[points_column].fillna(0)
    )

valid = (
    (df["paber_previous_maps"] >= MIN_HISTORY)
    & (df["paber_components_available"] >= MIN_COMPONENTS)
    & (available_weights > 0)
)

df.loc[valid, "paber_raw_score"] = (
    weighted_points[valid]
    / available_weights[valid]
) * 100

# Persistence: average of current and previous two
# available PABER raw scores for the same player.
print("Calculating deterioration persistence...")

for player_id, group in df.groupby("player_id", sort=False):

    persistence = (
        group["paber_raw_score"]
        .rolling(window=3, min_periods=1)
        .mean()
    )

    df.loc[group.index, "paber_persistence"] = (
        persistence.to_numpy()
    )

# Combine current deterioration and persistence.
# 80% current score + 20% recent-history persistence.
df["paber_score"] = (
    0.80 * df["paber_raw_score"]
    + 0.20 * df["paber_persistence"]
)

df["paber_score"] = df["paber_score"].clip(0, 100)

df.loc[~valid, "paber_score"] = np.nan

df["paber_data_available"] = (
    df["paber_score"].notna()
).astype(int)

# Risk categories use the same boundaries as GTI.
df["paber_level"] = np.select(
    [
        df["paber_score"].isna(),
        df["paber_score"] <= 25,
        df["paber_score"] <= 50,
        df["paber_score"] <= 75,
        df["paber_score"] > 75
    ],
    [
        "Insufficient Data",
        "Stable",
        "Mild Signs",
        "Elevated",
        "Strong Potential Tilt"
    ],
    default="Insufficient Data"
)

df["paber_confidence"] = np.select(
    [
        df["paber_previous_maps"] < MIN_HISTORY,
        df["paber_components_available"] < MIN_COMPONENTS,
        df["paber_previous_maps"] >= WINDOW
    ],
    [
        "Insufficient",
        "Insufficient",
        "High"
    ],
    default="Moderate"
)

# Summary.
print("\n========== PABER SUMMARY ==========")

print("Total records:", len(df))

print(
    "PABER available:",
    int(df["paber_data_available"].sum())
)

print(
    "PABER unavailable:",
    int((df["paber_data_available"] == 0).sum())
)

print("\nPABER category distribution:")

print(
    df["paber_level"].value_counts(dropna=False)
)

print("\nPABER score statistics:")

print(
    df["paber_score"].describe().round(2)
)

# Save independent PABER dataset.
Path("data/features").mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    PABER_OUTPUT,
    index=False
)

print("\nSaved:", PABER_OUTPUT)

# Attach PABER to a COPY of the existing validation data.
print("\nUpdating validation copy...")

validation = pd.read_csv(VALIDATION_INPUT)

for key in ["player_id", "match_id", "game_id"]:
    validation[key] = validation[key].astype(str)

paber_columns = [
    "player_id",
    "match_id",
    "game_id",
    "paber_previous_maps",
    "paber_components_available",
    "paber_raw_score",
    "paber_persistence",
    "paber_score",
    "paber_data_available",
    "paber_level",
    "paber_confidence"
]

paber_subset = df[paber_columns].copy()

validation = validation.merge(
    paber_subset,
    on=["player_id", "match_id", "game_id"],
    how="left"
)

validation.to_csv(
    VALIDATION_OUTPUT,
    index=False
)

print("Validation records:", len(validation))

print(
    "Validation PABER available:",
    int(validation["paber_data_available"].fillna(0).sum())
)

print("\nSaved:", VALIDATION_OUTPUT)

print("\nPABER BUILD COMPLETE!")

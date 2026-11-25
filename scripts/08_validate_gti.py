
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu

INPUT_FILE = "data/features/gamesense_gti.csv"
OUTPUT_FILE = "data/features/gamesense_validation.csv"

print("Loading GTI dataset...")
df = pd.read_csv(INPUT_FILE)

required = [
    "player_id",
    "match_id",
    "game_id",
    "utc_timestamp",
    "rating_all",
    "acs_all",
    "gti_score",
    "gti_data_available"
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

for col in ["rating_all", "acs_all", "gti_score"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["utc_timestamp"] = pd.to_datetime(
    df["utc_timestamp"], errors="coerce", utc=True
)

df = df.sort_values(
    ["player_id", "utc_timestamp", "match_id", "game_id"]
).reset_index(drop=True)

# Historical baselines use previous maps only.
grouped_player = df.groupby("player_id", sort=False)

df["rating_history_baseline"] = grouped_player[
    "rating_all"
].transform(
    lambda s: s.shift(1).rolling(5, min_periods=2).mean()
)

df["acs_history_baseline"] = grouped_player[
    "acs_all"
].transform(
    lambda s: s.shift(1).rolling(5, min_periods=2).mean()
)

# Pair each map with the player's next map in the same series.
grouped_series = df.groupby(
    ["player_id", "match_id"], sort=False
)

df["next_map_rating"] = grouped_series["rating_all"].shift(-1)
df["next_map_acs"] = grouped_series["acs_all"].shift(-1)
df["next_map_game_id"] = grouped_series["game_id"].shift(-1)

df["next_rating_change_pct"] = np.where(
    df["rating_history_baseline"] > 0,
    (
        (df["next_map_rating"] - df["rating_history_baseline"])
        / df["rating_history_baseline"]
    ) * 100,
    np.nan
)

df["next_acs_change_pct"] = np.where(
    df["acs_history_baseline"] > 0,
    (
        (df["next_map_acs"] - df["acs_history_baseline"])
        / df["acs_history_baseline"]
    ) * 100,
    np.nan
)

# Operational definition: next-map rating or ACS is at least
# 15% below the player's historical baseline.
df["next_map_poor_performance"] = np.where(
    df["next_map_rating"].notna()
    & df["next_map_acs"].notna()
    & df["rating_history_baseline"].gt(0)
    & df["acs_history_baseline"].gt(0),
    (
        (df["next_rating_change_pct"] <= -15)
        | (df["next_acs_change_pct"] <= -15)
    ).astype(int),
    np.nan
)

valid = df[
    df["gti_score"].notna()
    & (df["gti_data_available"] == 1)
    & df["next_map_rating"].notna()
    & df["next_map_acs"].notna()
].copy()

print("\nNEXT-MAP VALIDATION")
print("Eligible map pairs:", len(valid))
print("Unique players:", valid["player_id"].nunique())
print("Unique series:", valid["match_id"].nunique())

def report_correlation(x, y, label):
    subset = valid[[x, y]].dropna()
    if len(subset) >= 3 and subset[x].nunique() > 1 and subset[y].nunique() > 1:
        result = spearmanr(subset[x], subset[y])
        print(
            f"{label}: Spearman rho={result.statistic:.3f}, "
            f"p={result.pvalue:.4g}, n={len(subset)}"
        )
    else:
        print(f"{label}: insufficient variation/data")

report_correlation(
    "gti_score", "next_rating_change_pct",
    "GTI vs next-map rating change"
)

report_correlation(
    "gti_score", "next_acs_change_pct",
    "GTI vs next-map ACS change"
)

print("\nPOOR-PERFORMANCE RATE BY GTI GROUP")

valid["gti_group"] = pd.cut(
    valid["gti_score"],
    bins=[-0.01, 25, 50, 75, 100],
    labels=["Stable", "Mild Signs", "Elevated", "Strong Potential Tilt"]
)

print(
    valid.groupby("gti_group", observed=False)[
        "next_map_poor_performance"
    ].agg(["count", "mean"]).assign(
        poor_performance_pct=lambda x: x["mean"] * 100
    ).round(2)
)

high = valid.loc[
    valid["gti_score"] > 75, "next_map_poor_performance"
].dropna()

low = valid.loc[
    valid["gti_score"] <= 25, "next_map_poor_performance"
].dropna()

if len(high) and len(low):
    print("\nHigh-GTI vs Stable poor-performance rate")
    print("High GTI:", round(high.mean() * 100, 2), "%")
    print("Stable:", round(low.mean() * 100, 2), "%")

# Save the validation-ready paired dataset.
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
valid.to_csv(OUTPUT_FILE, index=False)

print("\nVALIDATION DATASET SAVED")
print("Rows:", len(valid))
print("Columns:", len(valid.columns))
print(OUTPUT_FILE)

import pandas as pd

df = pd.read_csv("gamesense_champions.csv")

df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"])

# Sort chronologically
df = df.sort_values(
    ["player_id", "utc_timestamp", "match_id", "game_id"]
)

# Count maps per player
player_maps = (
    df.groupby(["player_id", "player_name"])
      .size()
      .reset_index(name="map_count")
      .sort_values("map_count", ascending=False)
)

print("\n===== PLAYER MAP COUNTS =====")
print(player_maps.head(30).to_string(index=False))

print("\n===== SUMMARY =====")
print("Players:", df["player_id"].nunique())
print("Total records:", len(df))
print("Players with >= 2 maps:", (player_maps["map_count"] >= 2).sum())
print("Players with >= 3 maps:", (player_maps["map_count"] >= 3).sum())
print("Players with >= 5 maps:", (player_maps["map_count"] >= 5).sum())
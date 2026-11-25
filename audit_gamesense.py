import pandas as pd
import duckdb

CSV_FILE = "data/features/gamesense_advanced_features.csv"
DB_FILE = "vct.duckdb"

print("=" * 60)
print("GAMESENSE AI - DATA AUDIT")
print("=" * 60)

# 1. Inspect existing feature dataset
df = pd.read_csv(CSV_FILE)

print("\n1. DATASET")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nRelevant columns:")
for col in [
    "player_id", "match_id", "game_id",
    "utc_timestamp", "map_number",
    "total_rounds", "map_score_team0",
    "map_score_team1", "player_role"
]:
    if col in df.columns:
        print(f"{col}: {df[col].dtype}")

# 2. Timestamp audit
print("\n2. TIMESTAMP AUDIT")

df["utc_timestamp"] = pd.to_datetime(
    df["utc_timestamp"], errors="coerce", utc=True
)

print("Missing timestamps:", df["utc_timestamp"].isna().sum())
print("Unique timestamps:", df["utc_timestamp"].nunique())

if "match_id" in df.columns and "game_id" in df.columns:
    print("\nMaps per series:")
    print(
        df.groupby("match_id")["game_id"]
        .nunique()
        .describe()
    )

    print("\nTimestamp counts per series:")
    timestamp_counts = (
        df.groupby("match_id")["utc_timestamp"]
        .nunique()
    )
    print(timestamp_counts.describe())

    print("\nSample chronological records:")
    cols = [
        c for c in [
            "player_name", "player_id", "match_id",
            "game_id", "map_number", "utc_timestamp"
        ] if c in df.columns
    ]

    print(
        df[cols]
        .sort_values(["player_id", "utc_timestamp"])
        .head(20)
        .to_string(index=False)
    )

# 3. Round-count audit
print("\n3. ROUND COUNT AUDIT")

for col in [
    "total_rounds", "map_score_team0",
    "map_score_team1", "deaths_all"
]:
    if col in df.columns:
        print(
            col,
            "| Missing:", df[col].isna().sum(),
            "| Unique:", df[col].nunique()
        )

# 4. Inspect DuckDB
print("\n4. DUCKDB TABLES")

con = duckdb.connect(DB_FILE, read_only=True)

tables = con.execute(
    "SHOW TABLES"
).fetchdf()

print(tables.to_string(index=False))

# Inspect relevant table schemas
for table in ["player_map", "maps", "rounds", "matches"]:
    if table in tables["name"].tolist():
        print(f"\nSCHEMA: {table}")
        schema = con.execute(
            f"DESCRIBE SELECT * FROM {table}"
        ).fetchdf()

        print(schema.to_string(index=False))

        print(f"\nSAMPLE: {table}")
        sample = con.execute(
            f"SELECT * FROM {table} LIMIT 3"
        ).fetchdf()

        print(sample.to_string(index=False))

con.close()

print("\nAUDIT COMPLETE")
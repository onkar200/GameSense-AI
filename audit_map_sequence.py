import duckdb
import pandas as pd

con = duckdb.connect("vct.duckdb", read_only=True)

print("\n1. MAP SEQUENCE AUDIT")

query = """
SELECT
    m.match_id,
    m.game_id,
    m.map_name,
    m.score0,
    m.score1,
    ROW_NUMBER() OVER (
        PARTITION BY m.match_id
        ORDER BY CAST(m.game_id AS BIGINT)
    ) AS sequence_by_game_id,
    COUNT(r.round_num) AS actual_rounds
FROM maps m
LEFT JOIN rounds r
    ON m.match_id = r.match_id
    AND m.game_id = r.game_id
GROUP BY
    m.match_id,
    m.game_id,
    m.map_name,
    m.score0,
    m.score1
ORDER BY
    CAST(m.match_id AS BIGINT),
    CAST(m.game_id AS BIGINT)
"""

maps = con.execute(query).fetchdf()

print(maps.head(30).to_string(index=False))

print("\n2. ROUND COUNT CHECK")

maps["score_total"] = maps["score0"] + maps["score1"]

maps["round_difference"] = (
    maps["actual_rounds"] - maps["score_total"]
)

print("Total maps:", len(maps))
print(
    "Round count mismatches:",
    (maps["round_difference"] != 0).sum()
)

print("\n3. MAPS PER SERIES")

print(
    maps.groupby("match_id").size().describe()
)

print("\n4. MAP NUMBER ORDER CHECK")

df = pd.read_csv(
    "data/features/gamesense_advanced_features.csv"
)

check = df[
    ["match_id", "game_id", "map_number"]
].drop_duplicates()

check["game_id"] = pd.to_numeric(
    check["game_id"], errors="coerce"
)

print(
    check.sort_values(
        ["match_id", "game_id"]
    ).head(30).to_string(index=False)
)

con.close()

print("\nMAP SEQUENCE AUDIT COMPLETE")
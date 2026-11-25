import duckdb

con = duckdb.connect("vct.duckdb", read_only=True)

query = """
SELECT
    m.match_id,
    m.game_id,
    m.map_name,
    m.score0,
    m.score1,
    (m.score0 + m.score1) AS score_total,
    COUNT(r.round_num) AS actual_rounds,
    COUNT(r.round_num) - (m.score0 + m.score1) AS difference
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
HAVING COUNT(r.round_num) != (m.score0 + m.score1)
ORDER BY m.match_id, m.game_id
"""

result = con.execute(query).fetchdf()

print(result.to_string(index=False))
print("\nMismatch records:", len(result))

con.close()
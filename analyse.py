import duckdb
d = "2025-01-01.parquet"
duckdb.sql(f"DESCRIBE SELECT * FROM '{d}'").show(max_rows=60)
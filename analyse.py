import duckdb

from src.utils.constants import RAW_2025_01_01

duckdb.sql(f"DESCRIBE SELECT * FROM '{RAW_2025_01_01}'").show(max_rows=60)
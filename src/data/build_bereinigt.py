"""Schreibt data/02_events/final_bereinigt/*.parquet: wie final/, aber ohne Zeilen in Ausfall-Bloecken (scope 'alle':
A kein_update, B netz_nur_null, C station_nur_null). Zeilen in D-Stunden (codes fehlen) bleiben (nur_codes-Scope).
Aufruf: uv run python -m src.data.build_bereinigt"""
import duckdb
import pandas as pd

from src.utils.ausfall_bloecke import load_blocks
from src.utils.constants import DATA_DIR

SRC = DATA_DIR / "02_events" / "final"
DST = DATA_DIR / "02_events" / "final_bereinigt"


def main():
    DST.mkdir(exist_ok=True)
    b = load_blocks("alle")
    b["h"] = pd.to_datetime(b["von_lokal"]).dt.tz_localize(None)
    per = b[b["stop_id"] != -1][["stop_id", "richtung", "h"]].drop_duplicates()
    per["richtung"] = per["richtung"].where(per["richtung"].notna(), None)
    netz = b[b["stop_id"] == -1][["h"]].drop_duplicates()
    con = duckdb.connect()
    con.register("per", per); con.register("netz", netz)
    rows = []
    for f in sorted(SRC.glob("*.parquet")):
        out = DST / f.name
        con.execute(f"""COPY (
            SELECT e.* FROM read_parquet('{f}') e
            WHERE NOT EXISTS (SELECT 1 FROM per p WHERE p.stop_id = e.stop_id AND p.richtung IS NOT DISTINCT FROM e.richtung
                              AND p.h = date_trunc('hour', e.time_schedule AT TIME ZONE 'Europe/Berlin'))
              AND NOT EXISTS (SELECT 1 FROM netz n WHERE n.h = date_trunc('hour', e.time_schedule AT TIME ZONE 'Europe/Berlin'))
        ) TO '{out}' (FORMAT PARQUET)""")
        n0 = con.execute(f"select count(*) from read_parquet('{f}')").fetchone()[0]
        n1 = con.execute(f"select count(*) from read_parquet('{out}')").fetchone()[0]
        rows.append(dict(datei=f.name, zeilen_alt=n0, zeilen_neu=n1, entfernt=n0 - n1, anteil=(n0 - n1) / n0))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(DATA_DIR / "ausfall_bereinigung_zeilen.csv", index=False)


if __name__ == "__main__":
    main()

"""
Baut die Liste der Münchner S-Bahn-Stationen (Stufe 0.1 des Bauplans).

Der Bauplan geht von einer vorgefertigten Datei `muenchen_stationen_namen.csv` mit
187 DB-Stationen aus, die im Projekt nicht existierte. Ein erster Versuch, diese
Liste aus der bundesweiten, statischen Referenztabelle `stops.parquet` zu bauen
(location_type='STATION' + grobe Bbox), ergab 278 Kandidaten -- deutlich zu viele
und voller Duplikate (dieselbe physische Station mit mehreren stop_ids aus
unterschiedlichen Datenquellen, z.B. "Baierbrunn", "Grafrath", "Starnberg" doppelt/
dreifach) sowie Stationen, die klar nicht mehr zum S-Bahn-Netz gehören.

Robusterer Ansatz, analog zur Identifikation der 12 Stammstreckenhalte in
explore.py: Stationen NICHT aus einer statischen Referenztabelle ableiten,
sondern aus den tatsächlichen S-Bahn-Fahrten selbst. Ein stop_id gilt als
Münchner S-Bahn-Station, wenn dort mindestens eine Fahrt mit category='S' UND
operator='800725' (Betreiber der Münchner S-Bahn, siehe explore.py) gehalten hat.
Das schließt Duplikate und fremde Regionalbahnhöfe automatisch aus, weil nur
stop_ids zählen, die im Datensatz wirklich von einer S-Bahn-Linie bedient werden.

Ergebnis: 141 Stationen (Stand 2025) statt 187 -- diese Differenz bleibt unerklärt
und wird hier bewusst nicht auf 187 hochgezwungen. 141 ist die Zahl, die sich aus
dem tatsächlichen Betrieb 2025 ergibt.

Benötigt eine bereits (großzügig) auf den Münchner Raum vorgefilterte Rohdaten-
quelle als Input (z.B. data/01_muenchen/*.parquet aus einem ersten, groben
Filterlauf) -- ein Scan über die vollen 20GB 2025.tar ist dafür nicht nötig.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from src.utils.constants import DATA_DIR, OUTPUTS_DIR, STOPS_PARQUET

STOPS_PATH = STOPS_PARQUET
MUENCHEN_GLOB = str(DATA_DIR / "01_muenchen" / "*.parquet")
OUT_PATH = OUTPUTS_DIR / "muenchen_stationen_namen.csv"

SBAHN_OPERATOR = "800725"


def build() -> pd.DataFrame:
    con = duckdb.connect()
    traffic = con.execute(
        f"""
        SELECT stop_id, avg(lat) AS lat, avg(lon) AS lon,
               count(*) AS n_rows, count(DISTINCT trip_id) AS n_trips
        FROM read_parquet('{MUENCHEN_GLOB}')
        WHERE category = 'S' AND operator = '{SBAHN_OPERATOR}'
        GROUP BY stop_id
        """
    ).fetchdf()

    stops = pd.read_parquet(STOPS_PATH)
    name_lookup = (
        stops[stops["location_type"] == "STATION"]
        .drop_duplicates(subset="stop_id")
        .set_index("stop_id")["stop_name"]
    )
    traffic["stop_name"] = traffic["stop_id"].map(name_lookup)

    missing = traffic["stop_name"].isna().sum()
    if missing:
        print(f"WARNUNG: {missing} stop_ids ohne Namenstreffer in stops.parquet")

    traffic = traffic.sort_values("stop_name").reset_index(drop=True)
    return traffic[["stop_id", "stop_name", "lat", "lon", "n_rows", "n_trips"]].rename(
        columns={"lat": "stop_lat", "lon": "stop_lon"}
    )


if __name__ == "__main__":
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    result = build()
    result.to_csv(OUT_PATH, index=False)
    print(f"{len(result)} Münchner S-Bahn-Stationen (category='S', operator='{SBAHN_OPERATOR}')")
    print(f"-> geschrieben nach {OUT_PATH}")

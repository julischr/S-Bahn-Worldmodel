"""
Bootstrap-Kandidatenliste für den 0.1-Filterlauf (filter_muenchen.py).

Der eigentliche Zielprozess ist zweistufig:
  1. Rohdaten (2025.tar, 20 GB) grob auf den Münchner Raum filtern (dieses Skript
     liefert die dafür nötige, bewusst großzügige Kandidatenliste).
  2. Aus den gefilterten Daten die PRÄZISE S-Bahn-Stationsliste datengetrieben
     ableiten (build_muenchen_stationsliste.py: category='S', operator='800725').

Diese Datei ist NUR für Schritt 1 gedacht. Sie darf ruhig zu großzügig sein
(Duplikate/fremde Regionalbahnhöfe schaden hier nicht, siehe Bauplan 0.1:
"Alle Spalten behalten... wenn nur 1% der Zeilen bleibt") -- sie darf aber
NICHTS vom echten S-Bahn-Netz abschneiden.

Vorherige Version dieser Bbox (lat 47.85-48.45, lon 10.95-11.85) hat den
kompletten Erdinger Außenast (Grafing, Ebersberg, Kirchseeon, Markt Schwaben,
Erding, Altenerding, ...) abgeschnitten, weil deren Längengrad bei ca.
11.86-11.97 liegt -- knapp über der alten Obergrenze. Per Wikipedia-Abgleich
(Liste der Stationen der S-Bahn München, 150 aktive Stationen) korrigiert:
neue Obergrenze 12.05, mit Sicherheitsmarge.
"""

from __future__ import annotations

import pandas as pd

from src.utils.constants import OUTPUTS_DIR, STOPS_PARQUET

OUT_PATH = OUTPUTS_DIR / "muenchen_bootstrap_kandidaten.csv"

# Bbox um das komplette Münchner S-Bahn-Netz inkl. aller Außenäste, mit Marge.
# Extremfälle laut Netz: Geltendorf (lon 11.04), Ebersberg (lon 11.97),
# Holzkirchen (lat 47.88), Petershausen/Freising (lat ~48.40-48.42).
BBOX_LAT = (47.80, 48.45)
BBOX_LON = (10.95, 12.05)


def build() -> pd.DataFrame:
    df = pd.read_parquet(STOPS_PARQUET)
    is_station = df["location_type"] == "STATION"
    in_bbox = df["stop_lat"].between(*BBOX_LAT) & df["stop_lon"].between(*BBOX_LON)
    candidates = df[is_station & in_bbox].drop_duplicates(subset="stop_id")
    return candidates[["stop_id", "stop_name", "stop_lat", "stop_lon"]].sort_values("stop_name")


if __name__ == "__main__":
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    result = build()
    result.to_csv(OUT_PATH, index=False)
    print(f"{len(result)} Bootstrap-Kandidaten (Bbox lat={BBOX_LAT}, lon={BBOX_LON})")
    print(f"-> geschrieben nach {OUT_PATH}")

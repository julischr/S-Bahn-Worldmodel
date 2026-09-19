"""
Stufe 0.1 des Bauplans: 2025.tar (20 GB, deutschlandweit) einmalig durchlaufen,
pro Tag auf die Münchner DB-Stationen filtern, den Rest verwerfen.

- Liest jede Tagesdatei einzeln aus dem Tar (kein Vollextrakt nötig, das Tar ist
  unkomprimiert -> sequenzielles Lesen ist schnell und braucht keinen Extra-Diskplatz).
- Filter: stop_id in outputs/muenchen_stationen_namen.csv (siehe
  build_muenchen_stationsliste.py). NICHT auf die 12 Stammstreckenhalte verengt.
- Alle Spalten bleiben erhalten.
- Protokolliert pro Tag die Zeilenzahl (auch 0, auch fehlende Tage) nach
  outputs/filter_log.csv.
- Schreibt eine Parquet-Datei pro Monat nach data/01_muenchen/.

Laufzeit/Ressourcen: einmaliger Lauf, ca. 20 GB Lesevolumen. Der gefilterte Anteil
liegt laut Voranalyse (explore.py, ein Tag) bei grob 1% der Zeilen -> Ausgabe passt
in den Arbeitsspeicher, es muss nichts inkrementell auf Platte gestreamt werden.
"""

from __future__ import annotations

import tarfile
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.utils.constants import DATA_DIR, OUTPUTS_DIR, RAW_2025_TAR

RAW_TAR = RAW_2025_TAR
# Bewusst die breite Bootstrap-Liste, nicht die precise muenchen_stationen_namen.csv:
# letztere wird ERST aus dem Ergebnis dieses Laufs abgeleitet (siehe
# build_bootstrap_kandidaten.py für die Begründung).
STATIONS_CSV = OUTPUTS_DIR / "muenchen_bootstrap_kandidaten.csv"
OUT_DIR = DATA_DIR / "01_muenchen"
LOG_PATH = OUTPUTS_DIR / "filter_log.csv"

YEAR = 2025


def load_station_ids() -> set[int]:
    df = pd.read_csv(STATIONS_CSV)
    return set(df["stop_id"].astype("int64").tolist())


def expected_days(year: int) -> list[date]:
    d = date(year, 1, 1)
    days = []
    while d.year == year:
        days.append(d)
        d += timedelta(days=1)
    return days


def filter_day(tf: tarfile.TarFile, member: tarfile.TarInfo, station_ids: set[int]) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        fileobj = tf.extractfile(member)
        tmp.write(fileobj.read())
        tmp.flush()
        df = pd.read_parquet(tmp.name)
    return df[df["stop_id"].isin(station_ids)].copy()


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    station_ids = load_station_ids()
    print(f"{len(station_ids)} Münchner Stations-IDs geladen aus {STATIONS_CSV}")

    with tarfile.open(RAW_TAR, "r") as tf:
        members_by_day = {}
        for m in tf.getmembers():
            if not m.name.endswith(".parquet"):
                continue
            try:
                d = date.fromisoformat(m.name.removesuffix(".parquet"))
            except ValueError:
                continue
            members_by_day[d] = m

        log_rows = []

        def flush_month(month_key: str, frames: list[pd.DataFrame]) -> None:
            month_df = pd.concat(frames, ignore_index=True)
            out_path = OUT_DIR / f"{month_key}.parquet"
            month_df.to_parquet(out_path, index=False)
            size_mb = out_path.stat().st_size / 1e6
            print(f"  -> {month_key}: {len(month_df):,} Zeilen geschrieben nach {out_path} ({size_mb:.1f} MB)")

        current_month = None
        current_frames: list[pd.DataFrame] = []

        for d in expected_days(YEAR):
            month_key = f"{d.year:04d}-{d.month:02d}"
            if current_month is not None and month_key != current_month:
                flush_month(current_month, current_frames)
                current_frames = []
            current_month = month_key

            member = members_by_day.get(d)
            if member is None:
                print(f"{d}: FEHLT im Tar")
                log_rows.append({"date": d.isoformat(), "rows_muenchen": None, "status": "missing"})
                continue

            df_filtered = filter_day(tf, member, station_ids)
            n_rows = len(df_filtered)
            current_frames.append(df_filtered)

            print(f"{d}: {n_rows:,} Münchner Zeilen")
            log_rows.append({"date": d.isoformat(), "rows_muenchen": n_rows, "status": "ok"})

        if current_frames:
            flush_month(current_month, current_frames)

    pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)
    print(f"\nLog geschrieben nach {LOG_PATH}")


if __name__ == "__main__":
    run()

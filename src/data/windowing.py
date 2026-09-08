"""Build 5-minute Stammstrecke state matrices from raw parquet files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.constants import OUTPUTS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR


def _normalize_station_name(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())

def get_validated_stammstrecke_ids(csv_path: str | Path) -> dict:
    """
    Liest die Stations-CSV ein und validiert die 12 Stammstrecken-Stationen.
    """
    print(f"Lade und validiere Stationen aus {csv_path}...")
    df_stations = pd.read_csv(csv_path)

    col_name = "name"
    col_id = "stop_id"

    # Mehrere Schreibweisen pro Halt abfangen, damit die CSV robust validiert wird.
    expected_order = {
        "Pasing": ["pasing"],
        "Laim": ["laim"],
        "Hirschgarten": ["hirschgarten"],
        "Donnersbergerbrücke": ["donnersbergerbruecke", "donnersbergerbrücke"],
        "Hackerbrücke": ["hackerbruecke", "hackerbrücke"],
        "Hauptbahnhof (tief)": ["hauptbahnhoftief", "hauptbahnhof", "muenchenhauptbahnhof"],
        "Karlsplatz (Stachus)": ["karlsplatzstachus", "stachus"],
        "Marienplatz": ["marienplatz"],
        "Isartor": ["isartor"],
        "Rosenheimer Platz": ["rosenheimerplatz"],
        "Ostbahnhof": ["ostbahnhof", "muenchenost", "münchenost"],
        "Leuchtenbergring": ["leuchtenbergring"],
    }

    station_dict = {}
    normalized_names = df_stations[col_name].fillna("").map(_normalize_station_name)
    for expected_name, aliases in expected_order.items():
        match = df_stations[normalized_names.isin(aliases) | normalized_names.str.contains("|".join(aliases), regex=True)]

        assert not match.empty, f"FEHLER: Station '{expected_name}' nicht in CSV gefunden!"
        assert len(match) == 1, f"FEHLER: Station '{expected_name}' nicht eindeutig ({len(match)} Treffer)!"

        station_dict[expected_name] = match[col_id].values[0]

    print("✅ Stationen erfolgreich validiert.")
    return station_dict


def create_stammstrecke_state_matrix(
    data_dir: str | Path,
    stations_csv: str | Path,
    output_dir: str | Path,
    freq: str = '5min'
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. IDs sicher laden
    STAMM_STATIONEN = get_validated_stammstrecke_ids(stations_csv)

    # 2. Daten laden und exakte Duplikate entfernen
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Keine Parquet-Dateien in {data_dir} gefunden.")

    print(f"\nLade {len(parquet_files)} Datei(en)...")
    
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    initial_len = len(df)
    
    df = df.drop_duplicates()
    print(f"Exakte Datei-Duplikate entfernt: {initial_len - len(df)} Zeilen")

    # 3. Essenzielle Filter
    mask = (
        (df['is_final'] == True) & 
        (df['is_cancelled'] == False) & 
        (df['pickup_drop_off_type'] != 'NOT_AVAILABLE') &
        (df['is_arrival'] == True)
    )
    df = df[mask].copy()

    # Sortierung: na_position='first' drückt NULL-Timestamps nach oben.
    # keep='last' behält dadurch bevorzugt Zeilen MIT echtem Zeitstempel.
    df = df.sort_values('update_timestamp', na_position='first').drop_duplicates(
        subset=['trip_id', 'stop_id'], keep='last'
    )

    # 4. Hauptbahnhof-Alias auflösen und filtern
    df['stop_id'] = df['stop_id'].replace({8098261: 8098263})
    df = df[df['stop_id'].isin(list(STAMM_STATIONEN.values()))]

    # 5. Zeiten und Metriken berechnen
    df['actual_time'] = df['time_schedule'] + pd.to_timedelta(df['delay'], unit='s')
    df['delay_min'] = df['delay'] / 60.0
    df = df.set_index('actual_time')

    # 6. Aggregation (State und Count)
    print(f"Aggregiere auf {freq}-Fenster...")
    grouped = df.groupby([
        pd.Grouper(freq=freq, label='right', closed='right'),
        'stop_id'
    ])
    
    state_matrix = grouped['delay_min'].mean().unstack()
    count_matrix = grouped.size().unstack(fill_value=0)

    # 7. Spaltenreihenfolge erzwingen und umbenennen
    id_to_name = {v: k for k, v in STAMM_STATIONEN.items()}
    state_matrix = state_matrix.rename(columns=id_to_name)
    count_matrix = count_matrix.rename(columns=id_to_name)
    
    ordered_names = list(STAMM_STATIONEN.keys())
    state_matrix = state_matrix.reindex(columns=ordered_names)
    count_matrix = count_matrix.reindex(columns=ordered_names)

    # 8. Kontrollierte Imputation (30 Minuten überbrücken)
    state_matrix = state_matrix.ffill(limit=6)
    
    # Speichern
    state_path = output_dir / "stammstrecke_5min.parquet"
    count_path = output_dir / "stammstrecke_5min_counts.parquet"
    state_matrix.to_parquet(state_path)
    count_matrix.to_parquet(count_path)
    
    print(f"\nZustandsmatrix Shape: {state_matrix.shape}")
    print(f"Countmatrix Shape: {count_matrix.shape}")
    print(f"Gespeichert: {state_path}")
    print(f"Gespeichert: {count_path}")
    print("Fertig! Bereit für das episodische VAR-Training.")

if __name__ == "__main__":
    create_stammstrecke_state_matrix(
        data_dir=RAW_DATA_DIR,
        stations_csv=OUTPUTS_DIR / "muenchen_stationen.csv",
        output_dir=PROCESSED_DATA_DIR,
    )
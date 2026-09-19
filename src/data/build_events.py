"""
Stufe 0.2 des Bauplans: aus den gefilterten Münchner Rohzeilen (data/01_muenchen/)
zwei Ereignistabellen mit unterschiedlicher Korngröße bauen.

events_final        -- eine Zeile je (trip_id, stop_id, is_arrival), is_final=true
events_progression  -- alle Meldungen (Prognoseverlauf über update_timestamp)

Beide Tabellen bekommen dieselbe Bereinigung:
  - category = 'S' (nur Münchner S-Bahn-Verkehr; data/01_muenchen enthält an
    denselben stop_ids auch Regional-/Fernverkehr, siehe Analyse vor diesem Skript)
  - is_betriebshalt = false verworfen (~0,02%)
  - pickup_drop_off_type = 'NOT_AVAILABLE' verworfen (~0,44%)
  - delay < -1200 (unter -20 Minuten) verworfen. Ein Histogramm der negativen
    delays (siehe outputs/plots/delay_histogram_negativ.png) zeigt für die
    Münchner S-Bahn KEINE scharfe Lücke zwischen zwei Häufungen wie im
    bundesweiten Befund (BEFUNDE.md, dort bis -87180s durch Datumsüberträge) --
    stattdessen einen glatten Abfall. Nur 916 von 124 Mio. Zeilen (0,0007%)
    liegen unter -1200s; die Schwelle aus dem Bauplan entfernt diese kleine,
    physikalisch unplausible Ausreißer-Menge, ohne den legitimen
    Fahrplanpuffer-Bereich (bis grob -660s) zu beschneiden.
  - is_cancelled wird NICHT verworfen, bleibt als Spalte erhalten.

Richtung: pro trip_id wird aus der Reihenfolge der stop_sequence an den 12
Stammstreckenhalten abgeleitet, ob die Fahrt ost- oder westwärts läuft
(Spalte 'richtung'). Für Fahrten, die die Stammstrecke nicht oder nur an
einem einzigen Kernhalt berühren, bleibt richtung NaN (nicht bestimmbar).

Zeitzone: time_schedule/time_real/update_timestamp liegen bereits in UTC
(siehe Schema-Check) und werden hier NICHT nach Europe/Berlin konvertiert --
das 5-Minuten-Raster wird laut Bauplan in UTC gebaut und erst am Ende
(Stufe 0.3) umgerechnet, um die Sommerzeitumstellung sauber zu behandeln.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.constants import DATA_DIR, PLOTS_DIR

MUENCHEN_GLOB = str(DATA_DIR / "01_muenchen" / "*.parquet")
EVENTS_DIR = DATA_DIR / "02_events"
FINAL_DIR = EVENTS_DIR / "final"
PROGRESSION_DIR = EVENTS_DIR / "progression"

DELAY_CUTOFF = -1200  # -20 Minuten, siehe Docstring

# West -> Ost, siehe explore.py / output/muenchen_stationen.csv
STAMMSTRECKE_ORDER = [
    8004158,  # Pasing
    8004151,  # Laim
    8004179,  # Hirschgarten
    8004128,  # Donnersbergerbrücke
    8004129,  # Hackerbrücke
    8098263,  # Hauptbahnhof (tief)
    8004132,  # Karlsplatz (Stachus)
    8004135,  # Marienplatz
    8004131,  # Isartor
    8004136,  # Rosenheimer Platz
    8000262,  # Ostbahnhof
    8004134,  # Leuchtenbergring
]
CANONICAL_INDEX = {sid: i for i, sid in enumerate(STAMMSTRECKE_ORDER)}


def plot_negative_delay_histogram(con: duckdb.DuckDBPyConnection) -> None:
    df = con.execute(
        f"""
        SELECT CASE WHEN delay < -1800 THEN -1800 ELSE CAST(floor(delay/30.0)*30 AS INT) END AS bucket,
               count(*) AS c
        FROM read_parquet('{MUENCHEN_GLOB}')
        WHERE category = 'S' AND delay < 0
        GROUP BY 1 ORDER BY 1
        """
    ).fetchdf()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["bucket"], df["c"], width=25, color="#4C72B0")
    ax.axvline(DELAY_CUTOFF, color="#C44E52", linestyle="--", label=f"Cutoff {DELAY_CUTOFF}s (-20min)")
    ax.set_xlabel("delay (Sekunden, negativ = zu früh)")
    ax.set_ylabel("Anzahl Zeilen (log)")
    ax.set_yscale("log")
    ax.set_title("Verteilung negativer delays, Münchner S-Bahn 2025 (category='S')")
    ax.legend()
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / "delay_histogram_negativ.png", dpi=150)
    plt.close(fig)

    n_below_cutoff = con.execute(
        f"SELECT count(*) FROM read_parquet('{MUENCHEN_GLOB}') WHERE category='S' AND delay < {DELAY_CUTOFF}"
    ).fetchone()[0]
    n_total = con.execute(
        f"SELECT count(*) FROM read_parquet('{MUENCHEN_GLOB}') WHERE category='S'"
    ).fetchone()[0]
    print(f"delay < {DELAY_CUTOFF}s: {n_below_cutoff:,} von {n_total:,} Zeilen ({n_below_cutoff/n_total:.4%})")


def build_richtung_lookup(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    core_ids_sql = ",".join(str(x) for x in STAMMSTRECKE_ORDER)
    df = con.execute(
        f"""
        SELECT DISTINCT trip_id, stop_id, stop_sequence
        FROM read_parquet('{MUENCHEN_GLOB}')
        WHERE category = 'S' AND stop_id IN ({core_ids_sql})
        """
    ).fetchdf()
    df["canon"] = df["stop_id"].map(CANONICAL_INDEX)
    df = df.sort_values(["trip_id", "stop_sequence"])

    def direction(group: pd.DataFrame) -> str | float:
        canon = group["canon"].tolist()
        if len(set(canon)) < 2:
            return float("nan")
        if canon[-1] > canon[0]:
            return "Ost"
        elif canon[-1] < canon[0]:
            return "West"
        return float("nan")

    richtung = df.groupby("trip_id").apply(direction, include_groups=False)
    richtung = richtung.rename("richtung").reset_index()
    print(f"Richtung bestimmt für {richtung['richtung'].notna().sum():,} von {len(richtung):,} Fahrten "
          f"mit Stammstreckenkontakt")
    print(richtung["richtung"].value_counts(dropna=False))
    return richtung


def build_event_table(
    con: duckdb.DuckDBPyConnection,
    richtung_lookup: pd.DataFrame,
    *,
    require_final: bool,
    out_dir: Path,
) -> None:
    con.register("richtung_lookup", richtung_lookup)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_clause = "AND is_final = true" if require_final else ""
    # Für events_final: falls doch mal >1 is_final-Zeile je Gruppe vorkommt
    # (siehe BEFUNDE.md, 0.001% der Fälle), die chronologisch letzte behalten.
    dedup_select = (
        """
        QUALIFY row_number() OVER (
            PARTITION BY trip_id, stop_id, is_arrival
            ORDER BY update_timestamp DESC NULLS LAST
        ) = 1
        """
        if require_final
        else ""
    )

    for month in range(1, 13):
        month_str = f"2025-{month:02d}"
        src = str(DATA_DIR / "01_muenchen" / f"{month_str}.parquet")
        if not Path(src).exists():
            continue
        out_path = out_dir / f"{month_str}.parquet"
        query = f"""
            COPY (
                SELECT r.*, l.richtung
                FROM read_parquet('{src}') r
                LEFT JOIN richtung_lookup l USING (trip_id)
                WHERE r.category = 'S'
                  AND r.is_betriebshalt = false
                  AND r.pickup_drop_off_type != 'NOT_AVAILABLE'
                  AND r.delay >= {DELAY_CUTOFF}
                  {final_clause}
                {dedup_select}
            ) TO '{out_path}' (FORMAT PARQUET)
        """
        con.execute(query)
        n = con.execute(f"SELECT count(*) FROM read_parquet('{out_path}')").fetchone()[0]
        print(f"  {month_str}: {n:,} Zeilen -> {out_path}")


def run() -> None:
    con = duckdb.connect()

    print("=== Histogramm negative delays ===")
    plot_negative_delay_histogram(con)

    print("\n=== Richtung pro Fahrt ===")
    richtung_lookup = build_richtung_lookup(con)

    print("\n=== events_final ===")
    build_event_table(con, richtung_lookup, require_final=True, out_dir=FINAL_DIR)

    print("\n=== events_progression ===")
    build_event_table(con, richtung_lookup, require_final=False, out_dir=PROGRESSION_DIR)


if __name__ == "__main__":
    run()

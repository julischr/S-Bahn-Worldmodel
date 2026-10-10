"""Abschnitt 11.6.1: Klaerung der 335.673 Zugpaare mit Headway<=0 aus Abschnitt 11.3.

In 11.3 (`plausibilitaet_zugebene.py`) wurden von 3.433.425 moeglichen Vorgaenger/Folgezug-
Paaren an derselben Station/Richtung 335.673 (9,8%) mit Headway<=0 (time_schedule-Differenz)
kommentarlos ausgeschlossen ("vermutlich zeitgleich geplante Zuege unterschiedlicher Linien
oder Datenartefakte"). Dieses Modul untersucht diese Paare genauer:

1. Verteilung des (negativen) Headways -- exakt 0? kleine negative Werte? grosse Spruenge?
2. Hypothese "gekoppelter Zugverband" (zwei trip_ids, ein physischer Zugverband): Paare mit
   Headway<=0 UND nahezu identischen time_schedule-Zeiten an MEHREREN gemeinsamen
   Stammstrecke-Stationen (nicht nur der einen betrachteten) sind ein starkes Indiz fuer einen
   gekoppelten/verstaerkten Zugverband (z.B. zwei Kurzzuege, die als ein Verband fahren, aber
   in den GTFS-Realtime-Daten zwei trip_ids besitzen).
3. Alternative: Datenduplikate (identische trip_id mehrfach) -- wird direkt anhand der
   Rohdaten unter data/02_events/final/ geprueft (siehe check_no_duplicate_events()).
4. Alternative: echte Ueberholung (ein spaeter geplanter Zug faehrt vor einem frueher
   geplanten) -- da der Headway hier auf time_schedule (Fahrplanzeiten), NICHT auf time_real
   basiert, kann eine reale Ueberholung allein keinen negativen SCHEDULE-Headway erzeugen;
   das wird explizit dokumentiert/geprueft.
5. Quantifizierte Zuordnung der 335.673 Paare zu den Kategorien.

Nur lesender Zugriff auf data/02_events/final/, src/analysis/var/stations.py,
src/analysis/var/plausibilitaet_zugebene.py. Neue Outputs unter outputs/var/plausibilitaet/
mit Praefix "headwayartefakt_", neuer Plot 25.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.constants import DATA_DIR, PROJECT_ROOT
from .stations import STOP_ID_TO_NAME
from .plausibilitaet_zugebene import EVENTS_GLOB, STAMM_IDS

OUT_DIR = PROJECT_ROOT / "outputs" / "var" / "plausibilitaet"
PLOT_DIR = PROJECT_ROOT / "outputs" / "var" / "plots"

# Schwellen zur Kategorisierung
EXACT_ZERO_TOL_SEC = 0.5          # headway == 0 (numerisch exakt)
SMALL_NEG_MIN = -2.0              # -2min < headway < 0 => "klein negativ"
ALIGN_TOL_SEC = 60.0              # Toleranz fuer "nahezu identische" time_schedule an anderen Stationen


def load_events_full() -> pd.DataFrame:
    """Wie plausibilitaet_zugebene.load_events, aber zusaetzlich trip_headsign und line
    fuer die Verbands-Pruefung."""
    con = duckdb.connect()
    q = f"""
        SELECT stop_id, richtung, trip_id, time_schedule, delay, line, trip_headsign, stop_sequence
        FROM read_parquet('{EVENTS_GLOB}')
        WHERE is_arrival = true
          AND is_cancelled = false
          AND richtung IS NOT NULL
          AND stop_id IN ({STAMM_IDS})
        ORDER BY stop_id, richtung, time_schedule
    """
    df = con.execute(q).fetchdf()
    df["station_name"] = df["stop_id"].map(STOP_ID_TO_NAME)
    return df


def build_all_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Wie build_pairs in 11.3, aber OHNE den headway>0-Filter -- alle Paare inkl. headway<=0
    und headway>60min werden zurueckgegeben, mit Kategorie-Spalte."""
    df = df.sort_values(["station_name", "richtung", "time_schedule"]).reset_index(drop=True)
    g = df.groupby(["station_name", "richtung"], sort=False)
    df["trip_id_prev"] = g["trip_id"].shift(1)
    df["time_schedule_prev"] = g["time_schedule"].shift(1)
    df["delay_prev"] = g["delay"].shift(1)
    df["line_prev"] = g["line"].shift(1)
    df["headsign_prev"] = g["trip_headsign"].shift(1)

    pairs = df.dropna(subset=["trip_id_prev"]).copy()
    pairs = pairs[pairs["trip_id_prev"] != pairs["trip_id"]]
    pairs["headway_sec"] = (pairs["time_schedule"] - pairs["time_schedule_prev"]).dt.total_seconds()
    pairs["headway_min"] = pairs["headway_sec"] / 60.0
    pairs = pairs.rename(columns={"delay": "delay_next", "line": "line_next", "trip_headsign": "headsign_next"})
    return pairs


def check_no_duplicate_events() -> pd.DataFrame:
    """Prueft direkt in den Rohdaten, ob (trip_id, stop_id, is_arrival) je mehrfach vorkommt
    (Datenduplikat-Hypothese). Falls final/ bereits vollstaendig dedupliziert ist, sollte das
    Ergebnis 0 Duplikate sein."""
    con = duckdb.connect()
    q = f"""
        SELECT trip_id, stop_id, is_arrival, count(*) AS n
        FROM read_parquet('{EVENTS_GLOB}')
        WHERE stop_id IN ({STAMM_IDS})
        GROUP BY 1, 2, 3
        HAVING count(*) > 1
    """
    return con.execute(q).fetchdf()


def check_convoy_alignment(neg_pairs: pd.DataFrame, all_events: pd.DataFrame) -> pd.DataFrame:
    """Fuer jedes eindeutige (trip_id_prev, trip_id)-Paar aus den Headway<=0-Faellen: an wie
    vielen der 12 Stammstrecke-Stationen kommen BEIDE trip_ids gemeinsam vor, und an wie vielen
    davon liegt |time_schedule-Differenz| <= ALIGN_TOL_SEC (= "fast zeitgleich")? Ein Paar, das
    an >=2 gemeinsamen Stationen fast zeitgleich faehrt, ist ein starkes Indiz fuer einen
    gekoppelten Zugverband (nicht nur ein Zufallstreffer an einer Station)."""
    uniq_pairs = neg_pairs[["trip_id_prev", "trip_id"]].drop_duplicates()
    uniq_pairs.columns = ["t1", "t2"]

    ev = all_events[["trip_id", "station_name", "time_schedule", "line", "stop_sequence"]].copy()

    # Nur trip_ids betrachten, die tatsaechlich in den Headway<=0-Paaren vorkommen (Performance)
    relevant_trips = set(uniq_pairs["t1"]) | set(uniq_pairs["t2"])
    ev = ev[ev["trip_id"].isin(relevant_trips)]

    e1 = ev.rename(columns={"trip_id": "t1", "time_schedule": "t_t1", "line": "line_t1"})
    e2 = ev.rename(columns={"trip_id": "t2", "time_schedule": "t_t2", "line": "line_t2"})

    merged = uniq_pairs.merge(e1[["t1", "station_name", "t_t1", "line_t1"]], on="t1", how="left")
    merged = merged.merge(e2[["t2", "station_name", "t_t2", "line_t2"]], on=["t2", "station_name"], how="inner")
    merged["diff_sec"] = (merged["t_t1"] - merged["t_t2"]).dt.total_seconds().abs()
    merged["aligned"] = merged["diff_sec"] <= ALIGN_TOL_SEC

    agg = merged.groupby(["t1", "t2"]).agg(
        n_common_stations=("station_name", "nunique"),
        n_aligned_stations=("aligned", "sum"),
        gleiche_linie=("line_t1", lambda s: (s.values == merged.loc[s.index, "line_t2"].values).all()),
    ).reset_index()
    agg["konvoi_kandidat"] = agg["n_aligned_stations"] >= 2
    return agg


def categorize(neg_pairs: pd.DataFrame, convoy: pd.DataFrame) -> pd.DataFrame:
    df = neg_pairs.merge(
        convoy.rename(columns={"t1": "trip_id_prev", "t2": "trip_id"}),
        on=["trip_id_prev", "trip_id"], how="left"
    )
    df["n_aligned_stations"] = df["n_aligned_stations"].fillna(0)
    df["konvoi_kandidat"] = df["konvoi_kandidat"].fillna(False)

    def cat(row):
        if row["konvoi_kandidat"]:
            return "gekoppelter_verband"
        if abs(row["headway_sec"]) <= EXACT_ZERO_TOL_SEC:
            return "exakt_zeitgleich_ungeklaert"
        if row["headway_min"] > SMALL_NEG_MIN:
            return "klein_negativ_ungeklaert"
        return "gross_negativ_ungeklaert"

    df["kategorie"] = df.apply(cat, axis=1)
    return df


def plot_summary(dist: pd.Series, cat_counts: pd.Series, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(dist.clip(-30, 0), bins=60, color="#2A6F97")
    ax.set_xlabel("Headway (Minuten, <=0, bei -30min gekappt)")
    ax.set_ylabel("Anzahl Paare")
    ax.set_title("Verteilung des negativen Headways")
    ax.axvline(0, color="grey", linewidth=0.8)

    ax2 = axes[1]
    order = cat_counts.sort_values(ascending=True)
    ax2.barh(order.index, order.values, color="#C4471C")
    ax2.set_xlabel("Anzahl Paare")
    ax2.set_title("Kategorisierung der Headway<=0-Paare")
    for i, v in enumerate(order.values):
        ax2.text(v, i, f" {v:,}", va="center", fontsize=8)

    fig.suptitle("Plot 25: Klaerung Headway<=0-Paare (Abschnitt 11.6.1)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Lade Ankunfts-Ereignisse ...")
    df = load_events_full()
    print(f"{len(df):,} Ereignisse geladen.")

    all_pairs = build_all_pairs(df)
    neg_pairs = all_pairs[all_pairs["headway_min"] <= 0].copy()
    print(f"Paare mit headway<=0: {len(neg_pairs):,} (erwartet ~335.673, konsistent zu 11.3)")

    neg_pairs.to_csv(OUT_DIR / "headwayartefakt_negative_paare_rohdaten.csv.gz",
                      index=False, compression="gzip")

    print("\n--- Verteilung headway_min<=0 ---")
    print(neg_pairs["headway_min"].describe().to_string())
    n_exact_zero = int((neg_pairs["headway_sec"].abs() <= EXACT_ZERO_TOL_SEC).sum())
    n_small_neg = int(((neg_pairs["headway_min"] > SMALL_NEG_MIN) & (neg_pairs["headway_min"] < 0)).sum())
    n_large_neg = int((neg_pairs["headway_min"] <= SMALL_NEG_MIN).sum())
    print(f"exakt 0: {n_exact_zero:,} ({n_exact_zero/len(neg_pairs):.1%}); "
          f"klein negativ (-2,0)min: {n_small_neg:,} ({n_small_neg/len(neg_pairs):.1%}); "
          f"<=-2min: {n_large_neg:,} ({n_large_neg/len(neg_pairs):.1%})")

    print("\n--- Datenduplikat-Check (trip_id,stop_id,is_arrival) ---")
    dupes = check_no_duplicate_events()
    print(f"Anzahl (trip_id,stop_id,is_arrival)-Kombinationen mit >1 Vorkommen: {len(dupes):,}")
    dupes.to_csv(OUT_DIR / "headwayartefakt_duplikat_check.csv", index=False)

    print("\n--- Konvoi/gekoppelter-Verband-Check ---")
    convoy = check_convoy_alignment(neg_pairs, df)
    convoy.to_csv(OUT_DIR / "headwayartefakt_konvoi_kandidaten.csv.gz", index=False, compression="gzip")
    n_konvoi_paare = int(convoy["konvoi_kandidat"].sum())
    print(f"Eindeutige trip_id-Paare gesamt: {len(convoy):,}; davon Konvoi-Kandidaten "
          f"(>=2 fast-zeitgleiche gemeinsame Stationen): {n_konvoi_paare:,} "
          f"({n_konvoi_paare/len(convoy):.1%} der eindeutigen Paare)")

    categorized = categorize(neg_pairs, convoy)
    categorized.to_csv(OUT_DIR / "headwayartefakt_kategorisiert_rohdaten.csv.gz",
                        index=False, compression="gzip")

    cat_counts = categorized["kategorie"].value_counts()
    cat_summary = pd.DataFrame({
        "kategorie": cat_counts.index,
        "n_zeilen_paare": cat_counts.values,
        "anteil": (cat_counts.values / len(categorized)).round(4),
    })
    cat_summary.to_csv(OUT_DIR / "headwayartefakt_kategorien_zusammenfassung.csv", index=False)
    print("\n--- Kategorisierung (auf Zeilen-/Paar-Ebene, nicht eindeutige trip-Paare) ---")
    print(cat_summary.to_string(index=False))

    plot_summary(neg_pairs["headway_min"], cat_counts, PLOT_DIR / "25_headway_negativ_klaerung.png")
    print(f"\nPlot gespeichert: {PLOT_DIR / '25_headway_negativ_klaerung.png'}")

    return dict(neg_pairs=neg_pairs, convoy=convoy, categorized=categorized, cat_summary=cat_summary)


if __name__ == "__main__":
    main()

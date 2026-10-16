"""Abschnitt 11.1: Plausibilitaetstest -- fahrplanmaessige Fahrzeit Pasing<->Leuchtenbergring
im Vergleich zur VAR-Peak-Zeit von 25 Minuten (BEFUNDE_VAR.md Abschnitt 4.4/9.9).

Nutzt data/02_events/final/*.parquet (Zugebene, 12 Monatsdateien), NICHT den aggregierten
state_vector_5min.parquet der Hauptanalyse. Fuer jede trip_id, die sowohl Pasing (8004158)
als auch Leuchtenbergring (8004134) anfaehrt (Ankunfts-Ereignis, is_arrival=true, konsistent
zur Definition von delay_mean im Hauptlauf -- siehe src/data/build_state_vector.py), wird die
fahrplanmaessige Fahrzeit (time_schedule-Differenz) berechnet. Richtung wird PRIMAER ueber
stop_sequence bestimmt (kleinere stop_sequence = zuerst durchfahren) und gegen die
vorhandene richtung-Spalte der Events validiert.

Nur lesender Zugriff auf data/02_events/final/, src/analysis/var/*, outputs/var/* (bestehende
Teile). Neue Outputs unter outputs/var/plausibilitaet/, neuer Plot 21.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.ausfall_bloecke import events_final_dir
from src.utils.constants import DATA_DIR, PROJECT_ROOT

EVENTS_GLOB = str(events_final_dir() / "*.parquet")
OUT_DIR = PROJECT_ROOT / "outputs" / "var" / "plausibilitaet"
PLOT_DIR = PROJECT_ROOT / "outputs" / "var" / "plots"

PASING = 8004158
LEUCHTENBERGRING = 8004134

VAR_PEAK_MIN = 25.0


def load_trip_pairs() -> pd.DataFrame:
    """Fuer jede trip_id mit Ankunfts-Ereignis an BEIDEN Stationen: Zeitpunkt, stop_sequence
    und (falls vorhanden) richtung je Station."""
    con = duckdb.connect()
    q = f"""
    WITH ev AS (
        SELECT trip_id, stop_id, stop_sequence, time_schedule, richtung
        FROM read_parquet('{EVENTS_GLOB}')
        WHERE is_arrival = true
          AND is_cancelled = false
          AND stop_id IN ({PASING}, {LEUCHTENBERGRING})
    ),
    piv AS (
        SELECT trip_id,
            max(CASE WHEN stop_id = {PASING} THEN time_schedule END) AS t_pasing,
            max(CASE WHEN stop_id = {PASING} THEN stop_sequence END) AS seq_pasing,
            max(CASE WHEN stop_id = {PASING} THEN richtung END) AS richtung_pasing,
            max(CASE WHEN stop_id = {LEUCHTENBERGRING} THEN time_schedule END) AS t_leuchtenbergring,
            max(CASE WHEN stop_id = {LEUCHTENBERGRING} THEN stop_sequence END) AS seq_leuchtenbergring,
            max(CASE WHEN stop_id = {LEUCHTENBERGRING} THEN richtung END) AS richtung_leuchtenbergring,
            count(*) AS n_events
        FROM ev
        GROUP BY trip_id
    )
    SELECT * FROM piv
    WHERE t_pasing IS NOT NULL AND t_leuchtenbergring IS NOT NULL AND n_events = 2
    """
    df = con.execute(q).fetchdf()
    return df


def classify_and_compute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["richtung_seq"] = np.where(
        df["seq_pasing"] < df["seq_leuchtenbergring"], "Ost", "West"
    )
    fahrzeit_min = np.where(
        df["richtung_seq"] == "Ost",
        (df["t_leuchtenbergring"] - df["t_pasing"]).dt.total_seconds() / 60.0,
        (df["t_pasing"] - df["t_leuchtenbergring"]).dt.total_seconds() / 60.0,
    )
    df["fahrzeit_min"] = fahrzeit_min
    return df


def cross_validate_richtung(df: pd.DataFrame) -> dict:
    """Vergleicht die stop_sequence-basierte Richtungszuordnung mit der vorhandenen
    richtung-Spalte der Events (wo diese an BEIDEN Stationen uebereinstimmend vorliegt)."""
    both_present = df.dropna(subset=["richtung_pasing", "richtung_leuchtenbergring"])
    consistent_within = (both_present["richtung_pasing"] == both_present["richtung_leuchtenbergring"])
    matched = both_present[consistent_within]
    agree_with_seq = (matched["richtung_seq"] == matched["richtung_pasing"])
    return {
        "n_total_trips": int(len(df)),
        "n_mit_richtung_beidseitig": int(len(both_present)),
        "n_richtung_pasing_leuchtenbergring_uebereinstimmend": int(consistent_within.sum()),
        "anteil_seq_stimmt_mit_richtung_ueberein": float(agree_with_seq.mean()) if len(matched) else float("nan"),
        "n_verglichen": int(len(matched)),
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for richtung, g in df.groupby("richtung_seq"):
        x = g["fahrzeit_min"]
        # Physikalisch unplausible Werte (negative oder extrem lange Fahrzeiten, z.B. durch
        # falsch zugeordnete trip_id-Wiederverwendung ueber Tagesgrenzen) ausschliessen: eine
        # Fahrt Pasing<->Leuchtenbergring dauert real deutlich unter 2 Stunden.
        plausible = x[(x > 0) & (x < 120)]
        rows.append({
            "richtung": richtung,
            "n_trips_roh": int(len(x)),
            "n_trips_plausibel": int(len(plausible)),
            "n_ausgeschlossen_unplausibel": int(len(x) - len(plausible)),
            "median_min": float(plausible.median()),
            "mean_min": float(plausible.mean()),
            "std_min": float(plausible.std()),
            "p10_min": float(plausible.quantile(0.10)),
            "p90_min": float(plausible.quantile(0.90)),
            "min_min": float(plausible.min()),
            "max_min": float(plausible.max()),
            "diff_zu_var_peak_25min": float(plausible.median() - VAR_PEAK_MIN),
        })
    return pd.DataFrame(rows)


def plot_histogram(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    colors = {"Ost": "#2A6F97", "West": "#C4471C"}
    for ax, richtung in zip(axes, ["Ost", "West"]):
        g = df[df["richtung_seq"] == richtung]["fahrzeit_min"]
        g = g[(g > 0) & (g < 120)]
        ax.hist(g, bins=60, color=colors[richtung], alpha=0.85, edgecolor="white", linewidth=0.3)
        med = g.median()
        ax.axvline(med, color="black", linestyle="-", linewidth=1.5, label=f"Median {med:.1f} min")
        ax.axvline(VAR_PEAK_MIN, color="#333333", linestyle="--", linewidth=1.5,
                   label=f"VAR-Peak-Zeit {VAR_PEAK_MIN:.0f} min")
        ax.set_title(f"Richtung {richtung} (n={len(g):,})")
        ax.set_xlabel("Fahrplanmaessige Fahrzeit Pasing<->Leuchtenbergring [min]")
        ax.set_ylabel("Anzahl Fahrten")
        ax.legend(fontsize=8)
    fig.suptitle("Plot 21: Fahrplanmaessige Fahrzeit Pasing<->Leuchtenbergring vs. VAR-Peak-Zeit (25 min)\n"
                 "[Abschnitt 11.1 Plausibilitaetstest]")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Lade Trip-Paare (Pasing, Leuchtenbergring) aus data/02_events/final/ ...")
    raw = load_trip_pairs()
    print(f"{len(raw):,} trip_ids mit Ankunft an beiden Stationen gefunden.")

    df = classify_and_compute(raw)
    df.to_csv(OUT_DIR / "fahrzeit_pasing_leuchtenbergring_rohdaten.csv", index=False)

    cv = cross_validate_richtung(df)
    pd.Series(cv).to_csv(OUT_DIR / "richtung_kreuzvalidierung.csv")
    print("Kreuzvalidierung stop_sequence vs. richtung-Spalte:")
    for k, v in cv.items():
        print(f"  {k}: {v}")

    summary = summarize(df)
    summary.to_csv(OUT_DIR / "fahrzeit_zusammenfassung.csv", index=False)
    print("\nZusammenfassung:")
    print(summary.round(2).to_string(index=False))

    plot_histogram(df, PLOT_DIR / "21_fahrzeit_pasing_leuchtenbergring.png")
    print(f"\nPlot gespeichert: {PLOT_DIR / '21_fahrzeit_pasing_leuchtenbergring.png'}")

    return dict(raw=raw, df=df, cv=cv, summary=summary)


if __name__ == "__main__":
    main()

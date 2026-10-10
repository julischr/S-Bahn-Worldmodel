"""Abschnitt 11.8: Ausfaelle als moeglicher Uebertragungskanal.

Fragestellung: faellt der Folgezug haeufiger aus (is_cancelled=true), wenn der Vorgaengerzug
stark verspaetet war (v.a. bei kurzem Headway)? Das ist ein alternativer Uebertragungsmechanismus
zur reinen Verspaetungs-Weitergabe (11.3/11.4/11.6/11.7): statt selbst verspaetet zu werden,
faellt der Folgezug im Extremfall ganz aus (z.B. weil ein Fahrzeug/Triebfahrzeugfuehrer wegen
der Vorgaenger-Verspaetung nicht rechtzeitig zur Verfuegung steht).

WICHTIG -- Selektionseffekt in 11.3/11.4/11.6/11.7: dort wurden Vorgaenger/Folgezug-Paare aus
Ankunfts-Ereignissen MIT is_cancelled=false gebildet -- ein ausgefallener Folgezug taucht dort
also gar nicht als "Folgezug" auf, sondern wird stillschweigend uebersprungen (der naechste
NICHT ausgefallene Zug wird dann faelschlich zum unmittelbaren Nachfolger). Genau das wird hier
korrigiert: die Zugfolge wird aus ALLEN geplanten Ankunfts-Ereignissen gebildet (is_cancelled
IN {true, false}), sodass ausgefallene Zuege als eigener "Folgezug"-Fall sichtbar werden. Der
VORGAENGER muss weiterhin nicht ausgefallen sein (sonst ist seine delay-Beobachtung nicht
aussagekraeftig / es ist unklar, ob ER selbst schon Teil einer Ausfallkaskade ist) -- diese
Faelle (Vorgaenger ausgefallen) werden explizit separat ausgewiesen, nicht einfach verworfen.

Klassenbildung wie in 11.7: Vorgaenger-Verspaetung (<=2min, 2-5min, 5-10min, >10min) x
Headway-Bin (<5min, 5-10min, 10-15min, 15-20min, 20-60min, Headway weiterhin aus
time_schedule-Differenz, wie in 11.3, unabhaengig vom Ausfallstatus). Je Zelle: Ausfallrate des
Folgezugs mit 95%-Wilson-Konfidenzintervall.

Nur lesender Zugriff auf data/02_events/final/, stations.py, plausibilitaet_zugebene.py. Neue
Outputs unter outputs/var/plausibilitaet/ mit Praefix "ausfallkanal_", neuer Plot 28.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.constants import PROJECT_ROOT
from .stations import STOP_ID_TO_NAME
from .plausibilitaet_zugebene import EVENTS_GLOB, STAMM_IDS, BIN_EDGES, BIN_LABELS, MAX_HEADWAY_MIN
from .plausibilitaet_nichtlinear import DELAY_BIN_EDGES, DELAY_BIN_LABELS, MIN_N

OUT_DIR = PROJECT_ROOT / "outputs" / "var" / "plausibilitaet"
PLOT_DIR = PROJECT_ROOT / "outputs" / "var" / "plots"


def load_events_incl_cancelled() -> pd.DataFrame:
    """Wie plausibilitaet_zugebene.load_events, aber OHNE is_cancelled=false-Filter --
    ausgefallene Zuege bleiben in der Sequenz, damit sie als moeglicher 'Folgezug' erkennbar
    sind (siehe Docstring oben)."""
    con = duckdb.connect()
    q = f"""
        SELECT stop_id, richtung, trip_id, time_schedule, delay, is_cancelled
        FROM read_parquet('{EVENTS_GLOB}')
        WHERE is_arrival = true
          AND richtung IS NOT NULL
          AND stop_id IN ({STAMM_IDS})
        ORDER BY stop_id, richtung, time_schedule
    """
    df = con.execute(q).fetchdf()
    df["station_name"] = df["stop_id"].map(STOP_ID_TO_NAME)
    return df


def build_pairs_incl_cancelled(df: pd.DataFrame) -> pd.DataFrame:
    """Vorgaenger/Folgezug-Paare ueber die VOLLSTAENDIGE geplante Sequenz (Ausfaelle
    eingeschlossen). Der Vorgaenger muss nicht ausgefallen sein (delay_prev nur dann
    aussagekraeftig); Faelle mit ausgefallenem Vorgaenger werden separat gezaehlt, aber aus der
    Hauptauswertung ausgeschlossen."""
    df = df.sort_values(["station_name", "richtung", "time_schedule"]).reset_index(drop=True)
    g = df.groupby(["station_name", "richtung"], sort=False)
    df["trip_id_prev"] = g["trip_id"].shift(1)
    df["time_schedule_prev"] = g["time_schedule"].shift(1)
    df["delay_prev"] = g["delay"].shift(1)
    df["is_cancelled_prev"] = g["is_cancelled"].shift(1)

    pairs = df.dropna(subset=["trip_id_prev"]).copy()
    pairs = pairs[pairs["trip_id_prev"] != pairs["trip_id"]]
    pairs["headway_min"] = (pairs["time_schedule"] - pairs["time_schedule_prev"]).dt.total_seconds() / 60.0
    pairs = pairs.rename(columns={"is_cancelled": "is_cancelled_next"})

    n_before = len(pairs)
    n_vorgaenger_cancelled = int(pairs["is_cancelled_prev"].sum())
    pairs_clean = pairs[~pairs["is_cancelled_prev"].astype(bool)].copy()
    pairs_clean = pairs_clean[(pairs_clean["headway_min"] > 0) & (pairs_clean["headway_min"] <= MAX_HEADWAY_MIN)]
    print(f"Paare gesamt (inkl. Ausfaelle): {n_before:,}; davon Vorgaenger ausgefallen: "
          f"{n_vorgaenger_cancelled:,} (ausgeschlossen aus Hauptauswertung); "
          f"verbleibend nach headway-Filter: {len(pairs_clean):,}")

    pairs_clean["headway_bin"] = pd.cut(pairs_clean["headway_min"], bins=BIN_EDGES, labels=BIN_LABELS, right=True)
    pairs_clean["delay_prev_bin"] = pd.cut(pairs_clean["delay_prev"], bins=DELAY_BIN_EDGES,
                                            labels=DELAY_BIN_LABELS, right=True)
    return pairs_clean[["station_name", "richtung", "trip_id_prev", "trip_id", "headway_min",
                         "headway_bin", "delay_prev", "delay_prev_bin", "is_cancelled_next"]]


def cancel_rate_cells(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hb in BIN_LABELS:
        for db in DELAY_BIN_LABELS:
            sub = pairs[(pairs["headway_bin"] == hb) & (pairs["delay_prev_bin"] == db)]
            n = len(sub)
            if n < MIN_N:
                rows.append(dict(headway_bin=hb, delay_prev_bin=db, n=n, n_cancelled=int(sub["is_cancelled_next"].sum()),
                                  rate=np.nan, ci_lo=np.nan, ci_hi=np.nan))
                continue
            k = int(sub["is_cancelled_next"].sum())
            rate = k / n
            lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
            rows.append(dict(headway_bin=hb, delay_prev_bin=db, n=n, n_cancelled=k, rate=rate, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def plot_heatmap(cells: pd.DataFrame, out_path: Path):
    mat = cells.pivot(index="delay_prev_bin", columns="headway_bin", values="rate").reindex(
        index=DELAY_BIN_LABELS, columns=BIN_LABELS)
    n_mat = cells.pivot(index="delay_prev_bin", columns="headway_bin", values="n").reindex(
        index=DELAY_BIN_LABELS, columns=BIN_LABELS)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    vmax = np.nanmax(mat.values) if np.isfinite(mat.values).any() else 0.01
    im = ax.imshow(mat.values * 100, cmap="Reds", vmin=0, vmax=max(vmax * 100, 0.5), aspect="auto")
    ax.set_xticks(range(len(BIN_LABELS)))
    ax.set_xticklabels(BIN_LABELS, rotation=20)
    ax.set_yticks(range(len(DELAY_BIN_LABELS)))
    ax.set_yticklabels(DELAY_BIN_LABELS)
    ax.set_xlabel("Headway-Bin")
    ax.set_ylabel("Vorgaenger-Verspaetungsklasse")

    for i in range(len(DELAY_BIN_LABELS)):
        for j in range(len(BIN_LABELS)):
            v = mat.values[i, j]
            n = n_mat.values[i, j]
            if np.isnan(v):
                txt = f"n={n:.0f}\n(zu klein)" if n > 0 else "n=0"
            else:
                row = cells[(cells["delay_prev_bin"] == DELAY_BIN_LABELS[i]) &
                            (cells["headway_bin"] == BIN_LABELS[j])].iloc[0]
                txt = f"{v*100:.2f}%\n[{row['ci_lo']*100:.2f},{row['ci_hi']*100:.2f}]\nn={n:,.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, color="black")

    fig.colorbar(im, ax=ax, label="Ausfallrate Folgezug (%)")
    ax.set_title("Plot 28: Ausfallrate des Folgezugs nach Vorgaenger-Verspaetung x Headway\n"
                 "[Abschnitt 11.8]")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Lade Ankunfts-Ereignisse INKLUSIVE Ausfaelle ...")
    df = load_events_incl_cancelled()
    print(f"{len(df):,} Ereignisse geladen (davon ausgefallen: {int(df['is_cancelled'].sum()):,}, "
          f"{df['is_cancelled'].mean():.2%}).")

    pairs = build_pairs_incl_cancelled(df)
    pairs.to_csv(OUT_DIR / "ausfallkanal_paare_rohdaten.csv.gz", index=False, compression="gzip")

    overall_rate = pairs["is_cancelled_next"].mean()
    print(f"\nGesamt-Ausfallrate des Folgezugs (alle Paare, alle Headways/Verspaetungen): {overall_rate:.3%}")

    cells = cancel_rate_cells(pairs)
    cells.to_csv(OUT_DIR / "ausfallkanal_zellen.csv", index=False)
    print("\nAusfallrate je Zelle:")
    print(cells.round(5).to_string(index=False))

    print("\n--- Fokusvergleich: <5min Headway, Vorgaenger <=2min vs. >10min verspaetet ---")
    for db in ["<=2min", ">10min"]:
        row = cells[(cells["headway_bin"] == "<5min") & (cells["delay_prev_bin"] == db)]
        print(row.to_string(index=False))

    # Monotonie je Headway-Bin pruefen (steigt Ausfallrate mit delay_prev_bin?)
    print("\n--- Monotonie je Headway-Bin ---")
    for hb in BIN_LABELS:
        sub = cells[cells["headway_bin"] == hb].set_index("delay_prev_bin").reindex(DELAY_BIN_LABELS)
        print(f"{hb}: " + " -> ".join(f"{v:.3%}" if not np.isnan(v) else "n/a" for v in sub["rate"]))

    plot_heatmap(cells, PLOT_DIR / "28_ausfallkanal_heatmap.png")
    print(f"\nPlot gespeichert: {PLOT_DIR / '28_ausfallkanal_heatmap.png'}")

    return dict(pairs=pairs, cells=cells, overall_rate=overall_rate)


if __name__ == "__main__":
    main()

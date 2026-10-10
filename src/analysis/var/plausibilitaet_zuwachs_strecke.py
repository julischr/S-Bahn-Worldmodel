"""Abschnitt 11.6.2: Verspaetungs-Zuwachs des Folgezugs ueber die GESAMTE Stammstrecke
(Pasing <-> Leuchtenbergring), analog zu 11.4.2 (dort: nur Nachbarstationen-Paare).

Hintergrund: 11.4.2 fand fuer vier Paare geografisch BENACHBARTER Stationen nur r=0,00-0,07
zwischen Vorgaenger-Verspaetung und dem Zuwachs des Folgezugs. Hypothese des Nutzers: die
minutengenaue Aufloesung der delay-Werte koennte kleine Effekte PRO Abschnitt verdecken, die
sich erst kumuliert ueber die gesamte Kernstrecke zu einem messbaren Signal aufsummieren.

Endpunkte: Pasing (Rang 0) und Leuchtenbergring (Rang 11) sind gemaess
`stations.STATION_ORDER_WEST_OST` die tatsaechlichen West-/Ost-Endpunkte der 12-Stationen-
Kernstrecke (nicht Ostbahnhof, Rang 10 -- das ist die vorletzte Station). Das ist konsistent
mit Abschnitt 11.1 (`plausibilitaet_fahrzeit.py`), das ebenfalls Pasing<->Leuchtenbergring als
Kernstrecken-Endpunkte verwendet -- dieselbe Wahl wird hier fuer die direkte Vergleichbarkeit
der VAR-Peak-Zeit (25min, Abschnitt 9/11.1) mit der Zuwachs-Analyse beibehalten.

Wiederverwendet `analyze_pair` aus plausibilitaet_zugebene_vertiefung.py 1:1 (die Funktion ist
nicht auf geografisch benachbarte Stationen beschraenkt -- sie berechnet delta = delay(downstream)
- delay(upstream) fuer ein beliebiges Stationspaar plus Vorgaenger-Verknuepfung an der
upstream-Station).

Nur lesender Zugriff auf data/02_events/final/, stations.py, plausibilitaet_zugebene.py,
plausibilitaet_zugebene_vertiefung.py. Neue Outputs unter outputs/var/plausibilitaet/ mit
Praefix "streckenzuwachs_", neuer Plot 26.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.constants import PROJECT_ROOT
from .plausibilitaet_zugebene import BIN_LABELS
from .plausibilitaet_zugebene_vertiefung import analyze_pair, analyze_delta_by_headway

OUT_DIR = PROJECT_ROOT / "outputs" / "var" / "plausibilitaet"
PLOT_DIR = PROJECT_ROOT / "outputs" / "var" / "plots"

UP_STATION = "Pasing"
DOWN_STATION = "Leuchtenbergring"

# Vergleichswerte aus 11.4.2 (Nachbarstationen, gepoolt ueber 4 Stationspaare) zum direkten
# Gegenueberstellen im Plot.
NACHBAR_VERGLEICH = {
    "<5min": 0.064, "5-10min": 0.000, "10-15min": 0.074, "15-20min": 0.045, "20-60min": 0.011,
}


def plot_vergleich(strecke: pd.DataFrame, out_path: Path):
    strecke = strecke.set_index("headway_bin").reindex(BIN_LABELS).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(strecke))

    ax = axes[0]
    ax.errorbar(x, strecke["r"], yerr=[strecke["r"] - strecke["r_ci_lo"], strecke["r_ci_hi"] - strecke["r"]],
                marker="o", color="#2A6F97", capsize=3, label="Pasing<->Leuchtenbergring (ganze Strecke)")
    nachbar_vals = [NACHBAR_VERGLEICH.get(b, np.nan) for b in strecke["headway_bin"]]
    ax.plot(x, nachbar_vals, marker="s", color="#C4471C", linestyle="--",
            label="Nachbarstationen gepoolt (11.4.2)")
    ax.set_xticks(x)
    ax.set_xticklabels(strecke["headway_bin"], rotation=20)
    ax.set_ylabel("r(delay_vorgaenger, delta_folgezug)")
    ax.set_title("Korrelation: ganze Strecke vs. Nachbarstationen")
    ax.axhline(0, color="grey", linewidth=0.7)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.bar(x, strecke["n"], color="#2A6F97")
    ax2.set_xticks(x)
    ax2.set_xticklabels(strecke["headway_bin"], rotation=20)
    ax2.set_ylabel("n Paare")
    ax2.set_title("Stichprobengroesse je Headway-Bin (ganze Strecke)")

    fig.suptitle("Plot 26: Verspaetungs-Zuwachs Pasing->Leuchtenbergring (ganze Kernstrecke)\n"
                 "vs. Nachbarstationen-Zuwachs aus 11.4.2 [Abschnitt 11.6.2]")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Lade und matche Trips {UP_STATION} <-> {DOWN_STATION} (beide Richtungen) ...")
    trips = analyze_pair(UP_STATION, DOWN_STATION)
    print(f"{len(trips):,} gematchte Trips mit Vorgaenger-Verknuepfung (beide Richtungen).")
    trips.to_csv(OUT_DIR / "streckenzuwachs_rohdaten.csv.gz", index=False, compression="gzip")

    print("\nVerteilung ueber Headway-Bins:")
    print(trips["headway_bin"].value_counts().sort_index().to_string())

    pooled = analyze_delta_by_headway(trips, by=None)
    pooled.to_csv(OUT_DIR / "streckenzuwachs_gepoolt.csv", index=False)
    print("\nGepoolt (ganze Strecke Pasing<->Leuchtenbergring, beide Richtungen):")
    print(pooled.round(4).to_string(index=False))

    per_richtung = analyze_delta_by_headway(trips, by=["richtung_seq"])
    per_richtung.to_csv(OUT_DIR / "streckenzuwachs_je_richtung.csv", index=False)
    print("\nJe Richtung:")
    print(per_richtung.round(4).to_string(index=False))

    print("\nVergleich mit Nachbarstationen-Ergebnis aus 11.4.2:")
    cmp = pooled.set_index("headway_bin")["r"].reindex(BIN_LABELS)
    for b in BIN_LABELS:
        nb = NACHBAR_VERGLEICH.get(b, float("nan"))
        print(f"  {b}: ganze Strecke r={cmp.get(b, float('nan')):.3f} vs. Nachbarstationen r={nb:.3f}")

    plot_vergleich(pooled, PLOT_DIR / "26_streckenzuwachs_vs_nachbarstationen.png")
    print(f"\nPlot gespeichert: {PLOT_DIR / '26_streckenzuwachs_vs_nachbarstationen.png'}")

    return dict(trips=trips, pooled=pooled, per_richtung=per_richtung)


if __name__ == "__main__":
    main()

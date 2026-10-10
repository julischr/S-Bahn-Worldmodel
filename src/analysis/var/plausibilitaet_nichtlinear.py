"""Abschnitt 11.7: Nichtlineare Analyse -- gibt es einen "Auffahreffekt" NUR bei extremer
Vorgaenger-Verspaetung UND kurzem Headway, den die lineare Korrelation aus 11.4.2/11.6.2
uebersehen haben koennte?

11.4.2 und 11.6.2 pruefen die lineare Korrelation r(delay_vorgaenger, delta_folgezug) -- eine
OLS-Steigung/Pearson-r ueber Millionen ueberwiegend "normaler" Faelle kann einen seltenen,
aber starken Schwelleneffekt bei EXTREMEN Vorgaenger-Verspaetungen (>10min) UND sehr kurzem
Headway (<5min) leicht verdecken (Ausreisser mit geringem Gewicht in der Gesamtregression).
Deshalb hier: Klassierung statt linearer Fit.

Klassen:
- Vorgaenger-Verspaetung (delay_prev, aus derselben Datenbasis wie 11.4.2/11.6.2): <=2min,
  2-5min, 5-10min, >10min (Sekunden -> Minuten).
- Headway (wie in 11.3/11.4/11.6): <5min, 5-10min, 10-15min, 15-20min, 20-60min.

Je Zelle: mittlerer Zuwachs (delta) des Folgezugs mit 95%-Konfidenzintervall (analytisch,
Normalapproximation der Mittelwert-SE: mean +/- 1.96*std/sqrt(n); bei n<30 wird die Zelle als
zu klein markiert und nicht interpretiert).

Verwendet BEIDE Zuwachs-Definitionen aus 11.4.2 (Nachbarstationen, groessere Stichprobe,
Rohdaten aus outputs/var/plausibilitaet/vertiefung_zuwachs_vorgaenger_rohdaten.csv.gz) und
11.6.2 (ganze Kernstrecke Pasing<->Leuchtenbergring, Rohdaten aus
outputs/var/plausibilitaet/streckenzuwachs_rohdaten.csv.gz) als Robustheitscheck.

Setzt voraus, dass plausibilitaet_zugebene_vertiefung.py und plausibilitaet_zuwachs_strecke.py
bereits gelaufen sind (liest deren Rohdaten-CSVs, rechnet nichts aus den Events neu). Neue
Outputs unter outputs/var/plausibilitaet/ mit Praefix "nichtlinear_", neuer Plot 27.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.constants import PROJECT_ROOT
from .plausibilitaet_zugebene import BIN_LABELS

OUT_DIR = PROJECT_ROOT / "outputs" / "var" / "plausibilitaet"
PLOT_DIR = PROJECT_ROOT / "outputs" / "var" / "plots"

DELAY_BIN_EDGES = [-np.inf, 2 * 60, 5 * 60, 10 * 60, np.inf]
DELAY_BIN_LABELS = ["<=2min", "2-5min", "5-10min", ">10min"]

MIN_N = 30


def classify(df: pd.DataFrame, headway_col="headway_bin", delay_col="delay_prev", delta_col="delta") -> pd.DataFrame:
    df = df.copy()
    df["delay_prev_bin"] = pd.cut(df[delay_col], bins=DELAY_BIN_EDGES, labels=DELAY_BIN_LABELS, right=True)
    df["delta_min"] = df[delta_col] / 60.0
    return df


def cell_stats(df: pd.DataFrame, headway_col="headway_bin") -> pd.DataFrame:
    rows = []
    for hb in BIN_LABELS:
        for db in DELAY_BIN_LABELS:
            sub = df[(df[headway_col] == hb) & (df["delay_prev_bin"] == db)]
            n = len(sub)
            if n < MIN_N:
                rows.append(dict(headway_bin=hb, delay_prev_bin=db, n=n, mean_delta_min=np.nan,
                                  ci_lo=np.nan, ci_hi=np.nan))
                continue
            m = sub["delta_min"].mean()
            se = sub["delta_min"].std(ddof=1) / np.sqrt(n)
            rows.append(dict(headway_bin=hb, delay_prev_bin=db, n=n, mean_delta_min=m,
                              ci_lo=m - 1.96 * se, ci_hi=m + 1.96 * se))
    return pd.DataFrame(rows)


def plot_heatmap(cells: pd.DataFrame, title: str, out_path: Path):
    mat = cells.pivot(index="delay_prev_bin", columns="headway_bin", values="mean_delta_min")
    mat = mat.reindex(index=DELAY_BIN_LABELS, columns=BIN_LABELS)
    n_mat = cells.pivot(index="delay_prev_bin", columns="headway_bin", values="n").reindex(
        index=DELAY_BIN_LABELS, columns=BIN_LABELS)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    vmax = np.nanmax(np.abs(mat.values)) if np.isfinite(mat.values).any() else 1.0
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
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
                lo = cells[(cells["delay_prev_bin"] == DELAY_BIN_LABELS[i]) &
                           (cells["headway_bin"] == BIN_LABELS[j])]["ci_lo"].values[0]
                hi = cells[(cells["delay_prev_bin"] == DELAY_BIN_LABELS[i]) &
                           (cells["headway_bin"] == BIN_LABELS[j])]["ci_hi"].values[0]
                txt = f"{v:+.2f}min\n[{lo:+.2f},{hi:+.2f}]\nn={n:,.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                     color="black")

    fig.colorbar(im, ax=ax, label="mittlerer Zuwachs delta (Minuten)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Nachbarstationen (11.4.2-Rohdaten) ===")
    nb_path = OUT_DIR / "vertiefung_zuwachs_vorgaenger_rohdaten.csv.gz"
    nb = pd.read_csv(nb_path)
    nb = classify(nb)
    nb_cells = cell_stats(nb)
    nb_cells.to_csv(OUT_DIR / "nichtlinear_nachbarstationen_zellen.csv", index=False)
    print(nb_cells.round(4).to_string(index=False))
    plot_heatmap(nb_cells,
                 "Plot 27a: Zuwachs (Nachbarstationen) nach Vorgaenger-Verspaetung x Headway\n"
                 "[Abschnitt 11.7]",
                 PLOT_DIR / "27a_nichtlinear_heatmap_nachbarstationen.png")

    print("\n=== Ganze Strecke Pasing<->Leuchtenbergring (11.6.2-Rohdaten) ===")
    strecke_path = OUT_DIR / "streckenzuwachs_rohdaten.csv.gz"
    strecke = pd.read_csv(strecke_path)
    strecke = classify(strecke)
    strecke_cells = cell_stats(strecke)
    strecke_cells.to_csv(OUT_DIR / "nichtlinear_ganzestrecke_zellen.csv", index=False)
    print(strecke_cells.round(4).to_string(index=False))
    plot_heatmap(strecke_cells,
                 "Plot 27b: Zuwachs (ganze Strecke Pasing<->Leuchtenbergring) nach\n"
                 "Vorgaenger-Verspaetung x Headway [Abschnitt 11.7]",
                 PLOT_DIR / "27b_nichtlinear_heatmap_ganzestrecke.png")

    # Fokuszelle: Vorgaenger >10min, Headway <5min -- der spezifisch vermutete "Auffahreffekt"
    print("\n--- Fokuszelle (Vorgaenger >10min verspaetet, Headway <5min) ---")
    for name, cells in [("Nachbarstationen", nb_cells), ("Ganze Strecke", strecke_cells)]:
        row = cells[(cells["headway_bin"] == "<5min") & (cells["delay_prev_bin"] == ">10min")]
        print(f"{name}: {row.to_string(index=False)}")
        other_short_headway = cells[(cells["headway_bin"] == "<5min") & (cells["delay_prev_bin"] != ">10min")]
        print(f"  (zum Vergleich, gleicher Headway-Bin, andere Verspaetungsklassen:)")
        print(f"  {other_short_headway[['delay_prev_bin', 'mean_delta_min', 'n']].to_string(index=False)}")

    return dict(nb_cells=nb_cells, strecke_cells=strecke_cells)


if __name__ == "__main__":
    main()

"""Abschnitt 11.3: Plausibilitaetstest auf Zugebene -- ueberträgt sich Verspaetung vom
Vorgaenger- auf den direkt folgenden Zug DERSELBEN Station (nicht: Verspaetung EINES Zuges
ueber Stationen hinweg -- das ist eine andere, triviale Autokorrelation, hier NICHT gemessen)?

Vorgehen je Station/Richtung (12 Stammstrecke-Stationen, beide Richtungen):
- Alle Ankunfts-Ereignisse (is_arrival=true, konsistent mit der delay_mean-Definition des
  Hauptlaufs, is_cancelled=false) chronologisch nach time_schedule sortiert.
- Fuer jedes aufeinanderfolgende Paar unterschiedlicher trip_id: Headway (time_schedule-Diff)
  und Verspaetung (delay, Sekunden) von Vorgaenger und Folgezug.
- Headways > 60 Minuten werden ausgeschlossen (das sind i.d.R. Nachtluecken/Betriebspausen,
  kein sinnvoller "direkt folgender Zug" im Sinne von Zugfolge/Kapazitaetsdynamik).
- Stratifiziert nach Headway-Bins: <5min, 5-10min, 10-15min, 15-20min, 20-60min (als
  Naeherung fuer ">20min").
- Je Bin (gepoolt ueber alle 12 Stationen/2 Richtungen UND je Station/Richtung einzeln als
  Konsistenzcheck): Pearson-Korrelation und OLS-Steigung (delay_next ~ delay_prev) mit
  Konfidenzintervallen (Fisher-z fuer r, HC1-robuste SE fuer die Steigung).

Nur lesender Zugriff auf data/02_events/final/, src/analysis/var/stations.py. Neue Outputs
unter outputs/var/plausibilitaet/, neuer Plot 23.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.constants import DATA_DIR, PROJECT_ROOT
from .stations import STOP_ID_TO_NAME

EVENTS_GLOB = str(DATA_DIR / "02_events" / "final" / "*.parquet")
OUT_DIR = PROJECT_ROOT / "outputs" / "var" / "plausibilitaet"
PLOT_DIR = PROJECT_ROOT / "outputs" / "var" / "plots"

STAMM_IDS = ",".join(str(sid) for sid, _ in STOP_ID_TO_NAME.items())

MAX_HEADWAY_MIN = 60.0
BIN_EDGES = [0, 5, 10, 15, 20, MAX_HEADWAY_MIN]
BIN_LABELS = ["<5min", "5-10min", "10-15min", "15-20min", "20-60min"]


def load_events() -> pd.DataFrame:
    con = duckdb.connect()
    q = f"""
        SELECT stop_id, richtung, trip_id, time_schedule, delay
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


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["station_name", "richtung", "time_schedule"]).reset_index(drop=True)
    g = df.groupby(["station_name", "richtung"], sort=False)
    df["trip_id_prev"] = g["trip_id"].shift(1)
    df["time_schedule_prev"] = g["time_schedule"].shift(1)
    df["delay_prev"] = g["delay"].shift(1)

    pairs = df.dropna(subset=["trip_id_prev"]).copy()
    pairs = pairs[pairs["trip_id_prev"] != pairs["trip_id"]]
    pairs["headway_min"] = (pairs["time_schedule"] - pairs["time_schedule_prev"]).dt.total_seconds() / 60.0
    pairs = pairs.rename(columns={"delay": "delay_next"})

    n_before = len(pairs)
    n_nonpos = int((pairs["headway_min"] <= 0).sum())
    n_toolong = int((pairs["headway_min"] > MAX_HEADWAY_MIN).sum())
    pairs = pairs[(pairs["headway_min"] > 0) & (pairs["headway_min"] <= MAX_HEADWAY_MIN)]
    print(f"Paare vor Filter: {n_before:,}; ausgeschlossen: {n_nonpos:,} (headway<=0), "
          f"{n_toolong:,} (headway>{MAX_HEADWAY_MIN:.0f}min); verbleibend: {len(pairs):,}")

    pairs["headway_bin"] = pd.cut(pairs["headway_min"], bins=BIN_EDGES, labels=BIN_LABELS, right=True)
    return pairs[["station_name", "richtung", "trip_id_prev", "trip_id", "time_schedule",
                  "headway_min", "headway_bin", "delay_prev", "delay_next"]]


def _corr_slope_with_ci(x: np.ndarray, y: np.ndarray) -> dict:
    n = len(x)
    if n < 30:
        return dict(n=n, r=np.nan, r_ci_lo=np.nan, r_ci_hi=np.nan,
                    slope=np.nan, slope_se=np.nan, slope_ci_lo=np.nan, slope_ci_hi=np.nan)
    r, _ = stats.pearsonr(x, y)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se_z = 1.0 / np.sqrt(n - 3)
    z_lo, z_hi = z - 1.96 * se_z, z + 1.96 * se_z
    r_lo, r_hi = np.tanh(z_lo), np.tanh(z_hi)

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit(cov_type="HC1")
    slope = model.params[1]
    slope_se = model.bse[1]
    ci = model.conf_int(alpha=0.05)
    slope_ci_lo, slope_ci_hi = ci[1, 0], ci[1, 1]
    return dict(n=n, r=r, r_ci_lo=r_lo, r_ci_hi=r_hi,
                slope=slope, slope_se=slope_se, slope_ci_lo=slope_ci_lo, slope_ci_hi=slope_ci_hi)


def analyze_pooled(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b in BIN_LABELS:
        sub = pairs[pairs["headway_bin"] == b]
        res = _corr_slope_with_ci(sub["delay_prev"].to_numpy(dtype=float), sub["delay_next"].to_numpy(dtype=float))
        rows.append({"headway_bin": b, **res})
    return pd.DataFrame(rows)


def analyze_per_station(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (station, richtung, b), sub in pairs.groupby(["station_name", "richtung", "headway_bin"]):
        res = _corr_slope_with_ci(sub["delay_prev"].to_numpy(dtype=float), sub["delay_next"].to_numpy(dtype=float))
        rows.append({"station": station, "richtung": richtung, "headway_bin": b, **res})
    return pd.DataFrame(rows)


def plot_slope_by_headway(pooled: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(pooled))

    ax = axes[0]
    ax.errorbar(x, pooled["r"], yerr=[pooled["r"] - pooled["r_ci_lo"], pooled["r_ci_hi"] - pooled["r"]],
                marker="o", color="#2A6F97", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(pooled["headway_bin"], rotation=20)
    ax.set_ylabel("Pearson-Korrelation r(delay_prev, delay_next)")
    ax.set_title("Korrelation Vorgaenger->Folgezug nach Headway-Bin")
    ax.axhline(0, color="grey", linewidth=0.7)

    ax2 = axes[1]
    ax2.errorbar(x, pooled["slope"], yerr=[pooled["slope"] - pooled["slope_ci_lo"], pooled["slope_ci_hi"] - pooled["slope"]],
                 marker="o", color="#C4471C", capsize=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(pooled["headway_bin"], rotation=20)
    ax2.set_ylabel("OLS-Steigung delay_next ~ delay_prev")
    ax2.set_title("Steigung Vorgaenger->Folgezug nach Headway-Bin")
    ax2.axhline(0, color="grey", linewidth=0.7)

    fig.suptitle("Plot 23: Zug-zu-Zug-Ansteckung nach Headway (gepoolt ueber 12 Stationen, beide Richtungen)\n"
                 "[Abschnitt 11.3 Plausibilitaetstest]")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Lade Ankunfts-Ereignisse (12 Stammstrecke-Stationen) ...")
    df = load_events()
    print(f"{len(df):,} Ereignisse geladen.")

    pairs = build_pairs(df)
    pairs.to_csv(OUT_DIR / "zugpaare_headway_delay_rohdaten.csv.gz", index=False, compression="gzip")

    print("\nVerteilung ueber Headway-Bins:")
    print(pairs["headway_bin"].value_counts().sort_index().to_string())

    pooled = analyze_pooled(pairs)
    pooled.to_csv(OUT_DIR / "zugebene_headway_kontingenz_gepoolt.csv", index=False)
    print("\nGepoolt (alle 12 Stationen, beide Richtungen):")
    print(pooled.round(4).to_string(index=False))

    per_station = analyze_per_station(pairs)
    per_station.to_csv(OUT_DIR / "zugebene_headway_kontingenz_je_station.csv", index=False)

    # Konsistenzcheck: Vorzeichen/Groessenordnung von r je Station im Bin <5min vs. gepoolt
    kurz = per_station[per_station["headway_bin"] == "<5min"].dropna(subset=["r"])
    print(f"\nKonsistenzcheck Bin <5min ueber Stationen: r zwischen {kurz['r'].min():.3f} und "
          f"{kurz['r'].max():.3f} (Median {kurz['r'].median():.3f}), {len(kurz)} Station/Richtung-Kombinationen.")

    plot_slope_by_headway(pooled, PLOT_DIR / "23_zugebene_headway_ansteckung.png")
    print(f"\nPlot gespeichert: {PLOT_DIR / '23_zugebene_headway_ansteckung.png'}")

    return dict(pairs=pairs, pooled=pooled, per_station=per_station)


if __name__ == "__main__":
    main()

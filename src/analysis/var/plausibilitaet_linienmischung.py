"""Abschnitt 11.2: Plausibilitaetstest -- ist der 20-Minuten-ACF-/Eigenwert-Peak (Abschnitt
3.3/9.6/10.2 in BEFUNDE_VAR.md) ein Linienmischungs-Kompositionseffekt (S1-S8 im ~20-Min-Takt
mischen sich zufaellig in jedem 5-Min-Fenster) statt echter zeitlicher Dynamik?

Vorgehen:
1. Mittlere Verspaetung je Station/Richtung/Linie ueber das Jahr aus data/02_events/final/
   (gleiche Basisfilter wie src/data/build_state_vector.py: is_arrival=true, is_cancelled=false,
   richtung not null, 12 Stammstrecke-Stationen; zusaetzlich dieselben ausgeschlossenen
   Betriebstage wie im VAR-Hauptlauf, siehe run_all.EXCLUDED_DATES).
2. Je 5-Min-Fenster/Station/Richtung: "erwarteter" delay_mean-Wert NUR aus der
   Linienzusammensetzung im Fenster (nach Zuganzahl gewichteter Mittelwert der
   Linien-Jahres-mittelwerte). Composition-bereinigter Wert = tatsaechlicher delay_mean minus
   dieser Erwartungswert.
3. Auf die bereinigte Reihe wird EXAKT dieselbe Fourier-Detrending-Funktion angewendet wie im
   Hauptlauf (detrend.detrend_frame, n_daily=4, n_weekly=3, importiert, nicht veraendert),
   danach dieselbe Luecken-/Episodenbehandlung (episodes.py) und dieselbe episodengewichtete
   gepoolte ACF-Berechnung wie in acf_check.py (Funktionen importiert, nicht veraendert).
4. Vergleich der ACF bei Lag 4 (20 Minuten) vorher (aus outputs/var/lag_crosscheck/, bereits
   vorhandener Hauptlauf-ACF-Check) vs. nachher (linienbereinigt, hier neu berechnet).

Nur lesender Zugriff auf data/02_events/final/, src/analysis/var/{detrend,episodes,acf_check,
prepare,stations,run_all}.py (importiert, nicht veraendert), outputs/var/lag_crosscheck/
(bestehende Vergleichsbasis, nur gelesen). Neue Outputs unter outputs/var/plausibilitaet/,
neuer Plot 22.
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
from .stations import STATION_NAMES_WEST_OST, STOP_ID_TO_NAME
from .prepare import pivot_direction
from .detrend import detrend_frame
from .episodes import interpolate_short_gaps, build_daily_episodes
from .run_all import EXCLUDED_DATES
from .acf_check import acf_table_all_stations

EVENTS_GLOB = str(events_final_dir() / "*.parquet")
OUT_DIR = PROJECT_ROOT / "outputs" / "var" / "plausibilitaet"
PLOT_DIR = PROJECT_ROOT / "outputs" / "var" / "plots"
LAG_CROSSCHECK_DIR = PROJECT_ROOT / "outputs" / "var" / "lag_crosscheck"

STAMM_IDS = ",".join(str(sid) for sid, _ in STOP_ID_TO_NAME.items())
EXAMPLE_STATIONS = ["Pasing", "Hauptbahnhof_tief", "Ostbahnhof", "Leuchtenbergring"]
NLAGS = 25


def load_events() -> pd.DataFrame:
    """Ankunfts-Ereignisse an den 12 Stammstrecke-Stationen, is_cancelled=false,
    richtung IS NOT NULL, mit Linie, delay, 5-Min-Fenster (UTC-Floor von time_real, wie
    build_state_vector.py) und Berlin-lokalem Kalendertag (fuer den Tagesausschluss)."""
    con = duckdb.connect()
    q = f"""
        SELECT stop_id, richtung, line, delay,
               time_bucket(INTERVAL 5 MINUTE, time_real) AS window_utc,
               CAST(time_real AT TIME ZONE 'Europe/Berlin' AS DATE) AS berlin_date
        FROM read_parquet('{EVENTS_GLOB}')
        WHERE is_arrival = true
          AND is_cancelled = false
          AND richtung IS NOT NULL
          AND stop_id IN ({STAMM_IDS})
    """
    df = con.execute(q).fetchdf()
    df["station_name"] = df["stop_id"].map(STOP_ID_TO_NAME)
    df["window_utc"] = pd.to_datetime(df["window_utc"], utc=True)
    df["berlin_date"] = pd.to_datetime(df["berlin_date"]).dt.date
    return df


def line_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mittlere Verspaetung je (Station, Richtung, Linie) ueber das Jahr, OHNE die
    ausgeschlossenen Betriebstage (gleiche 12 Tage wie im VAR-Hauptlauf)."""
    d = df[~df["berlin_date"].isin(EXCLUDED_DATES)]
    prof = (
        d.groupby(["station_name", "richtung", "line"])["delay"]
        .agg(line_mean_delay="mean", n_linie="size")
        .reset_index()
    )
    return prof


def expected_vs_actual(df: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    """Je (Fenster, Station, Richtung): tatsaechlicher delay_mean vs. erwarteter Wert NUR aus
    der Linienzusammensetzung im Fenster (nach Zuganzahl gewichteter Mittelwert der
    Jahres-Linienmittelwerte aus line_profile)."""
    win = (
        df.groupby(["window_utc", "station_name", "richtung", "line"])["delay"]
        .agg(n="size", sum_delay="sum")
        .reset_index()
    )
    win = win.merge(prof[["station_name", "richtung", "line", "line_mean_delay"]],
                     on=["station_name", "richtung", "line"], how="left")
    win["gewichtet_erwartet"] = win["n"] * win["line_mean_delay"]

    agg = (
        win.groupby(["window_utc", "station_name", "richtung"])
        .agg(n_zuege=("n", "sum"), _sum_delay=("sum_delay", "sum"), _sum_expected=("gewichtet_erwartet", "sum"))
        .reset_index()
    )
    agg["actual_delay_mean"] = agg["_sum_delay"] / agg["n_zuege"]
    agg["expected_delay_mean_komposition"] = agg["_sum_expected"] / agg["n_zuege"]
    agg = agg.drop(columns=["_sum_delay", "_sum_expected"])
    agg["residual_komposition_bereinigt"] = agg["actual_delay_mean"] - agg["expected_delay_mean_komposition"]
    agg["window_berlin"] = agg["window_utc"].dt.tz_convert("Europe/Berlin")
    return agg


def build_adjusted_pooled_acf(agg: pd.DataFrame, richtung: str) -> tuple[pd.DataFrame, dict]:
    """Wendet auf die komposition-bereinigte Reihe dieselbe Pipeline an wie der Hauptlauf auf
    die rohe delay_mean-Reihe: Fourier-Detrending -> Tagesausschluss (bereits in agg entfernt,
    da line_profile schon ohne Ausschlusstage berechnet wurde, hier zusaetzlich nochmal explizit
    auf Fensterebene) -> Kurzluecken-Interpolation -> Episodenbildung -> gepoolte ACF."""
    wide = pivot_direction(agg, richtung, value_col="residual_komposition_bereinigt")
    wide = wide[STATION_NAMES_WEST_OST]

    n_daily, n_weekly = 4, 3
    resid, fourier_info = detrend_frame(wide, n_daily, n_weekly)

    mask_excl = pd.Series(resid.index.date, index=resid.index).isin(EXCLUDED_DATES)
    resid_f = resid[~mask_excl]

    interp = interpolate_short_gaps(resid_f, limit_steps=12)
    episodes, ep_meta = build_daily_episodes(interp, min_length=24)

    acf_tab = acf_table_all_stations(episodes, nlags=NLAGS)
    meta = {"n_episoden": len(episodes), "fourier_info": fourier_info}
    return acf_tab, meta


def load_original_acf(richtung: str) -> pd.DataFrame:
    path = LAG_CROSSCHECK_DIR / f"acf_{richtung.lower()}_je_station.csv"
    tab = pd.read_csv(path, index_col=0)
    tab.columns = [int(c) for c in tab.columns]
    return tab


def compare_lag4(acf_original: pd.DataFrame, acf_adjusted: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for st in acf_original.index:
        orig = acf_original.loc[st, 4]
        adj = acf_adjusted.loc[st, 4] if st in acf_adjusted.index else np.nan
        rows.append({
            "station": st,
            "acf_lag4_original_20min": orig,
            "acf_lag4_linienbereinigt_20min": adj,
            "differenz": adj - orig,
            "relative_abschwaechung_pct": 100.0 * (orig - adj) / orig if orig else np.nan,
        })
    return pd.DataFrame(rows)


def plot_comparison(acf_original: pd.DataFrame, acf_adjusted: pd.DataFrame, stations: list[str], out_path: Path):
    fig, axes = plt.subplots(1, len(stations), figsize=(4.2 * len(stations), 4.2), sharey=True)
    lags_min = np.arange(NLAGS + 1) * 5
    for ax, st in zip(axes, stations):
        ax.plot(lags_min, acf_original.loc[st, :NLAGS], color="#2A6F97", marker="o", markersize=3,
                label="Original (delay_mean)")
        ax.plot(lags_min, acf_adjusted.loc[st, :NLAGS], color="#C4471C", marker="o", markersize=3,
                label="Linienbereinigt")
        ax.axvline(20, color="grey", linestyle="--", linewidth=1)
        ax.set_title(st)
        ax.set_xlabel("Lag [min]")
        ax.axhline(0, color="black", linewidth=0.5)
    axes[0].set_ylabel("ACF")
    axes[0].legend(fontsize=8)
    fig.suptitle("Plot 22: ACF vor/nach Linienbereinigung, Richtung Ost [Abschnitt 11.2]\n"
                 "gestrichelte Linie = Lag 4 (20 Minuten)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Lade Ankunfts-Ereignisse (12 Stammstrecke-Stationen) ...")
    df = load_events()
    print(f"{len(df):,} Ereignisse geladen.")

    prof = line_profile(df)
    prof.to_csv(OUT_DIR / "linienprofil_mittlere_verspaetung.csv", index=False)
    print(f"Linienprofil: {len(prof)} (Station, Richtung, Linie)-Kombinationen.")

    agg = expected_vs_actual(df, prof)
    agg.to_parquet(OUT_DIR / "fenster_actual_vs_erwartet_komposition.parquet", index=False)
    print(f"Fensterebene: {len(agg):,} (Fenster, Station, Richtung)-Zeilen.")

    results = {}
    for richtung in ["Ost", "West"]:
        print(f"\n--- Richtung {richtung}: komposition-bereinigte ACF ---")
        acf_adj, meta = build_adjusted_pooled_acf(agg, richtung)
        acf_adj.to_csv(OUT_DIR / f"acf_linienbereinigt_{richtung.lower()}.csv")
        acf_orig = load_original_acf(richtung)
        cmp_tab = compare_lag4(acf_orig, acf_adj)
        cmp_tab.to_csv(OUT_DIR / f"vergleich_lag4_20min_{richtung.lower()}.csv", index=False)
        print(f"  n_episoden (linienbereinigt) = {meta['n_episoden']}")
        print(cmp_tab.round(4).to_string(index=False))
        results[richtung] = dict(acf_adj=acf_adj, acf_orig=acf_orig, cmp_tab=cmp_tab, meta=meta)

    plot_comparison(results["Ost"]["acf_orig"], results["Ost"]["acf_adj"], EXAMPLE_STATIONS,
                     PLOT_DIR / "22_linienbereinigte_acf_vergleich.png")
    print(f"\nPlot gespeichert: {PLOT_DIR / '22_linienbereinigte_acf_vergleich.png'}")

    return results


if __name__ == "__main__":
    main()

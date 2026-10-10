"""Validierung Abschnitt 10.2: Autokorrelationsstruktur der West- vs. Ost-Residualreihen im
Bereich 70-90 Minuten (Lags 14-18), als moegliche Erklaerung fuer das deutlich hoehere
BIC-optimale p bei West (p=17=85min) gegenueber Ost (p=5=25min).

Nutzt dieselbe Datenaufbereitung wie run_all.py (Fourier-Detrending, Tagesausschluss,
Episodenbildung), berechnet dann je Station eine EPISODENWEISE gepoolte ACF (Lags nie ueber
Episodengrenzen hinweg, analog zur Lag-Design-Matrix in var_model.build_lagged_design):
je Episode mit statsmodels.tsa.stattools.acf, anschliessend gewichteter Mittelwert ueber alle
Episoden (Gewicht = nutzbare Punktepaare je Lag, n_episode - lag), NICHT eine einzelne ACF auf
den durchgehend aneinandergehaengten Residuen (das wuerde an jeder Episodengrenze
kuenstliche Sprungkorrelationen erzeugen)."""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from src.utils.constants import PROJECT_ROOT
from .stations import STATION_NAMES_WEST_OST
from .lag_crosscheck import prepare_direction_episodes
from . import plots as P

OUT_DIR = PROJECT_ROOT / "outputs" / "var" / "lag_crosscheck"
PLOT_DIR = PROJECT_ROOT / "outputs" / "var" / "plots"
NLAGS = 25


def pooled_acf(episodes: list[pd.DataFrame], col: str, nlags: int = NLAGS, min_len: int = 40) -> pd.Series:
    """Gewichteter Mittelwert der episodenweisen ACF (Gewicht = n_episode - lag), je Lag."""
    acc = np.zeros(nlags + 1)
    weight = np.zeros(nlags + 1)
    for ep in episodes:
        n = len(ep)
        if n < min_len:
            continue
        x = ep[col].to_numpy()
        if np.std(x) < 1e-8:
            continue
        a = acf(x, nlags=min(nlags, n - 1), fft=True)
        k = len(a)
        w = np.array([max(n - lag, 0) for lag in range(k)], dtype=float)
        acc[:k] += a * w
        weight[:k] += w
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(weight > 0, acc / weight, np.nan)
    return pd.Series(out, index=range(nlags + 1), name=col)


def acf_table_all_stations(episodes: list[pd.DataFrame], nlags: int = NLAGS) -> pd.DataFrame:
    cols = episodes[0].columns.tolist()
    rows = {col: pooled_acf(episodes, col, nlags=nlags) for col in cols}
    return pd.DataFrame(rows).T  # index=station, columns=lag


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ep_ost, _ = prepare_direction_episodes("Ost")
    ep_west, _ = prepare_direction_episodes("West")

    acf_ost = acf_table_all_stations(ep_ost)
    acf_west = acf_table_all_stations(ep_west)
    acf_ost.to_csv(OUT_DIR / "acf_ost_je_station.csv")
    acf_west.to_csv(OUT_DIR / "acf_west_je_station.csv")

    # Fenster 70-90 Minuten = Lags 14-18 (bei 5-Min-Schritten)
    window = list(range(14, 19))
    tab = pd.DataFrame({
        "acf_west_mean_lag14_18": acf_west[window].mean(axis=1),
        "acf_ost_mean_lag14_18": acf_ost[window].mean(axis=1),
    })
    tab["differenz_west_minus_ost"] = tab["acf_west_mean_lag14_18"] - tab["acf_ost_mean_lag14_18"]
    tab.to_csv(OUT_DIR / "acf_70_90min_vergleich.csv")
    print(tab.round(4).to_string())

    # Vollstaendige Lag-14..18-Werte je Station (fuer Tabelle in BEFUNDE_VAR.md)
    detail_west = acf_west[window]
    detail_west.columns = [f"lag{c}_{c*5}min" for c in window]
    detail_west.to_csv(OUT_DIR / "acf_west_lag14_18_detail.csv")
    detail_ost = acf_ost[window]
    detail_ost.columns = [f"lag{c}_{c*5}min" for c in window]
    detail_ost.to_csv(OUT_DIR / "acf_ost_lag14_18_detail.csv")

    print("\nWest, Lags 14-18 (Minuten 70-90):")
    print(detail_west.round(4).to_string())
    print("\nOst, Lags 14-18 (Minuten 70-90):")
    print(detail_ost.round(4).to_string())

    # Globales Maximum der ACF ausserhalb Lag 0-3 je Station, zum Vergleich (wo liegt der
    # staerkste Nebenpeak wirklich, nicht nur im 70-90min-Fenster?)
    global_peak_west = acf_west.iloc[:, 4:].idxmax(axis=1) * 5
    global_peak_ost = acf_ost.iloc[:, 4:].idxmax(axis=1) * 5
    peak_tab = pd.DataFrame({
        "west_staerkster_nebenpeak_min": global_peak_west,
        "west_wert": [acf_west.loc[s, acf_west.iloc[:, 4:].idxmax(axis=1)[s]] for s in acf_west.index],
        "ost_staerkster_nebenpeak_min": global_peak_ost,
        "ost_wert": [acf_ost.loc[s, acf_ost.iloc[:, 4:].idxmax(axis=1)[s]] for s in acf_ost.index],
    })
    peak_tab.to_csv(OUT_DIR / "acf_staerkster_nebenpeak.csv")
    print("\nStaerkster Nebenpeak (Lag>=4) je Station:")
    print(peak_tab.round(4).to_string())

    # Episodenlaengen-Artefakt-Check: welchen Gewichtsanteil an lag=16 (80min) haben kurze
    # Episoden (70-90 Beobachtungen) gegenueber langen (>=150 Beobachtungen)?
    def weight_share(episodes, lag=16):
        lens = np.array([len(e) for e in episodes])
        w = np.maximum(lens - lag, 0).astype(float)
        short = (lens >= 70) & (lens <= 90)
        long_ = lens >= 150
        return {
            "n_episoden": len(lens),
            "n_episoden_70_90": int(short.sum()),
            "gewichtsanteil_70_90_obs_bei_lag16": float(w[short].sum() / w.sum()),
            "gewichtsanteil_ge150_obs_bei_lag16": float(w[long_].sum() / w.sum()),
        }

    artefakt_check = pd.DataFrame({
        "West": weight_share(ep_west),
        "Ost": weight_share(ep_ost),
    })
    artefakt_check.to_csv(OUT_DIR / "episodenlaengen_artefakt_check.csv")
    print("\nEpisodenlaengen-Artefakt-Check (Gewichtsanteil bei Lag 16 = 80min):")
    print(artefakt_check.to_string())

    # Plots
    example_stations = ["Pasing", "Hauptbahnhof_tief", "Ostbahnhof", "Leuchtenbergring"]
    P.plot_acf_west_vs_ost(acf_west, acf_ost, example_stations, 5.0, PLOT_DIR / "20a_acf_west_vs_ost.png")
    aic_bic_west = pd.read_csv(PROJECT_ROOT / "outputs/var/west/aic_bic_lag_wahl.csv")
    P.plot_acf_bic_step(aic_bic_west, PLOT_DIR / "20b_west_bic_stufe.png")
    print(f"Plots gespeichert: {PLOT_DIR / '20a_acf_west_vs_ost.png'}, {PLOT_DIR / '20b_west_bic_stufe.png'}")

    return dict(acf_ost=acf_ost, acf_west=acf_west, tab=tab,
                detail_west=detail_west, detail_ost=detail_ost, peak_tab=peak_tab,
                artefakt_check=artefakt_check)


if __name__ == "__main__":
    main()

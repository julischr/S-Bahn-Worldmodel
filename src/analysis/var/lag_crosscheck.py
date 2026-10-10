"""Validierung Abschnitt 10.1: Lag-Kreuzvergleich.

Schaetzt West zusaetzlich mit festem p=5 (statt BIC-optimal p=17) und Ost zusaetzlich mit
festem p=17 (statt BIC-optimal p=5), bei ansonsten IDENTISCHER Datenaufbereitung
(Fourier-Detrending, Tagesausschluss, Episodenbildung) wie in den jeweiligen Hauptlaeufen
(run_all.py). Fuer beide neuen Varianten wird die kausal korrekte Cholesky-Ordnung verwendet
(Ost: primaere geografische Ordnung West->Ost; West: umgekehrte Ordnung Ost->West), analog zum
Richtungsvergleich in Abschnitt 5 von BEFUNDE_VAR.md.

Aus Zeitgruenden wird hier KEIN Bootstrap gerechnet (nur Punktschaetzungen) -- die
Hauptlauf-Ergebnisse (Ost p=5, West p=17) inkl. Bootstrap bleiben unveraendert in
outputs/var/ bzw. outputs/var/west/ und werden hier nur eingelesen, nicht neu geschaetzt.
"""

import numpy as np
import pandas as pd

from src.utils.constants import PROJECT_ROOT
from .prepare import load_state_vector, pivot_direction
from .detrend import detrend_frame
from .episodes import interpolate_short_gaps, build_daily_episodes
from .var_model import fit_var, stability_eigenvalues
from .irf import structural_irf
from .irf_metrics import fevd, peak_amplitude_time, half_life
from .stations import STATION_NAMES_WEST_OST
from .run_all import EXCLUDED_DATES
from . import plots as P

OUT_DIR = PROJECT_ROOT / "outputs" / "var" / "lag_crosscheck"
PLOT_DIR = PROJECT_ROOT / "outputs" / "var" / "plots"
HORIZON_MIN = 240
STEP = 5.0


def prepare_direction_episodes(richtung: str):
    """Identische Aufbereitung wie run_all.run_direction, station_order fest = West->Ost
    (Original-Reihenfolge), damit Ost und West direkt vergleichbare Spaltennamen haben."""
    df = load_state_vector()
    wide = pivot_direction(df, richtung)[STATION_NAMES_WEST_OST]
    resid, _ = detrend_frame(wide, 4, 3)
    mask_excl = pd.Series(resid.index.date, index=resid.index).isin(EXCLUDED_DATES)
    resid_f = resid[~mask_excl]
    interp = interpolate_short_gaps(resid_f, limit_steps=12)
    episodes, ep_meta = build_daily_episodes(interp, min_length=24)
    return episodes, ep_meta


def net_coupling(fevd_h, cols):
    K = len(cols)
    export = np.zeros(K)
    import_ = np.zeros(K)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            export[j] += fevd_h[i, j]
            import_[i] += fevd_h[i, j]
    return pd.Series(export - import_, index=cols)


def run_variant(episodes, p, order, horizon_steps):
    res = fit_var(episodes, p)
    As = res.A_matrices()
    eig = stability_eigenvalues(As)
    theta = structural_irf(As, res.Sigma_u, order, horizon_steps)
    fevd_h = fevd(theta, horizon_steps)
    time_to_peak, amplitude = peak_amplitude_time(theta, step_minutes=STEP)
    hl = half_life(theta, step_minutes=STEP)
    net = net_coupling(fevd_h, episodes[0].columns.tolist())
    return dict(res=res, As=As, eig=eig, theta=theta, fevd_h=fevd_h,
                time_to_peak=time_to_peak, amplitude=amplitude, half_life=hl, net=net,
                mean_foreign_fevd=float(np.mean(1 - np.diag(fevd_h))),
                max_abs_eig=float(np.max(np.abs(eig))))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    horizon_steps = int(HORIZON_MIN / STEP)
    cols = STATION_NAMES_WEST_OST
    order_geo = list(range(len(cols)))       # West->Ost primaer (kausal richtig fuer Ost)
    order_rev = list(reversed(order_geo))     # Ost->West (kausal richtig fuer West)

    print("Aufbereitung Ost/West Episoden ...")
    ep_ost, meta_ost = prepare_direction_episodes("Ost")
    ep_west, meta_west = prepare_direction_episodes("West")
    print(f"Ost: {len(ep_ost)} Episoden, West: {len(ep_west)} Episoden")

    # WICHTIG: alle vier Varianten werden HIER mit konsistenter, kausal korrekter Ordnung neu
    # berechnet (Ost stets order_geo, West stets order_rev) -- auch die beiden
    # "Hauptlauf"-Varianten (Ost p=5, West p=17). Das ist noetig fuer einen fairen
    # Vier-Varianten-Vergleich: die in outputs/var/west/*.csv gespeicherten Haupttabellen
    # (fevd.csv, netto_kopplung.csv, irf_amplitude.csv) nutzen dort die PRIMAERE (fuer
    # West-Zuege kausal falsche) Ordnung order_geo -- nur die Prosa/Plot 11 in Abschnitt 5
    # nutzte fuer die eine Pasing<->Leuchtenbergring-Vergleichszahl bereits order_rev. Um
    # Aepfel mit Aepfeln zu vergleichen, wird hier fuer West durchgaengig order_rev verwendet;
    # nichts an den bestehenden outputs/var/west/*.csv wird veraendert oder ueberschrieben.
    print("Fitte Ost p=5 (Hauptlauf-Replikation, fuer konsistenten Vergleich) ...")
    ost_p5 = run_variant(ep_ost, 5, order_geo, horizon_steps)
    print("Fitte Ost p=17 (Kreuzvergleich, Hauptlauf hat p=5) ...")
    ost_p17 = run_variant(ep_ost, 17, order_geo, horizon_steps)
    print("Fitte West p=17 (Hauptlauf-Replikation mit order_rev, fuer konsistenten Vergleich) ...")
    west_p17 = run_variant(ep_west, 17, order_rev, horizon_steps)
    print("Fitte West p=5 (Kreuzvergleich, Hauptlauf hat p=17) ...")
    west_p5 = run_variant(ep_west, 5, order_rev, horizon_steps)

    # Speichere Tabellen (nur die neuen/re-berechneten Varianten, nichts Bestehendes wird beruehrt)
    for name, r in [("ost_p5_rep", ost_p5), ("ost_p17", ost_p17), ("west_p17_rep", west_p17), ("west_p5", west_p5)]:
        pd.DataFrame(r["time_to_peak"], index=cols, columns=cols).to_csv(OUT_DIR / f"irf_zeit_bis_peak_min_{name}.csv")
        pd.DataFrame(r["amplitude"], index=cols, columns=cols).to_csv(OUT_DIR / f"irf_amplitude_{name}.csv")
        pd.DataFrame(r["fevd_h"], index=cols, columns=cols).to_csv(OUT_DIR / f"fevd_{name}.csv")
        r["net"].to_csv(OUT_DIR / f"netto_kopplung_{name}.csv", header=["netto_kopplung"])

    def peak_amp(r, response, shock):
        ttp = pd.DataFrame(r["time_to_peak"], index=cols, columns=cols)
        amp = pd.DataFrame(r["amplitude"], index=cols, columns=cols)
        return ttp.loc[response, shock], amp.loc[response, shock]

    rows = []
    for label, r, response, shock in [
        ("Ost p=5 (Hauptlauf, BIC-optimal)", ost_p5, "Leuchtenbergring", "Pasing"),
        ("Ost p=17 (Kreuzvergleich, AIC-Kandidat von West)", ost_p17, "Leuchtenbergring", "Pasing"),
        ("West p=17 (Hauptlauf, BIC-optimal)", west_p17, "Pasing", "Leuchtenbergring"),
        ("West p=5 (Kreuzvergleich, BIC-optimal von Ost)", west_p5, "Pasing", "Leuchtenbergring"),
    ]:
        t, a = peak_amp(r, response, shock)
        rows.append({
            "variante": label, "peak_min": t, "amplitude_s": a,
            "mean_foreign_fevd": r["mean_foreign_fevd"], "max_abs_eig": r["max_abs_eig"],
            "top_quelle": r["net"].idxmax(), "top_quelle_wert": r["net"].max(),
            "top_senke": r["net"].idxmin(), "top_senke_wert": r["net"].min(),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "vier_varianten_vergleich.csv", index=False)
    print(summary.to_string(index=False))

    meta = {
        "ost_n_episoden": len(ep_ost), "west_n_episoden": len(ep_west),
        "ost_p5_T": ost_p5["res"].T, "ost_p17_T": ost_p17["res"].T,
        "west_p17_T": west_p17["res"].T, "west_p5_T": west_p5["res"].T,
    }
    pd.Series(meta).to_json(OUT_DIR / "lag_crosscheck_meta.json", indent=2)

    # Vergleichsplot: Pasing<->Leuchtenbergring-Kurve (kausal richtige Ordnung) je Variante
    i_p, j_l = cols.index("Leuchtenbergring"), cols.index("Pasing")  # Ost: Antwort Leuchtenbergring auf Schock Pasing
    i_l, j_p = cols.index("Pasing"), cols.index("Leuchtenbergring")  # West: Antwort Pasing auf Schock Leuchtenbergring
    curves = {
        "Ost p=5 (Hauptlauf)": ost_p5["theta"][:, i_p, j_l],
        "Ost p=17 (Kreuzvergleich)": ost_p17["theta"][:, i_p, j_l],
        "West p=17 (Hauptlauf)": west_p17["theta"][:, i_l, j_p],
        "West p=5 (Kreuzvergleich)": west_p5["theta"][:, i_l, j_p],
    }
    P.plot_lag_crosscheck(curves, summary, STEP, PLOT_DIR / "19_lag_crosscheck_vergleich.png", max_minutes=90)
    print(f"Plot gespeichert: {PLOT_DIR / '19_lag_crosscheck_vergleich.png'}")

    return dict(ost_p5=ost_p5, ost_p17=ost_p17, west_p17=west_p17, west_p5=west_p5,
                summary=summary, ep_ost=ep_ost, ep_west=ep_west, cols=cols)


if __name__ == "__main__":
    main()

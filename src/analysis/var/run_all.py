"""Einstiegspunkt: fuehrt die komplette VAR-Analyse (Stufe 2) fuer eine Richtung aus
und schreibt alle Outputs/Plots. Aufruf: uv run python -m src.analysis.var.run_all
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.constants import PROJECT_ROOT
from .prepare import load_state_vector, pivot_direction, pivot_aux
from .detrend import detrend_frame, select_harmonics
from .episodes import interpolate_short_gaps, build_daily_episodes, daily_anomaly_table
from .var_model import fit_var, aic_bic_for_p, stability_eigenvalues
from .diagnostics import adf_per_episode, residual_diagnostics
from .irf import structural_irf, block_bootstrap_irf
from .irf_metrics import fevd, peak_amplitude_time, half_life
from . import plots as P
from .stations import STATION_NAMES_WEST_OST, STATION_NAMES_OST_WEST

OUT_DIR = PROJECT_ROOT / "outputs" / "var"
PLOT_DIR = OUT_DIR / "plots"

# Auffaellige Betriebstage: datengetrieben identifiziert (n_zuege < 30% des Medians des
# jeweiligen Wochentagtyps -- siehe Abschnitt "Auffaellige Betriebstage" in BEFUNDE_VAR.md).
EXCLUDED_DATES = pd.to_datetime([
    "2025-05-10", "2025-05-11",  # Wochenende, quasi Komplettsperrung (92/85 Zuege statt ~4700 Median)
    "2025-10-18", "2025-10-19",  # Wochenende, quasi Komplettsperrung (125/123 Zuege) -- vom Nutzer benannt
    "2026-01-01",                 # Jahresgrenze, nur 1. Betriebsstunde in den Daten (Rand-Artefakt)
    "2025-01-11", "2025-01-12",  # Wochenende, stark reduzierter Betrieb (548/855 Zuege)
    "2025-06-07", "2025-06-08", "2025-06-09", "2025-06-10", "2025-06-11", "2025-06-12",  # ganze Woche reduziert (~1200-1450 statt ~4700-5400)
]).date


def run_direction(richtung: str, station_order: list[str], p_range=range(1, 21), n_boot=500, horizon_min=240, seed=0):
    step = 5.0
    horizon = int(horizon_min / step)
    df = load_state_vector()
    wide = pivot_direction(df, richtung)
    wide = wide[station_order]

    n_daily, n_weekly = 4, 3
    resid, fourier_info = detrend_frame(wide, n_daily, n_weekly)

    mask_excl = pd.Series(resid.index.date, index=resid.index).isin(EXCLUDED_DATES)
    resid_f = resid[~mask_excl]

    interp = interpolate_short_gaps(resid_f, limit_steps=12)
    episodes, ep_meta = build_daily_episodes(interp, min_length=24)

    aic_bic = aic_bic_for_p(episodes, p_range)
    # BIC-Minimum waehlen (siehe Begruendung in BEFUNDE_VAR.md)
    chosen_p = int(aic_bic.loc[aic_bic["bic"].idxmin(), "p"])
    aic_p = int(aic_bic.loc[aic_bic["aic"].idxmin(), "p"])

    res = fit_var(episodes, chosen_p)
    As = res.A_matrices()
    eig = stability_eigenvalues(As)

    adf_agg, adf_detail = adf_per_episode(episodes, min_length=150)
    resid_diag = residual_diagnostics(res.U, res.columns, lags=20)

    order_geo = list(range(len(station_order)))
    order_rev = list(reversed(order_geo))
    theta_geo = structural_irf(As, res.Sigma_u, order_geo, horizon)
    theta_rev = structural_irf(As, res.Sigma_u, order_rev, horizon)

    rng = np.random.default_rng(seed)
    t0 = time.time()
    boot_theta = block_bootstrap_irf(episodes, chosen_p, order_geo, horizon, n_boot=n_boot, rng=rng)
    boot_time = time.time() - t0

    fevd_h = fevd(theta_geo, horizon)
    time_to_peak, amplitude = peak_amplitude_time(theta_geo, step_minutes=step)
    hl = half_life(theta_geo, step_minutes=step)

    export = np.zeros(len(station_order))
    import_ = np.zeros(len(station_order))
    K = len(station_order)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            export[j] += fevd_h[i, j]  # wie viel Station j zur Varianz von i beitraegt
            import_[i] += fevd_h[i, j]  # wie viel Station i von anderen importiert
    net = pd.Series(export - import_, index=station_order)

    return dict(
        richtung=richtung,
        station_order=station_order,
        wide=wide,
        resid=resid,
        fourier_info=fourier_info,
        ep_meta=ep_meta,
        episodes=episodes,
        aic_bic=aic_bic,
        chosen_p=chosen_p,
        aic_p=aic_p,
        var_result=res,
        As=As,
        eig=eig,
        adf_agg=adf_agg,
        adf_detail=adf_detail,
        resid_diag=resid_diag,
        theta_geo=theta_geo,
        theta_rev=theta_rev,
        boot_theta=boot_theta,
        boot_time=boot_time,
        n_boot=n_boot,
        fevd_h=fevd_h,
        horizon=horizon,
        time_to_peak=time_to_peak,
        amplitude=amplitude,
        half_life=hl,
        net=net,
        step=step,
    )


def save_outputs_ost(r: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    cols = r["station_order"]

    r["fourier_info"].to_csv(OUT_DIR / "fourier_info.csv", index=False)
    r["ep_meta"].to_csv(OUT_DIR / "episoden_meta.csv", index=False)
    r["aic_bic"].to_csv(OUT_DIR / "aic_bic_lag_wahl.csv", index=False)
    r["adf_agg"].to_csv(OUT_DIR / "adf_ergebnisse.csv", index=False)
    r["adf_detail"].to_csv(OUT_DIR / "adf_ergebnisse_je_episode.csv", index=False)
    r["resid_diag"].to_csv(OUT_DIR / "residual_diagnostik.csv", index=False)

    pd.DataFrame(r["time_to_peak"], index=cols, columns=cols).to_csv(OUT_DIR / "irf_zeit_bis_peak_min.csv")
    pd.DataFrame(r["amplitude"], index=cols, columns=cols).to_csv(OUT_DIR / "irf_amplitude.csv")
    pd.DataFrame(r["half_life"], index=cols, columns=cols).to_csv(OUT_DIR / "irf_halbwertszeit_min.csv")
    pd.DataFrame(r["fevd_h"], index=cols, columns=cols).to_csv(OUT_DIR / "fevd.csv")
    r["net"].to_csv(OUT_DIR / "netto_kopplung.csv", header=["netto_kopplung"])

    eig = r["eig"]
    eig_df = pd.DataFrame({"real": eig.real, "imag": eig.imag, "modulus": np.abs(eig), "angle_rad": np.angle(eig)})
    eig_df["period_minuten"] = np.where(np.abs(eig_df["angle_rad"]) > 1e-9, 2 * np.pi / np.abs(eig_df["angle_rad"]) * r["step"], np.inf)
    eig_df["halbwertszeit_min"] = np.log(0.5) / np.log(eig_df["modulus"]) * r["step"]
    eig_df.sort_values("modulus", ascending=False).to_csv(OUT_DIR / "eigenwerte.csv", index=False)

    excl_df = pd.DataFrame({"date": EXCLUDED_DATES})
    excl_df.to_csv(OUT_DIR / "ausgeschlossene_betriebstage.csv", index=False)

    meta = {
        "richtung": r["richtung"],
        "n_daily_harmonics": 4,
        "n_weekly_harmonics": 3,
        "chosen_p": r["chosen_p"],
        "aic_optimal_p": r["aic_p"],
        "T_obs": r["var_result"].T,
        "n_episodes": len(r["episodes"]),
        "n_boot": r["n_boot"],
        "boot_time_sec": r["boot_time"],
        "horizon_steps": r["horizon"],
        "horizon_minutes": r["horizon"] * r["step"],
        "max_abs_eigenvalue": float(np.max(np.abs(r["eig"]))),
    }
    (OUT_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def make_plots_ost(r: dict):
    cols = r["station_order"]
    step = r["step"]

    P.plot_periodogram(r["wide"], r["resid"], ["Pasing", "Hauptbahnhof_tief", "Leuchtenbergring"], PLOT_DIR / "01_periodogramm.png")
    P.plot_aic_bic(r["aic_bic"], r["chosen_p"], PLOT_DIR / "02_aic_bic.png")
    P.plot_eigenvalues(r["eig"], PLOT_DIR / "03_eigenwerte.png")
    P.plot_irf_grid(r["theta_geo"], cols, step, PLOT_DIR / "04_irf_gitter_12x12.png", max_minutes=120)

    shock_set = ["Pasing", "Hauptbahnhof_tief", "Ostbahnhof", "Leuchtenbergring"]
    P.plot_selected_shocks(r["theta_geo"], cols, shock_set, step, PLOT_DIR / "05_irf_ausgewaehlte_schocks.png", max_minutes=90, boot_theta=r["boot_theta"])

    P.plot_time_to_peak_heatmap(r["time_to_peak"], cols, PLOT_DIR / "06_heatmap_zeit_bis_peak.png")
    P.plot_amplitude_heatmap(r["amplitude"], cols, PLOT_DIR / "07_heatmap_amplitude.png")
    P.plot_halflife_heatmap(r["half_life"], cols, PLOT_DIR / "08_heatmap_halbwertszeit.png")
    P.plot_fevd(r["fevd_h"], cols, r["horizon"] * step, PLOT_DIR / "09_fevd.png")
    P.plot_net_coupling(r["net"], PLOT_DIR / "10a_netto_kopplung_balken.png", PLOT_DIR / "10b_netto_kopplung_strecke.png")
    P.plot_cholesky_robustness(r["theta_geo"], r["theta_rev"], cols, shock_set, "Cholesky West->Ost (primaer)", "Cholesky Ost->West (Alternative)", step, PLOT_DIR / "12_cholesky_robustheit.png", max_minutes=90)

    # Plot 13: komplexe Eigenwerte / gedaempfte Schwingung, falls relevant vorhanden
    eig = r["eig"]
    complex_mask = np.abs(eig.imag) > 1e-6
    complex_eigs = eig[complex_mask]
    relevant = complex_eigs[np.abs(complex_eigs) > 0.5]
    if len(relevant) > 0:
        idx = np.argmax(np.abs(relevant))
        z = relevant[idx]
        period_min = 2 * np.pi / abs(np.angle(z)) * step
        # zeige die Eigenantwort einer Station auf sich selbst (deutlichste Bahn fuer Interpretation)
        comp = r["theta_geo"][:, 0, 0]
        P.plot_complex_oscillation(comp, (cols[0], cols[0]), {"mod": abs(z), "period_min": period_min}, step, PLOT_DIR / "13_komplexe_schwingung.png")

    return True

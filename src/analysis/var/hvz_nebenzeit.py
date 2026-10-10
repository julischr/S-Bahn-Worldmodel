"""Validierung 9.5: getrennte VAR-Schaetzung fuer Hauptverkehrszeit (HVZ) vs. Nebenzeit,
Richtung Ost. Nutzt dieselbe Datenaufbereitung (Fourier-Detrending, Tagesausschluss,
Kurzluecken-Interpolation) wie run_all.py, teilt die Residuenreihe dann nach Uhrzeit/
Wochentag in zwei Teilmengen und baut je Teilmenge eigene, kurze Episoden (jedes HVZ-Fenster
= eine Episode, weil Zeilen ausserhalb der Teilmenge auf NaN gesetzt werden und
build_daily_episodes automatisch an NaN-Zeilen abbricht)."""

import numpy as np
import pandas as pd

from .prepare import load_state_vector, pivot_direction
from .detrend import detrend_frame
from .episodes import interpolate_short_gaps, build_daily_episodes
from .var_model import fit_var, aic_bic_for_p, stability_eigenvalues
from .irf import structural_irf, block_bootstrap_irf
from .irf_metrics import fevd, peak_amplitude_time, half_life
from .stations import STATION_NAMES_WEST_OST
from .run_all import EXCLUDED_DATES

HVZ_WINDOWS = [(6, 9), (16, 19)]  # Stunden, jeweils [start, end)


def is_hvz(index: pd.DatetimeIndex) -> np.ndarray:
    weekday = index.weekday  # 0=Mo .. 6=So
    hour = index.hour
    in_window = np.zeros(len(index), dtype=bool)
    for start, end in HVZ_WINDOWS:
        in_window |= (hour >= start) & (hour < end)
    return in_window & (weekday < 5)  # Mo-Fr


def prepare_base_resid():
    df = load_state_vector()
    wide = pivot_direction(df, "Ost")[STATION_NAMES_WEST_OST]
    n_daily, n_weekly = 4, 3
    resid, _ = detrend_frame(wide, n_daily, n_weekly)
    mask_excl = pd.Series(resid.index.date, index=resid.index).isin(EXCLUDED_DATES)
    resid_f = resid[~mask_excl]
    interp = interpolate_short_gaps(resid_f, limit_steps=12)
    return interp


def build_subset_episodes(interp: pd.DataFrame, hvz: bool, min_length: int = 24):
    hvz_mask = is_hvz(interp.index)
    keep_mask = hvz_mask if hvz else ~hvz_mask
    subset = interp.copy()
    subset.loc[~keep_mask, :] = np.nan
    episodes, ep_meta = build_daily_episodes(subset, min_length=min_length)
    return episodes, ep_meta


def run_subset(episodes, p, horizon, order, n_boot, rng):
    res = fit_var(episodes, p)
    As = res.A_matrices()
    eig = stability_eigenvalues(As)
    theta = structural_irf(As, res.Sigma_u, order, horizon)
    boot_theta = block_bootstrap_irf(episodes, p, order, horizon, n_boot=n_boot, rng=rng)
    fevd_h = fevd(theta, horizon)
    time_to_peak, amplitude = peak_amplitude_time(theta, step_minutes=5.0)
    hl = half_life(theta, step_minutes=5.0)
    K = len(episodes[0].columns)
    export = np.zeros(K); import_ = np.zeros(K)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            export[j] += fevd_h[i, j]
            import_[i] += fevd_h[i, j]
    net = pd.Series(export - import_, index=episodes[0].columns)
    return dict(res=res, As=As, eig=eig, theta=theta, boot_theta=boot_theta, fevd_h=fevd_h,
                time_to_peak=time_to_peak, amplitude=amplitude, half_life=hl, net=net)


if __name__ == "__main__":
    interp = prepare_base_resid()
    print("HVZ-Anteil an Zeilen (vor Episodenbildung):", is_hvz(interp.index).mean())
    ep_hvz, meta_hvz = build_subset_episodes(interp, hvz=True)
    ep_neb, meta_neb = build_subset_episodes(interp, hvz=False)
    print("HVZ Episoden:", len(ep_hvz), "n_obs total:", meta_hvz["n_obs"].sum())
    print("Nebenzeit Episoden:", len(ep_neb), "n_obs total:", meta_neb["n_obs"].sum())

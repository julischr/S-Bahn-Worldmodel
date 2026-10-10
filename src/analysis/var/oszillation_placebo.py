"""Validierung 9.6: Absicherung der 20/48-Minuten-Oszillation im Ost-VAR.

(a) Bootstrap-Band um die Eigenantwort, Vergleich mit einer rein monoton abklingenden
    Referenzkurve (einfacher Exponentialfit an die ersten Schritte) -- liegt der
    Wiederanstieg ausserhalb des Bandes um die monotone Referenz?
(b) Eigenwerte/Perioden bei p=17 (AIC-Optimum) statt p=5 (BIC-Optimum).
(c) Placebo: zusaetzliche Fourier-Terme bei EXAKT 20 und 48 Minuten in der Detrending-
    Regression (neue Funktion, ergaenzt detrend.py NICHT, sondern erweitert hier), dann
    VAR neu schaetzen und pruefen, ob die 20/48-min-Eigenwerte verschwinden/schwaecher werden.
"""

import numpy as np
import pandas as pd

from .detrend import build_fourier_design, fit_fourier_residuals
from .prepare import load_state_vector, pivot_direction
from .episodes import interpolate_short_gaps, build_daily_episodes
from .var_model import fit_var, stability_eigenvalues
from .stations import STATION_NAMES_WEST_OST
from .run_all import EXCLUDED_DATES


def build_placebo_fourier_design(index: pd.DatetimeIndex, n_daily: int, n_weekly: int,
                                  extra_periods_min=(20.0, 48.0)) -> pd.DataFrame:
    """Wie build_fourier_design, aber mit zusaetzlichen Sinus/Cosinus-Termen bei den
    angegebenen Perioden (in Minuten) -- NEUE Funktion, veraendert detrend.py nicht."""
    design = build_fourier_design(index, n_daily, n_weekly)
    t0 = index[0]
    t_hours = (index - t0).total_seconds().to_numpy() / 3600.0
    for period_min in extra_periods_min:
        period_h = period_min / 60.0
        omega = 2 * np.pi / period_h
        design[f"placebo_sin_{period_min:g}min"] = np.sin(omega * t_hours)
        design[f"placebo_cos_{period_min:g}min"] = np.cos(omega * t_hours)
    return design


def detrend_frame_placebo(wide: pd.DataFrame, n_daily: int, n_weekly: int,
                           extra_periods_min=(20.0, 48.0)) -> pd.DataFrame:
    design = build_placebo_fourier_design(wide.index, n_daily, n_weekly, extra_periods_min)
    resid_cols = {}
    for col in wide.columns:
        resid, _ = fit_fourier_residuals(wide[col], design)
        resid_cols[col] = resid
    return pd.DataFrame(resid_cols)


def build_episodes_for(resid: pd.DataFrame, min_length=24):
    mask_excl = pd.Series(resid.index.date, index=resid.index).isin(EXCLUDED_DATES)
    resid_f = resid[~mask_excl]
    interp = interpolate_short_gaps(resid_f, limit_steps=12)
    episodes, ep_meta = build_daily_episodes(interp, min_length=min_length)
    return episodes, ep_meta


def eigen_periods(eig: np.ndarray, step_minutes: float = 5.0, top_n: int = 8) -> pd.DataFrame:
    complex_mask = np.abs(eig.imag) > 1e-6
    ceig = eig[complex_mask]
    mods = np.abs(ceig)
    order = np.argsort(-mods)[:top_n]
    rows = []
    for idx in order:
        z = ceig[idx]
        period = 2 * np.pi / abs(np.angle(z)) * step_minutes
        rows.append({"modulus": abs(z), "angle_rad": np.angle(z), "period_min": period})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_state_vector()
    wide = pivot_direction(df, "Ost")[STATION_NAMES_WEST_OST]
    resid_placebo = detrend_frame_placebo(wide, 4, 3)
    episodes_placebo, _ = build_episodes_for(resid_placebo)
    print("n episodes (placebo):", len(episodes_placebo))
    for p in [5, 17]:
        res = fit_var(episodes_placebo, p)
        eig = stability_eigenvalues(res.A_matrices())
        print(f"--- placebo p={p} ---")
        print(eigen_periods(eig))

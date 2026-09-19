"""Fourier-Regression zur Entfernung des Tages-/Wochentakts je Stationsreihe."""

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86400.0


def _time_features(index: pd.DatetimeIndex) -> np.ndarray:
    """Stunden seit Start des Index (float), fuer Fourier-Terme."""
    t0 = index[0]
    return (index - t0).total_seconds().to_numpy() / 3600.0


def build_fourier_design(index: pd.DatetimeIndex, n_daily: int, n_weekly: int) -> pd.DataFrame:
    """Design-Matrix mit n_daily Tages- und n_weekly Wochen-Harmonischen + Konstante."""
    t = _time_features(index)
    cols = {"const": np.ones_like(t)}
    for k in range(1, n_daily + 1):
        omega = 2 * np.pi * k / 24.0
        cols[f"day_sin_{k}"] = np.sin(omega * t)
        cols[f"day_cos_{k}"] = np.cos(omega * t)
    for k in range(1, n_weekly + 1):
        omega = 2 * np.pi * k / (24.0 * 7)
        cols[f"week_sin_{k}"] = np.sin(omega * t)
        cols[f"week_cos_{k}"] = np.cos(omega * t)
    return pd.DataFrame(cols, index=index)


def fit_fourier_residuals(series: pd.Series, design: pd.DataFrame) -> tuple[pd.Series, dict]:
    """OLS von series auf design (nur an nicht-NaN Stellen), gibt Residuen (volle Laenge, NaN erhalten) zurueck."""
    mask = series.notna().to_numpy()
    y = series.to_numpy()[mask]
    X = design.to_numpy()[mask]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted_full = design.to_numpy() @ coef
    resid = series.to_numpy() - fitted_full
    resid[~mask] = np.nan
    resid_series = pd.Series(resid, index=series.index, name=series.name)
    resid_var = float(np.var(y - fitted_full[mask]))
    n = mask.sum()
    k = X.shape[1]
    # AIC/BIC auf Basis der Residualvarianz (Gaussian-Loglik-Naeherung)
    aic = n * np.log(resid_var) + 2 * k
    bic = n * np.log(resid_var) + np.log(n) * k
    info = {"n": int(n), "k": int(k), "resid_var": resid_var, "aic": aic, "bic": bic}
    return resid_series, info


def select_harmonics(series: pd.Series, index: pd.DatetimeIndex, daily_range=range(1, 9), weekly_range=(0, 1, 2)) -> pd.DataFrame:
    """Rastert n_daily x n_weekly Harmonische und berichtet AIC/BIC, fuer die Wahl der Ordnung."""
    rows = []
    for n_daily in daily_range:
        for n_weekly in weekly_range:
            design = build_fourier_design(index, n_daily, n_weekly)
            _, info = fit_fourier_residuals(series, design)
            rows.append({"n_daily": n_daily, "n_weekly": n_weekly, **info})
    return pd.DataFrame(rows)


def detrend_frame(wide: pd.DataFrame, n_daily: int, n_weekly: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wendet Fourier-Entfernung auf alle Spalten an. Gibt (Residuen-Frame, Info-Tabelle) zurueck."""
    design = build_fourier_design(wide.index, n_daily, n_weekly)
    resid_cols = {}
    infos = []
    for col in wide.columns:
        resid, info = fit_fourier_residuals(wide[col], design)
        resid_cols[col] = resid
        infos.append({"station": col, **info})
    resid_df = pd.DataFrame(resid_cols)
    info_df = pd.DataFrame(infos)
    return resid_df, info_df

"""Stationaritaets- und Residualdiagnostik."""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy import stats


def adf_per_episode(episodes: list[pd.DataFrame], min_length: int = 150) -> pd.DataFrame:
    """ADF-Test je Station, ausgefuehrt auf jeder ausreichend langen Episode einzeln.
    Aggregiert ueber Episoden (Median Teststatistik/p-Wert, Anteil stationaer bei 5%)."""
    cols = list(episodes[0].columns)
    rows = []
    for ep in episodes:
        if len(ep) < min_length:
            continue
        for col in cols:
            x = ep[col].to_numpy()
            if np.std(x) < 1e-8:
                continue
            try:
                stat, pval, *_ = adfuller(x, autolag="AIC")
            except Exception:
                continue
            rows.append({"station": col, "adf_stat": stat, "p_value": pval})
    detail = pd.DataFrame(rows)
    agg = detail.groupby("station").agg(
        n_episoden=("adf_stat", "size"),
        median_adf_stat=("adf_stat", "median"),
        median_p_value=("p_value", "median"),
        anteil_stationaer_5pct=("p_value", lambda s: float((s < 0.05).mean())),
    ).reset_index()
    return agg, detail


def residual_diagnostics(U: np.ndarray, columns: list[str], lags: int = 20) -> pd.DataFrame:
    """Ljung-Box, Jarque-Bera, ARCH-Test je Gleichung (Spalte von U)."""
    rows = []
    for i, col in enumerate(columns):
        u = U[:, i]
        lb = acorr_ljungbox(u, lags=[lags], return_df=True)
        lb_stat = float(lb["lb_stat"].iloc[0])
        lb_p = float(lb["lb_pvalue"].iloc[0])
        jb_stat, jb_p = stats.jarque_bera(u)
        try:
            arch_stat, arch_p, _, _ = het_arch(u, nlags=lags)
        except Exception:
            arch_stat, arch_p = np.nan, np.nan
        rows.append(
            {
                "station": col,
                "ljungbox_stat": lb_stat,
                "ljungbox_p": lb_p,
                "ljungbox_autokorreliert_5pct": lb_p < 0.05,
                "jarque_bera_stat": float(jb_stat),
                "jarque_bera_p": float(jb_p),
                "normal_5pct": jb_p >= 0.05,
                "arch_stat": arch_stat,
                "arch_p": arch_p,
                "arch_heterosked_5pct": (arch_p < 0.05) if not np.isnan(arch_p) else np.nan,
            }
        )
    return pd.DataFrame(rows)

"""Validierung 9.8 Variante A: VAR auf ersten Differenzen (Delta delay_mean) statt Level-
Residuen, als Robustheitscheck gegen die dokumentierte Nicht-Stationaritaet eines Teils der
Episoden (Abschnitt 2 in BEFUNDE_VAR.md). Nutzt dieselbe Episodenbildung/Detrending wie
run_all.py, differenziert dann INNERHALB jeder Episode (kein Differenzieren ueber
Episodengrenzen hinweg), schaetzt VAR(5) neu und macht die IRF durch Kumulation wieder mit
der Level-IRF des Hauptlaufs vergleichbar (Kumulative Summe einer Delta-IRF = Level-Antwort
auf einen einmaligen Delta-Schock)."""

import numpy as np
import pandas as pd

from .var_model import fit_var, stability_eigenvalues
from .irf import structural_irf, block_bootstrap_irf
from .irf_metrics import peak_amplitude_time


def difference_episodes(episodes: list[pd.DataFrame]) -> list[pd.DataFrame]:
    diffed = []
    for ep in episodes:
        d = ep.diff().iloc[1:]
        if len(d) > 0:
            diffed.append(d)
    return diffed


def cumulative_theta(theta_diff: np.ndarray) -> np.ndarray:
    """Kumuliert eine auf Differenzen geschaetzte IRF entlang der Zeitachse (axis 0) ->
    entspricht der Level-Antwort auf einen einmaligen Schock in Delta y_0."""
    return np.cumsum(theta_diff, axis=0)


def run_diff_var(episodes, p, order, horizon, n_boot, rng):
    diff_eps = difference_episodes(episodes)
    res = fit_var(diff_eps, p)
    As = res.A_matrices()
    eig = stability_eigenvalues(As)
    theta_diff = structural_irf(As, res.Sigma_u, order, horizon)
    theta_level = cumulative_theta(theta_diff)
    boot_theta_diff = block_bootstrap_irf(diff_eps, p, order, horizon, n_boot=n_boot, rng=rng)
    boot_theta_level = np.cumsum(boot_theta_diff, axis=1)
    return dict(res=res, As=As, eig=eig, theta_diff=theta_diff, theta_level=theta_level,
                boot_theta_level=boot_theta_level, diff_episodes=diff_eps)

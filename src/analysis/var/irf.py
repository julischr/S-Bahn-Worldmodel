"""Impulsantwortfunktionen (MA-Darstellung, Cholesky-Orthogonalisierung) und Bootstrap-Konfidenzbaender."""

import numpy as np
import pandas as pd

from .var_model import fit_var, VARResult


def phi_matrices(As: list[np.ndarray], horizon: int) -> list[np.ndarray]:
    """Reduced-form MA-Koeffizienten Phi_0..Phi_horizon aus A_1..A_p."""
    K = As[0].shape[0]
    p = len(As)
    Phi = [np.eye(K)]
    for s in range(1, horizon + 1):
        acc = np.zeros((K, K))
        for k in range(1, min(s, p) + 1):
            acc += As[k - 1] @ Phi[s - k]
        Phi.append(acc)
    return Phi


def permutation_matrix(order: list[int]) -> np.ndarray:
    K = len(order)
    Pm = np.zeros((K, K))
    for i, orig_idx in enumerate(order):
        Pm[i, orig_idx] = 1.0
    return Pm


def structural_irf(As: list[np.ndarray], Sigma_u: np.ndarray, order: list[int], horizon: int) -> np.ndarray:
    """Orthogonalisierte IRF Theta_0..Theta_horizon (Cholesky, gegebene Ordnung `order`,
    Liste von Original-Spaltenindizes). Rueckgabe in ORIGINALER Variablenreihenfolge:
    Theta[s][i, j] = Effekt eines Schocks an Station j auf Station i, s Schritte spaeter."""
    K = As[0].shape[0]
    Pm = permutation_matrix(order)
    As_perm = [Pm @ A @ Pm.T for A in As]
    Sigma_perm = Pm @ Sigma_u @ Pm.T
    # Cholesky kann bei (fast) singulaeren Sigma numerisch knapp scheitern -> kleine Ridge
    try:
        P_perm = np.linalg.cholesky(Sigma_perm)
    except np.linalg.LinAlgError:
        Sigma_perm = Sigma_perm + np.eye(K) * 1e-8
        P_perm = np.linalg.cholesky(Sigma_perm)
    Phi_perm = phi_matrices(As_perm, horizon)
    Theta_perm = [phi @ P_perm for phi in Phi_perm]
    Theta_orig = [Pm.T @ th @ Pm for th in Theta_perm]
    return np.stack(Theta_orig, axis=0)  # (horizon+1, K, K)


def one_std_irf(theta: np.ndarray) -> np.ndarray:
    """theta bereits in Einheiten einer 1-Standardabweichungs-Schockgroesse (Cholesky-Faktor
    beinhaltet die Schock-Staerke), daher identisch zu structural_irf-Ausgabe."""
    return theta


def block_bootstrap_irf(
    episodes: list[pd.DataFrame],
    p: int,
    order: list[int],
    horizon: int,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Episodenweiser Block-Bootstrap: zieht len(episodes) Episoden mit Zuruecklegen,
    schaetzt VAR(p) neu, berechnet Theta. Gibt Array (n_boot, horizon+1, K, K) zurueck."""
    n_ep = len(episodes)
    K = episodes[0].shape[1]
    out = np.full((n_boot, horizon + 1, K, K), np.nan)
    idx_all = np.arange(n_ep)
    for b in range(n_boot):
        draw = rng.choice(idx_all, size=n_ep, replace=True)
        boot_eps = [episodes[i] for i in draw]
        try:
            res = fit_var(boot_eps, p)
            As = res.A_matrices()
            theta = structural_irf(As, res.Sigma_u, order, horizon)
            out[b] = theta
        except Exception:
            continue
    return out

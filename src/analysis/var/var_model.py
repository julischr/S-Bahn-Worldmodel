"""Episodenweise gepoolte VAR(p)-Schaetzung per Equation-by-Equation OLS."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def build_lagged_design(episodes: list[pd.DataFrame], p: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Baut gepoolte (Y, X) Matrizen ueber alle Episoden, Lags nie ueber Episodengrenzen hinweg.

    Y: T x K (abhaengige Variablen, y_t)
    X: T x (K*p + 1) (Konstante + y_{t-1..t-p} gestapelt)
    """
    cols = list(episodes[0].columns)
    K = len(cols)
    Y_list = []
    X_list = []
    for ep in episodes:
        arr = ep.to_numpy()
        T_e = arr.shape[0]
        if T_e <= p:
            continue
        y_block = arr[p:T_e]  # (T_e-p) x K
        n = y_block.shape[0]
        lag_blocks = [arr[p - lag: T_e - lag] for lag in range(1, p + 1)]  # each (T_e-p) x K
        const_col = np.ones((n, 1))
        x_block = np.hstack([const_col] + lag_blocks)
        Y_list.append(y_block)
        X_list.append(x_block)
    if not Y_list:
        return np.empty((0, K)), np.empty((0, K * p + 1)), cols
    Y = np.vstack(Y_list)
    X = np.vstack(X_list)
    return Y, X, cols


@dataclass
class VARResult:
    p: int
    K: int
    columns: list[str]
    B: np.ndarray  # (K*p+1) x K coefficient matrix (const + lags), columns=equations
    Sigma_u: np.ndarray  # K x K residual covariance (MLE, divide by T)
    Sigma_u_df: np.ndarray  # K x K residual covariance (df-corrected, divide by T-(Kp+1))
    T: int
    U: np.ndarray = field(repr=False)  # residuals T x K
    X: np.ndarray = field(repr=False)
    Y: np.ndarray = field(repr=False)

    def A_matrices(self) -> list[np.ndarray]:
        """Gibt Liste der Lag-Koeffizientenmatrizen A_1..A_p zurueck, je K x K,
        A_l[i,j] = Effekt von Station j (Lag l) auf Station i."""
        K, p = self.K, self.p
        As = []
        for lag in range(p):
            block = self.B[1 + lag * K: 1 + (lag + 1) * K, :]  # K x K, rows=regressor j, cols=eq i
            As.append(block.T)  # -> A_l[i,j]
        return As

    def const(self) -> np.ndarray:
        return self.B[0, :]


def fit_var(episodes: list[pd.DataFrame], p: int) -> VARResult:
    Y, X, cols = build_lagged_design(episodes, p)
    K = len(cols)
    T = Y.shape[0]
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)  # (Kp+1) x K
    U = Y - X @ B
    Sigma_u = (U.T @ U) / T
    n_params = X.shape[1]
    df = max(T - n_params, 1)
    Sigma_u_df = (U.T @ U) / df
    return VARResult(p=p, K=K, columns=cols, B=B, Sigma_u=Sigma_u, Sigma_u_df=Sigma_u_df, T=T, U=U, X=X, Y=Y)


def aic_bic_for_p(episodes: list[pd.DataFrame], p_range: range) -> pd.DataFrame:
    rows = []
    for p in p_range:
        res = fit_var(episodes, p)
        K, T = res.K, res.T
        sign, logdet = np.linalg.slogdet(res.Sigma_u)
        n_params_per_eq = K * p + 1
        total_params = K * n_params_per_eq
        aic = logdet + (2.0 / T) * total_params
        bic = logdet + (np.log(T) / T) * total_params
        rows.append({"p": p, "T": T, "logdet_Sigma_u": logdet, "aic": aic, "bic": bic})
    return pd.DataFrame(rows)


def companion_matrix(As: list[np.ndarray]) -> np.ndarray:
    K = As[0].shape[0]
    p = len(As)
    top = np.hstack(As)  # K x Kp
    if p == 1:
        return top
    eye_block = np.eye(K * (p - 1))
    bottom = np.hstack([eye_block, np.zeros((K * (p - 1), K))])
    return np.vstack([top, bottom])


def stability_eigenvalues(As: list[np.ndarray]) -> np.ndarray:
    F = companion_matrix(As)
    return np.linalg.eigvals(F)

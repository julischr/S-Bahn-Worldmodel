"""Abgeleitete Kennzahlen aus den Impulsantworten: FEVD, Peak-Zeit, Amplitude, Halbwertszeit."""

import numpy as np


def fevd(theta: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast Error Variance Decomposition bei gegebenem Horizont.
    theta: (H+1, K, K) Theta[s,i,j]. Rueckgabe: FEVD[i,j] = Anteil der h-Schritt-Prognosefehler-
    varianz von Variable i, der durch Schocks in Variable j erklaert wird (Zeilensumme = 1)."""
    K = theta.shape[1]
    h = min(horizon, theta.shape[0] - 1)
    cum_sq = np.sum(theta[: h + 1] ** 2, axis=0)  # K x K, sum_s Theta[s,i,j]^2
    row_sum = cum_sq.sum(axis=1, keepdims=True)
    return cum_sq / row_sum


def peak_amplitude_time(theta: np.ndarray, step_minutes: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Fuer jedes Paar (i,j): Zeit bis zum betragsgroessten Ausschlag (Minuten) und dessen
    (signierter) Wert. theta: (H+1, K, K)."""
    H1, K, _ = theta.shape
    absval = np.abs(theta)
    t_idx = np.argmax(absval, axis=0)  # K x K
    time_to_peak = t_idx * step_minutes
    i_idx, j_idx = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
    amplitude = theta[t_idx, i_idx, j_idx]
    return time_to_peak, amplitude


def half_life(theta: np.ndarray, step_minutes: float = 5.0) -> np.ndarray:
    """Halbwertszeit (Minuten) je Paar (i,j): Zeit vom Peak bis der |Betrag| erstmals unter
    die Haelfte des Peak-Betrags faellt. NaN falls im Horizont nicht erreicht."""
    H1, K, _ = theta.shape
    absval = np.abs(theta)
    t_peak = np.argmax(absval, axis=0)
    peak_val = np.max(absval, axis=0)
    hl = np.full((K, K), np.nan)
    for i in range(K):
        for j in range(K):
            tp = t_peak[i, j]
            pv = peak_val[i, j]
            if pv < 1e-9:
                continue
            found = None
            for s in range(tp, H1):
                if absval[s, i, j] < 0.5 * pv:
                    found = s
                    break
            if found is not None:
                hl[i, j] = (found - tp) * step_minutes
    return hl

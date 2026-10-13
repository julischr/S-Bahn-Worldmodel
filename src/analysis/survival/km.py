"""Kaplan-Meier mit tageweisem Bootstrap, Log-Rank (k Gruppen). Eigene Implementierung
(keine lifelines-Abhaengigkeit im Projekt)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Ueberlebensniveaus fuer Zeit-Quartile: S<=0.75 -> 25 %-Quantil der Erholungszeit usw.
QS = (0.75, 0.5, 0.25)


def _prep(t, e):
    o = np.argsort(t, kind="stable")
    ts = np.asarray(t)[o]
    es = np.asarray(e)[o]
    u, start = np.unique(ts, return_index=True)
    return o, es, u, start


def km_weighted(t, e, w=None):
    """Gewichtete KM-Kurve. Rueckgabe: (u, S) an den eindeutigen Zeiten."""
    t = np.asarray(t, float); e = np.asarray(e, float)
    w = np.ones(len(t)) if w is None else np.asarray(w, float)
    o, es, u, start = _prep(t, e)
    ws = w[o]
    D = np.add.reduceat(ws * es, start)
    C = np.add.reduceat(ws, start)
    N = C[::-1].cumsum()[::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.cumprod(np.where(N > 0, 1 - D / N, 1.0))
    return u, S


def quantile_from_curve(u, S, q):
    idx = np.flatnonzero(S <= q)
    return u[idx[0]] if len(idx) else np.inf


def km_quantiles(t, e, w=None, qs=QS):
    u, S = km_weighted(t, e, w)
    return np.array([quantile_from_curve(u, S, q) for q in qs])


def km_at(u, S, grid):
    """S(t) an Gitterpunkten (rechtsstetig)."""
    idx = np.searchsorted(u, grid, side="right") - 1
    return np.where(idx >= 0, S[np.maximum(idx, 0)], 1.0)


def make_day_weights(n_days: int, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_days, np.full(n_days, 1 / n_days), size=B).astype(np.float32)


def km_bootstrap(t, e, day_idx, Wdays, grid=None, qs=QS, chunk=100):
    """Tageweiser Bootstrap. Wdays (B, n_days) Multiplizitaeten; day_idx: Tagesindex je Zeile.
    Rueckgabe dict: q_point (len(qs)), q_boot (B, len(qs)), grid_point, grid_boot (B, G)."""
    t = np.asarray(t, float); e = np.asarray(e, float); day_idx = np.asarray(day_idx)
    o, es, u, start = _prep(t, e)
    day_s = day_idx[o]
    B = Wdays.shape[0]
    q_boot = np.empty((B, len(qs)))
    g_boot = np.empty((B, len(grid))) if grid is not None else None
    gidx = np.searchsorted(u, grid, side="right") - 1 if grid is not None else None
    for s in range(0, B, chunk):
        W = Wdays[s:s + chunk][:, day_s].astype(np.float64)
        D = np.add.reduceat(W * es, start, axis=1)
        C = np.add.reduceat(W, start, axis=1)
        N = np.cumsum(C[:, ::-1], axis=1)[:, ::-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            S = np.cumprod(np.where(N > 0, 1 - D / np.where(N > 0, N, 1), 1.0), axis=1)
        for k, q in enumerate(qs):
            hit = S <= q
            first = hit.argmax(axis=1)
            q_boot[s:s + chunk, k] = np.where(hit.any(axis=1), u[first], np.inf)
        if grid is not None:
            g_boot[s:s + chunk] = np.where(gidx >= 0, S[:, np.maximum(gidx, 0)], 1.0)
    u0, S0 = km_weighted(t, e)
    out = dict(q_point=np.array([quantile_from_curve(u0, S0, q) for q in qs]), q_boot=q_boot)
    if grid is not None:
        out["grid_point"] = km_at(u0, S0, grid)
        out["grid_boot"] = g_boot
    return out


def ci_perc(x, lo=2.5, hi=97.5):
    """Perzentil-KI; inf-Werte zaehlen als 'nicht erreicht'."""
    x = np.asarray(x, float)
    return np.percentile(x, lo, method="nearest"), np.percentile(x, hi, method="nearest")


def fmt_q(point, boot):
    lo, hi = ci_perc(boot)
    f = lambda v: "n.e." if not np.isfinite(v) else f"{v:.1f}"
    return f(point), f(lo), f(hi)


def logrank_k(t, e, g):
    """k-Stichproben-Log-Rank. Rueckgabe (chi2, df, p)."""
    t = np.asarray(t, float); e = np.asarray(e, float); g = np.asarray(g)
    groups = np.unique(g)
    k = len(groups)
    if k < 2:
        return np.nan, 0, np.nan
    et = np.unique(t[e > 0])
    if len(et) == 0:
        return np.nan, k - 1, np.nan
    n = np.zeros((k, len(et))); d = np.zeros((k, len(et)))
    for j, gr in enumerate(groups):
        m = g == gr
        tj = np.sort(t[m])
        n[j] = len(tj) - np.searchsorted(tj, et, side="left")
        te = np.sort(t[m & (e > 0)])
        d[j] = np.searchsorted(te, et, side="right") - np.searchsorted(te, et, side="left")
    N = n.sum(0); Dt = d.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        exp = n * (Dt / N)
        c = np.where(N > 1, Dt * (N - Dt) / (N - 1), 0.0)
    O_E = (d - exp).sum(1)[: k - 1]
    V = np.zeros((k - 1, k - 1))
    for tt in range(len(et)):
        if N[tt] <= 1:
            continue
        nn = n[: k - 1, tt]
        V += c[tt] / N[tt] * (np.diag(nn) - np.outer(nn, nn) / N[tt])
    try:
        chi2 = float(O_E @ np.linalg.solve(V, O_E))
    except np.linalg.LinAlgError:
        return np.nan, k - 1, np.nan
    return chi2, k - 1, float(stats.chi2.sf(chi2, k - 1))


def logrank_robust(t, e, z, cluster):
    """Clusterrobuster Log-Rank-Score-Test (2 Gruppen, z in {0,1}), Cluster = Betriebstag.
    Score-Residuen psi_i = int (z_i - zbar(s)) dM_i(s), M unter H0 (gepoolter Nelson-Aalen).
    Rueckgabe (chi2, p, log_hr_einschritt, se_log_hr_robust). HR>1: Gruppe z=1 erholt schneller."""
    t = np.asarray(t, float); e = np.asarray(e, float); z = np.asarray(z, float)
    o = np.argsort(t, kind="stable")
    t, e, z = t[o], e[o], z[o]
    cl = np.asarray(cluster)[o]
    u, start = np.unique(t, return_index=True)
    n_tot = np.add.reduceat(np.ones(len(t)), start)
    n_z = np.add.reduceat(z, start)
    d_tot = np.add.reduceat(e, start)
    N = n_tot[::-1].cumsum()[::-1]
    Nz = n_z[::-1].cumsum()[::-1]
    zbar = Nz / N
    dL = d_tot / N
    L = np.cumsum(dL)
    A = np.cumsum(zbar * dL)
    idx = np.searchsorted(u, t)
    zb_i = zbar[idx]
    psi = e * (z - zb_i) - np.where(z > 0, L[idx] - A[idx], -A[idx])
    U = psi.sum()
    # modellbasierte Information (Hypergeometrische Varianz) fuer Ein-Schritt-HR
    Vm = np.sum(np.where(N > 1, d_tot * (N - d_tot) / np.maximum(N - 1, 1) * zbar * (1 - zbar), 0.0))
    cs = pd.Series(psi).groupby(cl).sum().to_numpy()
    Vr = float((cs ** 2).sum())
    chi2 = U ** 2 / Vr if Vr > 0 else np.nan
    return chi2, float(stats.chi2.sf(chi2, 1)), U / Vm, np.sqrt(Vr) / Vm

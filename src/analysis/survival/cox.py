"""Schritte 5/6: Cox (clusterrobust), PH-Test (Grambsch-Therneau), AFT (Log-Normal/Weibull),
Restricted Cubic Splines. Eigene Implementierung auf statsmodels/scipy."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.duration.hazard_regression import PHReg


def rcs_basis(x, knots):
    """Restricted cubic spline (Harrell): Spalten [x, h_1 ... h_{k-2}]; die h_j sind die
    nichtlinearen Terme."""
    x = np.asarray(x, float)
    k = np.asarray(knots, float)
    kk = len(k)
    def pos3(v):
        return np.maximum(v, 0) ** 3
    cols = [x]
    for j in range(kk - 2):
        h = (pos3(x - k[j]) - pos3(x - k[kk - 2]) * (k[kk - 1] - k[j]) / (k[kk - 1] - k[kk - 2])
             + pos3(x - k[kk - 1]) * (k[kk - 2] - k[j]) / (k[kk - 1] - k[kk - 2])) / (k[kk - 1] - k[0]) ** 2
        cols.append(h)
    return np.column_stack(cols)


def fit_cox(t, e, X: pd.DataFrame, cluster, strata=None):
    mod = PHReg(np.asarray(t, float), X.to_numpy(float), status=np.asarray(e, float), ties="efron",
                strata=None if strata is None else np.asarray(strata))
    res = mod.fit(groups=np.asarray(cluster), disp=0)
    return mod, res


def coef_table(res, names):
    b = res.params
    se = res.bse
    return pd.DataFrame({"term": names, "coef": b, "se_robust": se, "hr": np.exp(b),
                         "hr_lo": np.exp(b - 1.96 * se), "hr_hi": np.exp(b + 1.96 * se),
                         "p": 2 * stats.norm.sf(np.abs(b / se))})


def wald_robust(res, idx):
    """Gemeinsamer robuster Wald-Test fuer Koeffizienten idx."""
    idx = list(idx)
    b = res.params[idx]
    V = res.cov_params()[np.ix_(idx, idx)]
    chi2 = float(b @ np.linalg.solve(V, b))
    return chi2, len(idx), float(stats.chi2.sf(chi2, len(idx)))


def zph_test(mod, res, t, e, names):
    """Grambsch-Therneau-Test (Zeittransformation: Rang) auf Basis der Schoenfeld-Residuen
    des Modells mit modellbasierter Kovarianz. Rueckgabe: je Kovariate chi2(1), p; global."""
    t = np.asarray(t, float); e = np.asarray(e, float)
    r = np.asarray(res.schoenfeld_residuals)
    ev = e > 0
    r = r[ev]
    tt = t[ev]
    d = ev.sum()
    mb = mod.fit(disp=0)                     # modellbasierte Kovarianz
    V = np.asarray(mb.cov_params())
    beta = np.asarray(mb.params)
    ok = ~np.isnan(r).any(axis=1)
    r, tt = r[ok], tt[ok]
    d = len(tt)
    g = stats.rankdata(tt) / d
    xx = g - g.mean()
    sc = r @ V * d + beta                    # skalierte Residuen
    num = (xx @ sc) ** 2
    den = d * np.diag(V) * (xx @ xx)
    chi2 = num / den
    u = xx @ sc                               # Teststatistik-Vektor je Kovariate
    glob = float(u @ np.linalg.solve(V, u) / (d * (xx @ xx)))
    p = stats.chi2.sf(chi2, 1)
    return pd.DataFrame({"term": names, "chi2": chi2, "p": p}), (glob, len(names), float(stats.chi2.sf(glob, len(names))))


# ---------------------------------------------------------------------------
# AFT (Log-Normal / Weibull) per ML mit clusterrobuster Sandwich-Kovarianz
# ---------------------------------------------------------------------------

def _aft_ll(theta, X, y, e, dist):
    k = X.shape[1]
    beta, ls = theta[:k], theta[k]
    s = np.exp(ls)
    z = (y - X @ beta) / s
    if dist == "lognormal":
        logf = stats.norm.logpdf(z) - ls - y            # y = log t -> Dichte in t: f(t)=phi(z)/(s t)
        logS = stats.norm.logsf(z)
    else:  # weibull: log t = X b + s * W, W ~ Gumbel(min)
        logf = z - np.exp(z) - ls - y
        logS = -np.exp(z)
    return np.where(e > 0, logf, logS)


def fit_aft(t, e, X: pd.DataFrame, cluster, dist="lognormal", offset_min=1.0):
    """AFT: log(T+offset) = X b + s*eps. Rueckgabe Tabelle (Zeitverhaeltnis exp(b)) und dict."""
    Xm = np.column_stack([np.ones(len(X)), X.to_numpy(float)])
    y = np.log(np.asarray(t, float) + offset_min)
    e = np.asarray(e, float)
    k = Xm.shape[1]
    f = lambda th: -_aft_ll(th, Xm, y, e, dist).sum()
    b0 = np.linalg.lstsq(Xm, y, rcond=None)[0]
    th0 = np.r_[b0, np.log(np.std(y - Xm @ b0) + 1e-3)]
    r = optimize.minimize(f, th0, method="BFGS", options=dict(maxiter=500, gtol=1e-5))
    th = r.x

    def score_obs(th):
        h = 1e-5
        g = np.empty((len(y), len(th)))
        for j in range(len(th)):
            a = th.copy(); b = th.copy(); a[j] += h; b[j] -= h
            g[:, j] = (_aft_ll(a, Xm, y, e, dist) - _aft_ll(b, Xm, y, e, dist)) / (2 * h)
        return g
    S = score_obs(th)
    # Hesse numerisch aus Scores
    h = 1e-4
    H = np.empty((len(th), len(th)))
    for j in range(len(th)):
        a = th.copy(); b = th.copy(); a[j] += h; b[j] -= h
        H[:, j] = (score_obs(a).sum(0) - score_obs(b).sum(0)) / (2 * h)
    H = -(H + H.T) / 2
    Hinv = np.linalg.inv(H)
    cl = pd.Series(np.asarray(cluster))
    Sc = pd.DataFrame(S).groupby(cl.to_numpy()).sum().to_numpy()
    meat = Sc.T @ Sc
    V = Hinv @ meat @ Hinv
    se = np.sqrt(np.diag(V))
    names = ["(Intercept)"] + list(X.columns) + ["log_sigma"]
    tab = pd.DataFrame({"term": names, "coef": th, "se_robust": se})
    tab["zeitverhaeltnis"] = np.exp(tab["coef"]); tab["tr_lo"] = np.exp(tab["coef"] - 1.96 * se)
    tab["tr_hi"] = np.exp(tab["coef"] + 1.96 * se)
    tab["p"] = 2 * stats.norm.sf(np.abs(tab["coef"] / se))
    return tab, dict(theta=th, V=V, loglik=-r.fun, converged=bool(r.success), grad_max=float(np.abs(r.jac).max()), message=str(r.message), n=len(y), names=names, aic=2 * r.fun + 2 * len(th))

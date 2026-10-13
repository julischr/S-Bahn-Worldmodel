"""Schritt 4: Kaplan-Meier je Gruppe, Placebo, Bootstrap, Log-Rank.
Aufruf: uv run python -m src.analysis.survival.run_km"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg

from . import data as D
from . import km as K
from .groups import code_table, gruppe_map

B = 1000
SEED = 7
GRID = np.arange(0, 241, 2.0)
LEVELS = {"st": "Meldestation", "ln": "Strecke"}


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = np.sum(~np.isnan(p))
    adj = np.full(len(p), np.nan)
    run = 0.0
    for rank, i in enumerate(o[:m]):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def robust_cox(t, e, ind, cluster):
    """Ein-Schritt-HR (Ereignis vs Placebo) und clusterrobuster Score-Test (Log-Rank).
    HR>1 = Erholung schneller als Placebo."""
    chi2, p, lhr, se = K.logrank_robust(t, e, np.asarray(ind, float), cluster)
    return float(np.exp(lhr)), float(p)


def load():
    ep = pd.read_parquet(D.ZW / "episoden.parquet")
    pls = {k: pd.read_parquet(D.ZW / f"{k}.parquet") for k in ("placebo", "placebo2")}
    for k, p in pls.items():
        p["day_nr"] = np.array(p["parent_opday"], dtype="datetime64[D]").astype("int64")
    t = code_table(ep)
    gm = gruppe_map(t)
    ep["gruppe"] = ep["code"].map(gm)
    for p in pls.values():
        p["gruppe"] = p["code"].map(gm)
    return ep, pls, t


def analyse(ep, pl, level, thr, cens="std", variant="R0", Wd=None, days=None, label=""):
    tc, ec = f"t_{variant}_{thr}_{level}_{cens}", f"e_{variant}_{thr}_{level}_{cens}"
    x = ep.dropna(subset=[tc]); y = pl.dropna(subset=[tc])
    rows, curves = [], {}
    order = ["alle"] + sorted(ep["gruppe"].unique(), key=lambda g: (g == "selten", -(ep["gruppe"] == g).sum()))
    for g in order:
        xg = x if g == "alle" else x[x["gruppe"] == g]
        yg = y if g == "alle" else y[y["gruppe"] == g]
        if len(xg) < 10 or len(yg) < 10:
            continue
        bx = K.km_bootstrap(xg[tc], xg[ec], np.searchsorted(days, xg["opday_nr"]), Wd, grid=GRID)
        by = K.km_bootstrap(yg[tc], yg[ec], np.searchsorted(days, yg["day_nr"]), Wd, grid=GRID)
        row = dict(gruppe=g, ebene=level, schwelle=thr, zensur=cens, ursprung=variant,
                   n_episoden=len(xg), n_placebo=len(yg),
                   anteil_zensiert=1 - xg[ec].mean(), anteil_zensiert_placebo=1 - yg[ec].mean())
        for k, nm in enumerate(("q25", "median", "q75")):
            for who, bb in (("", bx), ("_placebo", by)):
                pt, lo, hi = K.fmt_q(bb["q_point"][k], bb["q_boot"][:, k])
                row[f"{nm}{who}"], row[f"{nm}{who}_lo"], row[f"{nm}{who}_hi"] = pt, lo, hi
        dm = bx["q_boot"][:, 1] - by["q_boot"][:, 1]
        row["diff_median"] = bx["q_point"][1] - by["q_point"][1]
        row["diff_median_lo"], row["diff_median_hi"] = (np.nan, np.nan) if not np.isfinite(dm).all() and np.isinf(dm).mean() > .025 else K.ci_perc(dm[np.isfinite(dm)])
        row["diff_median_boot_nicht_erreicht"] = float(np.mean(~np.isfinite(dm)))
        t_ = np.r_[xg[tc], yg[tc]]; e_ = np.r_[xg[ec], yg[ec]]; gi = np.r_[np.zeros(len(xg)), np.ones(len(yg))]
        row["logrank_chi2"], _, row["logrank_p"] = K.logrank_k(t_, e_, gi)
        row["hr_vs_placebo"], row["cox_robust_p"] = robust_cox(t_, e_, gi == 0, np.r_[xg["opday_nr"], yg["day_nr"]])
        nc = f"noeff_{variant}_{thr}"
        if level == "st" and nc in xg:
            row["ohne_wirkung"] = xg[nc].mean(); row["ohne_wirkung_placebo"] = yg[nc].mean()
        rows.append(row)
        curves[g] = dict(x_point=bx["grid_point"], x_lo=np.percentile(bx["grid_boot"], 2.5, axis=0),
                         x_hi=np.percentile(bx["grid_boot"], 97.5, axis=0), p_point=by["grid_point"],
                         p_lo=np.percentile(by["grid_boot"], 2.5, axis=0), p_hi=np.percentile(by["grid_boot"], 97.5, axis=0),
                         n=len(xg))
    df = pd.DataFrame(rows)
    df["cox_robust_p_holm"] = np.r_[[np.nan], holm(df.loc[df.gruppe != "alle", "cox_robust_p"])] if df.gruppe.iloc[0] == "alle" else holm(df["cox_robust_p"])
    return df, curves


def main():
    ep, pls, tab = load()
    days = np.array(sorted(ep["opday_nr"].unique()))
    Wd = K.make_day_weights(len(days), B, SEED)
    out = []
    curves_all = {}
    ep3 = ep[ep["n_fahrten"] >= 3]
    for pname, pl in pls.items():
        for level in LEVELS:
            for thr in (180, 360):
                for cens in ("std", "nofolge"):
                    df, cv = analyse(ep, pl, level, thr, cens, "R0", Wd, days)
                    df["placebo"] = pname; df["teilmenge"] = "alle"
                    out.append(df)
                    curves_all[(pname, level, thr, cens)] = cv
                if level == "st" and thr == 180:
                    df, _ = analyse(ep3, pl[pl["parent_episode"].isin(ep3["episode_id"])], level, thr, "std", "R0", Wd, days)
                    df["placebo"] = pname; df["teilmenge"] = "n_fahrten>=3"
                    out.append(df)
    res = pd.concat(out, ignore_index=True)
    res.to_csv(D.OUT / "km_ergebnisse.csv", index=False)
    pd.to_pickle(curves_all, D.ZW / "km_kurven.pkl")

    # Log-Rank zwischen Codes (R0, Meldestation, 180 s, nur Ereignisse), global + paarweise
    x = ep.dropna(subset=["t_R0_180_st_std"])
    x = x[x["gruppe"] != "selten"]
    chi2, df_, p = K.logrank_k(x["t_R0_180_st_std"], x["e_R0_180_st_std"], x["gruppe"])
    gl = pd.DataFrame([dict(test="Log-Rank k Gruppen (Codes >=50)", chi2=chi2, df=df_, p_naiv=p, n=len(x))])
    # clusterrobuster Gesamttest: Cox mit Code-Dummies, Wald-Test
    Xd = pd.get_dummies(x["gruppe"], drop_first=True, dtype=float)
    r = PHReg(x["t_R0_180_st_std"].to_numpy(), Xd.to_numpy(), status=x["e_R0_180_st_std"].to_numpy(), ties="breslow").fit(groups=x["opday_nr"].to_numpy(), disp=0)
    w = r.wald_test(np.eye(Xd.shape[1]), scalar=True)
    gl["wald_robust_chi2"] = float(w.statistic); gl["wald_robust_p"] = float(w.pvalue)
    gl.to_csv(D.OUT / "km_logrank_codes_global.csv", index=False)
    gs = list(x["gruppe"].unique())
    pr = []
    for i, a in enumerate(gs):
        for b in gs[i + 1:]:
            m = x["gruppe"].isin([a, b])
            c_, _, p_ = K.logrank_k(x.loc[m, "t_R0_180_st_std"], x.loc[m, "e_R0_180_st_std"], x.loc[m, "gruppe"])
            pr.append(dict(a=a, b=b, chi2=c_, p=p_))
    pr = pd.DataFrame(pr); pr["p_holm"] = holm(pr["p"])
    pr.to_csv(D.OUT / "km_logrank_codes_paarweise.csv", index=False)
    print(gl.T)
    print(f"Paarweise: {len(pr)} Tests, {int((pr.p_holm < 0.05).sum())} mit Holm-p<0.05")


if __name__ == "__main__":
    main()

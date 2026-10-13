"""Schritte 5 und 6: Cox, PH-Test, AFT, Spline/Kipppunkt.
Aufruf: uv run python -m src.analysis.survival.run_cox"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg

from . import data as D
from . import cox as C
from . import km as K
from .groups import code_table, gruppe_map

SEV_CAP = 120.0
SEV_BINS = [0, 3, 5, 10, 20, np.inf]
SEV_LABELS = ["<3", "3-5", "5-10", "10-20", ">20"]
B = 1000


def load_events():
    ep = pd.read_parquet(D.ZW / "episoden.parquet")
    gm = gruppe_map(code_table(ep))
    ep["gruppe"] = ep["code"].map(gm)
    return ep


def design(df, ref_code, ref_station, sev_mode="linear", knots=None, interact=False, code_dummies=True):
    X = pd.DataFrame(index=df.index)
    if code_dummies:
        cd = pd.get_dummies(df["gruppe"], dtype=float)
        cd = cd.drop(columns=[ref_code])
        cd.columns = [f"code_{c}" for c in cd.columns]
        X = X.join(cd)
    sd = pd.get_dummies(df["station"], dtype=float).drop(columns=[ref_station])
    sd.columns = [f"st_{c}" for c in sd.columns]
    X = X.join(sd)
    X["richtung_West"] = (df["richtung"] == "West").astype(float)
    X["hvz"] = df["hvz"].astype(float)
    for w in ("Fr", "Sa", "So+Feiertag"):
        X[f"wt_{w}"] = (df["wochentagstyp"] == w).astype(float)
    X["stoerungslast"] = df["stoerungslast"].astype(float)
    X["vorbelastung_5min"] = df["vorbelastung_min"] / 5
    sev = df["schwere_min"].clip(upper=SEV_CAP)
    if sev_mode == "linear":
        X["schwere_5min"] = sev / 5
    else:
        B_ = C.rcs_basis(sev, knots)
        for j in range(B_.shape[1]):
            X[f"schwere_s{j}"] = B_[:, j]
        if interact:
            for j in range(B_.shape[1]):
                X[f"schwere_s{j}_x_hvz"] = B_[:, j] * df["hvz"].astype(float)
    X["log2_fahrten15"] = np.log2(df["n_fahrten_15"].clip(lower=1))
    return X


def prep(ep, variant="R15", thr=180, level="st", cens="std"):
    tc, ec = f"t_{variant}_{thr}_{level}_{cens}", f"e_{variant}_{thr}_{level}_{cens}"
    cols = [tc, ec, "schwere_min", "vorbelastung_min", "stoerungslast", "n_fahrten_15"]
    n0 = len(ep)
    d = ep.dropna(subset=cols).copy()
    d["t"] = d[tc]; d["e"] = d[ec]
    info = dict(n_episoden=n0, n_vollstaendig=len(d), n_ausgeschlossen=n0 - len(d),
                n_ereignisse=int(d["e"].sum()))
    return d, info


def main():
    ep = load_events()
    ref_code = ep["gruppe"].value_counts().index[0]
    ref_station = ep["station"].value_counts().index[0]
    log = []

    def run_cox(d, name, strata=None, code_dummies=True):
        X = design(d, ref_code, ref_station, code_dummies=code_dummies)
        mod, res = C.fit_cox(d["t"], d["e"], X, d["opday_nr"], strata=strata)
        tab = C.coef_table(res, list(X.columns))
        tab["modell"] = name; tab["n"] = len(d); tab["ereignisse"] = int(d["e"].sum())
        zt, glob = C.zph_test(mod, res, d["t"], d["e"], list(X.columns))
        zt["modell"] = name
        zt["global_chi2"], zt["global_df"], zt["global_p"] = glob
        return tab, zt, X, mod, res

    main_models = {}
    tabs, zts, infos = [], [], []
    for name, (variant, thr, level, cens) in {
        "haupt_R15_180_station": ("R15", 180, "st", "std"),
        "neben_R15_180_strecke": ("R15", 180, "ln", "std"),
        "neben_R15_360_station": ("R15", 360, "st", "std"),
        "sens_R15_180_station_ohneFolgeZensur": ("R15", 180, "st", "nofolge"),
    }.items():
        d, info = prep(ep, variant, thr, level, cens)
        info["modell"] = name; infos.append(info)
        tab, zt, X, mod, res = run_cox(d, name)
        tabs.append(tab); zts.append(zt)
        main_models[name] = (d, X, mod, res)
        print(name, info)
    pd.concat(tabs).to_csv(D.OUT / "cox_koeffizienten.csv", index=False)
    pd.concat(zts).to_csv(D.OUT / "cox_ph_test.csv", index=False)
    pd.DataFrame(infos).to_csv(D.OUT / "cox_fallzahlen.csv", index=False)

    # PH-Groessenordnung: HR frueh (0-T*) vs spaet (> T*), T* = Median der Ereigniszeiten
    d, X, mod, res = main_models["haupt_R15_180_station"]
    tstar = float(d.loc[d["e"] > 0, "t"].median())
    e_t = np.where(d["t"] <= tstar, d["e"], 0.0); t_e = np.minimum(d["t"], tstar)
    r1 = PHReg(t_e.to_numpy(), X.to_numpy(float), status=e_t, ties="efron").fit(groups=d["opday_nr"].to_numpy(), disp=0)
    late = d["t"] > tstar
    r2 = PHReg(d.loc[late, "t"].to_numpy(), X[late].to_numpy(float), status=d.loc[late, "e"].to_numpy(),
               entry=np.full(late.sum(), tstar), ties="efron").fit(groups=d.loc[late, "opday_nr"].to_numpy(), disp=0)
    ph = pd.DataFrame({"term": X.columns, "hr_frueh": np.exp(r1.params), "hr_frueh_se": r1.bse,
                       "hr_spaet": np.exp(r2.params), "hr_spaet_se": r2.bse})
    ph["verhaeltnis_spaet_frueh"] = ph["hr_spaet"] / ph["hr_frueh"]
    ph["t_split_min"] = tstar
    ph.to_csv(D.OUT / "cox_ph_frueh_spaet.csv", index=False)

    # stratifiziert nach Code (Stratum = Gruppe)
    tab, zt, *_ = run_cox(d, "strat_code", strata=d["gruppe"].to_numpy(), code_dummies=False)
    tab.to_csv(D.OUT / "cox_stratifiziert_code.csv", index=False); zt.to_csv(D.OUT / "cox_stratifiziert_code_ph.csv", index=False)

    # AFT
    aft_tabs = []
    X = design(d, ref_code, ref_station)
    for dist in ("lognormal", "weibull"):
        tb, meta = C.fit_aft(d["t"], d["e"], X, d["opday_nr"], dist=dist)
        tb["verteilung"] = dist; tb["loglik"] = meta["loglik"]; tb["aic"] = meta["aic"]; tb["konvergiert"] = meta["converged"]; tb["grad_max"] = meta["grad_max"]; tb["meldung"] = meta["message"]
        aft_tabs.append(tb)
        print("AFT", dist, "loglik", round(meta["loglik"], 1), "aic", round(meta["aic"], 1), meta["converged"], "grad_max", meta["grad_max"], meta["message"])
    pd.concat(aft_tabs).to_csv(D.OUT / "aft_koeffizienten.csv", index=False)

    # ---------------- Kipppunkt ----------------
    d["klasse"] = pd.cut(d["schwere_min"], SEV_BINS, labels=SEV_LABELS, right=False)
    sev_cap = d["schwere_min"].clip(upper=SEV_CAP)
    knots = np.percentile(sev_cap, [5, 35, 65, 95])
    print("Knoten (min):", knots)
    kn = dict(knoten=list(knots))
    spl = {}
    Xs = design(d, ref_code, ref_station, "spline", knots)
    mod_s, res_s = C.fit_cox(d["t"], d["e"], Xs, d["opday_nr"])
    Xl = design(d, ref_code, ref_station, "linear")
    mod_l, res_l = C.fit_cox(d["t"], d["e"], Xl, d["opday_nr"])
    nl_idx = [i for i, c in enumerate(Xs.columns) if c.startswith("schwere_s") and not c.endswith("s0") and "_x_" not in c]
    chi2, df_, p = C.wald_robust(res_s, nl_idx)
    llr = 2 * (mod_s.fit(disp=0).llf - mod_l.fit(disp=0).llf)
    from scipy import stats
    out = [dict(modell="Cox", test="Nichtlinearitaet Schwere (robuster Wald)", chi2=chi2, df=df_, p=p),
           dict(modell="Cox", test="Nichtlinearitaet Schwere (partieller LR, modellbasiert)", chi2=llr, df=len(nl_idx), p=float(stats.chi2.sf(llr, len(nl_idx))))]
    Xi = design(d, ref_code, ref_station, "spline", knots, interact=True)
    mod_i, res_i = C.fit_cox(d["t"], d["e"], Xi, d["opday_nr"])
    ia = [i for i, c in enumerate(Xi.columns) if "_x_hvz" in c]
    c2, dfi, pi = C.wald_robust(res_i, ia)
    out.append(dict(modell="Cox", test="Interaktion Schwere x HVZ (robuster Wald)", chi2=c2, df=dfi, p=pi))
    # AFT-Version (Zeitskala)
    tb_s, meta_s = C.fit_aft(d["t"], d["e"], Xs, d["opday_nr"], dist="lognormal")
    tb_l, meta_l = C.fit_aft(d["t"], d["e"], Xl, d["opday_nr"], dist="lognormal")
    names = meta_s["names"]; th = meta_s["theta"]; V = meta_s["V"]
    idx = [names.index(Xs.columns[i]) for i in nl_idx]
    b = th[idx]; Vi = V[np.ix_(idx, idx)]
    c3 = float(b @ np.linalg.solve(Vi, b))
    out.append(dict(modell="AFT log-normal", test="Nichtlinearitaet Schwere (robuster Wald)", chi2=c3, df=len(idx), p=float(stats.chi2.sf(c3, len(idx)))))
    lr3 = 2 * (meta_s["loglik"] - meta_l["loglik"])
    out.append(dict(modell="AFT log-normal", test="Nichtlinearitaet Schwere (LR, modellbasiert)", chi2=lr3, df=len(idx), p=float(stats.chi2.sf(lr3, len(idx)))))
    Xia = design(d, ref_code, ref_station, "spline", knots, interact=True)
    tb_i, meta_i = C.fit_aft(d["t"], d["e"], Xia, d["opday_nr"], dist="lognormal")
    ia2 = [meta_i["names"].index(c) for c in Xia.columns if "_x_hvz" in c]
    b = meta_i["theta"][ia2]; Vi = meta_i["V"][np.ix_(ia2, ia2)]
    c4 = float(b @ np.linalg.solve(Vi, b))
    out.append(dict(modell="AFT log-normal", test="Interaktion Schwere x HVZ (robuster Wald)", chi2=c4, df=len(ia2), p=float(stats.chi2.sf(c4, len(ia2)))))
    pd.DataFrame(out).to_csv(D.OUT / "kipppunkt_tests.csv", index=False)
    print(pd.DataFrame(out).to_string())

    # Partialeffekte (log HR bzw. log Zeitverhaeltnis) relativ zu Referenz 3 min, Cox + AFT, mit HVZ-Interaktion
    grid = np.linspace(max(0, knots[0]), min(60, knots[-1] * 1.0), 60)
    ref = 3.0
    def contrast(vals, hvz, cols, with_inter):
        Bv = C.rcs_basis(vals, knots) - C.rcs_basis(np.array([ref]), knots)
        rows = np.zeros((len(vals), len(cols)))
        for j in range(Bv.shape[1]):
            rows[:, cols.index(f"schwere_s{j}")] = Bv[:, j]
            if with_inter and hvz:
                rows[:, cols.index(f"schwere_s{j}_x_hvz")] = Bv[:, j]
        return rows
    pe = []
    for modell, beta, Vm, cols in (("Cox", res_i.params, res_i.cov_params(), list(Xi.columns)),
                                   ("AFT", meta_i["theta"][1:-1], meta_i["V"][1:-1, 1:-1], list(Xia.columns))):
        for hvz in (0, 1):
            Rm = contrast(grid, hvz, cols, True)
            est = Rm @ beta
            se = np.sqrt(np.einsum("ij,jk,ik->i", Rm, Vm, Rm))
            for g_, e_, s_ in zip(grid, est, se):
                pe.append(dict(modell=modell, hvz=hvz, schwere_min=g_, effekt=e_, lo=e_ - 1.96 * s_, hi=e_ + 1.96 * s_))
    pd.DataFrame(pe).to_csv(D.OUT / "kipppunkt_partialeffekt.csv", index=False)
    pd.Series(kn).to_json(D.OUT / "kipppunkt_knoten.json")

    # Klassen: Median je Klasse x HVZ, Ereignis und Placebo (P1, P2), Bootstrap
    days = np.array(sorted(ep["opday_nr"].unique()))
    Wd = K.make_day_weights(len(days), B, 11)
    rows = []
    boots = {}
    for pname in ("placebo", "placebo2"):
        pl = pd.read_parquet(D.ZW / f"{pname}.parquet")
        pl["day_nr"] = np.array(pl["parent_opday"], dtype="datetime64[D]").astype("int64")
        pl = pl.dropna(subset=["t_R15_180_st_std", "schwere_min"]).copy()
        pl["klasse"] = pd.cut(pl["schwere_min"], SEV_BINS, labels=SEV_LABELS, right=False)
        for who, src, dcol in (("ereignis", d, "opday_nr"), (pname, pl, "day_nr")):
            for hv in ("alle", 0, 1):
                for kl in SEV_LABELS:
                    m = (src["klasse"] == kl)
                    if hv != "alle":
                        m &= src["hvz"].astype(int) == hv
                    s = src[m]
                    if len(s) < 20:
                        continue
                    tcol, ecol = ("t", "e") if who == "ereignis" else ("t_R15_180_st_std", "e_R15_180_st_std")
                    bb = K.km_bootstrap(s[tcol], s[ecol], np.searchsorted(days, s[dcol]), Wd)
                    boots[(who, hv, kl)] = bb["q_boot"][:, 1]
                    r = dict(quelle=who, hvz=hv, klasse=kl, n=len(s), anteil_zensiert=1 - s[ecol].mean(),
                             schwere_median=float(s["schwere_min"].median()))
                    for k, nm in enumerate(("q25", "median", "q75")):
                        r[nm], r[nm + "_lo"], r[nm + "_hi"] = K.fmt_q(bb["q_point"][k], bb["q_boot"][:, k])
                    rows.append(r)
    kl = pd.DataFrame(rows).drop_duplicates(["quelle", "hvz", "klasse"])
    kl.to_csv(D.OUT / "kipppunkt_klassen_median.csv", index=False)

    # Steigungsverhaeltnis oben/unten (Median-Erholungszeit pro Minute Schwere), Bootstrap
    def slope_ratio(who):
        def med(kls, hv="alle"):
            return boots.get((who, hv, kls)), kl[(kl.quelle == who) & (kl.hvz == hv) & (kl.klasse == kls)]
        res_ = {}
        for hv in ("alle", 0, 1):
            try:
                (b1, r1), (b2, r2), (b3, r3), (b4, r4) = (med("3-5", hv), med("5-10", hv), med("10-20", hv), med(">20", hv))
                s_lo = (b2 - b1) / (r2["schwere_median"].iloc[0] - r1["schwere_median"].iloc[0])
                s_hi = (b4 - b3) / (r4["schwere_median"].iloc[0] - r3["schwere_median"].iloc[0])
                s_lo_pt = (float(r2["median"].iloc[0]) - float(r1["median"].iloc[0])) / (r2["schwere_median"].iloc[0] - r1["schwere_median"].iloc[0])
                s_hi_pt = (float(r4["median"].iloc[0]) - float(r3["median"].iloc[0])) / (r4["schwere_median"].iloc[0] - r3["schwere_median"].iloc[0])
                ok = np.isfinite(s_lo) & np.isfinite(s_hi) & (s_lo > 0)
                ratio = s_hi[ok] / s_lo[ok]
                res_[hv] = dict(quelle=who, hvz=hv, steigung_unten=s_lo_pt, steigung_oben=s_hi_pt, verhaeltnis_punkt=s_hi_pt / s_lo_pt if s_lo_pt else np.nan,
                                verh_lo=np.percentile(ratio, 2.5), verh_hi=np.percentile(ratio, 97.5), n_boot_gueltig=int(ok.sum()))
            except Exception as ex:  # Median nicht erreicht o.ae.
                res_[hv] = dict(quelle=who, hvz=hv, fehler=str(ex)[:80])
        return list(res_.values())
    sr = slope_ratio("ereignis") + slope_ratio("placebo") + slope_ratio("placebo2")
    pd.DataFrame(sr).to_csv(D.OUT / "kipppunkt_steigungsverhaeltnis.csv", index=False)
    print(pd.DataFrame(sr).to_string())


if __name__ == "__main__":
    main()

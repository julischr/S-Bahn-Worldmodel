"""Stufe 4, Schritt 2/3: Kalibrierung, Ein-Schritt-Pruefung auf Okt-Dez, SHAP.

Schreibt outputs/simulator/einschritt_*.csv, kalibrierung_*.csv, shap_*.csv, modelle/calib.pkl.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import shap
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import roc_auc_score

from .data import OUT
from .features import CLASS_HI, CLASS_LO, FEATURES, N_CLASS, load_dataset
from .model_io import Models, apply_cancel_cal, softmax_temp
from .train import MODELS, split

V = np.arange(CLASS_LO, CLASS_HI + 1)   # Klassenwerte -3 ... 15
RELI_EDGES_CANCEL = np.array([0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0001])
RELI_EDGES = np.array([0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1.0001])


def wilson(k, n, z=1.96):
    k = np.asarray(k, float); n = np.asarray(n, float)
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    den = 1 + z**2 / np.maximum(n, 1)
    c = (p + z**2 / (2 * np.maximum(n, 1))) / den
    h = z * np.sqrt(p * (1 - p) / np.maximum(n, 1) + z**2 / (4 * np.maximum(n, 1) ** 2)) / den
    return c - h, c + h


def logloss_cls(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))


def crps_cls(P, y):
    F = np.cumsum(P, axis=1)[:, :-1]                  # Schwellen -3..14
    ind = (np.arange(N_CLASS - 1)[None, :] >= y[:, None]).astype(float)
    return ((F - ind) ** 2).sum(axis=1)


def median_cls(P):
    return V[(np.cumsum(P, axis=1) >= 0.5).argmax(axis=1)]


def fit_temperature(P, y):
    f = lambda lt: logloss_cls(softmax_temp(P, float(np.exp(lt))), y)
    r = minimize_scalar(f, bounds=(-1, 1), method="bounded")
    return float(np.exp(r.x))


def fit_platt(p, y):
    z = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))

    def nll(ab):
        q = 1 / (1 + np.exp(-(ab[0] * z + ab[1])))
        q = np.clip(q, 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    r = minimize(nll, [1.0, 0.0], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-12, maxiter=2000))
    return dict(a=float(r.x[0]), b=float(r.x[1]))


def cancel_metrics(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return dict(n=len(y), n_cancel=int(y.sum()), rate=float(y.mean()), brier=float(np.mean((p - y) ** 2)),
                logloss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                auc=float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan)


def reliability(p, y, edges):
    b = np.digitize(p, edges) - 1
    rows = []
    for i in range(len(edges) - 1):
        m = b == i
        n = int(m.sum())
        if n == 0:
            continue
        k = int(y[m].sum())
        lo, hi = wilson(k, n)
        rows.append(dict(bin=f"[{edges[i]:g}, {min(edges[i+1], 1):g})", n=n, mean_pred=float(p[m].mean()),
                         obs_rate=k / n, obs_lo=float(lo), obs_hi=float(hi)))
    t = pd.DataFrame(rows)
    ece = float((t["n"] * (t["mean_pred"] - t["obs_rate"]).abs()).sum() / t["n"].sum())
    return t, ece


def main():
    OUT.mkdir(exist_ok=True)
    d = load_dataset()
    parts = split(d)
    M = Models(calibrated=False)
    res = {}

    # ---------- Vorhersagen ----------
    for name in ("train", "val", "cal", "test"):
        p = parts[name]
        X = p[FEATURES].to_numpy(dtype=np.float32)
        p = p.assign(pc_raw=M.p_cancel_raw(X))
        parts[name] = p
        if name in ("cal", "test"):
            pass
    def class_probs(p):
        m = p["cancel_next"].to_numpy() == 0
        X = p.loc[m, FEATURES].to_numpy(dtype=np.float32)
        return m, M.p_class_raw(X)

    # ---------- Kalibrierung (Plan Abschnitt 3): Fit auf Sep, Entscheidung per Halbierung nach Tag ----------
    cal = parts["cal"]
    m_cal, P_cal = class_probs(cal)
    ycal = cal.loc[m_cal, "dclass"].to_numpy()
    day = pd.to_datetime(cal["opday"]).dt.day.to_numpy()
    odd = day % 2 == 1
    odd_c = odd[m_cal]
    # Klassenmodell: Temperatur
    T_odd = fit_temperature(P_cal[odd_c], ycal[odd_c])
    ll_raw = logloss_cls(P_cal[~odd_c], ycal[~odd_c])
    ll_cal = logloss_cls(softmax_temp(P_cal[~odd_c], T_odd), ycal[~odd_c])
    T_all = fit_temperature(P_cal, ycal)
    use_T = ll_cal < ll_raw - 1e-5
    # Ausfallmodell: Platt
    yc = cal["cancel_next"].to_numpy().astype(float)
    pc = cal["pc_raw"].to_numpy()
    ab_odd = fit_platt(pc[odd], yc[odd])
    br_raw = cancel_metrics(pc[~odd], yc[~odd])
    br_cal = cancel_metrics(apply_cancel_cal(ab_odd, pc[~odd]), yc[~odd])
    ab_all = fit_platt(pc, yc)
    use_platt = br_cal["logloss"] < br_raw["logloss"] - 1e-6 and br_cal["brier"] <= br_raw["brier"]
    calib = dict(cancel=ab_all if use_platt else None, temp=T_all if use_T else 1.0)
    pickle.dump(calib, open(MODELS / "calib.pkl", "wb"))
    cal_info = dict(
        temp_odd=T_odd, temp_all=T_all, class_logloss_even_raw=ll_raw, class_logloss_even_cal=ll_cal, class_use_temp=bool(use_T),
        platt_odd=ab_odd, platt_all=ab_all, cancel_even_raw=br_raw, cancel_even_cal=br_cal, cancel_use_platt=bool(use_platt),
        n_cal_class=int(len(ycal)), n_cal_cancel=int(yc.sum()))
    json.dump(cal_info, open(OUT / "kalibrierung_entscheidung.json", "w"), indent=1)
    print(json.dumps(cal_info, indent=1))

    # ---------- Baselines (Trainingsdaten Jan-Jul) ----------
    tr = parts["train"]
    trm = tr[tr["cancel_next"] == 0]
    Pglob = np.bincount(trm["dclass"], minlength=N_CLASS) / len(trm)
    rglob = tr["cancel_next"].mean()
    key = lambda df: (df["rank"].astype(int) * 2 + df["richtung"].astype(int)) * 24 + np.floor(df["hour"]).astype(int)
    trm_key = key(trm).to_numpy()
    cnt = np.zeros((48 * 12 // 2 * 2 + 1, N_CLASS))
    nk = 12 * 2 * 24
    cnt = np.zeros((nk, N_CLASS))
    np.add.at(cnt, (trm_key, trm["dclass"].to_numpy()), 1)
    ALPHA = 5.0
    Pgrp = (cnt + ALPHA * Pglob[None, :]) / (cnt.sum(1, keepdims=True) + ALPHA)
    mean_grp = Pgrp @ V
    med_grp = V[(np.cumsum(Pgrp, axis=1) >= 0.5).argmax(axis=1)]
    tr_key = key(tr).to_numpy()
    nc = np.bincount(tr_key, minlength=nk)
    kc = np.bincount(tr_key, weights=tr["cancel_next"].to_numpy(), minlength=nk)
    rgrp = (kc + ALPHA * rglob) / (nc + ALPHA)

    # ---------- Test ----------
    out = {}
    for label, use_cal in (("roh", False), ("kalibriert", True)):
        te = parts["test"]
        m_te = te["cancel_next"].to_numpy() == 0
        X = te.loc[m_te, FEATURES].to_numpy(dtype=np.float32)
        P_raw = M.p_class_raw(X)
        P = softmax_temp(P_raw, calib["temp"]) if use_cal else P_raw
        pcv = te["pc_raw"].to_numpy()
        pcv = apply_cancel_cal(calib["cancel"], pcv) if (use_cal and calib["cancel"] is not None) else pcv
        tt = te.loc[m_te]
        y = tt["dclass"].to_numpy()
        dk = tt["d_k"].to_numpy()
        dn = tt["d_next"].to_numpy()
        kk = key(tt).to_numpy()
        rows = []
        for gname, mask in (("gesamt", np.ones(len(tt), bool)), ("Ost", tt["richtung"].to_numpy() == 0),
                            ("West", tt["richtung"].to_numpy() == 1)):
            ym, dkm, dnm = y[mask], dk[mask], dn[mask]
            Pm = P[mask]; kkm = kk[mask]
            def row(modell, point, P_dist):
                r = dict(gruppe=gname, modell=modell, n=int(mask.sum()),
                         mae=float(np.mean(np.abs(dnm - point))), rmse=float(np.sqrt(np.mean((dnm - point) ** 2))),
                         bias=float(np.mean(point - dnm)))
                if P_dist is not None:
                    r["crps"] = float(crps_cls(P_dist, ym).mean())
                    r["logloss"] = logloss_cls(P_dist, ym)
                return r
            rows.append(row("LightGBM (Median)", dkm + median_cls(Pm), Pm))
            rows.append(row("LightGBM (Mittel)", dkm + Pm @ V, Pm))
            r1 = row("V1 Verspaetung bleibt (Delta=0)", dkm, None)
            F0 = np.tile((V >= 0).astype(float), (len(ym), 1))
            r1["crps"] = float(np.abs(np.clip(dnm - dkm, CLASS_LO, CLASS_HI)).mean())
            r1["logloss"] = np.nan
            rows.append(r1)
            rows.append(row("V1b Persistenz + globale Delta-Verteilung", dkm + V[(np.cumsum(Pglob) >= .5).argmax()], np.tile(Pglob, (len(ym), 1))))
            rows.append(row("V2 Gruppenverteilung (Median)", dkm + med_grp[kkm], Pgrp[kkm]))
            rows.append(row("V2 Gruppenmittel (Punkt)", dkm + mean_grp[kkm], Pgrp[kkm]))
        t_cls = pd.DataFrame(rows)
        t_cls.to_csv(OUT / f"einschritt_verspaetung_{label}.csv", index=False)
        # Ausfall (alle Test-Uebergaenge)
        yc_te = te["cancel_next"].to_numpy().astype(float)
        kk_all = key(te).to_numpy()
        rows = []
        for gname, mask in (("gesamt", np.ones(len(te), bool)), ("Ost", te["richtung"].to_numpy() == 0),
                            ("West", te["richtung"].to_numpy() == 1)):
            for modell, p in (("LightGBM", pcv), ("V1 globale Rate", np.full(len(te), rglob)), ("V2 Rate je Station/Richtung/Stunde", rgrp[kk_all])):
                r = cancel_metrics(p[mask], yc_te[mask]); r.update(gruppe=gname, modell=modell)
                rows.append(r)
        t_can = pd.DataFrame(rows)
        t_can["brier_skill_vs_V2"] = np.nan
        for g in t_can["gruppe"].unique():
            b2 = t_can[(t_can.gruppe == g) & t_can.modell.str.startswith("V2")]["brier"].iloc[0]
            m = t_can.gruppe == g
            t_can.loc[m, "brier_skill_vs_V2"] = 1 - t_can.loc[m, "brier"] / b2
        t_can.to_csv(OUT / f"einschritt_ausfall_{label}.csv", index=False)
        out[label] = dict(cls=t_cls, can=t_can)
        print(f"\n=== Test ({label}) Verspaetung ===\n", t_cls.to_string())
        print(f"\n=== Test ({label}) Ausfall ===\n", t_can.to_string())

        # Reliability
        rel_c, ece_c = reliability(pcv, yc_te, RELI_EDGES_CANCEL)
        rel_c.to_csv(OUT / f"reliability_ausfall_{label}.csv", index=False)
        rels = []
        eces = {}
        cum_events = {"Delta<=0": (np.arange(N_CLASS) <= 3), "Delta>=+2": (np.arange(N_CLASS) >= 5), "Delta>=+5": (np.arange(N_CLASS) >= 8)}
        for en, sel in cum_events.items():
            pe = P[:, sel].sum(1)
            ye = sel[y].astype(float)
            t, ece = reliability(pe, ye, RELI_EDGES)
            t.insert(0, "ereignis", en)
            rels.append(t); eces[en] = ece
        pd.concat(rels).to_csv(OUT / f"reliability_zuwachs_{label}.csv", index=False)
        # je Klasse: Mittel
        pcls = pd.DataFrame(dict(klasse=[f"{v:+d}" + ("(<=)" if v == CLASS_LO else "(>=)" if v == CLASS_HI else "") for v in V],
                                 pred_mittel=P.mean(0), beobachtet=np.bincount(y, minlength=N_CLASS) / len(y),
                                 n_beobachtet=np.bincount(y, minlength=N_CLASS)))
        pcls.to_csv(OUT / f"reliability_klassen_{label}.csv", index=False)
        res[label] = dict(ece_cancel=ece_c, ece_zuwachs=eces)
        print(label, res[label])

    # ---------- SHAP ----------
    rng = np.random.default_rng(3)
    te = parts["test"]
    sel = rng.choice(len(te), 40000, replace=False)
    Xc = te.iloc[sel][FEATURES].astype(float)
    sv = shap.TreeExplainer(M.cancel).shap_values(Xc)
    sv = sv[1] if isinstance(sv, list) else sv
    shc = pd.DataFrame(dict(merkmal=FEATURES, mean_abs_shap=np.abs(sv).mean(0))).sort_values("mean_abs_shap", ascending=False)
    shc.to_csv(OUT / "shap_ausfall.csv", index=False)
    tm = te[te["cancel_next"] == 0]
    sel2 = rng.choice(len(tm), 20000, replace=False)
    Xk = tm.iloc[sel2][FEATURES].astype(float)
    sv = shap.TreeExplainer(M.klass).shap_values(Xk)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else sv     # (n, F, C)
    a = np.abs(sv).mean(0)                                          # (F, C)
    shk = pd.DataFrame(dict(merkmal=FEATURES, mean_abs_shap_summe_klassen=a.sum(1), mean_abs_shap_klasse_Delta0=a[:, 3],
                            mean_abs_shap_klasse_Delta_plus2=a[:, 5])).sort_values("mean_abs_shap_summe_klassen", ascending=False)
    shk.to_csv(OUT / "shap_zuwachs.csv", index=False)
    print(shc.head(15).to_string()); print(shk.head(15).to_string())

    # Rohvorhersagen fuer spaetere Auswertung ablegen
    json.dump(res, open(OUT / "einschritt_ece.json", "w"), indent=1)


if __name__ == "__main__":
    main()

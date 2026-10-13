"""Stufe 4, Schritt 5: Pruefsteine auf simulierten Testtagen.

Aufruf: uv run python -m src.simulator.eval_sim --days 30 --runs 100 --workers 5
Schreibt outputs/simulator/pruefstein_*.csv und zwischen/sim_tage/*.npz (erste Laeufe je Tag fuer (e)).
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .data import EXCL, OUT, ZW, load_trains, weekday_type
from .metrics import (DL_LABELS, HW_LABELS, NCELL, overall, plausibility, stone_a_b, stone_d)
from .model_io import Models
from .sim import Day, simulate

SEED_DAYS = 4
RUN_SEED = 11
N_SAVE = 20           # Laeufe je Tag, die fuer (e) / Bildfahrplan gespeichert werden
SAVE_DIR = ZW / "sim_tage"


def pick_days(trains: pd.DataFrame, n: int, seed: int = SEED_DAYS) -> list:
    days = sorted(d for d in trains["opday"].unique() if d.month >= 10 and d not in EXCL)
    wt = {d: weekday_type(d) for d in days}
    rng = np.random.default_rng(seed)
    out = []
    types = sorted(set(wt.values()))
    shares = {t: sum(1 for d in days if wt[d] == t) / len(days) for t in types}
    alloc = {t: max(2, int(round(n * shares[t]))) for t in types}
    while sum(alloc.values()) > n:
        t = max(alloc, key=alloc.get); alloc[t] -= 1
    while sum(alloc.values()) < n:
        t = max(shares, key=lambda x: shares[x] - alloc[x] / n); alloc[t] += 1
    for t in types:
        cand = [d for d in days if wt[d] == t]
        out += list(rng.choice(np.array(cand, dtype=object), size=min(alloc[t], len(cand)), replace=False))
    return sorted(out)


_G = {}


def _init(threads):
    _G["trains"] = load_trains()
    _G["models"] = Models(num_threads=threads)


def day_job(args):
    opday, R, save = args
    t0 = time.time()
    trains, M = _G["trains"], _G["models"]
    day = Day(trains[trains["opday"] == opday], opday)
    real = day.real_result()
    res = simulate(day, M, R=R, seed=RUN_SEED)
    out = dict(opday=opday, secs=time.time() - t0, meta={k: v for k, v in res.meta.items() if k != "cancm0"},
               N=day.N, a_b=stone_a_b(res, real), d=stone_d(res, real),
               plaus_sim=plausibility(res), plaus_real=plausibility(real),
               ov_sim=overall(res), ov_real=overall(real))
    if save:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(SAVE_DIR / f"{opday}.npz", arr_t=res.arr_t[:N_SAVE], delay=res.delay[:N_SAVE],
                            canc=res.canc[:N_SAVE])
    return out


def boot_ci(per_day: list[tuple[np.ndarray, np.ndarray]], fn, B=1000, seed=1):
    """per_day: Liste (Summen, Zaehler) je Tag; fn(sum_total, n_total) -> Kennzahl."""
    rng = np.random.default_rng(seed)
    S = np.array([p[0] for p in per_day]); Nn = np.array([p[1] for p in per_day])
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(per_day), len(per_day))
        vals.append(fn(S[i].sum(0), Nn[i].sum(0)))
    vals = np.array(vals)
    return np.nanpercentile(vals, 2.5, axis=0), np.nanpercentile(vals, 97.5, axis=0)


def aggregate(results: list[dict], tag: str = "basis") -> dict:
    verdict = {}
    # ---------------- (a) ----------------
    A = []
    for r in results:
        if "a" in r["a_b"]:
            a = pd.DataFrame(r["a_b"]["a"]); a["opday"] = r["opday"]; A.append(a)
    A = pd.concat(A, ignore_index=True)
    A = A[np.isfinite(A["sim_q50"])]
    A["in90"] = (A["real_end"] >= A["sim_q05"]) & (A["real_end"] <= A["sim_q95"])
    A["in50"] = (A["real_end"] >= A["sim_q25"]) & (A["real_end"] <= A["sim_q75"])
    A["err"] = A["sim_q50"] - A["real_end"]
    A["abs_sim"] = A["err"].abs()
    A["abs_pers"] = (A["real_entry"] - A["real_end"]).abs()
    A["err_mean"] = A["sim_mean"] - A["real_end"]
    A.to_parquet(OUT / f"pruefstein_a_rohdaten_{tag}.parquet")
    days = sorted(A["opday"].unique())
    g = A.groupby("opday").agg(n=("in90", "size"), s90=("in90", "sum"), s50=("in50", "sum"), e=("err", "sum"),
                              asim=("abs_sim", "sum"), apers=("abs_pers", "sum"), em=("err_mean", "sum"))
    rows = []
    for name, sub in (("alle", A), ("Ost", A[A.rich == 0]), ("West", A[A.rich == 1]), ("HVZ", A[A.hvz == 1]), ("Nicht-HVZ", A[A.hvz == 0])):
        gg = sub.groupby("opday").agg(n=("in90", "size"), s90=("in90", "sum"), s50=("in50", "sum"), e=("err", "sum"),
                                      asim=("abs_sim", "sum"), apers=("abs_pers", "sum"), em=("err_mean", "sum"))
        per = [(gg.loc[d, ["s90", "s50", "e", "asim", "apers", "em"]].to_numpy(float), np.array([gg.loc[d, "n"]], float)) for d in gg.index]
        tot = gg.sum()
        f = lambda s, n: s / n
        lo, hi = boot_ci(per, f)
        for i, k in enumerate(["abdeckung_90", "abdeckung_50", "bias_median_min", "mae_sim_median_min", "mae_persistenz_min", "bias_mittel_min"]):
            v = tot[["s90", "s50", "e", "asim", "apers", "em"]].iloc[i] / tot["n"]
            rows.append(dict(gruppe=name, kennzahl=k, wert=float(v), ki_lo=float(lo[i]), ki_hi=float(hi[i]), n_zuege=int(tot["n"]), n_tage=len(gg)))
    ta = pd.DataFrame(rows)
    ta.to_csv(OUT / f"pruefstein_a_{tag}.csv", index=False)
    al = ta[ta.gruppe == "alle"].set_index("kennzahl")["wert"]
    verdict["a1_abdeckung"] = (bool(0.85 <= al["abdeckung_90"] <= 0.95), float(al["abdeckung_90"]))
    verdict["a2_bias"] = (bool(abs(al["bias_median_min"]) <= 0.5), float(al["bias_median_min"]))
    verdict["a3_mae"] = (bool(al["mae_sim_median_min"] <= al["mae_persistenz_min"]), float(al["mae_sim_median_min"] - al["mae_persistenz_min"]))
    verdict["a"] = all(v[0] for k, v in verdict.items() if k.startswith("a"))

    # ---------------- (b) ----------------
    B = []
    for r in results:
        if "b" in r["a_b"]:
            b = pd.DataFrame(r["a_b"]["b"]); b["opday"] = r["opday"]; B.append(b)
    B = pd.concat(B, ignore_index=True)
    B = B[np.isfinite(B["sim_build_mean"])]
    B.to_parquet(OUT / f"pruefstein_b_rohdaten_{tag}.parquet")
    rows = []
    for name, sub in (("alle", B), ("Ost", B[B.rich == 0]), ("West", B[B.rich == 1]), ("HVZ", B[B.hvz == 1]), ("Nicht-HVZ", B[B.hvz == 0]),
                      ("Eintritt <=2 min", B[B.entry_delay <= 2]), ("Eintritt >10 min", B[B.entry_delay > 10])):
        gg = sub.groupby("opday").agg(n=("real_build", "size"), sr=("real_build", "sum"), ss=("sim_build_mean", "sum"))
        per = [(gg.loc[d, ["sr", "ss"]].to_numpy(float), np.array([gg.loc[d, "n"]], float)) for d in gg.index]
        tot = gg.sum()
        lo, hi = boot_ci(per, lambda s, n: np.array([s[0] / n[0], s[1] / n[0], (s[1] - s[0]) / n[0]]))
        rows.append(dict(gruppe=name, n_zuege=int(tot["n"]), echt_mittel_min=tot["sr"] / tot["n"], echt_lo=lo[0], echt_hi=hi[0],
                         sim_mittel_min=tot["ss"] / tot["n"], sim_lo=lo[1], sim_hi=hi[1],
                         diff_sim_minus_echt=(tot["ss"] - tot["sr"]) / tot["n"], diff_lo=lo[2], diff_hi=hi[2]))
    tb = pd.DataFrame(rows)
    tb.to_csv(OUT / f"pruefstein_b_{tag}.csv", index=False)
    d_all = tb[tb.gruppe == "alle"].iloc[0]
    verdict["b"] = (bool(abs(d_all.diff_sim_minus_echt) <= 0.5), float(d_all.diff_sim_minus_echt))

    # ---------------- (c) ----------------
    cr = [r["a_b"]["c"] for r in results if "c" in r["a_b"]]
    r_sum = np.array([c["r_sum"] for c in cr]); r_n = np.array([c["r_n"] for c in cr])
    s_sum = np.array([c["s_sum"] for c in cr]); s_n = np.array([c["s_n"] for c in cr])
    R_ = s_sum.shape[1]
    rm = r_sum.sum(0) / np.maximum(r_n.sum(0), 1)
    lo, hi = boot_ci(list(zip(r_sum, r_n)), lambda s, n: s / np.maximum(n, 1))
    sm_runs = s_sum.sum(0) / np.maximum(s_n.sum(0), 1)         # (R, 20)
    rows = []
    for c in range(NCELL):
        rows.append(dict(abstand=HW_LABELS[c // 4], vorgaenger=DL_LABELS[c % 4], n_echt=int(r_n.sum(0)[c]),
                         echt_mittel=float(rm[c]) if r_n.sum(0)[c] > 0 else np.nan, echt_lo=float(lo[c]), echt_hi=float(hi[c]),
                         n_sim_pro_lauf=float(s_n.sum(0)[:, c].mean()), sim_mittel=float(np.nanmean(sm_runs[:, c])),
                         sim_lauf_q025=float(np.nanpercentile(sm_runs[:, c], 2.5)), sim_lauf_q975=float(np.nanpercentile(sm_runs[:, c], 97.5))))
    tc = pd.DataFrame(rows)
    tc["diff"] = tc["sim_mittel"] - tc["echt_mittel"]
    q = tc["n_echt"] >= 200
    tc["zelle_ok"] = np.where(q, tc["diff"].abs() <= 0.3, np.nan)
    tc.to_csv(OUT / f"pruefstein_c_{tag}.csv", index=False)
    frac = float(tc.loc[q, "zelle_ok"].mean())
    grad_ok = []
    for h in HW_LABELS:
        t = tc[tc.abstand == h].set_index("vorgaenger")
        if t.loc["<=2min", "n_echt"] >= 200 and t.loc[">10min", "n_echt"] >= 200:
            grad_ok.append((h, bool((t.loc[">10min", "sim_mittel"] > t.loc["<=2min", "sim_mittel"]) == (t.loc[">10min", "echt_mittel"] > t.loc["<=2min", "echt_mittel"]))))
    verdict["c"] = (bool(frac >= 0.75 and all(x[1] for x in grad_ok)), dict(anteil_zellen_ok=frac, n_zellen=int(q.sum()), gradient=grad_ok))

    # ---------------- (d) ----------------
    dr = [r["d"] for r in results]
    rn = np.array([x["r_n"] for x in dr]); rc = np.array([x["r_c"] for x in dr])
    sn = np.array([x["s_n"] for x in dr]); sc = np.array([x["s_c"] for x in dr])
    rr_ = rc.sum(0) / np.maximum(rn.sum(0), 1)
    lo, hi = boot_ci(list(zip(rc, rn)), lambda s, n: s / np.maximum(n, 1))
    sr_runs = sc.sum(0) / np.maximum(sn.sum(0), 1)
    rows = []
    for c in range(NCELL):
        rows.append(dict(abstand=HW_LABELS[c // 4], vorgaenger=DL_LABELS[c % 4], n_echt=int(rn.sum(0)[c]), ausfaelle_echt=int(rc.sum(0)[c]),
                         echt_rate=float(rr_[c]), echt_lo=float(lo[c]), echt_hi=float(hi[c]),
                         n_sim_pro_lauf=float(sn.sum(0)[:, c].mean()), sim_rate=float(np.mean(sr_runs[:, c])),
                         sim_lauf_q025=float(np.percentile(sr_runs[:, c], 2.5)), sim_lauf_q975=float(np.percentile(sr_runs[:, c], 97.5))))
    td = pd.DataFrame(rows)
    td["diff_pp"] = (td["sim_rate"] - td["echt_rate"]) * 100
    q = td["n_echt"] >= 500
    tol = np.maximum(0.003, 0.4 * td["echt_rate"])
    td["zelle_ok"] = np.where(q, (td["sim_rate"] - td["echt_rate"]).abs() <= tol, np.nan)
    td.to_csv(OUT / f"pruefstein_d_{tag}.csv", index=False)
    frac = float(td.loc[q, "zelle_ok"].mean())
    row0 = td[td.abstand == "<5min"].set_index("vorgaenger")
    mono_r = bool(row0.loc[DL_LABELS, "echt_rate"].is_monotonic_increasing)
    mono_s = bool(row0.loc[DL_LABELS, "sim_rate"].is_monotonic_increasing)
    verdict["d"] = (bool(frac >= 0.75 and mono_s == mono_r and mono_s), dict(anteil_zellen_ok=frac, n_zellen=int(q.sum()), monoton_echt=mono_r, monoton_sim=mono_s))

    # ---------------- Zusatz: Plausibilitaet, Profile, Ausfaelle ----------------
    pr = {}
    for who in ("sim", "real"):
        ps = [r["plaus_" + who] for r in results]
        inv = sum(p["inv"].sum() for p in ps); opp = sum(p["inv_opp"].sum() for p in ps)
        same = sum(p["same"].sum() for p in ps); near = sum(p["near"].sum() for p in ps); gt = sum(p["gap_tot"].sum() for p in ps)
        pr[who] = dict(ueberholungen=float(inv), paare=float(opp), ueberholquote=float(inv / opp), gleiche_minute=float(same),
                       gleiche_minute_quote=float(same / gt), abstand_hoechstens_1min_quote=float(near / gt), n_abstaende=float(gt))
    pd.DataFrame(pr).T.to_csv(OUT / f"plausibilitaet_{tag}.csv")
    ov = {}
    for who in ("sim", "real"):
        o = [r["ov_" + who] for r in results]
        ps = sum(x["prof_sum"].sum(0) for x in o); pn = sum(x["prof_n"].sum(0) for x in o)
        nm = sum(x["n_mod"] for x in o); nc = sum(x["n_canc"].sum() for x in o)
        ov[who] = dict(profil=(ps / np.maximum(pn, 1)).tolist(), n_mod=nm, ausfallquote=float(nc / (nm * (len(o[0]["n_canc"])))))
    prof = pd.DataFrame({"station_rang": np.repeat(np.arange(12), 2), "richtung": ["Ost", "West"] * 12,
                         "echt_mittel_verspaetung": ov["real"]["profil"], "sim_mittel_verspaetung": ov["sim"]["profil"]})
    prof.to_csv(OUT / f"profil_verspaetung_{tag}.csv", index=False)
    meta = dict(n_clamp=sum(r["meta"]["n_clamp"] for r in results), n_hold=sum(r["meta"]["n_hold"] for r in results),
                ausfallquote_modelliert_echt=ov["real"]["ausfallquote"], ausfallquote_modelliert_sim=ov["sim"]["ausfallquote"],
                n_positionen_modelliert=sum(r["ov_real"]["n_mod"] for r in results),
                tage=[str(r["opday"]) for r in results], plausibilitaet=pr)
    json.dump(meta, open(OUT / f"simlauf_meta_{tag}.json", "w"), indent=1, default=str)
    verdict["meta"] = meta
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tag", default="basis")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    trains = load_trains()
    days = pick_days(trains, args.days)
    if args.only:
        days = [pd.Timestamp(args.only).date()]
    print(len(days), "Tage:", [str(d) for d in days], flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.threads,)) as ex:
        results = []
        for i, r in enumerate(ex.map(day_job, [(d, args.runs, True) for d in days])):
            results.append(r)
            print(f"{i+1}/{len(days)} {r['opday']} {r['secs']:.0f}s N={r['N']} clamp={r['meta']['n_clamp']}", flush=True)
    print("Laufzeit", time.time() - t0)
    import pickle
    pickle.dump(results, open(ZW / f"eval_results_{args.tag}.pkl", "wb"))
    v = aggregate(results, args.tag)
    print(json.dumps({k: v[k] for k in v if k != "meta"}, indent=1, default=str))


if __name__ == "__main__":
    main()

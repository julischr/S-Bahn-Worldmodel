"""Stufe 4, Schritt 6: Szenarien a) Verspaetung einspeisen, b) Fahrt streichen, c) Sperrung (experimentell).

Aufruf: uv run python -m src.simulator.run_scenarios --runs 200 --k 3 --workers 4
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .data import OUT, ZW, load_trains
from .model_io import Models
from .scenarios import effects, find_gaps, pick_trains
from .sim import NS_MIN, Day, Scenario, simulate

DAYS = ["2025-11-12", "2025-12-03"]       # Mittwoche im Testzeitraum
NS_ = [5, 10, 20]
_G = {}


def _init(threads):
    _G["trains"] = load_trains()
    _G["models"] = Models(num_threads=threads)


def group_job(args):
    opday, hvz, R, k, seed = args
    trains, M = _G["trains"], _G["models"]
    opday = pd.Timestamp(opday).date()
    day = Day(trains[trains["opday"] == opday], opday)
    rng = np.random.default_rng([seed, int(hvz), opday.toordinal()])
    zs = pick_trains(day, hvz, k, rng)
    if not zs:
        return dict(opday=opday, hvz=hvz, rows=[], curves=[], zs=[])
    idxs = [day.zug_index[z] for z in zs]
    t_inj = [int(day.real_min[n, int(day.entry_pos[n])]) for n in idxs]
    t_start, t_end = max(min(t_inj) - 60, 0), min(max(t_inj) + 360, day.M - 1)
    t00 = time.time()
    base = simulate(day, M, R=R, seed=seed, scenario=None, t_start=t_start, t_end=t_end)
    print(f'Basis {opday} hvz={hvz} {time.time()-t00:.0f}s Fenster {t_end-t_start} min', flush=True)
    rows, curves, examples = [], [], {}
    for z, n, ti in zip(zs, idxs, t_inj):
        e = int(day.entry_pos[n])
        for art, mins in [("verspaetung", N) for N in NS_] + [("ausfall", 0)]:
            sc = Scenario(kind="delay" if art == "verspaetung" else "cancel", zug=int(z), delay_abs_min=float(mins))
            scen = simulate(day, M, R=R, seed=seed, scenario=sc, t_start=t_start, t_end=t_end)
            lab = dict(art=art, minuten=mins, hvz=hvz, opday=str(opday), zug=int(z), linie=int(day.line_at[n, e]),
                       eintritt_uhrzeit=float(day.hour_of_min[ti]), eintritt_basis_verspaetung=float(day.delay_real[n, e]))
            eff, cur = effects(base, scen, ti, t_end, n, lab)
            print(f'{time.strftime("%H:%M:%S")} {opday} hvz={hvz} {art} {mins} fertig', flush=True)
            rows.append(eff); curves.append(cur)
            if art == "verspaetung" and mins == 10 and len(examples) == 0:
                examples["ex"] = (scen.arr_t[:30].copy(), scen.delay[:30].copy(), scen.canc[:30].copy(), base.arr_t[:30].copy(),
                                  base.delay[:30].copy(), base.canc[:30].copy(), int(z), ti)
    out = dict(opday=opday, hvz=hvz, rows=rows, curves=curves, zs=zs, secs=0, example=examples.get("ex"))
    return out


def closure_job(args):
    case, R, seed = args
    trains, M = _G["trains"], _G["models"]
    opday = pd.Timestamp(case["opday"]).date()
    day = Day(trains[trains["opday"] == opday], opday)
    rich = 0 if case["richtung"] == "Ost" else 1
    t1, t2 = int(case["t1_m"]), int(case["t2_m"])
    t_start, t_end = max(t1 - day.m0 - 60, 0), min(t2 - day.m0 + 150, day.M - 1)
    sc = Scenario(kind="closure", rank=int(case["rank"]), rich=rich, t1=t1, t2=t2)
    base = simulate(day, M, R=R, seed=seed, scenario=None, t_start=t_start, t_end=t_end)
    scen = simulate(day, M, R=R, seed=seed, scenario=sc, t_start=t_start, t_end=t_end)
    real = day.real_result()
    rk = int(case["rank"])
    nxt = rk + (1 if rich == 0 else -1)
    out = dict(case)
    for name, R_ in (("echt", real), ("basis", base), ("sperrung", scen)):
        for lab, r_ in (("an_s", rk), ("naechste", nxt)):
            if not 0 <= r_ <= 11:
                continue
            ns_ = np.flatnonzero((day.rich == rich) & (day.pos_of_rank[:, r_] >= 0))
            pa = day.pos_of_rank[ns_, r_].astype(int)
            arr = R_.arr_t[:, ns_, pa]; dl = R_.delay[:, ns_, pa]
            ok = (arr < 10**7) & np.isfinite(dl) & ~R_.canc[:, ns_, pa]
            win = ok & (arr + day.m0 >= t2) & (arr + day.m0 < t2 + 60)
            cnt = win.sum(1)
            mean_d = np.where(cnt > 0, np.where(win, dl, 0).sum(1) / np.maximum(cnt, 1), np.nan)
            first = np.where(win, arr + day.m0, 10**9).min(1)
            out[f"{name}_{lab}_zuege_60min_nach_t2"] = float(cnt.mean())
            out[f"{name}_{lab}_mean_verspaetung_60min"] = float(np.nanmean(mean_d)) if np.isfinite(mean_d).any() else np.nan
            out[f"{name}_{lab}_erste_ankunft_nach_t2_min"] = float(np.mean(first[first < 10**9] - t2)) if (first < 10**9).any() else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--teil", default="ab", help="a, b (Verspaetung/Ausfall zusammen), c (Sperrung)")
    ap.add_argument("--n-sperr", type=int, default=8)
    ap.add_argument("--tag", default="szenarien")
    a = ap.parse_args()
    t0 = time.time()
    if "a" in a.teil or "b" in a.teil:
        jobs = [(d, h, a.runs, a.k, a.seed) for d in DAYS for h in (True, False)]
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.threads,)) as ex:
            res = list(ex.map(group_job, jobs))
        rows = [r for g in res for r in g["rows"]]
        curves = pd.concat([c for g in res for c in g["curves"]], ignore_index=True)
        pd.DataFrame(rows).to_csv(OUT / f"szenario_einzel_{a.tag}.csv", index=False)
        curves.to_csv(OUT / f"szenario_kurven_{a.tag}.csv", index=False)
        pickle.dump([g.get("example") for g in res], open(ZW / f"szenario_beispiele_{a.tag}.pkl", "wb"))
        print("Szenarien a/b fertig", time.time() - t0, len(rows), flush=True)
    if "c" in a.teil:
        tr = load_trains()
        g = find_gaps(tr, 15)
        g["monat"] = pd.to_datetime(g.opday).dt.month
        g = g[(g.monat >= 10) & (g.zuege_vorstation_in_luecke >= 8) & (g.luecke_min >= 20)]
        g = g.sort_values("luecke_min", ascending=False)
        g = g.drop_duplicates(["opday", "rank"]).head(a.n_sperr)
        g.to_csv(OUT / "sperrung_faelle.csv", index=False)
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.threads,)) as ex:
            out = list(ex.map(closure_job, [(r, min(a.runs, 100), a.seed) for r in g.to_dict("records")]))
        pd.DataFrame(out).to_csv(OUT / f"sperrung_test_{a.tag}.csv", index=False)
        print("Sperrung fertig", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()

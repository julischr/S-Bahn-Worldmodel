"""Stufe 4, Pruefstein (e): Erholungszeit je Schwereklasse auf simulierten Tagen.

Dieselbe Messung wie Stufe 3 (`src.analysis.survival.recovery.measure`, R15, Schwelle 180 s,
Meldestation, Standardzensierung), angewendet auf (i) die echten Testtage, (ii) jeden
gespeicherten simulierten Lauf. Episoden (Zeitpunkt, Meldestation, Richtung) kommen aus den echten
Stoerungsmeldungen der Survival-Stufe (nur gelesen).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.analysis.survival import data as SD
from src.analysis.survival import km as K
from src.analysis.survival import recovery as R
from src.analysis.var.stations import STATION_ORDER_WEST_OST

from .data import OUT, ZW, load_trains
from .eval_sim import N_SAVE, SAVE_DIR
from .sim import INF, Day

SEV_BINS = [0, 3, 5, 10, 20, np.inf]
SEV_LABELS = ["<3", "3-5", "5-10", "10-20", ">20"]
STOP_OF_RANK = [s for s, _ in STATION_ORDER_WEST_OST]
NS = 10**9


def survival_df_real(trains: pd.DataFrame, days) -> pd.DataFrame:
    t = trains[trains["opday"].isin(days)]
    sched = pd.to_datetime(t["sched"].to_numpy(), utc=True)
    real = pd.to_datetime(np.where(t["cancelled"], t["sched"], t["real"]), utc=True)
    return pd.DataFrame({"stop_id": t["stop_id"].to_numpy(), "richtung": t["richtung"].astype(str).to_numpy(),
                         "time_schedule": sched, "time_real": real,
                         "delay": (t["delay_min"] * 60).to_numpy(), "cancelled": t["cancelled"].to_numpy()})


def survival_df_sim(day: Day, arr_t, delay, canc) -> pd.DataFrame:
    """Ein Lauf: Arrays (N, P) -> Tabelle im Format der Survival-Stufe."""
    nn, qq = np.nonzero(np.arange(day.P)[None, :] < day.plen[:, None])
    c = canc[nn, qq] | ~np.isfinite(delay[nn, qq])
    sched = (day.sched_min[nn, qq].astype(np.int64) + day.m0) * 60
    real = np.where(c, sched, (arr_t[nn, qq].astype(np.int64) + day.m0) * 60)
    return pd.DataFrame({"stop_id": np.array(STOP_OF_RANK)[day.rank_at[nn, qq].astype(int)],
                         "richtung": np.where(day.rich[nn] == 0, "Ost", "West"),
                         "time_schedule": pd.to_datetime(sched, unit="s", utc=True),
                         "time_real": pd.to_datetime(real, unit="s", utc=True),
                         "delay": np.where(c, np.nan, delay[nn, qq] * 60.0), "cancelled": c})


def measure_df(df: pd.DataFrame, ep: pd.DataFrame, fp) -> pd.DataFrame:
    series = R.build_series(df)
    frame = ep[["stop_id", "richtung"]].copy()
    frame["t0_s"] = ep["t0_s"].to_numpy()
    m = R.measure(frame, series, fp, ep["episode_id"].to_numpy())
    out = m[["schwere_min", "t_R15_180_st_std", "e_R15_180_st_std"]].copy()
    out["klasse"] = pd.cut(out["schwere_min"], SEV_BINS, labels=SEV_LABELS, right=False)
    out["opday"] = ep["opday"].to_numpy()
    return out.dropna(subset=["t_R15_180_st_std", "klasse"])


def km_median(t, e):
    return float(K.km_quantiles(np.asarray(t, float), np.asarray(e, float), qs=(0.5,))[0])


def main(days, tag="basis"):
    trains = load_trains()
    ep = pd.read_parquet(SD.ZW / "episoden.parquet", columns=["episode_id", "code", "t0", "stop_id", "richtung", "opday"])
    h = pd.read_parquet(SD.ZW / "hits_ep.parquet")
    fp = R.Footprints(h)
    ep = ep[ep["opday"].isin(days)].copy()
    ep["t0_s"] = SD.ns(ep["t0"]) // NS
    print("Episoden auf den Tagen:", len(ep), flush=True)

    # --- Echt ---
    real = measure_df(survival_df_real(trains, days), ep, fp)
    day_codes = {d: i for i, d in enumerate(sorted(days))}
    # --- Sim ---
    sims = []
    parts = {d: np.load(SAVE_DIR / f"{d}.npz") for d in days}
    dayobj = {d: Day(trains[trains["opday"] == d], d) for d in days}
    nrun = min(N_SAVE, min(p["arr_t"].shape[0] for p in parts.values()))
    for r in range(nrun):
        dfs = [survival_df_sim(dayobj[d], parts[d]["arr_t"][r], parts[d]["delay"][r], parts[d]["canc"][r]) for d in days]
        m = measure_df(pd.concat(dfs, ignore_index=True), ep, fp)
        m["lauf"] = r
        sims.append(m)
        print("Lauf", r, len(m), flush=True)
    sims = pd.concat(sims, ignore_index=True)
    real.to_parquet(OUT / f"pruefstein_e_rohdaten_echt_{tag}.parquet")
    sims.to_parquet(OUT / f"pruefstein_e_rohdaten_sim_{tag}.parquet")

    # --- Kennzahlen ---
    rng = np.random.default_rng(5)
    rows = []
    d_idx = real["opday"].map(day_codes).to_numpy()
    for kl in SEV_LABELS:
        rr = real[real["klasse"] == kl]
        row = dict(klasse=kl, n_echt=len(rr), zensiert_echt=float(1 - rr["e_R15_180_st_std"].mean()) if len(rr) else np.nan)
        if len(rr) >= 10:
            row["median_echt"] = km_median(rr["t_R15_180_st_std"], rr["e_R15_180_st_std"])
            Wd = K.make_day_weights(len(days), 1000, 3)
            bb = K.km_bootstrap(rr["t_R15_180_st_std"], rr["e_R15_180_st_std"], d_idx[(real["klasse"] == kl).to_numpy()], Wd, qs=(0.5,))
            lo, hi = K.ci_perc(bb["q_boot"][:, 0])
            row["echt_lo"], row["echt_hi"] = float(lo), float(hi)
        ms, ns = [], []
        for r, g in sims[sims["klasse"] == kl].groupby("lauf"):
            if len(g) >= 10:
                ms.append(km_median(g["t_R15_180_st_std"], g["e_R15_180_st_std"]))
            ns.append(len(g))
        ms = np.array(ms, float)
        row.update(n_sim_pro_lauf=float(np.mean(ns)) if ns else 0.0, n_laeufe=len(ms),
                   zensiert_sim=float(1 - sims.loc[sims["klasse"] == kl, "e_R15_180_st_std"].mean()) if ns else np.nan)
        if len(ms):
            fin = ms[np.isfinite(ms)]
            row.update(median_sim_lauf_mittel=float(np.mean(ms[np.isfinite(ms)])) if len(fin) else np.inf,
                       median_sim_q025=float(np.percentile(ms, 2.5)), median_sim_q975=float(np.percentile(ms, 97.5)),
                       anteil_laeufe_ne=float((~np.isfinite(ms)).mean()))
        rows.append(row)
    t = pd.DataFrame(rows)
    # Pruefstein-Regel
    ok, used = [], []
    for _, r in t.iterrows():
        if r["n_echt"] < 50 or "median_echt" not in r or pd.isna(r.get("median_sim_lauf_mittel", np.nan)):
            ok.append(np.nan); continue
        me, ms_ = r["median_echt"], r["median_sim_lauf_mittel"]
        if not np.isfinite(me):
            good = r["anteil_laeufe_ne"] > 0.5
        else:
            good = abs(ms_ - me) <= max(0.25 * me, 5.0) or (r["median_sim_q025"] <= me <= r["median_sim_q975"])
        ok.append(bool(good))
    t["klasse_ok"] = ok
    t.to_csv(OUT / f"pruefstein_e_{tag}.csv", index=False)
    valid = t["klasse_ok"].dropna()
    simm = t["median_sim_lauf_mittel"].to_numpy(float)
    mono_sim = bool(np.all(np.diff(simm[np.isfinite(simm)]) >= 0)) if np.isfinite(simm).sum() > 1 else False
    realm = t["median_echt"].to_numpy(float)
    mono_real = bool(np.all(np.diff(realm[np.isfinite(realm)]) >= 0))
    verdict = dict(klassen_ok=int(valid.sum()), klassen_gesamt=len(valid), monoton_sim=mono_sim, monoton_echt=mono_real,
                   bestanden=bool(valid.sum() >= 4 and mono_sim))
    json.dump(verdict, open(OUT / f"pruefstein_e_urteil_{tag}.json", "w"), indent=1)
    print(t.to_string()); print(verdict)
    return verdict


if __name__ == "__main__":
    import pickle
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else "basis"
    res = pickle.load(open(ZW / f"eval_results_{tag}.pkl", "rb"))
    main([r["opday"] for r in res], tag)

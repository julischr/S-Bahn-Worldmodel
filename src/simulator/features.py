"""Stufe 4: Uebergangsdatensatz (Zug an Station k -> k+1) mit strikt kausalen Merkmalen.

Stichtag T = Realzeit der Ankunft des Zuges an k (ganze Minute). Nur Ereignisse mit
Realzeit < T gehen in Merkmale ein (strikt: gleiche Minute ist nicht sichtbar).
Siehe outputs/simulator/plan.md, Abschnitt 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import HVZ_HOURS, RANK, TZ, ZW, load_trains, weekday_type

NS_MIN = 60 * 10**9
N_CLASS = 19          # Delta -3 ... +15 min
CLASS_LO, CLASS_HI = -3, 15
LINES = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "other"]
LINE_ID = {l: i for i, l in enumerate(LINES)}

FEATURES = [
    "d_k", "d_km1", "rank", "richtung", "line", "hour", "hvz", "wtype",
    "plan_gap_pred", "act_gap_pred", "d_pred_k", "d_pred_k1",
    "net_mean15", "net_n15", "cancel_sd30", "cancel_net30", "plan_run",
]
CAT_FEATURES = ["line"]


def line_id(s: pd.Series) -> np.ndarray:
    return s.map(LINE_ID).fillna(LINE_ID["other"]).astype("int8").to_numpy()


def path_order(t: pd.DataFrame) -> pd.DataFrame:
    """Sortiert nach (zug, Fahrtrichtungsreihenfolge) und ergaenzt pos."""
    t = t.copy()
    t["rich_i"] = (t["richtung"] == "West").astype("int8")
    t["pos"] = np.where(t["rich_i"] == 0, t["rank"], -t["rank"]).astype("int16")
    return t.sort_values(["zug", "pos"]).reset_index(drop=True)


def add_neighbours(t: pd.DataFrame) -> pd.DataFrame:
    """Naechster/vorheriger Halt desselben Zuges (nur wenn Rang-Differenz genau 1)."""
    z = t["zug"].to_numpy()
    pos = t["pos"].to_numpy()
    n = len(t)
    nxt = np.full(n, -1, dtype=np.int64)
    prv = np.full(n, -1, dtype=np.int64)
    same = (z[1:] == z[:-1]) & (pos[1:] - pos[:-1] == 1)
    idx = np.flatnonzero(same)
    nxt[idx] = idx + 1
    prv[idx + 1] = idx
    t["nxt"] = nxt
    t["prv"] = prv
    return t


def known_time(t: pd.DataFrame) -> np.ndarray:
    """Bekanntzeit (ns) eines Ausfalls je Zeile (nur fuer cancelled sinnvoll):
    max(Planzeit, Realzeit der letzten Ankunft DESSELBEN Zuges davor); ohne vorherige Ankunft
    (Zug fuhr nie bzw. Praefix-Ausfall) = Planzeit. Der Ausfall ist erst dann bekannt, wenn der
    Zug seinen letzten gefahrenen Halt erreicht hat (im Simulator: Zeitpunkt der Ziehung)."""
    sched = t["sched"].to_numpy()
    canc = t["cancelled"].to_numpy()
    lastrun = pd.Series(np.where(canc, np.nan, t["real"].to_numpy().astype(float)))
    lastrun = lastrun.groupby(t["zug"].to_numpy()).ffill().to_numpy()
    out = np.where(np.isnan(lastrun), sched.astype(float), np.maximum(sched.astype(float), lastrun))
    # float64 hat 53 Bit; ns-Zeiten ~1.7e18 sind nicht exakt -> auf Minuten gerundet rechnen
    mins = np.maximum(sched // NS_MIN, np.where(np.isnan(lastrun), -1, np.round(lastrun / NS_MIN))).astype(np.int64)
    return mins * NS_MIN


def build_dataset(trains: pd.DataFrame | None = None) -> pd.DataFrame:
    t = load_trains() if trains is None else trains
    t = add_neighbours(path_order(t))
    n = len(t)
    sched = t["sched"].to_numpy()
    real = t["real"].to_numpy()
    canc = t["cancelled"].to_numpy()
    delay = t["delay_min"].to_numpy(dtype=float)
    rank = t["rank"].to_numpy().astype(int)
    rich = t["rich_i"].to_numpy().astype(int)
    stop = t["stop_id"].to_numpy()
    zug = t["zug"].to_numpy()
    kt = known_time(t)

    # ---- Zeilen, die Uebergaenge bilden: gefahren an k, k+1 existiert ----
    nxt = t["nxt"].to_numpy()
    src = np.flatnonzero((~canc) & (nxt >= 0))
    dst = nxt[src]
    prv = t["prv"].to_numpy()[src]
    T = real[src]

    d = pd.DataFrame({"row": src, "zug": zug[src], "rank": rank[src].astype("int8"),
                      "richtung": rich[src].astype("int8"), "T": T, "sched": sched[src]})
    d["d_k"] = delay[src]
    prv_ok = (prv >= 0)
    prv_run = np.zeros(len(src), bool)
    prv_run[prv_ok] = ~canc[prv[prv_ok]]
    d["d_km1"] = np.where(prv_run, delay[np.where(prv_ok, prv, 0)], np.nan)
    d["line"] = line_id(t["line"].iloc[src].reset_index(drop=True))
    d["plan_run"] = (sched[dst] - sched[src]) / NS_MIN
    # Ziele
    d["cancel_next"] = canc[dst].astype("int8")
    dn = delay[dst]
    d["d_next"] = np.where(canc[dst], np.nan, dn)
    dlt = dn - delay[src]
    d["delta"] = np.where(canc[dst], np.nan, dlt)
    d["dclass"] = np.where(canc[dst], -1, np.clip(dlt, CLASS_LO, CLASS_HI) - CLASS_LO).astype("int16")

    # ---- Zeit-/Kalendermerkmale ----
    loc = pd.to_datetime(T, utc=True).tz_convert(TZ)
    hr = (loc.hour + loc.minute / 60).to_numpy()
    d["hour"] = hr
    d["hvz"] = np.isin(loc.hour, list(HVZ_HOURS)).astype("int8")
    opd = (loc - pd.Timedelta(hours=3)).date
    d["opday"] = opd
    uniq = pd.unique(opd)
    wmap = {u: weekday_type(u) for u in uniq}
    d["wtype"] = pd.Series(opd).map(wmap).astype("int8").to_numpy()

    # ---- Vorgaenger je (Station, Richtung) ----
    run_idx = np.flatnonzero(~canc)
    grp_key = stop.astype(np.int64) * 2 + rich
    d["pred_row"] = -1
    pred_row = np.full(len(src), -1, dtype=np.int64)
    d_stop_key = grp_key[src]
    ord_run = run_idx[np.lexsort((zug[run_idx], sched[run_idx], real[run_idx], grp_key[run_idx]))]
    gk_run = grp_key[ord_run]
    bounds = np.flatnonzero(np.r_[True, gk_run[1:] != gk_run[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        gk = gk_run[a]
        rows = ord_run[a:b]
        rr = real[rows]
        m = np.flatnonzero(d_stop_key == gk)
        if len(m) == 0:
            continue
        i = np.searchsorted(rr, T[m], side="left") - 1
        ok = i >= 0
        pred_row[m[ok]] = rows[i[ok]]
    has_p = pred_row >= 0
    pr = np.where(has_p, pred_row, 0)
    d["plan_gap_pred"] = np.where(has_p, (sched[src] - sched[pr]) / NS_MIN, np.nan)
    d["act_gap_pred"] = np.where(has_p, (T - real[pr]) / NS_MIN, np.nan)
    d["d_pred_k"] = np.where(has_p, delay[pr], np.nan)
    # Vorgaenger an k+1: gleicher Zug, Rang gleich dem Zielrang
    key_all = zug.astype(np.int64) * 16 + rank
    # t ist nach (zug, pos) sortiert, nicht nach key_all; eigener Index
    o = np.argsort(key_all, kind="stable")
    ks = key_all[o]
    tgt_rank = rank[dst]
    q = zug[np.where(has_p, pred_row, 0)].astype(np.int64) * 16 + tgt_rank
    j = np.searchsorted(ks, q)
    j = np.minimum(j, len(ks) - 1)
    found = has_p & (ks[j] == q)
    prow = o[j]
    arr_ok = found & (~canc[prow]) & (real[prow] < T)
    d["d_pred_k1"] = np.where(arr_ok, delay[prow], np.nan)

    # ---- Netz-Fenster 15 min (Mittel und Zahl der Ankuenfte, Realzeit in [T-15, T)) ----
    o_all = np.argsort(real[run_idx], kind="stable")
    rs = real[run_idx][o_all]
    cs = np.r_[0.0, np.cumsum(delay[run_idx][o_all])]
    lo = np.searchsorted(rs, T - 15 * NS_MIN, side="left")
    hi = np.searchsorted(rs, T, side="left")
    cnt = hi - lo
    d["net_n15"] = cnt
    d["net_mean15"] = np.where(cnt > 0, (cs[hi] - cs[lo]) / np.maximum(cnt, 1), np.nan)

    # ---- Ausfaelle 30 min (Bekanntzeit in [T-30, T)) ----
    ci = np.flatnonzero(canc)
    kts = np.sort(kt[ci])
    lo = np.searchsorted(kts, T - 30 * NS_MIN, side="left")
    hi = np.searchsorted(kts, T, side="left")
    d["cancel_net30"] = hi - lo
    c_sd = np.zeros(len(src), dtype=np.int64)
    ck = grp_key[ci]
    for gk in np.unique(ck):
        kk = np.sort(kt[ci][ck == gk])
        m = np.flatnonzero(d_stop_key == gk)
        c_sd[m] = np.searchsorted(kk, T[m], side="left") - np.searchsorted(kk, T[m] - 30 * NS_MIN, side="left")
    d["cancel_sd30"] = c_sd
    d["month"] = pd.Series(opd).map(lambda x: x.month).astype("int8").to_numpy()
    d = d.drop(columns=["pred_row"])
    return d


def load_dataset(rebuild: bool = False) -> pd.DataFrame:
    p = ZW / "transitions.parquet"
    if p.exists() and not rebuild:
        return pd.read_parquet(p)
    d = build_dataset()
    d.to_parquet(p)
    return d


if __name__ == "__main__":
    d = load_dataset(rebuild=True)
    print(d.shape)
    print(d.describe().T.to_string())

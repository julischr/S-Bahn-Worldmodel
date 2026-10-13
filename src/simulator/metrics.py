"""Stufe 4, Schritt 5: Kennzahlen je Betriebstag fuer Echt (R=1) und Simulation (R Laeufe).

Alle Funktionen arbeiten auf SimResult-Arrays, damit Echt und Simulation identisch gemessen werden.
"""

from __future__ import annotations

import numpy as np

from .sim import INF, Day, SimResult

HW_EDGES = [0, 5, 10, 15, 20, 60]           # Abstand (min), (lo, hi]; ">0" und "<=60" wie 11.3
HW_LABELS = ["<5min", "5-10min", "10-15min", "15-20min", "20-60min"]
DL_EDGES = [-np.inf, 2, 5, 10, np.inf]       # Vorgaengerverspaetung (min), (lo, hi]
DL_LABELS = ["<=2min", "2-5min", "5-10min", ">10min"]
NCELL = len(HW_LABELS) * len(DL_LABELS)


def cell_index(hw, dl):
    """hw, dl -> Zellnummer 0..19 (hw-Zeile * 4 + dl-Spalte), -1 ausserhalb."""
    hb = np.digitize(hw, HW_EDGES[1:-1], right=True)
    ok = (hw > 0) & (hw <= 60)
    db = np.digitize(dl, DL_EDGES[1:-1], right=True)
    return np.where(ok & np.isfinite(dl), hb * 4 + db, -1)


def _ran(res: SimResult):
    return np.isfinite(res.delay) & (~res.canc)


def stone_a_b(res: SimResult, real: SimResult) -> dict:
    """(a) Ankunft am Pfadende, (b) Tunnelaufbau fuer Zuege mit Rang 0 und 11 im Pfad."""
    d = res.day
    N = d.N
    last = d.plen.astype(int) - 1
    e = d.entry_pos.astype(int)
    idx = np.arange(N)
    rr = real.delay[0]
    real_ran_last = (e >= 0) & (last > e) & ~real.canc[0][idx, last] & np.isfinite(rr[idx, last])
    sel = np.flatnonzero(real_ran_last)
    out = {}
    if len(sel):
        simd = res.delay[:, sel, last[sel]]                      # (R, n)
        ok = np.isfinite(simd) & ~res.canc[:, sel, last[sel]]
        simd = np.where(ok, simd, np.nan)
        with np.errstate(all="ignore"):
            q05 = np.nanquantile(simd, 0.05, axis=0, method="lower")
            q50 = np.nanquantile(simd, 0.50, axis=0)
            q25 = np.nanquantile(simd, 0.25, axis=0, method="lower")
            q75 = np.nanquantile(simd, 0.75, axis=0, method="higher")
            q95 = np.nanquantile(simd, 0.95, axis=0, method="higher")
            mean = np.nanmean(simd, axis=0)
        out["a"] = dict(zug=d.zug_ids[sel], last_rank=d.rank_at[sel, last[sel]], rich=d.rich[sel],
                        real_end=rr[sel, last[sel]], real_entry=rr[sel, e[sel]],
                        sim_q05=q05, sim_q25=q25, sim_q50=q50, sim_q75=q75, sim_q95=q95, sim_mean=mean,
                        frac_arrived=ok.mean(0), entry_min=d.real_min[sel, e[sel]] + d.m0,
                        hvz=d.hvz_of_min[np.clip(d.real_min[sel, e[sel]], 0, d.M - 1)],
                        n_stops=last[sel] - e[sel] + 1)
    # (b): beide Endstationen im Pfad, Echtlauf fuhr durch
    p0 = d.pos_of_rank[:, 0].astype(int)
    p11 = d.pos_of_rank[:, 11].astype(int)
    both = (p0 >= 0) & (p11 >= 0)
    first = np.where(d.rich == 0, p0, p11)
    lastp = np.where(d.rich == 0, p11, p0)
    ok_real = both & (e == first) & (first >= 0)
    ok_real &= ~real.canc[0][idx, np.maximum(lastp, 0)] & np.isfinite(rr[idx, np.maximum(lastp, 0)])
    sel = np.flatnonzero(ok_real)
    if len(sel):
        simd = res.delay[:, sel, lastp[sel]] - res.delay[:, sel, first[sel]]
        okk = ~res.canc[:, sel, lastp[sel]] & np.isfinite(simd)
        simd = np.where(okk, simd, np.nan)
        out["b"] = dict(zug=d.zug_ids[sel], rich=d.rich[sel], real_build=rr[sel, lastp[sel]] - rr[sel, first[sel]],
                        sim_build_mean=np.nanmean(simd, axis=0), frac_arrived=okk.mean(0),
                        entry_delay=rr[sel, first[sel]],
                        hvz=d.hvz_of_min[np.clip(d.real_min[sel, first[sel]], 0, d.M - 1)])
        # (c): Tabelle 11.7, Vorgaenger = vorheriger gefahrener Zug (Plan) am Eintrittshalt
        cells = np.full(len(sel), -1)
        real_delta = rr[sel, lastp[sel]] - rr[sel, first[sel]]
        for rich_v in (0, 1):
            ur = 0 if rich_v == 0 else 11
            m_all = np.flatnonzero((d.rich == rich_v) & (d.pos_of_rank[:, ur] >= 0))
            posu = d.pos_of_rank[m_all, ur].astype(int)
            ran_u = ~real.canc[0][m_all, posu] & np.isfinite(rr[m_all, posu])
            m_run = m_all[ran_u]
            pu = posu[ran_u]
            sch = d.sched_min[m_run, pu]
            order = np.lexsort((m_run, sch))
            m_run, pu, sch = m_run[order], pu[order], sch[order]
            dl_u = rr[m_run, pu]
            hw = np.r_[np.nan, np.diff(sch)]
            prev_dl = np.r_[np.nan, dl_u[:-1]]
            ci = cell_index(hw, prev_dl)
            where = {int(n): k for k, n in enumerate(m_run)}
            for j, n in enumerate(sel):
                if d.rich[n] == rich_v:
                    cells[j] = ci[where[int(n)]]
        simdelta = np.where(okk, res.delay[:, sel, lastp[sel]] - res.delay[:, sel, first[sel]], np.nan)
        s_sum = np.zeros((res.delay.shape[0], NCELL)); s_n = np.zeros((res.delay.shape[0], NCELL))
        r_sum = np.zeros(NCELL); r_n = np.zeros(NCELL)
        for c in range(NCELL):
            m = cells == c
            if m.any():
                r_sum[c] = real_delta[m].sum(); r_n[c] = m.sum()
                v = simdelta[:, m]
                s_sum[:, c] = np.nansum(v, axis=1); s_n[:, c] = np.isfinite(v).sum(1)
        out["c"] = dict(r_sum=r_sum, r_n=r_n, s_sum=s_sum, s_n=s_n)
    return out


def stone_d(res: SimResult, real: SimResult) -> dict:
    """Tabelle 11.8 an modellierten Uebergaengen: Folgezug fuhr am Vorhalt (Position > Eintritt)."""
    d = res.day
    N, P = d.N, d.P
    e = d.entry_pos.astype(int)
    # alle Ereignisse (n, q) je (rank, rich) nach Plan sortiert
    nn, qq = np.nonzero(np.arange(P)[None, :] < d.plen[:, None])
    rk = d.rank_at[nn, qq].astype(int)
    key = rk * 2 + d.rich[nn]
    sch = d.sched_min[nn, qq]
    order = np.lexsort((nn, sch, key))
    nn, qq, key, sch = nn[order], qq[order], key[order], sch[order]
    samegrp = np.r_[False, key[1:] == key[:-1]]
    hw = np.where(samegrp, np.r_[np.nan, np.diff(sch)], np.nan)
    pn = np.r_[-1, nn[:-1]]; pq = np.r_[-1, qq[:-1]]
    elig = samegrp & (hw > 0) & (hw <= 60) & (qq > e[nn]) & (e[nn] >= 0)
    nn, qq, hw, pn, pq = nn[elig], qq[elig], hw[elig], pn[elig], pq[elig]

    def tab(R_):
        cnt = np.zeros((R_.delay.shape[0], NCELL)); canc = np.zeros_like(cnt)
        dl = R_.delay[:, pn, pq]                                      # Vorgaengerverspaetung
        pred_ok = np.isfinite(dl) & ~R_.canc[:, pn, pq]
        ci = cell_index(np.broadcast_to(hw, dl.shape), np.where(pred_ok, dl, np.nan))
        fc = R_.canc[:, nn, qq]
        for c in range(NCELL):
            m = ci == c
            cnt[:, c] = m.sum(1)
            canc[:, c] = (m & fc).sum(1)
        return cnt, canc
    rc, rk_ = tab(real)
    sc, sk = tab(res)
    return dict(r_n=rc[0], r_c=rk_[0], s_n=sc, s_c=sk)


def plausibility(res: SimResult) -> dict:
    """Ueberholungen (Reihenfolge benachbarter Zuege zwischen k und k+1 vertauscht) und
    gleiche Ankunftsminute aufeinanderfolgender Zuege an einer Station."""
    d = res.day
    R = res.arr_t.shape[0]
    ran = _ran(res)
    inv = np.zeros(R); opp = np.zeros(R)
    for rich in (0, 1):
        ns_ = np.flatnonzero(d.rich == rich)
        for k in range(11):
            ka, kb = (k, k + 1) if rich == 0 else (k + 1, k)
            pa, pb = d.pos_of_rank[ns_, ka].astype(int), d.pos_of_rank[ns_, kb].astype(int)
            m = (pa >= 0) & (pb >= 0)
            if m.sum() < 2:
                continue
            n2, pa, pb = ns_[m], pa[m], pb[m]
            ta = np.where(ran[:, n2, pa], res.arr_t[:, n2, pa], INF)
            tb = np.where(ran[:, n2, pb], res.arr_t[:, n2, pb], INF)
            both = (ta < INF) & (tb < INF)
            key = np.where(both, ta.astype(np.int64) * 10000 + d.sched_min[n2, pa][None, :] % 10000, np.int64(INF) * 10000)
            o = np.argsort(key, axis=1)
            tb_s = np.take_along_axis(tb, o, axis=1)
            ok_s = np.take_along_axis(both, o, axis=1)
            adj = ok_s[:, 1:] & ok_s[:, :-1]
            inv += ((tb_s[:, :-1] > tb_s[:, 1:]) & adj).sum(1)
            opp += adj.sum(1)
    # gleiche Minute
    same = np.zeros(R); tot = np.zeros(R); near = np.zeros(R)
    for rich in (0, 1):
        ns_ = np.flatnonzero(d.rich == rich)
        for rk_ in range(12):
            pa = d.pos_of_rank[ns_, rk_].astype(int)
            m = pa >= 0
            if m.sum() < 2:
                continue
            n2, pa = ns_[m], pa[m]
            ta = np.where(ran[:, n2, pa], res.arr_t[:, n2, pa], INF)
            ts = np.sort(ta, axis=1)
            ok = (ts[:, 1:] < INF) & (ts[:, :-1] < INF)
            df = ts[:, 1:] - ts[:, :-1]
            same += ((df == 0) & ok).sum(1); near += ((df <= 1) & ok).sum(1); tot += ok.sum(1)
    return dict(inv=inv, inv_opp=opp, same=same, near=near, gap_tot=tot)


def overall(res: SimResult) -> dict:
    """Mittlere Verspaetung je Station/Richtung und Ausfallanteil (modellierte Positionen > Eintritt)."""
    d = res.day
    R = res.arr_t.shape[0]
    e = d.entry_pos.astype(int)
    pos = np.arange(d.P)[None, :]
    inpath = pos < d.plen[:, None]
    ran = _ran(res)
    prof_sum = np.zeros((R, 24)); prof_n = np.zeros((R, 24))
    for n_rich in (0, 1):
        for rk in range(12):
            m = (d.rich == n_rich) & (d.pos_of_rank[:, rk] >= 0)
            ns_ = np.flatnonzero(m)
            if len(ns_) == 0:
                continue
            pa = d.pos_of_rank[ns_, rk].astype(int)
            v = np.where(ran[:, ns_, pa], res.delay[:, ns_, pa], np.nan)
            prof_sum[:, rk * 2 + n_rich] = np.nansum(v, axis=1)
            prof_n[:, rk * 2 + n_rich] = np.isfinite(v).sum(1)
    modelled = inpath & (pos > e[:, None]) & (e[:, None] >= 0)
    n_mod = modelled.sum()
    n_c = (res.canc & modelled[None]).sum(axis=(1, 2))
    return dict(prof_sum=prof_sum, prof_n=prof_n, n_mod=int(n_mod), n_canc=n_c)

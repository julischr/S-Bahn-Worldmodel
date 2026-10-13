"""Stufe 4, Schritt 4: Rollout-Simulator (Minutentakt, alle Laeufe eines Tages parallel).

Ablauf je Minute t: alle Ankuenfte dieser Minute (Zuege, deren naechste Ankunft t ist) bilden
Merkmale NUR aus Zustaenden mit Zeit < t (gleiche Minute ist unsichtbar), dann werden je
Zug Ausfall und Verspaetungszuwachs bis zur naechsten Station aus den Modellen gezogen. Erst am
Minutenende werden die Ankuenfte der Minute ins Protokoll geschrieben. Das ist identisch
zur Definition der Trainingsmerkmale (features.py); `forced=True` (Teacher Forcing) prueft das.
Siehe outputs/simulator/plan.md, Abschnitt 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import HVZ_HOURS, TZ, load_trains, weekday_type
from .features import CLASS_HI, CLASS_LO, FEATURES, N_CLASS, line_id, path_order
from .model_io import Models

NS_MIN = 60 * 10**9
INF = 10**8
NSD = 24


@dataclass
class Scenario:
    kind: str = "none"                 # none | delay | cancel | closure
    zug: int | None = None             # Zug-ID (trip-Verband-ID) fuer delay/cancel
    delay_abs_min: float = 0.0         # delay: Verspaetung am Eintritt (absolut, min)
    rank: int | None = None            # closure: Station (Rang)
    rich: int | None = None            # closure: nur diese Richtung (0 Ost, 1 West), None = beide
    t1: int | None = None              # closure: Minute (absolut, UTC-Minuten seit Epoche)
    t2: int | None = None
    min_sep: int = 2                   # closure: Mindestabstand nach Wiederoeffnung (min)


class Day:
    """Tagesdaten (Betriebstag) im Array-Layout des Simulators."""

    def __init__(self, trains_day: pd.DataFrame, opday):
        self.opday = opday
        t = path_order(trains_day)
        z = t["zug"].to_numpy()
        pos = t["pos"].to_numpy()
        brk = np.r_[True, (z[1:] != z[:-1]) | (pos[1:] - pos[:-1] != 1)]
        seg = np.cumsum(brk)
        t["seg"] = seg
        seglen = t.groupby("seg")["zug"].transform("size")
        t["seglen"] = seglen
        # je Zug laengstes Segment (bei Gleichstand das erste)
        best = t.groupby("zug")["seglen"].transform("max")
        first_best = t[t["seglen"] == best].groupby("zug")["seg"].transform("min")
        keep = (t["seglen"] == best) & (t["seg"] == first_best.reindex(t.index))
        self.n_rows_dropped = int((~keep).sum())
        t = t[keep].reset_index(drop=True)
        zs = t["zug"].to_numpy()
        self.zug_ids, inv = np.unique(zs, return_inverse=True)
        N = len(self.zug_ids)
        cc = t.groupby("zug").cumcount().to_numpy()
        P = int(cc.max()) + 1
        self.N, self.P = N, P
        self.m0 = int(t["sched"].min() // NS_MIN) - 60
        sm = (t["sched"].to_numpy() // NS_MIN - self.m0).astype(np.int32)
        self.plen = np.bincount(inv, minlength=N).astype(np.int16)
        self.rank_at = np.full((N, P), -1, np.int8)
        self.sched_min = np.full((N, P), INF, np.int32)
        self.canc_real = np.zeros((N, P), bool)
        self.delay_real = np.full((N, P), np.nan, np.float32)
        self.real_min = np.full((N, P), INF, np.int32)
        self.rank_at[inv, cc] = t["rank"].to_numpy()
        self.sched_min[inv, cc] = sm
        self.canc_real[inv, cc] = t["cancelled"].to_numpy()
        self.delay_real[inv, cc] = t["delay_min"].to_numpy(dtype=np.float32)
        rm = (t["real"].to_numpy() // NS_MIN - self.m0).astype(np.int32)
        ok = ~t["cancelled"].to_numpy()
        self.real_min[inv[ok], cc[ok]] = rm[ok]
        first = t.groupby("zug").first()
        self.line_at = np.full((N, P), 8, np.int8)
        self.line_at[inv, cc] = line_id(t["line"])
        self.rich = first["rich_i"].reindex(self.zug_ids).to_numpy().astype(np.int8)
        self.pos_of_rank = np.full((N, 12), -1, np.int16)
        self.pos_of_rank[inv, t["rank"].to_numpy()] = cc
        # Entry = erster gefahrener Halt
        runm = ~self.canc_real & (np.arange(P)[None, :] < self.plen[:, None])
        self.entry_pos = np.where(runm.any(1), runm.argmax(1), -1).astype(np.int16)
        last_t = int(max(self.real_min[self.real_min < INF].max(), self.sched_min[self.sched_min < INF].max()))
        self.M = last_t + 400
        # Zeitgrundlagen je Minute
        absmin = (np.arange(self.M, dtype=np.int64) + self.m0) * 60
        loc = pd.to_datetime(absmin, unit="s", utc=True).tz_convert(TZ)
        self.hour_of_min = (loc.hour + loc.minute / 60).to_numpy().astype(np.float32)
        self.hvz_of_min = np.isin(loc.hour, list(HVZ_HOURS)).astype(np.int8)
        self.wtype = weekday_type(opday)
        self.zug_index = {int(zv): i for i, zv in enumerate(self.zug_ids)}

    # ---- Echtlauf im Layout von SimResult ----
    def real_result(self) -> "SimResult":
        arr = np.where(self.real_min < INF, self.real_min, INF)[None].astype(np.int32)
        canc = self.canc_real[None].copy()
        # Replay-Konvention: nur Ausfaelle am Eintritt/davor als solche, spaeter wie Daten
        return SimResult(self, arr, self.delay_real[None].copy(), canc, dict(n_clamp=0))


@dataclass
class SimResult:
    day: Day
    arr_t: np.ndarray       # (R, N, P) Ankunftsminute (INF = keine Ankunft)
    delay: np.ndarray       # (R, N, P) Verspaetung min (NaN = keine Ankunft)
    canc: np.ndarray        # (R, N, P) ausgefallen
    meta: dict = field(default_factory=dict)
    collected: pd.DataFrame | None = None


def simulate(day: Day, models: Models | None, R: int, seed: int = 0, scenario: Scenario | None = None,
             t_start: int | None = None, forced: bool = False, collect: bool = False,
             crn_seed: int | None = None, t_end: int | None = None) -> SimResult:
    N, P, M = day.N, day.P, day.M
    sc = scenario or Scenario()
    rng = np.random.default_rng([seed if crn_seed is None else crn_seed, int(pd.Timestamp(day.opday).strftime("%Y%m%d"))])
    U = rng.random((R, N, P, 3), dtype=np.float32) if not forced else None
    arr_t = np.full((R, N, P), INF, np.int32)
    delay = np.full((R, N, P), np.nan, np.float32)
    canc = np.zeros((R, N, P), bool)
    next_t = np.full((R, N), INF, np.int32)
    cur_pos = np.full((R, N), -1, np.int16)
    sumd = np.zeros((R, M), np.float32)
    cntd = np.zeros((R, M), np.int16)
    cancm = np.zeros((R, M), np.int16)
    cancsd = np.zeros((R, NSD, M), np.int16)
    last_tr = np.full((R, NSD), -1, np.int32)
    rel_next = np.zeros((R, NSD), np.int32)    # Sperrung: naechste freie Ankunftsminute
    ar = np.arange(P)
    sd_of = (day.rank_at.astype(np.int16) * 2 + day.rich[:, None]).astype(np.int16)
    sd_of = np.where(day.rank_at >= 0, sd_of, 0)
    n_clamp = 0
    n_hold = 0
    collected = []

    def add_cancel_known(rr, nn, qq, kt):
        kt = np.clip(kt, 0, M - 1)
        sd = sd_of[nn, qq]
        np.add.at(cancm, (rr, kt), 1)
        np.add.at(cancsd, (rr, sd, kt), 1)

    # ---- Szenario: Ausfall der Fahrt ----
    cancel_zug = None
    if sc.kind in ("delay", "cancel") and sc.zug is not None:
        cancel_zug = day.zug_index[int(sc.zug)]

    # ---- Replay-Initialisierung ----
    ent = day.entry_pos
    started = np.zeros(N, bool)
    t_cut = -10**9 if t_start is None else t_start
    # Zuege ohne Fahrt / Praefix-Ausfaelle: Replay der Datenausfaelle (bekannt zur Planzeit)
    base_c = np.zeros((NSD, M), np.int16)
    base_net = np.zeros(M, np.int16)
    for n in range(N):
        pl = int(day.plen[n])
        e = int(ent[n])
        force_cancel = (cancel_zug == n and sc.kind == "cancel")
        lim = pl if (e < 0 or force_cancel) else e
        for q in range(lim):
            kt = int(min(day.sched_min[n, q], M - 1))
            base_c[sd_of[n, q], kt] += 1
            base_net[kt] += 1
        if e < 0 or force_cancel:
            canc[:, n, :pl] = True
        else:
            canc[:, n, :e] = True
    cancsd += base_c[None]
    cancm += base_net[None]

    # Zuege mit gefahrenen Halten vor t_start: echten Verlauf uebernehmen
    pre_events = []
    for n in range(N):
        if canc[0, n, :day.plen[n]].all():
            continue
        e = int(ent[n])
        pl = int(day.plen[n])
        p_last = -1
        for q in range(e, pl):
            if day.canc_real[n, q] or day.real_min[n, q] >= INF:
                break
            if day.real_min[n, q] < t_cut:
                p_last = q
            else:
                break
        if p_last < 0:
            continue
        for q in range(e, p_last + 1):
            arr_t[:, n, q] = day.real_min[n, q]
            delay[:, n, q] = day.delay_real[n, q]
            pre_events.append((day.real_min[n, q], day.sched_min[n, q], n, q))
        started[n] = True
        cur_pos[:, n] = p_last
        # naechster Halt: echter Wert (Entscheidung fiel vor t_start)
        q = p_last + 1
        if q < pl:
            if day.canc_real[n, q] or day.real_min[n, q] >= INF:
                canc[:, n, q:pl] = True
                kt0 = max(int(day.sched_min[n, q]), int(day.real_min[n, p_last]))
                for qq in range(q, pl):
                    kt = max(int(day.sched_min[n, qq]), int(day.real_min[n, p_last]))
                    add_cancel_known(np.arange(R), np.full(R, n), np.full(R, qq), np.full(R, kt))
                next_t[:, n] = INF
            else:
                arr_t[:, n, q] = day.real_min[n, q]
                delay[:, n, q] = day.delay_real[n, q]
                next_t[:, n] = day.real_min[n, q]
                cur_pos[:, n] = q
        else:
            next_t[:, n] = INF
    # Eintritte nach t_start (oder alle)
    ent_adj = {}
    for n in range(N):
        e = int(ent[n])
        if e < 0 or started[n] or canc[0, n, e]:
            continue
        a = int(day.real_min[n, e])
        dl = float(day.delay_real[n, e])
        if sc.kind == "delay" and cancel_zug == n:
            dl = float(sc.delay_abs_min)
            a = int(day.sched_min[n, e] + dl)
        ent_adj[n] = (a, dl)
    if sc.kind == "closure":
        # Zuege, die direkt an der gesperrten Station eintreten
        lst = sorted([(a, n) for n, (a, dl) in ent_adj.items() if day.rank_at[n, int(ent[n])] == sc.rank and (sc.rich is None or day.rich[n] == sc.rich) and sc.t1 <= a + day.m0 < sc.t2])
        for n_a, (a, n) in enumerate(lst):
            sdx = sd_of[n, int(ent[n])]
            na = sc.t2 - day.m0 + sc.min_sep * n_a
            ent_adj[n] = (na, float(na - day.sched_min[n, int(ent[n])]))
    for n, (a, dl) in ent_adj.items():
        e = int(ent[n])
        arr_t[:, n, e] = a
        delay[:, n, e] = dl
        next_t[:, n] = a
        cur_pos[:, n] = e
    if t_start is not None:
        # Protokoll der Vor-Ereignisse
        pre_events.sort()
        for (tm, sch, n, q) in pre_events:
            sumd[:, tm] += day.delay_real[n, q]
            cntd[:, tm] += 1
            last_tr[:, sd_of[n, q]] = n
        t_begin = t_start
    else:
        t_begin = max(0, int(next_t[next_t < INF].min()) - 1) if (next_t < INF).any() else 0

    rr_all = np.arange(R)
    rich_a = day.rich
    hi_len, lo_len = len(models.rand_hi) if models else 1, len(models.rand_lo) if models else 1
    t = t_begin
    t_stop = M - 1 if t_end is None else min(M - 1, t_end)
    while t < t_stop:
        rr, nn = np.nonzero(next_t == t)
        if len(rr) == 0:
            if t % 64 == 0 and not (next_t < INF).any():
                break
            t += 1
            continue
        p = cur_pos[rr, nn].astype(np.int64)
        rk = day.rank_at[nn, p].astype(np.int64)
        sd = sd_of[nn, p].astype(np.int64)
        d_k = delay[rr, nn, p]
        dec = (p + 1) < day.plen[nn]
        # --- Entscheidungen ---
        if dec.any():
            ri, ni, pi, rki, sdi, dki = rr[dec], nn[dec], p[dec], rk[dec], sd[dec], d_k[dec]
            E = len(ri)
            e_pos = ent[ni].astype(np.int64)
            prevok = pi > e_pos
            d_km1 = np.where(prevok, delay[ri, ni, np.maximum(pi - 1, 0)], np.nan)
            lo15, lo30 = max(t - 15, 0), max(t - 30, 0)
            wsum = sumd[:, lo15:t].sum(1)
            wcnt = cntd[:, lo15:t].sum(1).astype(np.int64)
            net_mean = np.where(wcnt[ri] > 0, wsum[ri] / np.maximum(wcnt[ri], 1), np.nan)
            net_n = wcnt[ri]
            c_net = cancm[:, lo30:t].sum(1).astype(np.int64)[ri]
            c_sd = cancsd[:, :, lo30:t].sum(2).astype(np.int64)[ri, sdi]
            pr = last_tr[ri, sdi]
            hp = pr >= 0
            prc = np.where(hp, pr, 0)
            pp = np.where(hp, day.pos_of_rank[prc, rki], 0).astype(np.int64)
            plan_gap = np.where(hp, day.sched_min[ni, pi] - day.sched_min[prc, pp], np.nan)
            act_gap = np.where(hp, t - arr_t[ri, prc, pp], np.nan)
            d_pk = np.where(hp, delay[ri, prc, pp], np.nan)
            nr = rki + np.where(rich_a[ni] == 0, 1, -1)
            nr_c = np.clip(nr, 0, 11)
            pos2 = np.where(hp, day.pos_of_rank[prc, nr_c], -1).astype(np.int64)
            pos2c = np.maximum(pos2, 0)
            ok2 = hp & (pos2 >= 0) & (arr_t[ri, prc, pos2c] < t) & (~canc[ri, prc, pos2c])
            d_pk1 = np.where(ok2, delay[ri, prc, pos2c], np.nan)
            plan_run = (day.sched_min[ni, pi + 1] - day.sched_min[ni, pi]).astype(np.float32)
            X = np.column_stack([dki, d_km1, rki, rich_a[ni], day.line_at[ni, pi], np.full(E, day.hour_of_min[t]),
                                 np.full(E, day.hvz_of_min[t]), np.full(E, day.wtype),
                                 plan_gap, act_gap, d_pk, d_pk1, net_mean, net_n, c_sd, c_net, plan_run]).astype(np.float32)
            if collect:
                collected.append(pd.DataFrame(X, columns=FEATURES).assign(
                    r=ri, n=ni, p=pi, t=t, zug=day.zug_ids[ni]))
            if forced:
                canc_next = day.canc_real[ni, pi + 1] | (day.real_min[ni, pi + 1] >= INF)
                dn = day.delay_real[ni, pi + 1].astype(np.float64)
                a_new = day.real_min[ni, pi + 1].astype(np.int64)
            else:
                pc = models.p_cancel(X)
                Pc = models.p_class(X)
                u = U[ri, ni, pi]
                canc_next = u[:, 0] < pc
                cdf = np.cumsum(Pc, axis=1)
                cls = np.minimum((cdf < u[:, 1:2]).sum(1), N_CLASS - 1)
                delta = (cls + CLASS_LO).astype(np.float64)
                ih = cls == N_CLASS - 1
                il = cls == 0
                if ih.any():
                    delta[ih] = CLASS_HI + models.rand_hi[np.minimum((u[ih, 2] * hi_len).astype(int), hi_len - 1)]
                if il.any():
                    delta[il] = CLASS_LO + models.rand_lo[np.minimum((u[il, 2] * lo_len).astype(int), lo_len - 1)]
                dn = dki.astype(np.float64) + delta
                a_new = day.sched_min[ni, pi + 1].astype(np.int64) + np.rint(dn).astype(np.int64)
                # Sperrung
                if sc.kind == "closure":
                    tgt_rank = day.rank_at[ni, pi + 1]
                    hit = (tgt_rank == sc.rank) & ((sc.rich is None) | (rich_a[ni] == (sc.rich if sc.rich is not None else 0))) & (a_new + day.m0 >= sc.t1) & (a_new + day.m0 < sc.t2) & (~canc_next)
                    if hit.any():
                        for j in np.flatnonzero(hit):
                            s2 = sd_of[ni[j], pi[j] + 1]
                            na = max(sc.t2 - day.m0 + int(plan_run[j]), int(rel_next[ri[j], s2]))
                            rel_next[ri[j], s2] = na + sc.min_sep
                            a_new[j] = na
                            dn[j] = na - day.sched_min[ni[j], pi[j] + 1]
                            n_hold += 1
                    # Zuege, die nach dem Wiederoeffnen kommen, aber noch in der Warteschlange stehen
                    late = (tgt_rank == sc.rank) & ((sc.rich is None) | (rich_a[ni] == (sc.rich if sc.rich is not None else 0))) & (a_new + day.m0 >= sc.t2) & (~canc_next) & (~hit)
                    if late.any():
                        for j in np.flatnonzero(late):
                            s2 = sd_of[ni[j], pi[j] + 1]
                            if rel_next[ri[j], s2] > a_new[j] and a_new[j] + day.m0 < sc.t2 + 120:
                                a_new[j] = rel_next[ri[j], s2]
                                rel_next[ri[j], s2] += sc.min_sep
                                dn[j] = a_new[j] - day.sched_min[ni[j], pi[j] + 1]
            bad = (a_new < t + 1) & (~canc_next)
            n_clamp += int(bad.sum())
            a_new = np.maximum(a_new, t + 1)
            if not forced:
                dn = np.where(bad, a_new - day.sched_min[ni, pi + 1], dn)
            okm = ~canc_next
            if okm.any():
                r1, n1, p1 = ri[okm], ni[okm], pi[okm] + 1
                arr_t[r1, n1, p1] = a_new[okm]
                delay[r1, n1, p1] = dn[okm]
                next_t[r1, n1] = a_new[okm]
                cur_pos[r1, n1] = p1
            if canc_next.any():
                rc, nc_, pc_ = ri[canc_next], ni[canc_next], pi[canc_next]
                next_t[rc, nc_] = INF
                for qoff in range(1, P):
                    sel = (pc_ + qoff) < day.plen[nc_]
                    if not sel.any():
                        break
                    rq, nq, qq = rc[sel], nc_[sel], pc_[sel] + qoff
                    canc[rq, nq, qq] = True
                    kt = np.maximum(day.sched_min[nq, qq], t)
                    add_cancel_known(rq, nq, qq, kt.astype(np.int64))
        term = ~dec
        if term.any():
            next_t[rr[term], nn[term]] = INF
        # --- Protokoll der Ankuenfte dieser Minute ---
        np.add.at(sumd, (rr, np.full(len(rr), t)), d_k)
        np.add.at(cntd, (rr, np.full(len(rr), t)), 1)
        order = np.lexsort((nn, day.sched_min[nn, p], sd, rr))
        rs, ss, ns_ = rr[order], sd[order], nn[order]
        lastmask = np.r_[(rs[1:] != rs[:-1]) | (ss[1:] != ss[:-1]), True]
        last_tr[rs[lastmask], ss[lastmask]] = ns_[lastmask]
        t += 1

    meta = dict(n_clamp=n_clamp, n_hold=n_hold, t_begin=t_begin, rows_dropped=day.n_rows_dropped, cancm0=cancm[0].copy())
    res = SimResult(day, arr_t, delay, canc, meta)
    if collect and collected:
        res.collected = pd.concat(collected, ignore_index=True)
    return res


def load_day(trains: pd.DataFrame, opday) -> Day:
    return Day(trains[trains["opday"] == opday], opday)

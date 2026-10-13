"""Stufe 3 (Survival): Erholungsmessung, Placebo-Ziehung, Episoden-Kennzahlen.
Siehe outputs/survival/auswertungsplan.md (Abschnitte 2-4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.var.stations import STATION_ORDER_WEST_OST
from .data import EXCL, FEIERTAGE, TZ, ns

NS = 10**9
MIN = 60
THRS = (180, 360)
PLACEBO_K = 5
PLACEBO_SEED = 20250
DATE_TOL_DAYS = 42
MAX_TRIES = 60
EXCL_DAYS = np.array(sorted(EXCL), dtype="datetime64[D]")
HOL_DAYS = np.array(sorted(FEIERTAGE), dtype="datetime64[D]")
WT_NAMES = np.array(["Mo-Do", "Fr", "Sa", "So+Feiertag"])


def wtype_code(days: np.ndarray) -> np.ndarray:
    """datetime64[D] -> 0 Mo-Do, 1 Fr, 2 Sa, 3 So+Feiertag."""
    dow = (days.astype("int64") + 3) % 7  # Montag = 0
    code = np.where(dow == 4, 1, np.where(dow == 5, 2, np.where(dow == 6, 3, 0)))
    return np.where(np.isin(days, HOL_DAYS), 3, code)


def opday_day(utc_s: np.ndarray) -> np.ndarray:
    """UTC-Sekunden (int64) -> Betriebstag als datetime64[D] (lokal minus 3h)."""
    idx = pd.DatetimeIndex((utc_s.astype("int64") * NS).astype("datetime64[ns]")).tz_localize("UTC").tz_convert(TZ)
    return (idx.tz_localize(None) - pd.Timedelta(hours=3)).floor("D").to_numpy().astype("datetime64[D]")


class Series:
    """Zugfolge an (Station, Richtung), nach Planzeit sortiert (Sekunden seit Epoche)."""

    def __init__(self, g: pd.DataFrame):
        g = g.sort_values("time_schedule")
        self.sched = ns(g["time_schedule"]) // NS
        self.n = n = len(g)
        self.delay = g["delay"].to_numpy(dtype=float)  # NaN bei Ausfall
        self.cancel = g["cancelled"].to_numpy(dtype=bool)
        self.opday = opday_day(self.sched).astype("int64")
        self.opday_unique = np.unique(self.opday)
        real = ns(g["time_real"]) // NS
        o = np.argsort(real, kind="stable")
        self.real_sorted = real[o]
        self.delay_by_real = self.delay[o]
        self.nxt = {}
        for thr in THRS:
            ok = (~self.cancel) & (np.nan_to_num(self.delay, nan=1e9) < thr)
            tri = np.zeros(n, dtype=bool)
            tri[:-2] = ok[:-2] & ok[1:-1] & ok[2:] & (self.opday[:-2] == self.opday[2:])
            idx = np.where(tri, np.arange(n), n)
            self.nxt[thr] = np.r_[np.minimum.accumulate(idx[::-1])[::-1], n]
        last = pd.Series(self.sched).groupby(self.opday).transform("max").to_numpy()
        self.day_end = np.r_[last, 0]
        self.opday_ext = np.r_[self.opday, -1]
        self.sched_ext = np.r_[self.sched, 0, 0]

    def recover(self, o: np.ndarray, od: np.ndarray, thr: int):
        """o: Ursprung (s), od: Betriebstag-Nummer des Ursprungs. Rueckgabe: valid (es gibt
        Zuege ab o im selben Betriebstag), found, T (Planzeit des ersten Zuges der ersten
        puenktlichen Dreierfolge - o), done (Planzeit des dritten), end (Betriebstagsende - o),
        noeff (die ersten 3 Zuege ab o sind puenktlich)."""
        i0 = np.minimum(np.searchsorted(self.sched, o, side="left"), self.n)
        valid = (i0 < self.n) & (self.opday_ext[i0] == od)
        j = np.minimum(self.nxt[thr][i0], self.n)
        found = valid & (j < self.n) & (self.opday_ext[j] == od)
        T = np.where(found, self.sched_ext[j] - o, np.nan)
        done = np.where(found, self.sched_ext[np.minimum(j + 2, self.n + 1)], np.nan)
        end = np.where(valid, self.day_end[i0] - o, np.nan)
        return valid, found, T, done, end, valid & found & (j == i0)

    def window_stats(self, o: np.ndarray):
        """Schwere (max Verspaetung min, Planzeit in [o, o+15)), Vorbelastung (mittlere
        Verspaetung min, time_real in [o-30, o]), Ausfallquote (Planzeit in [o+15, o+135))."""
        m = len(o)
        sev = np.full(m, np.nan); vor = np.full(m, np.nan); auq = np.full(m, np.nan)
        nsev = np.zeros(m, dtype=int); nauq = np.zeros(m, dtype=int)
        a = np.searchsorted(self.sched, o, side="left")
        b = np.searchsorted(self.sched, o + 15 * MIN, side="left")
        c = np.searchsorted(self.sched, o + 135 * MIN, side="left")
        r0 = np.searchsorted(self.real_sorted, o - 30 * MIN, side="left")
        r1 = np.searchsorted(self.real_sorted, o, side="right")
        for k in range(m):
            d = self.delay[a[k]:b[k]]
            nsev[k] = b[k] - a[k]
            d = d[~np.isnan(d)]
            if len(d):
                sev[k] = d.max() / 60
            dv = self.delay_by_real[r0[k]:r1[k]]
            dv = dv[~np.isnan(dv)]
            if len(dv):
                vor[k] = dv.mean() / 60
            cn = self.cancel[b[k]:c[k]]
            nauq[k] = len(cn)
            if len(cn):
                auq[k] = cn.mean()
        return sev, vor, auq, nsev, nauq


def build_series(trains: pd.DataFrame) -> dict:
    return {(int(s), d): Series(g) for (s, d), g in trains.groupby(["stop_id", "richtung"])}


class Footprints:
    """Erste Meldung je Episode an (Station, Richtung) bzw. je Richtung (Zensierung durch
    Folgeepisoden); alle Meldezeiten je (Station, Richtung) fuer die Placebo-Bedingung."""

    def __init__(self, h: pd.DataFrame):
        h = h.dropna(subset=["richtung"]).assign(t=lambda x: ns(x["update_timestamp"]) // NS)
        f = h.groupby(["stop_id", "richtung", "episode_id"])["t"].min().reset_index()
        self.st = {}
        for (s, d), g in f.groupby(["stop_id", "richtung"]):
            g = g.sort_values("t")
            self.st[(int(s), d)] = (g["t"].to_numpy(), g["episode_id"].to_numpy())
        f2 = h.groupby(["richtung", "episode_id"])["t"].min().reset_index()
        self.dir = {}
        for d, g in f2.groupby("richtung"):
            g = g.sort_values("t")
            self.dir[d] = (g["t"].to_numpy(), g["episode_id"].to_numpy())
        self.allhits = {(int(s), d): np.sort(g["t"].to_numpy()) for (s, d), g in h.groupby(["stop_id", "richtung"])}

    @staticmethod
    def next_after(arr, t, own):
        times, ids = arr
        out = np.full(len(t), np.inf)
        idx = np.searchsorted(times, t, side="right")
        for k in range(len(t)):
            i = idx[k]
            while i < len(times) and ids[i] == own[k]:
                i += 1
            if i < len(times):
                out[k] = times[i]
        return out

    def hit_in_window(self, s, d, tau, lo_min, hi_min):
        a = self.allhits.get((s, d))
        if a is None:
            return np.zeros(len(tau), dtype=bool)
        i = np.searchsorted(a, tau + lo_min * MIN, side="left")
        j = np.searchsorted(a, tau + hi_min * MIN, side="right")
        return j > i


def make_placebo(ep: pd.DataFrame, fp: Footprints, series: dict, anchored: bool = False) -> pd.DataFrame:
    """K Placebo-Zeitpunkte je Episode (Plan Abschnitt 4).
    anchored=False: gleichverteilter Zeitpunkt in der Ortsstunde (Plan, "P1").
    anchored=True: Zeitpunkt = time_real eines zufaelligen Zuges derselben Station/Richtung
    in dieser Stunde + 120 s ("P2", Abweichung 2 im Plan), damit die Phase zum Zugtakt wie
    bei echten Meldungen liegt (Median-Verzoegerung der ersten Meldung: 1,9 min).
    Zusaetzlich: der Betriebstag muss Zuege an (Station, Richtung) enthalten."""
    rng = np.random.default_rng(PLACEBO_SEED + int(anchored))
    n = len(ep)
    rows = np.repeat(np.arange(n), PLACEBO_K)
    loc = ep["t0"].dt.tz_convert(TZ)
    base_date = loc.dt.tz_localize(None).dt.floor("D").to_numpy().astype("datetime64[D]")[rows]
    hour = loc.dt.hour.to_numpy()[rows]
    own_op = np.array(ep["opday"], dtype="datetime64[D]")[rows]
    wt_target = wtype_code(own_op)
    stop = ep["stop_id"].to_numpy()[rows]
    rich = ep["richtung"].to_numpy()[rows]
    tau = np.full(len(rows), np.nan)
    todo = np.arange(len(rows))
    for _ in range(MAX_TRIES):
        if len(todo) == 0:
            break
        off = rng.integers(-DATE_TOL_DAYS, DATE_TOL_DAYS + 1, size=len(todo))
        sec = rng.integers(0, 3600, size=len(todo))
        day = base_date[todo] + off.astype("timedelta64[D]")
        naive = day.astype("datetime64[s]") + (hour[todo] * 3600 + sec).astype("timedelta64[s]")
        idx = pd.DatetimeIndex(naive).tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
        good = ~idx.isna()
        utc = np.zeros(len(todo), dtype="int64")
        utc[good] = idx[good].tz_convert("UTC").as_unit("s").asi8
        opd = np.full(len(todo), np.datetime64("2025-01-01"), dtype="datetime64[D]")
        if good.any():
            opd[good] = opday_day(utc[good])
        if anchored:
            for key in set(zip(stop[todo], rich[todo])):
                m = np.flatnonzero((stop[todo] == key[0]) & (rich[todo] == key[1]) & good)
                if len(m) == 0:
                    continue
                rs = series[(int(key[0]), key[1])].real_sorted
                a = np.searchsorted(rs, utc[m] - sec[m], side="left")
                b = np.searchsorted(rs, utc[m] - sec[m] + 3600, side="left")
                has = b > a
                pick = a + np.floor(rng.random(len(m)) * np.maximum(b - a, 1)).astype(int)
                newt = np.where(has, rs[np.minimum(pick, len(rs) - 1)] + 120, 0)
                good[m] = has
                utc[m] = np.where(has, newt, utc[m])
            opd = np.full(len(todo), np.datetime64("2025-01-01"), dtype="datetime64[D]")
            if good.any():
                opd[good] = opday_day(utc[good])
        ok = good & (opd != own_op[todo]) & (off != 0)
        for key in set(zip(stop[todo][ok], rich[todo][ok])):
            m = ok & (stop[todo] == key[0]) & (rich[todo] == key[1])
            ok[np.flatnonzero(m)] = np.isin(opd[m].astype("int64"), series[(int(key[0]), key[1])].opday_unique)
        ok &= ~np.isin(opd, EXCL_DAYS)
        ok &= wtype_code(opd) == wt_target[todo]
        for key in set(zip(stop[todo][ok], rich[todo][ok])):
            m = ok & (stop[todo] == key[0]) & (rich[todo] == key[1])
            hit = fp.hit_in_window(int(key[0]), key[1], utc[m].astype(float), -60, 240)
            ok[np.flatnonzero(m)] = ~hit
        tau[todo[ok]] = utc[ok]
        todo = todo[~ok]
    out = pd.DataFrame({
        "parent_episode": ep["episode_id"].to_numpy()[rows], "code": ep["code"].to_numpy()[rows],
        "stop_id": stop, "richtung": rich, "t0_s": tau, "wochentagstyp": WT_NAMES[wt_target],
    })
    n_fail = int(np.isnan(tau).sum())
    print(f"Placebo: {len(out):,} angefordert, {n_fail:,} ohne gueltigen Zeitpunkt nach {MAX_TRIES} Versuchen")
    out = out.dropna(subset=["t0_s"]).reset_index(drop=True)
    out["t0_s"] = out["t0_s"].astype("int64")
    return out


def measure(frame: pd.DataFrame, series: dict, fp: Footprints, own_ids: np.ndarray) -> pd.DataFrame:
    """Erholung (R0/R15; 180/360; Station/Strecke; mit/ohne Folgeepisoden-Zensierung)
    plus Schwere, Vorbelastung, Ausfallquote. frame: stop_id, richtung, t0_s (UTC-s)."""
    n = len(frame)
    res = {}
    t0 = frame["t0_s"].to_numpy().astype("int64")
    od0 = opday_day(t0).astype("int64")
    od15 = opday_day(t0 + 15 * MIN).astype("int64")
    stop = frame["stop_id"].to_numpy()
    rich = frame["richtung"].to_numpy()
    sev = np.full(n, np.nan); vor = np.full(n, np.nan); auq = np.full(n, np.nan)
    nsev = np.zeros(n, dtype=int); nauq = np.zeros(n, dtype=int)
    tn_st = np.full(n, np.inf); tn_dir = np.full(n, np.inf)
    keys = set(zip(stop, rich))
    for key in keys:
        m = (stop == key[0]) & (rich == key[1])
        s = series[(int(key[0]), key[1])]
        sev[m], vor[m], auq[m], nsev[m], nauq[m] = s.window_stats(t0[m])
        if key in fp.st:
            tn_st[m] = fp.next_after(fp.st[key], t0[m], own_ids[m])
    for d in ("Ost", "West"):
        m = rich == d
        tn_dir[m] = fp.next_after(fp.dir[d], t0[m], own_ids[m])
    res.update(opday_nr=od0, schwere_min=sev, vorbelastung_min=vor, ausfallquote=auq,
               n_zuege_15=nsev, n_zuege_ausf=nauq)

    for variant, shift in (("R0", 0), ("R15", 15 * MIN)):
        o = (t0 + shift).astype(float)
        od = od0 if shift == 0 else od15
        for thr in THRS:
            valid = np.zeros(n, bool); found = np.zeros(n, bool)
            T = np.full(n, np.nan); done = np.full(n, np.nan); end = np.full(n, np.nan)
            noeff = np.zeros(n, bool)
            for key in keys:
                m = (stop == key[0]) & (rich == key[1])
                valid[m], found[m], T[m], done[m], end[m], noeff[m] = series[(int(key[0]), key[1])].recover(o[m], od[m], thr)
            res[f"noeff_{variant}_{thr}"] = np.where(valid, noeff.astype(float), np.nan)
            for cens in ("std", "nofolge"):
                tn = tn_st if cens == "std" else np.full(n, np.inf)
                rec = found & (done < tn)
                time = np.where(rec, T, np.minimum(end, tn - o))
                bad = ~valid | ((time <= 0) & ~rec)
                res[f"t_{variant}_{thr}_st_{cens}"] = np.where(bad, np.nan, time / 60)
                res[f"e_{variant}_{thr}_st_{cens}"] = np.where(bad, np.nan, rec.astype(float))
            Ts = np.full((n, 12), np.nan); Ds = np.full((n, 12), np.nan); Es = np.full((n, 12), np.nan)
            Fs = np.zeros((n, 12), bool); Vs = np.zeros((n, 12), bool)
            for si, (sid, _) in enumerate(STATION_ORDER_WEST_OST):
                for d in ("Ost", "West"):
                    m = rich == d
                    if m.any():
                        v, f, t_, dn, en, _ = series[(sid, d)].recover(o[m], od[m], thr)
                        Vs[m, si] = v; Fs[m, si] = f; Ts[m, si] = t_; Ds[m, si] = dn; Es[m, si] = en
            nvalid = Vs.sum(axis=1)
            allfound = (Fs | ~Vs).all(axis=1) & (nvalid > 0)
            lower = np.where(Fs, Ts, np.where(Vs, Es, -np.inf)).max(axis=1)
            done_max = np.where(Fs, Ds, -np.inf).max(axis=1)
            for cens in ("std", "nofolge"):
                tn = tn_dir if cens == "std" else np.full(n, np.inf)
                rec = allfound & (done_max < tn)
                time = np.where(rec, lower, np.minimum(lower, tn - o))
                bad = (nvalid < 6) | ((time <= 0) & ~rec)
                res[f"t_{variant}_{thr}_ln_{cens}"] = np.where(bad, np.nan, time / 60)
                res[f"e_{variant}_{thr}_ln_{cens}"] = np.where(bad, np.nan, rec.astype(float))
    return pd.DataFrame(res, index=frame.index)

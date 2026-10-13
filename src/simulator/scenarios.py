"""Stufe 4, Schritt 6: Szenarien (Vergleich Szenario gegen Basislauf desselben Tages, gleiche Zufallszahlen)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import EXCL, OUT, ZW, load_trains
from .sim import INF, NS_MIN, Day, Scenario, SimResult, simulate

BUCKET = 10          # Minuten
AFFECT_MIN = 1.0     # "betroffen": |Verspaetungsaenderung am Pfadende| >= 1 min


def pick_trains(day: Day, hvz: bool, k: int, rng, line: str = "S3", rich: int = 0, max_delay: float = 3.0) -> list[int]:
    """Zuege, die am Stammstreckenbeginn (Pasing Ost / Leuchtenbergring West) eintreten, pünktlich sind
    (<= max_delay am Eintritt), in HVZ bzw. ausserhalb; bevorzugt Linie S3."""
    from .features import LINE_ID
    cand = []
    for n in range(day.N):
        e = int(day.entry_pos[n])
        if e < 0 or day.rich[n] != rich:
            continue
        if day.rank_at[n, e] != (0 if rich == 0 else 11):
            continue
        if not np.isfinite(day.delay_real[n, e]) or day.delay_real[n, e] > max_delay:
            continue
        if day.plen[n] - e < 8:
            continue
        tm = int(day.real_min[n, e])
        if bool(day.hvz_of_min[min(tm, day.M - 1)]) != hvz:
            continue
        if hvz and not (7.0 <= day.hour_of_min[tm] < 9.0 or 16.0 <= day.hour_of_min[tm] < 18.5):
            continue
        if not hvz and not (10.0 <= day.hour_of_min[tm] < 14.0):
            continue
        cand.append((day.line_at[n, e] != LINE_ID.get(line, 8), n))
    pref = [n for fl, n in cand if not fl]
    pool = pref if len(pref) >= k else [n for _, n in cand]
    if len(pool) == 0:
        return []
    pick = rng.choice(pool, size=min(k, len(pool)), replace=False)
    return [int(day.zug_ids[n]) for n in pick]


def window_for(day: Day, n: int, before: int = 60, after: int = 360) -> tuple[int, int]:
    e = int(day.entry_pos[n])
    t_inj = int(day.real_min[n, e])
    return max(t_inj - before, 0), min(t_inj + after, day.M - 1)


def effects(base: SimResult, scen: SimResult, t_inj: int, t_end: int, exclude_n: int | None, label: dict) -> tuple[dict, pd.DataFrame]:
    """Kennzahlen Szenario - Basis. t_inj/t_end: Tagesminuten. exclude_n: Zugindex des beeinflussten Zuges."""
    d = base.day
    R = base.arr_t.shape[0]
    N, P = d.N, d.P
    keep = np.ones(N, bool)
    if exclude_n is not None:
        keep[exclude_n] = False
    arrived_b = np.isfinite(base.delay) & (~base.canc) & (base.arr_t < t_end)
    arrived_s = np.isfinite(scen.delay) & (~scen.canc) & (scen.arr_t < t_end)
    both = arrived_b & arrived_s & keep[None, :, None]
    diff = np.where(both, scen.delay - base.delay, np.nan)
    tb = np.where(both, base.arr_t, INF)
    # ---- je Zug: Aenderung am Pfadende (letzter Halt) ----
    last = d.plen.astype(int) - 1
    idx = np.arange(N)
    end_diff = diff[:, idx, last]                                   # (R, N)
    with np.errstate(all="ignore"):
        end_mean = np.nanmean(end_diff, axis=0)
        n_valid = np.isfinite(end_diff).sum(0)
    ok_tr = (n_valid >= 0.5 * R) & keep
    aff_run = (np.abs(end_diff) >= AFFECT_MIN) & np.isfinite(end_diff)
    n_aff = aff_run.sum(1)
    mean_aff_tr = end_mean[ok_tr & (np.abs(end_mean) >= 0.25)]
    tot_extra = np.nansum(np.where(keep[None, :], end_diff, 0.0), axis=1)          # Zug-Minuten je Lauf (andere Zuege)
    # ---- Ausfaelle ----
    mod = (np.arange(P)[None, :] < d.plen[:, None]) & (np.arange(P)[None, :] > d.entry_pos[:, None]) & (d.entry_pos[:, None] >= 0)
    mod &= keep[:, None]
    c_b = (base.canc & mod[None]).sum((1, 2)); c_s = (scen.canc & mod[None]).sum((1, 2))
    extra_c = c_s - c_b
    # ---- Zeitverlauf: mittlere Differenz je 10-min-Block nach t_inj ----
    nb = int(np.ceil((t_end - t_inj) / BUCKET))
    bidx = np.clip((tb.astype(np.int64) - t_inj) // BUCKET, -1, nb)
    sums = np.zeros((R, nb)); cnts = np.zeros((R, nb))
    for r in range(R):
        m = both[r] & (bidx[r] >= 0) & (bidx[r] < nb)
        np.add.at(sums[r], bidx[r][m], diff[r][m])
        np.add.at(cnts[r], bidx[r][m], 1)
    with np.errstate(all="ignore"):
        per_run = sums / cnts                                       # (R, nb)
    mean_b = np.nanmean(per_run, axis=0)
    se = np.nanstd(per_run, axis=0, ddof=1) / np.sqrt(np.isfinite(per_run).sum(0))
    lo, hi = mean_b - 1.96 * se, mean_b + 1.96 * se
    # Abklingzeit: erster Block, ab dem 3 Bloecke in Folge das 95%-KI die 0 einschliesst (und danach nie mehr klar von 0 verschieden)
    nonzero = (lo > 0) | (hi < 0)
    nonzero = np.where(np.isfinite(nonzero.astype(float)), nonzero, False)
    peak_b = int(np.nanargmax(np.abs(mean_b))) if np.isfinite(mean_b).any() else 0
    decay = np.nan
    last_nz = np.flatnonzero(nonzero[peak_b:])
    if len(last_nz):
        decay = float((peak_b + last_nz[-1] + 1) * BUCKET)
    else:
        decay = float(peak_b * BUCKET)
    # Variante: Block-Mittel < 0.1 min
    small = np.abs(mean_b) < 0.1
    dec2 = np.nan
    for b in range(peak_b, nb - 2):
        if small[b:b + 3].all():
            dec2 = float(b * BUCKET); break
    curve = pd.DataFrame(dict(minuten_nach_eingriff=np.arange(nb) * BUCKET, diff_mittel=mean_b, ki_lo=lo, ki_hi=hi, ereignisse_pro_lauf=np.nanmean(cnts, axis=0)))
    for k_, v_ in label.items():
        curve[k_] = v_
    eff = dict(label)
    own = {}
    if exclude_n is not None:
        e_ = int(d.entry_pos[exclude_n])
        own_end = scen.delay[:, exclude_n, last[exclude_n]] - base.delay[:, exclude_n, last[exclude_n]]
        own = dict(eigener_zug_aenderung_ende_mittel=float(np.nanmean(own_end)) if np.isfinite(own_end).any() else np.nan,
                   eigener_zug_eintritt_verspaetung_basis=float(base.delay[0, exclude_n, e_]),
                   eigener_zug_eintritt_verspaetung_szen=float(scen.delay[0, exclude_n, e_]) if np.isfinite(scen.delay[0, exclude_n, e_]) else np.nan)
    eff.update(
        n_zuege_fenster=int(ok_tr.sum()),
        betroffen_pro_lauf_mittel=float(n_aff.mean()), betroffen_q05=float(np.percentile(n_aff, 5)), betroffen_q95=float(np.percentile(n_aff, 95)),
        zugaenderung_ende_n=int(len(mean_aff_tr)),
        zugaenderung_mittel=float(mean_aff_tr.mean()) if len(mean_aff_tr) else 0.0,
        zugaenderung_median=float(np.median(mean_aff_tr)) if len(mean_aff_tr) else 0.0,
        zugaenderung_q10=float(np.percentile(mean_aff_tr, 10)) if len(mean_aff_tr) else 0.0,
        zugaenderung_q90=float(np.percentile(mean_aff_tr, 90)) if len(mean_aff_tr) else 0.0,
        zugaenderung_max=float(mean_aff_tr.max()) if len(mean_aff_tr) else 0.0,
        zugaenderung_min=float(mean_aff_tr.min()) if len(mean_aff_tr) else 0.0,
        gesamt_mehrverspaetung_zugmin_mittel=float(tot_extra.mean()), gesamt_zugmin_q05=float(np.percentile(tot_extra, 5)), gesamt_zugmin_q95=float(np.percentile(tot_extra, 95)),
        zusaetzliche_ausfaelle_mittel=float(extra_c.mean()), zus_ausf_q05=float(np.percentile(extra_c, 5)), zus_ausf_q95=float(np.percentile(extra_c, 95)),
        spitze_diff_min=float(mean_b[peak_b]) if np.isfinite(mean_b).any() else np.nan, spitze_nach_min=float(peak_b * BUCKET),
        abklingzeit_ki_min=decay, abklingzeit_unter_0_1min=dec2, n_laeufe=R, **own)
    return eff, curve


def run_pair(day: Day, models, sc: Scenario, R: int, n_idx: int | None, t_start: int, t_end: int, seed: int = 21):
    base = simulate(day, models, R=R, seed=seed, scenario=None, t_start=t_start, t_end=t_end)
    scen = simulate(day, models, R=R, seed=seed, scenario=sc, t_start=t_start, t_end=t_end)
    return base, scen


def find_gaps(trains: pd.DataFrame, min_gap: int = 15) -> pd.DataFrame:
    """Faelle, in denen an (Station, Richtung) > min_gap min kein Zug ankam (tagsueber 05-23 Uhr Ortszeit),
    waehrend an der Vorstation im Luecken-Intervall Zuege ankamen."""
    t = trains[~trains["cancelled"]].copy()
    t["m"] = t["real"] // NS_MIN
    t = t.sort_values(["stop_id", "richtung", "m"])
    t["prev_m"] = t.groupby(["stop_id", "richtung"], observed=True)["m"].shift(1)
    t["gap"] = t["m"] - t["prev_m"]
    g = t[t["gap"] > min_gap].copy()
    loc = pd.to_datetime(g["prev_m"].astype("int64") * 60, unit="s", utc=True).dt.tz_convert("Europe/Berlin")
    locm = pd.to_datetime(g["m"].astype("int64") * 60, unit="s", utc=True).dt.tz_convert("Europe/Berlin")
    g = g[(loc.dt.hour >= 5).to_numpy() & (locm.dt.hour < 23).to_numpy() & (loc.dt.date == locm.dt.date).to_numpy()]
    rows = []
    allt = t.groupby(["rank", "richtung"], observed=True)["m"].apply(lambda s: np.sort(s.to_numpy()))
    for _, r in g.iterrows():
        rk, rich = int(r["rank"]), str(r["richtung"])
        up = rk - 1 if rich == "Ost" else rk + 1
        n_up = np.nan
        if 0 <= up <= 11:
            a = allt.get((up, rich))
            if a is not None:
                n_up = int(((a > r["prev_m"]) & (a < r["m"])).sum())
        rows.append(dict(opday=r["opday"], rank=rk, richtung=rich, t1_m=int(r["prev_m"]) + 1, t2_m=int(r["m"]),
                         luecke_min=int(r["gap"]), zuege_vorstation_in_luecke=n_up))
    return pd.DataFrame(rows)

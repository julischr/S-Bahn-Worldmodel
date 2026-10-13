"""Stufe 4, Schritt 7: einfache Schnittstelle und Bildfahrplan.

    from src.simulator.api import simulate_day
    tab, res = simulate_day("2025-11-12", {"art": "verspaetung", "zug": 123, "minuten": 10}, runs=200)

CLI:
    uv run python -m src.simulator.api 2025-11-12 --art verspaetung --linie S3 --uhrzeit 08:00 --minuten 10 \
        --runs 200 --plot outputs/simulator/plots/demo.png
Szenario-Arten: keins | verspaetung | ausfall | sperrung (--station RANG --von HH:MM --bis HH:MM).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .data import OUT, load_trains
from .model_io import Models
from .scenarios import window_for
from .sim import INF, NS_MIN, Day, Scenario, simulate

STATIONS = ["Pasing", "Laim", "Hirschgarten", "Donnersbergerbrücke", "Hackerbrücke", "Hauptbahnhof (tief)",
            "Karlsplatz (Stachus)", "Marienplatz", "Isartor", "Rosenheimer Platz", "Ostbahnhof", "Leuchtenbergring"]


def _hhmm_to_min(day: Day, hhmm: str) -> int:
    """Tageszeit (Ortszeit, HH:MM) -> Tagesminute im Day-Raster (naechster Treffer)."""
    h, m = map(int, hhmm.split(":"))
    target = h + m / 60
    hrs = day.hour_of_min
    cand = np.flatnonzero(np.abs(hrs - target) < 1e-3)
    if len(cand) == 0:
        raise ValueError("Uhrzeit ausserhalb des Betriebstages")
    # Betriebstag beginnt ~03:00; fuer 00:00-03:00 den spaeteren Treffer nehmen
    return int(cand[-1] if target < 3 else cand[0])


def find_train(day: Day, line: str, hhmm: str, rich: int = 0) -> int:
    """Zug-ID der Linie `line`, dessen Eintritt am naechsten an hhmm liegt."""
    from .features import LINE_ID
    t0 = _hhmm_to_min(day, hhmm)
    best, bn = 10**9, None
    for n in range(day.N):
        e = int(day.entry_pos[n])
        if e < 0 or day.rich[n] != rich or day.line_at[n, e] != LINE_ID.get(line, 8):
            continue
        dt = abs(int(day.real_min[n, e]) - t0)
        if dt < best:
            best, bn = dt, n
    if bn is None:
        raise ValueError(f"Kein Zug der Linie {line} gefunden")
    return int(day.zug_ids[bn])


def result_table(day: Day, res: SimResult_t, real=None) -> pd.DataFrame:
    """Tabelle je (Zug, Halt): Plan, echt, simulierte Bandbreite."""
    nn, qq = np.nonzero(np.arange(day.P)[None, :] < day.plen[:, None])
    arr = res.arr_t[:, nn, qq].astype(float)
    ok = (arr < INF) & np.isfinite(res.delay[:, nn, qq]) & ~res.canc[:, nn, qq]
    arr = np.where(ok, arr, np.nan)
    dl = np.where(ok, res.delay[:, nn, qq], np.nan)
    with np.errstate(all="ignore"):
        q = np.nanquantile(dl, [0.05, 0.5, 0.95], axis=0)
        qa = np.nanquantile(arr, [0.05, 0.5, 0.95], axis=0)
    tab = pd.DataFrame(dict(
        zug=day.zug_ids[nn], linie=[("S%d" % 1) if False else l for l in day.line_at[nn, qq]], richtung=np.where(day.rich[nn] == 0, "Ost", "West"),
        station_rang=day.rank_at[nn, qq].astype(int),
        station=[STATIONS[i] for i in day.rank_at[nn, qq].astype(int)],
        plan_min=day.sched_min[nn, qq] + day.m0,
        echt_verspaetung_min=day.delay_real[nn, qq], echt_ausfall=day.canc_real[nn, qq],
        sim_verspaetung_q05=q[0], sim_verspaetung_median=q[1], sim_verspaetung_q95=q[2],
        sim_ausfallwahrscheinlichkeit=1 - ok.mean(0),
        sim_ankunft_q05_min=qa[0] + day.m0, sim_ankunft_median_min=qa[1] + day.m0, sim_ankunft_q95_min=qa[2] + day.m0))
    from .features import LINES
    tab["linie"] = [LINES[i] for i in day.line_at[nn, qq]]
    tab["plan_uhrzeit"] = pd.to_datetime(tab["plan_min"] * 60, unit="s", utc=True).dt.tz_convert("Europe/Berlin").dt.strftime("%H:%M")
    return tab


def bildfahrplan(day: Day, res, fname, t_from: str = "07:30", t_to: str = "09:30", real=None, title="", base=None, rich=None):
    """Zeit-Weg-Diagramm: Plan (grau), echt (schwarz), Simulation (Median-Linie, 90-%-Band)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a, b = _hhmm_to_min(day, t_from), _hhmm_to_min(day, t_to)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    real = real or day.real_result()
    R_ = res.arr_t.shape[0]
    for n in range(day.N):
        if rich is not None and day.rich[n] != rich:
            continue
        pl = int(day.plen[n])
        ps = np.arange(pl)
        y = day.rank_at[n, :pl].astype(float)
        sm = day.sched_min[n, :pl]
        if sm.max() < a - 30 or sm.min() > b + 30:
            continue
        ax.plot(sm, y, color="#b0b0b0", lw=0.6, zorder=1)
        rm = real.arr_t[0, n, :pl].astype(float)
        okr = rm < INF
        if okr.sum() >= 2:
            ax.plot(np.where(okr, rm, np.nan), y, color="#222222", lw=0.9, zorder=3)
        arr = res.arr_t[:, n, :pl].astype(float)
        ok = (arr < INF) & ~res.canc[:, n, :pl] & np.isfinite(res.delay[:, n, :pl])
        arr = np.where(ok, arr, np.nan)
        if ok.any(0).sum() >= 2:
            with np.errstate(all="ignore"):
                lo, md, hi = np.nanquantile(arr, [0.05, 0.5, 0.95], axis=0)
            m = np.isfinite(md)
            ax.fill_betweenx(y[m], lo[m], hi[m], color="#d1495b", alpha=0.18, lw=0, zorder=2)
            ax.plot(md[m], y[m], color="#d1495b", lw=0.8, zorder=2)
        if base is not None:
            arr0 = base.arr_t[:, n, :pl].astype(float)
            ok0 = (arr0 < INF) & ~base.canc[:, n, :pl]
            arr0 = np.where(ok0, arr0, np.nan)
            with np.errstate(all="ignore"):
                md0 = np.nanmedian(arr0, axis=0)
            m0 = np.isfinite(md0)
            if m0.sum() >= 2:
                ax.plot(md0[m0], y[m0], color="#2a6f97", lw=0.7, ls="--", zorder=2)
    ax.set_xlim(a, b)
    ax.set_ylim(-0.3, 11.3)
    ax.set_yticks(range(12)); ax.set_yticklabels(STATIONS, fontsize=8)
    ticks = np.arange(a - a % 10 + 10, b, 10)
    lab = pd.to_datetime((ticks + day.m0) * 60, unit="s", utc=True).tz_convert("Europe/Berlin").strftime("%H:%M")
    ax.set_xticks(ticks); ax.set_xticklabels(lab, fontsize=8, rotation=45)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([0], [0], color="#b0b0b0", lw=1.2, label="Plan"), Line2D([0], [0], color="#222222", lw=1.2, label="echt"),
         Line2D([0], [0], color="#d1495b", lw=1.2, label="Simulation (Median)"), Patch(color="#d1495b", alpha=0.25, label=f"Simulation 90-%-Band ({R_} Läufe)")]
    if base is not None:
        h.append(Line2D([0], [0], color="#2a6f97", lw=1.2, ls="--", label="Basislauf (Median)"))
    ax.legend(handles=h, loc="upper right", fontsize=8)
    ax.set_title(title or f"Bildfahrplan {day.opday} {t_from}–{t_to} (Stationen äquidistant)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)


SimResult_t = object


def simulate_day(datum, szenario: dict | None = None, runs: int = 200, seed: int = 21, models: Models | None = None,
                 t_from: str | None = None, t_to: str | None = None):
    """simulate(datum, szenario) -> (Tabelle, Ergebnis). Szenario-Dict:
    {"art": "verspaetung", "linie": "S3", "uhrzeit": "08:00", "minuten": 10} oder mit "zug": ID;
    {"art": "ausfall", ...gleiche Zugangaben}; {"art": "sperrung", "station": 7, "von": "08:00", "bis": "08:30", "richtung": None}."""
    datum = pd.Timestamp(datum).date()
    trains = load_trains()
    day = Day(trains[trains["opday"] == datum], datum)
    models = models or Models()
    sc = None
    t_start = _hhmm_to_min(day, t_from) if t_from else None
    t_end = _hhmm_to_min(day, t_to) if t_to else None
    if szenario and szenario.get("art") in ("verspaetung", "ausfall"):
        zug = szenario.get("zug") or find_train(day, szenario.get("linie", "S3"), szenario.get("uhrzeit", "08:00"), szenario.get("richtung", 0))
        sc = Scenario(kind="delay" if szenario["art"] == "verspaetung" else "cancel", zug=int(zug), delay_abs_min=float(szenario.get("minuten", 10)))
    elif szenario and szenario.get("art") == "sperrung":
        sc = Scenario(kind="closure", rank=int(szenario["station"]), rich=szenario.get("richtung"),
                      t1=_hhmm_to_min(day, szenario["von"]) + day.m0, t2=_hhmm_to_min(day, szenario["bis"]) + day.m0)
    res = simulate(day, models, R=runs, seed=seed, scenario=sc, t_start=t_start, t_end=t_end)
    return result_table(day, res), res, day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datum")
    ap.add_argument("--art", default="keins", choices=["keins", "verspaetung", "ausfall", "sperrung"])
    ap.add_argument("--linie", default="S3"); ap.add_argument("--uhrzeit", default="08:00")
    ap.add_argument("--minuten", type=float, default=10); ap.add_argument("--zug", type=int, default=None)
    ap.add_argument("--station", type=int, default=7); ap.add_argument("--von", default="08:00"); ap.add_argument("--bis", default="08:30")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--von-zeit", default=None); ap.add_argument("--bis-zeit", default=None)
    ap.add_argument("--plot", default=None); ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    sz = None if a.art == "keins" else dict(art=a.art, linie=a.linie, uhrzeit=a.uhrzeit, minuten=a.minuten, zug=a.zug,
                                            station=a.station, von=a.von, bis=a.bis)
    tab, res, day = simulate_day(a.datum, sz, a.runs, t_from=a.von_zeit, t_to=a.bis_zeit)
    if a.csv:
        tab.to_csv(a.csv, index=False)
    if a.plot:
        bildfahrplan(day, res, a.plot)
    print(tab.head(20).to_string())


if __name__ == "__main__":
    main()

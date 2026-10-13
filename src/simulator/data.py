"""Stufe 4 (Simulator): Zugtabelle der Stammstrecke.

Eigene Implementierung (kopiert die Verbandsregel aus
src/analysis/survival/data.py, schreibt aber NICHTS nach outputs/survival).
Nur lesender Zugriff auf data/02_events/final/.

Einheit: ein Zug (Verband = ein Zug) an einer der 12 Stammstreckenstationen
(Ankunft). delay in den Rohdaten ist in SEKUNDEN; hier in Minuten (float).
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from src.analysis.var.run_all import EXCLUDED_DATES
from src.analysis.var.stations import STATION_ORDER_WEST_OST, STOP_ID_TO_NAME
from src.utils.constants import DATA_DIR, PROJECT_ROOT

EVENTS_GLOB = str(DATA_DIR / "02_events" / "final" / "*.parquet")
OUT = PROJECT_ROOT / "outputs" / "simulator"
ZW = OUT / "zwischen"
TZ = "Europe/Berlin"
EXCL = set(EXCLUDED_DATES)
STAMM_IDS = [s for s, _ in STATION_ORDER_WEST_OST]
RANK = {s: i for i, (s, _) in enumerate(STATION_ORDER_WEST_OST)}  # 0 Pasing .. 11 Leuchtenbergring
HVZ_HOURS = set(range(6, 10)) | set(range(15, 20))

FEIERTAGE = set(pd.to_datetime([
    "2025-01-01", "2025-01-06", "2025-04-18", "2025-04-21",
    "2025-05-01", "2025-05-29", "2025-06-09", "2025-06-19",
    "2025-10-03", "2025-11-01", "2025-12-25", "2025-12-26",
]).date)


def opday_of(ts_utc: pd.Series) -> pd.Series:
    """Betriebstag = lokale Zeit minus 3h."""
    return (ts_utc.dt.tz_convert(TZ) - pd.Timedelta(hours=3)).dt.date


def weekday_type(d) -> int:
    """0 Mo-Do, 1 Fr, 2 Sa, 3 So/Feiertag."""
    if d in FEIERTAGE or d.weekday() == 6:
        return 3
    return {5: 2, 4: 1}.get(d.weekday(), 0)


def load_arrivals() -> pd.DataFrame:
    ids = ",".join(map(str, STAMM_IDS))
    df = duckdb.connect().execute(f"""
        SELECT stop_id, richtung, trip_id, line, time_schedule, time_real, delay, is_cancelled
        FROM read_parquet('{EVENTS_GLOB}')
        WHERE is_arrival = true AND richtung IS NOT NULL AND stop_id IN ({ids})
    """).fetchdf()
    for c in ("time_schedule", "time_real"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


def convoy_map(df: pd.DataFrame) -> pd.Series:
    """trip_id -> Verband-ID. Regel wie Stufe 3: gleiche Planzeit an >= 2
    gemeinsamen Stationen (gleiche Richtung)."""
    key = df[["stop_id", "richtung", "time_schedule", "trip_id"]].copy()
    key["ts"] = key["time_schedule"].astype("int64")
    grp = key.groupby(["stop_id", "richtung", "ts"])["trip_id"].agg(lambda s: sorted(set(s)))
    grp = grp[grp.map(len) > 1]
    pair_counts: dict[tuple[int, int], int] = {}
    for lst in grp:
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                p = (lst[i], lst[j])
                pair_counts[p] = pair_counts.get(p, 0) + 1
    parent: dict[int, int] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), c in pair_counts.items():
        if c >= 2:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    return pd.Series({t: find(t) for t in list(parent)}, dtype="int64")


def normalize_paths(out: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Macht die Zugtabelle simulatorkompatibel (gleiche Regeln in Training und Simulator):
    (1) je Zug nur das laengste zusammenhaengende Pfadsegment (Rang-Differenz 1), (2) Ausfall
    ist absorbierend: ab dem ersten Ausfall nach dem ersten gefahrenen Halt gelten alle
    weiteren Halte als ausgefallen (Wiederaufnahme der Fahrt in den Daten wird verworfen)."""
    out = out.copy()
    out["pos"] = np.where(out["richtung"] == "Ost", out["rank"], -out["rank"].astype(int)).astype("int16")
    out = out.sort_values(["zug", "pos"]).reset_index(drop=True)
    z = out["zug"].to_numpy()
    pos = out["pos"].to_numpy()
    brk = np.r_[True, (z[1:] != z[:-1]) | (pos[1:] - pos[:-1] != 1)]
    out["seg"] = np.cumsum(brk)
    out["seglen"] = out.groupby("seg")["zug"].transform("size")
    best = out.groupby("zug")["seglen"].transform("max")
    cand = out["seglen"] == best
    first_seg = out["seg"].where(cand).groupby(out["zug"]).transform("min")
    keep = cand & (out["seg"] == first_seg)
    n_gap = int((~keep).sum())
    out = out[keep].copy()
    ran = ~out["cancelled"]
    seen_run = ran.groupby(out["zug"]).cummax()
    canc_after = seen_run & out["cancelled"]
    dead = canc_after.groupby(out["zug"]).cummax()
    resumed = dead & ran
    n_resumed = int(resumed.sum())
    n_zug_resumed = int(out.loc[resumed, "zug"].nunique())
    out.loc[dead, "cancelled"] = True
    out.loc[dead, "delay_min"] = np.nan
    out = out.drop(columns=["seg", "seglen"]).reset_index(drop=True)
    info = dict(norm_zeilen_luecke_entfernt=n_gap, norm_zeilen_wiederaufnahme_verworfen=n_resumed,
                norm_zuege_mit_wiederaufnahme=n_zug_resumed)
    return out, info


def build_trains() -> pd.DataFrame:
    """Eine Zeile je (Zug, Station). Spalten: zug, stop_id, rank, richtung,
    sched/real (int64 ns UTC), delay_min (NaN bei Ausfall), cancelled, line, opday."""
    df = load_arrivals()
    n_raw = len(df)
    cmap = convoy_map(df)
    df["zug"] = df["trip_id"].map(cmap).fillna(df["trip_id"]).astype("int64")
    df["ok_delay"] = (df["delay"] / 60.0).where(~df["is_cancelled"])
    # Linie: S-Praefix vereinheitlichen
    df["line"] = df["line"].astype(str).str.replace(r"^S?", "S", regex=True)
    g = df.groupby(["stop_id", "richtung", "zug"], sort=False)
    out = g.agg(
        time_schedule=("time_schedule", "min"),
        time_real=("time_real", "min"),
        delay_min=("ok_delay", "max"),
        n_members=("trip_id", "size"),
        cancelled=("is_cancelled", "all"),
        line=("line", "first"),
    ).reset_index()
    out["delay_min"] = out["delay_min"].where(~out["cancelled"])
    out["opday"] = opday_of(out["time_schedule"])
    n_all = len(out)
    out = out[~out["opday"].isin(EXCL)].copy()
    out["sched"] = out["time_schedule"].dt.as_unit("ns").astype("int64")
    out["real"] = out["time_real"].dt.as_unit("ns").astype("int64")
    out["rank"] = out["stop_id"].map(RANK).astype("int8")
    out["richtung"] = out["richtung"].astype("category")
    out = out.drop(columns=["time_schedule", "time_real"])
    out, norm_info = normalize_paths(out)
    info = dict(ankuenfte_roh=n_raw, zug_station_nach_verband=n_all,
                zug_station_final=len(out), zuege_final=out["zug"].nunique(), **norm_info)
    print(info)
    pd.Series(info).to_csv(ZW / "trains_info.csv", header=["wert"])
    return out


def load_trains(rebuild: bool = False) -> pd.DataFrame:
    p = ZW / "trains.parquet"
    if p.exists() and not rebuild:
        return pd.read_parquet(p)
    t = build_trains()
    t.to_parquet(p)
    return t


if __name__ == "__main__":
    load_trains(rebuild=True)

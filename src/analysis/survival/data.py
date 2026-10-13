"""Stufe 3 (Survival): Datenaufbereitung. Zuege (mit Verband-Zusammenfassung),
Stoerungsmeldungen, Episoden. Siehe outputs/survival/auswertungsplan.md.
Nur lesender Zugriff auf data/02_events/final/.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from src.analysis.var.run_all import EXCLUDED_DATES
from src.analysis.var.stations import STOP_ID_TO_NAME
from src.utils.constants import DATA_DIR, PROJECT_ROOT

EVENTS_GLOB = str(DATA_DIR / "02_events" / "final" / "*.parquet")
OUT = PROJECT_ROOT / "outputs" / "survival"
ZW = OUT / "zwischen"

STAMM_IDS = list(STOP_ID_TO_NAME.keys())
FOLGECODES = {43, 44, 45, 46, 48}
AUSLOESER = [c for c in range(2, 70) if c not in FOLGECODES]
CLUSTER_GAP_MIN = 15
HVZ_HOURS = set(range(6, 10)) | set(range(15, 20))
TZ = "Europe/Berlin"

FEIERTAGE = set(pd.to_datetime([
    "2025-01-01", "2025-01-06", "2025-04-18", "2025-04-21",
    "2025-05-01", "2025-05-29", "2025-06-09", "2025-06-19",
    "2025-10-03", "2025-11-01", "2025-12-25", "2025-12-26",
    "2026-01-01",
]).date)
EXCL = set(EXCLUDED_DATES)


def ns(s: pd.Series) -> np.ndarray:
    """datetime64 (beliebige Einheit, tz-aware) -> int64 Nanosekunden seit Epoche."""
    return s.dt.as_unit("ns").astype("int64").to_numpy()


def opday_of(ts_utc: pd.Series) -> pd.Series:
    """Betriebstag = lokale Zeit minus 3h, als Datum."""
    return (ts_utc.dt.tz_convert(TZ) - pd.Timedelta(hours=3)).dt.date


def weekday_type_of_date(d) -> str:
    if d in FEIERTAGE or d.weekday() == 6:
        return "So+Feiertag"
    if d.weekday() == 5:
        return "Sa"
    if d.weekday() == 4:
        return "Fr"
    return "Mo-Do"


# ---------------------------------------------------------------------------
# Zuege
# ---------------------------------------------------------------------------

def load_arrivals() -> pd.DataFrame:
    ids = ",".join(map(str, STAMM_IDS))
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT stop_id, richtung, trip_id, time_schedule, time_real, delay, is_cancelled
        FROM read_parquet('{EVENTS_GLOB}')
        WHERE is_arrival = true AND richtung IS NOT NULL AND stop_id IN ({ids})
    """).fetchdf()
    for c in ("time_schedule", "time_real"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


def convoy_map(df: pd.DataFrame) -> pd.Series:
    """trip_id -> Verband-ID (kleinste trip_id der Komponente). Verband: gleiche
    time_schedule an >= 2 gemeinsamen Stationen (gleiche Richtung)."""
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

    n_pairs = 0
    for (a, b), c in pair_counts.items():
        if c >= 2:
            n_pairs += 1
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    m = pd.Series({t: find(t) for t in list(parent)}, dtype="int64")
    info = dict(n_pairs_total=len(pair_counts), n_pairs_convoy=n_pairs,
                n_trips_in_convoy=len(m), n_convoys=m.nunique())
    print("Verbaende:", info)
    pd.Series(info).to_csv(OUT / "verbaende_info.csv", header=["wert"])
    return m


def build_trains() -> pd.DataFrame:
    df = load_arrivals()
    n_raw = len(df)
    cmap = convoy_map(df)
    df["zug"] = df["trip_id"].map(cmap).fillna(df["trip_id"]).astype("int64")
    df["ok_delay"] = df["delay"].where(~df["is_cancelled"])
    g = df.groupby(["stop_id", "richtung", "zug"], sort=False)
    out = g.agg(
        time_schedule=("time_schedule", "min"),
        time_real=("time_real", "min"),
        delay=("ok_delay", "max"),
        n_members=("trip_id", "size"),
        all_cancelled=("is_cancelled", "all"),
    ).reset_index()
    out["cancelled"] = out["all_cancelled"]
    out["delay"] = out["delay"].where(~out["cancelled"])
    out["opday"] = opday_of(out["time_schedule"])
    n_excl = out["opday"].isin(EXCL).sum()
    out = out[~out["opday"].isin(EXCL)].copy()
    out = out.sort_values(["stop_id", "richtung", "time_schedule"]).reset_index(drop=True)
    print(f"Ankuenfte roh {n_raw:,} -> Zuege {len(out) + n_excl:,} (nach Verband); "
          f"{n_excl:,} auf ausgeschlossenen Betriebstagen entfernt -> {len(out):,}")
    pd.Series(dict(ankuenfte_roh=n_raw, zuege_nach_verband=len(out) + n_excl,
                   zuege_ausgeschlossene_tage=int(n_excl), zuege_final=len(out),
                   zuege_ausgefallen=int(out["cancelled"].sum()))).to_csv(
        OUT / "zuege_info.csv", header=["wert"])
    return out.drop(columns=["all_cancelled"])


# ---------------------------------------------------------------------------
# Meldungen und Episoden
# ---------------------------------------------------------------------------

def load_hits() -> pd.DataFrame:
    ids = ",".join(map(str, STAMM_IDS))
    codes = ",".join(map(str, AUSLOESER))
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT stop_id, richtung, trip_id, update_timestamp, time_real, code FROM (
          SELECT stop_id, richtung, trip_id, update_timestamp, time_real, unnest(message_codes) AS code
          FROM read_parquet('{EVENTS_GLOB}')
          WHERE is_arrival = true AND stop_id IN ({ids}) AND update_timestamp IS NOT NULL
        ) WHERE code IN ({codes})
    """).fetchdf()
    df["update_timestamp"] = pd.to_datetime(df["update_timestamp"], utc=True)
    df["time_real"] = pd.to_datetime(df["time_real"], utc=True)
    df["opday"] = opday_of(df["update_timestamp"])
    n = len(df)
    df = df[~df["opday"].isin(EXCL)].copy()
    print(f"Ausloeser-Meldungen: {n:,}, nach Ausschluss Betriebstage {len(df):,}")
    return df.sort_values("update_timestamp").reset_index(drop=True)


def build_episodes(hits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gibt (episoden, hits mit episode_id) zurueck."""
    h = hits.sort_values(["code", "stop_id", "update_timestamp"]).reset_index(drop=True)
    gap = h.groupby(["code", "stop_id"])["update_timestamp"].diff().dt.total_seconds() / 60
    newc = gap.isna() | (gap > CLUSTER_GAP_MIN)
    h["stationscluster"] = newc.cumsum()
    sc = h.groupby("stationscluster").agg(
        code=("code", "first"), stop_id=("stop_id", "first"),
        t0=("update_timestamp", "min")).reset_index()
    sc = sc.sort_values(["code", "t0", "stop_id"]).reset_index(drop=True)
    g2 = sc.groupby("code")["t0"].diff().dt.total_seconds() / 60
    sc["episode_id"] = (g2.isna() | (g2 > CLUSTER_GAP_MIN)).cumsum() - 1
    h = h.merge(sc[["stationscluster", "episode_id"]], on="stationscluster")
    h = h.sort_values(["episode_id", "update_timestamp", "stop_id"]).reset_index(drop=True)

    first = h.groupby("episode_id").first()
    last_t = h.groupby("episode_id")["update_timestamp"].max()
    maj = h.dropna(subset=["richtung"]).groupby(["episode_id", "richtung"]).size().unstack(fill_value=0)
    maj_dir = maj.idxmax(axis=1)
    tie = (maj.nunique(axis=1) == 1) & (maj.shape[1] > 1)
    first_dir = first["richtung"]
    maj_dir = maj_dir.where(~tie, first_dir.reindex(maj_dir.index))
    ep = pd.DataFrame({
        "episode_id": first.index,
        "code": first["code"].values,
        "t0": first["update_timestamp"].values,
        "stop_id": first["stop_id"].values,
    })
    ep["t0"] = pd.to_datetime(ep["t0"], utc=True)
    ep["richtung"] = ep["episode_id"].map(maj_dir)
    ep = ep.dropna(subset=["richtung"]).reset_index(drop=True)
    ep["melde_ende"] = ep["episode_id"].map(last_t)
    ep["meldedauer_min"] = (ep["melde_ende"] - ep["t0"]).dt.total_seconds() / 60
    ep["n_fahrten"] = ep["episode_id"].map(h.groupby("episode_id")["trip_id"].nunique())
    t0map = ep.set_index("episode_id")["t0"]
    h["t0_ep"] = h["episode_id"].map(t0map)
    h15 = h[(h["update_timestamp"] - h["t0_ep"]) < pd.Timedelta(minutes=15)]
    ep["n_fahrten_15"] = ep["episode_id"].map(h15.groupby("episode_id")["trip_id"].nunique()).fillna(0)
    ep["n_stationen"] = ep["episode_id"].map(h.groupby("episode_id")["stop_id"].nunique())
    ep["lag_erste_meldung_min"] = ((first["update_timestamp"] - first["time_real"]).dt.total_seconds() / 60).reindex(ep["episode_id"]).to_numpy()

    loc = ep["t0"].dt.tz_convert(TZ)
    ep["stunde"] = loc.dt.hour + loc.dt.minute / 60
    ep["hvz"] = loc.dt.hour.isin(HVZ_HOURS)
    ep["monat"] = loc.dt.month
    ep["opday"] = opday_of(ep["t0"])
    ep["wochentagstyp"] = ep["opday"].map(weekday_type_of_date)
    ep["station"] = ep["stop_id"].map(STOP_ID_TO_NAME)
    return ep, h


def add_stoerungslast(ep: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    """Zahl verschiedener ANDERER Ausloesercodes mit Meldung (update_timestamp) in
    [t0-30, t0], beliebige Station/Richtung der Stammstrecke (Plan, Abschnitt 2)."""
    hs = hits.sort_values("update_timestamp")
    ts = ns(hs["update_timestamp"])
    codes = hs["code"].to_numpy()
    t0 = ns(ep["t0"])
    w = 30 * 60 * 10**9
    lo = np.searchsorted(ts, t0 - w, side="left")
    hi = np.searchsorted(ts, t0, side="right")
    own = ep["code"].to_numpy()
    load = np.empty(len(ep), dtype=int)
    for i in range(len(ep)):
        s = set(codes[lo[i]:hi[i]])
        s.discard(own[i])
        load[i] = len(s)
    ep["stoerungslast"] = load
    return ep

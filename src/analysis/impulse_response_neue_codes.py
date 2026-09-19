"""
Impulsantwort-Analyse für seltenere, gravierendere Codes -- Vergleich zu
Code 34. Gleiche Grundpipeline wie impulse_response.py, aber:
  - Fenster [-60,+120] statt [-60,+240] (kleinere Fallzahl je Code)
  - drei Codes: 5 (Ärztliche Versorgung, Hauptfall, 522 Episoden lt.
    Störungsinventur), 8 (Notarzteinsatz, Vergleich, 117 Episoden), 2
    (Polizeieinsatz, zusätzlich angefragt, 603 Episoden)

Da mehrere Codes mit demselben Fenster in einem Prozess laufen, werden
cluster_events/load_state_vector/Event/weekday_type aus impulse_response.py
wiederverwendet (code-/fensterunabhängig), aber load_code_hits,
drop_overlapping, find_controls und curve_for hier parametrisiert neu
geschrieben (kein Monkeypatching über mehrere Codes hinweg). Original-Datei
und Original-Ergebnisse für Code 34 bleiben unverändert.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from src.analysis.impulse_response import (
    STAMMSTRECKE_IDS,
    Event,
    cluster_events,
    load_state_vector,
    weekday_type,
)
from src.analysis.impulse_response_diagnostics import STATION_NAMES
from src.utils.constants import DATA_DIR, OUTPUTS_DIR, PLOTS_DIR

EVENTS_FINAL_GLOB = str(DATA_DIR / "02_events" / "final" / "*.parquet")

PRE_MIN = -60
POST_MIN = 120
K_CONTROLS = 10
DATE_TOL_WEEKS = 6
RNG_SEED = 42
N_BOOTSTRAP = 1000
REL_MINUTES = np.arange(PRE_MIN, POST_MIN + 1, 5)

CODES = {5: "Ärztliche Versorgung eines Fahrgastes (Hauptfall)",
         8: "Notarzteinsatz auf der Strecke (Vergleich)",
         2: "Polizeieinsatz (zusätzlich)"}


def load_code_hits(code: int) -> pd.DataFrame:
    con = duckdb.connect()
    stop_ids_sql = ",".join(str(x) for x in STAMMSTRECKE_IDS)
    df = con.execute(
        f"""
        SELECT stop_id, richtung, trip_id, update_timestamp
        FROM read_parquet('{EVENTS_FINAL_GLOB}')
        WHERE is_arrival = true
          AND stop_id IN ({stop_ids_sql})
          AND list_contains(message_codes, {code})
        """
    ).fetchdf()
    df["update_timestamp"] = pd.to_datetime(df["update_timestamp"], utc=True)
    return df.dropna(subset=["update_timestamp"]).sort_values(["stop_id", "update_timestamp"])


def drop_overlapping(events: list[Event]) -> tuple[list[Event], int]:
    by_key: dict[tuple[int, str], list[Event]] = {}
    for e in events:
        by_key.setdefault((e.stop_id, e.richtung), []).append(e)
    clean, n_dropped = [], 0
    for key, group in by_key.items():
        group = sorted(group, key=lambda e: e.t0)
        t0s = pd.Series([e.t0 for e in group])
        for i, e in enumerate(group):
            window_start = e.t0 + pd.Timedelta(minutes=PRE_MIN)
            window_end = e.t0 + pd.Timedelta(minutes=POST_MIN)
            others = t0s.drop(i)
            if ((others >= window_start) & (others <= window_end)).any():
                n_dropped += 1
            else:
                clean.append(e)
    return clean, n_dropped


def find_controls(event: Event, all_events_by_key: dict, rng: np.random.Generator) -> list[pd.Timestamp]:
    e_type = weekday_type(event.t0)
    e_minute_of_day = event.t0.hour * 60 + event.t0.minute
    key = (event.stop_id, event.richtung)
    conflicting_t0s = pd.Series([e.t0 for e in all_events_by_key.get(key, [])])
    date_range = pd.date_range(event.t0.normalize() - pd.Timedelta(weeks=DATE_TOL_WEEKS),
                                event.t0.normalize() + pd.Timedelta(weeks=DATE_TOL_WEEKS),
                                freq="D", tz="UTC")
    candidates = []
    for day in date_range:
        if weekday_type(day) != e_type:
            continue
        candidate_t0 = day + pd.Timedelta(minutes=e_minute_of_day)
        if abs((candidate_t0 - event.t0).total_seconds()) < 1:
            continue
        candidates.append(candidate_t0)
    rng.shuffle(candidates)
    chosen = []
    for c in candidates:
        window_start = c + pd.Timedelta(minutes=PRE_MIN)
        window_end = c + pd.Timedelta(minutes=POST_MIN)
        if not ((conflicting_t0s >= window_start) & (conflicting_t0s <= window_end)).any():
            chosen.append(c)
        if len(chosen) >= K_CONTROLS:
            break
    return chosen


def curve_for(sv: pd.Series, stop_id: int, richtung: str, t0: pd.Timestamp) -> pd.Series:
    windows = [t0.floor("5min") + pd.Timedelta(minutes=int(m)) for m in REL_MINUTES]
    vals = pd.Series(index=REL_MINUTES, dtype=float)
    for m, w in zip(REL_MINUTES, windows):
        try:
            vals[m] = sv.loc[(stop_id, richtung, w)]
        except KeyError:
            vals[m] = np.nan
    return vals


def run_pipeline(code: int, sv) -> dict:
    hits = load_code_hits(code)
    all_events = cluster_events(hits)
    clean_events, n_dropped = drop_overlapping(all_events)
    by_key: dict[tuple[int, str], list[Event]] = {}
    for e in all_events:
        by_key.setdefault((e.stop_id, e.richtung), []).append(e)

    rng = np.random.default_rng(RNG_SEED)
    events_with_controls = []
    for e in clean_events:
        controls = find_controls(e, by_key, rng)
        if len(controls) >= K_CONTROLS:
            events_with_controls.append((e, controls))

    event_rows, control_rows, diff_rows, meta = [], [], [], []
    for e, controls in events_with_controls:
        ec = curve_for(sv, e.stop_id, e.richtung, e.t0)
        cc = pd.concat([curve_for(sv, e.stop_id, e.richtung, c) for c in controls], axis=1).mean(axis=1)
        event_rows.append(ec); control_rows.append(cc); diff_rows.append(ec - cc)
        meta.append({"event_id": e.event_id, "stop_id": e.stop_id, "richtung": e.richtung,
                     "t0": e.t0, "n_trips": e.n_trips, "t0_date": e.t0.date()})
    event_df = pd.DataFrame(event_rows); event_df.columns = REL_MINUTES
    control_df = pd.DataFrame(control_rows); control_df.columns = REL_MINUTES
    diff_df = pd.DataFrame(diff_rows); diff_df.columns = REL_MINUTES
    meta_df = pd.DataFrame(meta)

    print(f"Code {code}: {len(hits):,} Zeilen, {len(all_events)} Ereignisse, "
          f"{n_dropped} verworfen, {len(clean_events)} sauber, "
          f"{len(events_with_controls)} mit Kontrollen")

    return dict(code=code, hits=hits, all_events=all_events, clean_events=clean_events,
                events_with_controls=events_with_controls, event_df=event_df,
                control_df=control_df, diff_df=diff_df, meta_df=meta_df)


def bootstrap_ci(diff_df, meta_df, seed=RNG_SEED + 1):
    mean_curve = diff_df.mean(axis=0, skipna=True)
    days_arr = meta_df["t0_date"].values
    unique_days = np.unique(days_arr)
    rng = np.random.default_rng(seed)
    day_to_rows = {d: np.where(days_arr == d)[0] for d in unique_days}
    boot_means = np.zeros((N_BOOTSTRAP, len(REL_MINUTES)))
    for b in range(N_BOOTSTRAP):
        sampled_days = rng.choice(unique_days, size=len(unique_days), replace=True)
        rows = np.concatenate([day_to_rows[d] for d in sampled_days])
        boot_means[b] = diff_df.iloc[rows].mean(axis=0, skipna=True).values
    ci_low = np.nanpercentile(boot_means, 2.5, axis=0)
    ci_high = np.nanpercentile(boot_means, 97.5, axis=0)
    return mean_curve.values, ci_low, ci_high, len(unique_days)


def duration_for_events(hits, meta_df):
    hits_by_stop = {k: v.sort_values("update_timestamp") for k, v in hits.groupby("stop_id")}
    durations = {}
    for _, row in meta_df.iterrows():
        g = hits_by_stop[row["stop_id"]]
        window = g[(g["update_timestamp"] >= row["t0"]) & (g["update_timestamp"] <= row["t0"] + pd.Timedelta(hours=6))]
        gap = window["update_timestamp"].diff().dt.total_seconds() / 60
        if len(gap):
            gap.iloc[0] = 0
        break_idx = (gap > 15).idxmax() if (gap > 15).any() else None
        cluster_rows = window.loc[:break_idx].iloc[:-1] if break_idx is not None and gap.loc[break_idx] > 15 else window
        t_end = cluster_rows["update_timestamp"].max()
        durations[row["event_id"]] = (t_end - row["t0"]).total_seconds() / 60
    return meta_df["event_id"].map(durations)

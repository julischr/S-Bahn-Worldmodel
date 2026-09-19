"""
Diagnose: Sommerzeit im Kontroll-Matching der Impulsantwort (Code 34).

find_controls() in impulse_response.py matcht Kontrollfenster auf
Uhrzeit (Stunde:Minute) OHNE Zeitzonenumrechnung -- die Uhrzeit wird direkt
aus dem UTC-Zeitstempel von t0 gelesen (`event.t0.hour`, `event.t0.minute`).
Über eine Sommerzeitumstellung hinweg (30.03. und 26.10.2025) bedeutet das:
ein Ereignis um 8 Uhr Ortszeit (UTC+2 im Sommer) wird ggf. gegen ein
Kontrollfenster um 8 Uhr UTC (= 9 Uhr Ortszeit im Winter) gematcht, wenn
die beiden Zeitpunkte auf verschiedenen Seiten der Umstellung liegen.

Dieses Skript rekonstruiert exakt dieselben Ereignisse/Kontrollen wie im
Original (gleicher Seed), markiert für jedes Ereignis-Kontroll-Paar, ob
beide Zeitpunkte denselben Berlin-UTC-Offset haben oder nicht, und
vergleicht die Sockelhöhe [-60,-15] zwischen beiden Gruppen. Nur bei einem
gefundenen Unterschied wird zusätzlich eine lokalzeit-basierte
Kontrollauswahl gerechnet -- als eigene Datei, die Originalergebnisse
bleiben unverändert.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.impulse_response import (
    CODE,
    K_CONTROLS,
    POST_MIN,
    PRE_MIN,
    RNG_SEED,
    Event,
    cluster_events,
    curve_for,
    drop_overlapping,
    find_controls,
    load_code_hits,
    load_state_vector,
    weekday_type,
)
from src.utils.constants import OUTPUTS_DIR

BERLIN = "Europe/Berlin"


def reconstruct_original() -> tuple[list[tuple[Event, list]], dict]:
    hits = load_code_hits()
    all_events = cluster_events(hits)
    clean_events, _ = drop_overlapping(all_events)
    by_key: dict[tuple[int, str], list[Event]] = {}
    for e in all_events:
        by_key.setdefault((e.stop_id, e.richtung), []).append(e)

    rng = np.random.default_rng(RNG_SEED)
    events_with_controls = []
    for e in clean_events:
        controls = find_controls(e, by_key, rng)
        if len(controls) >= K_CONTROLS:
            events_with_controls.append((e, controls))
    return events_with_controls, by_key


def berlin_offset(ts: pd.Timestamp) -> pd.Timedelta:
    return ts.tz_convert(BERLIN).utcoffset()


def analyse_straddling(events_with_controls, sv) -> pd.DataFrame:
    rows = []
    for e, controls in events_with_controls:
        off_event = berlin_offset(e.t0)
        event_curve = curve_for(sv, e.stop_id, e.richtung, e.t0)
        event_baseline = event_curve.loc[-60:-15].mean()
        for c in controls:
            off_control = berlin_offset(c)
            straddling = off_event != off_control
            control_curve = curve_for(sv, e.stop_id, e.richtung, c)
            control_baseline = control_curve.loc[-60:-15].mean()
            rows.append({
                "event_id": e.event_id, "t0": e.t0, "control_t0": c,
                "straddling_dst": straddling,
                "event_baseline": event_baseline, "control_baseline": control_baseline,
                "paar_diff_baseline": event_baseline - control_baseline,
            })
    return pd.DataFrame(rows)


def local_weekday_type(ts_local: pd.Timestamp) -> str:
    return weekday_type(ts_local)  # weekday_type nutzt nur .date()/.weekday(), tz-agnostisch


def find_controls_localtime(event: Event, all_events_by_key: dict, rng: np.random.Generator,
                             k: int = K_CONTROLS) -> list[pd.Timestamp]:
    """Wie find_controls(), aber Wochentagstyp und Uhrzeit werden in
    Europe/Berlin bestimmt, nicht aus dem rohen UTC-Zeitstempel."""
    t0_local = event.t0.tz_convert(BERLIN)
    e_type = local_weekday_type(t0_local)
    e_minute_of_day = t0_local.hour * 60 + t0_local.minute
    key = (event.stop_id, event.richtung)
    conflicting_t0s = pd.Series([e.t0 for e in all_events_by_key.get(key, [])])

    local_date_range = pd.date_range(
        t0_local.normalize() - pd.Timedelta(weeks=6),
        t0_local.normalize() + pd.Timedelta(weeks=6),
        freq="D", tz=BERLIN,
    )
    candidates_utc = []
    for day_local in local_date_range:
        if local_weekday_type(day_local) != e_type:
            continue
        candidate_local = day_local + pd.Timedelta(minutes=e_minute_of_day)
        try:
            candidate_utc = candidate_local.tz_convert("UTC") if candidate_local.tzinfo else None
        except Exception:
            continue
        if candidate_utc is None:
            continue
        if abs((candidate_utc - event.t0).total_seconds()) < 1:
            continue
        candidates_utc.append(candidate_utc)
    rng.shuffle(candidates_utc)

    chosen = []
    for c in candidates_utc:
        window_start = c + pd.Timedelta(minutes=PRE_MIN)
        window_end = c + pd.Timedelta(minutes=POST_MIN)
        overlap = ((conflicting_t0s >= window_start) & (conflicting_t0s <= window_end)).any()
        if not overlap:
            chosen.append(c)
        if len(chosen) >= k:
            break
    return chosen


def run_localtime_variant(clean_events: list[Event], by_key: dict, sv) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    events_with_controls = []
    for e in clean_events:
        controls = find_controls_localtime(e, by_key, rng)
        if len(controls) >= K_CONTROLS:
            events_with_controls.append((e, controls))

    rel_minutes = np.arange(PRE_MIN, POST_MIN + 1, 5)
    diff_curves = []
    event_days = []
    for e, controls in events_with_controls:
        event_curve = curve_for(sv, e.stop_id, e.richtung, e.t0)
        control_curves = [curve_for(sv, e.stop_id, e.richtung, c) for c in controls]
        control_mean = pd.concat(control_curves, axis=1).mean(axis=1)
        diff_curves.append(event_curve - control_mean)
        event_days.append(e.t0.date())
    diff_df = pd.DataFrame(diff_curves)
    diff_df.columns = rel_minutes
    mean_curve = diff_df.mean(axis=0, skipna=True)

    days_arr = np.array(event_days)
    unique_days = np.unique(days_arr)
    rng2 = np.random.default_rng(RNG_SEED + 1)
    day_to_rows = {d: np.where(days_arr == d)[0] for d in unique_days}
    boot_means = np.zeros((1000, len(rel_minutes)))
    for b in range(1000):
        sampled_days = rng2.choice(unique_days, size=len(unique_days), replace=True)
        rows = np.concatenate([day_to_rows[d] for d in sampled_days])
        boot_means[b] = diff_df.iloc[rows].mean(axis=0, skipna=True).values
    ci_low = np.nanpercentile(boot_means, 2.5, axis=0)
    ci_high = np.nanpercentile(boot_means, 97.5, axis=0)

    result = pd.DataFrame({
        "minute_relativ_t0": rel_minutes,
        "impulsantwort_delay_s": mean_curve.values,
        "ci_low": ci_low,
        "ci_high": ci_high,
    })
    print(f"Lokalzeit-Variante: n={len(events_with_controls)} Ereignisse, {len(unique_days)} Tage")
    return result


if __name__ == "__main__":
    print("Rekonstruiere Original-Ereignisse und -Kontrollen (gleicher Seed)...")
    events_with_controls, by_key = reconstruct_original()
    print(f"{len(events_with_controls)} Ereignisse mit Kontrollen (sollte 3253 sein)")

    print("Lade Zustandsvektor...")
    sv = load_state_vector()

    print("\nAnalysiere Sommerzeit-Überschneidung je Ereignis-Kontroll-Paar...")
    pairs = analyse_straddling(events_with_controls, sv)
    pairs.to_csv(OUTPUTS_DIR / "qc_sommerzeit_paare.csv", index=False)

    n_total = len(pairs)
    n_straddling = pairs["straddling_dst"].sum()
    print(f"{n_straddling:,} von {n_total:,} Ereignis-Kontroll-Paaren ({n_straddling/n_total:.2%}) "
          f"überspannen eine Sommerzeitumstellung (unterschiedlicher Berlin-UTC-Offset).")

    baseline_straddling = pairs.loc[pairs["straddling_dst"], "paar_diff_baseline"].mean()
    baseline_normal = pairs.loc[~pairs["straddling_dst"], "paar_diff_baseline"].mean()
    print(f"\nMittlere paarweise Sockeldifferenz (Ereignis - Kontrolle, [-60,-15]min):")
    print(f"  Paare OHNE Umstellung:  {baseline_normal:.1f}s (n={(~pairs['straddling_dst']).sum():,})")
    print(f"  Paare MIT Umstellung:   {baseline_straddling:.1f}s (n={n_straddling:,})")
    print(f"  Differenz: {baseline_straddling - baseline_normal:.1f}s")

    summary = pd.DataFrame([{
        "gruppe": "ohne_umstellung", "n_paare": int((~pairs['straddling_dst']).sum()),
        "mittlere_sockeldiff_s": baseline_normal,
    }, {
        "gruppe": "mit_umstellung", "n_paare": int(n_straddling),
        "mittlere_sockeldiff_s": baseline_straddling,
    }])
    summary.to_csv(OUTPUTS_DIR / "qc_sommerzeit_sockel_vergleich.csv", index=False)

    print("\nBerechne Lokalzeit-Kontrollvariante zum Vergleich...")
    hits = load_code_hits()
    all_events = cluster_events(hits)
    clean_events, _ = drop_overlapping(all_events)
    local_result = run_localtime_variant(clean_events, by_key, sv)
    local_result.to_csv(OUTPUTS_DIR / "impulsantwort_code34_kontrolle_lokalzeit.csv", index=False)

    baseline_orig = pd.read_csv(OUTPUTS_DIR / "impulsantwort_code34.csv")
    baseline_orig_sockel = baseline_orig.loc[
        baseline_orig["minute_relativ_t0"].between(-60, -15), "impulsantwort_delay_s"
    ].mean()
    baseline_local_sockel = local_result.loc[
        local_result["minute_relativ_t0"].between(-60, -15), "impulsantwort_delay_s"
    ].mean()
    print(f"\nSockel [-60,-15] Original (UTC-Matching):    {baseline_orig_sockel:.1f}s")
    print(f"Sockel [-60,-15] Lokalzeit-Matching:          {baseline_local_sockel:.1f}s")
    print(f"Differenz: {baseline_local_sockel - baseline_orig_sockel:.1f}s")

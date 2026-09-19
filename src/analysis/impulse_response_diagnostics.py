"""
Diagnose + Anpassung der Stufe-1-Impulsantwort (message_code 34), auf
Anfrage nach Durchsicht des ersten Ergebnisses. Baut auf
src/analysis/impulse_response.py auf (importiert dessen Funktionen statt sie
zu duplizieren), überschreibt aber NICHTS von dessen Ausgaben -- alle neuen
Dateien haben eigene Namen.

Reihenfolge exakt wie angefragt: erst Diagnose (1a-1d, 2a-2b, 3a), dann
Anpassung (Baseline-Korrektur, strengere Kontrollen, Episoden-Ebene,
Dauer-Stratifizierung).
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from src.analysis.impulse_response import (
    CLUSTER_GAP_MIN,
    CODE,
    K_CONTROLS,
    MIN_TRIPS,
    N_BOOTSTRAP,
    POST_MIN,
    PRE_MIN,
    RNG_SEED,
    STAMMSTRECKE_IDS,
    Event,
    cluster_events,
    curve_for,
    drop_overlapping,
    load_code_hits,
    load_state_vector,
    weekday_type,
)
from src.utils.constants import DATA_DIR, OUTPUTS_DIR, PLOTS_DIR

EVENTS_FINAL_GLOB = str(DATA_DIR / "02_events" / "final" / "*.parquet")

# "Echte" Störungscodes für die Kontamination-Prüfung: IRIS-Codes 2-69, die
# tatsächliche Betriebsstörungen markieren (nicht Komfort-/Ausstattungshinweise
# wie 70-98, nicht die undokumentierten 0/1000/1001, nicht der reine
# Platzhalter 1 "Nähere Informationen in Kürze"). Quelle der Klassifikation:
# IRIS_CODES-Tabelle aus explore.py (Travel::Status::DE::IRIS::Result, derf).
ALLE_STOERUNGSCODES = list(range(2, 70))

STATION_NAMES = {
    8004158: "Pasing", 8004151: "Laim", 8004179: "Hirschgarten",
    8004128: "Donnersbergerbrücke", 8004129: "Hackerbrücke", 8098263: "Hauptbahnhof (tief)",
    8004132: "Karlsplatz (Stachus)", 8004135: "Marienplatz", 8004131: "Isartor",
    8004136: "Rosenheimer Platz", 8000262: "Ostbahnhof", 8004134: "Leuchtenbergring",
}


# ---------------------------------------------------------------------------
# Gemeinsamer Unterbau: Ereignisse wie im Original neu berechnen (identisch,
# damit Diagnose und Anpassung auf denselben Ereignissen laufen).
# ---------------------------------------------------------------------------

def build_base_events() -> tuple[pd.DataFrame, list[Event], list[Event]]:
    hits = load_code_hits()
    all_events = cluster_events(hits)
    clean_events, n_dropped = drop_overlapping(all_events)
    print(f"Basis: {len(hits):,} Code-{CODE}-Zeilen, {len(all_events)} Ereignisse, "
          f"{n_dropped} wegen Überlappung verworfen, {len(clean_events)} sauber.")
    return hits, all_events, clean_events


def find_controls_all(event: Event, all_events_by_key: dict, rng: np.random.Generator,
                       max_candidates: int | None = None) -> list[pd.Timestamp]:
    """Wie find_controls in impulse_response.py, aber OHNE frühen Abbruch bei
    k=10 -- liefert alle gültigen Kandidaten (für die Verteilungsdiagnose in 1b)."""
    e_type = weekday_type(event.t0)
    e_minute_of_day = event.t0.hour * 60 + event.t0.minute
    key = (event.stop_id, event.richtung)
    conflicting_t0s = pd.Series([e.t0 for e in all_events_by_key.get(key, [])])

    date_range = pd.date_range(
        event.t0.normalize() - pd.Timedelta(weeks=6),
        event.t0.normalize() + pd.Timedelta(weeks=6),
        freq="D", tz="UTC",
    )
    candidates = []
    for day in date_range:
        if weekday_type(day) != e_type:
            continue
        candidate_t0 = day + pd.Timedelta(minutes=e_minute_of_day)
        if abs((candidate_t0 - event.t0).total_seconds()) < 1:
            continue
        candidates.append(candidate_t0)
    rng.shuffle(candidates)

    valid = []
    for c in candidates:
        window_start = c + pd.Timedelta(minutes=PRE_MIN)
        window_end = c + pd.Timedelta(minutes=POST_MIN)
        overlap = ((conflicting_t0s >= window_start) & (conflicting_t0s <= window_end)).any()
        if not overlap:
            valid.append(c)
        if max_candidates and len(valid) >= max_candidates:
            break
    return valid


# ---------------------------------------------------------------------------
# PROBLEM 1a: Zusammensetzung Ereignis- vs. Kontrollfenster
# ---------------------------------------------------------------------------

def diagnose_1a(clean_events: list[Event], events_with_controls: list[tuple[Event, list]]) -> None:
    print("\n=== 1a: Zusammensetzung Ereignis- vs. Kontrollfenster ===")
    ev_rows = [{"hour": e.t0.hour, "weekday_type": weekday_type(e.t0), "month": e.t0.month,
                "stop_id": e.stop_id, "richtung": e.richtung} for e, _ in events_with_controls]
    ctrl_rows = []
    for e, controls in events_with_controls:
        for c in controls:
            ctrl_rows.append({"hour": c.hour, "weekday_type": weekday_type(c), "month": c.month,
                               "stop_id": e.stop_id, "richtung": e.richtung})
    ev_df = pd.DataFrame(ev_rows)
    ctrl_df = pd.DataFrame(ctrl_rows)

    for col in ["hour", "weekday_type", "month", "stop_id", "richtung"]:
        ev_share = ev_df[col].value_counts(normalize=True).sort_index()
        ctrl_share = ctrl_df[col].value_counts(normalize=True).sort_index()
        comp = pd.DataFrame({"ereignis_anteil": ev_share, "kontroll_anteil": ctrl_share}).fillna(0)
        comp["differenz_pp"] = (comp["ereignis_anteil"] - comp["kontroll_anteil"]) * 100
        print(f"\n-- {col} --")
        print(comp.round(4).to_string())
        comp.to_csv(OUTPUTS_DIR / f"diagnose_1a_verteilung_{col}.csv")


# ---------------------------------------------------------------------------
# PROBLEM 1b: wie viele Kontrollfenster pro Ereignis gefunden?
# ---------------------------------------------------------------------------

def diagnose_1b(clean_events: list[Event], by_key: dict, rng: np.random.Generator) -> dict[int, list]:
    print("\n=== 1b: Anzahl gefundener Kontrollfenster je Ereignis ===")
    n_found = {}
    all_controls = {}
    for e in clean_events:
        found = find_controls_all(e, by_key, rng, max_candidates=30)
        n_found[e.event_id] = len(found)
        all_controls[e.event_id] = found

    dist = pd.Series(n_found.values())
    print(dist.describe())
    print("\nVerteilung (Histogramm-Bins):")
    bins = [0, 1, 5, 9, 10, 15, 20, 25, 30, 999]
    labels = ["0", "1-4", "5-8", "9", "10-14", "15-19", "20-24", "25-29", "30+"]
    counts = pd.cut(dist, bins=bins, labels=labels, right=False).value_counts().sort_index()
    print(counts)
    print(f"\nEreignisse mit < {K_CONTROLS} Kontrollen: {(dist < K_CONTROLS).sum()} von {len(dist)} "
          f"({(dist < K_CONTROLS).mean():.1%}) -- diese wurden im Original komplett aus der "
          f"Analyse ausgeschlossen (nicht aufgefüllt, nicht mit weniger als k gerechnet).")

    out = pd.DataFrame({"event_id": list(n_found.keys()), "n_kontrollen_gefunden": list(n_found.values())})
    out.to_csv(OUTPUTS_DIR / "diagnose_1b_kontrollen_pro_ereignis.csv", index=False)
    return all_controls


# ---------------------------------------------------------------------------
# PROBLEM 1c: enthalten Kontrollfenster andere Störungscodes?
# ---------------------------------------------------------------------------

def load_all_disruption_hits() -> pd.DataFrame:
    con = duckdb.connect()
    stop_ids_sql = ",".join(str(x) for x in STAMMSTRECKE_IDS)
    codes_sql = ",".join(str(c) for c in ALLE_STOERUNGSCODES)
    df = con.execute(
        f"""
        SELECT DISTINCT stop_id, richtung, update_timestamp,
               unnest(list_filter(message_codes, x -> x IN ({codes_sql}))) AS code
        FROM read_parquet('{EVENTS_FINAL_GLOB}')
        WHERE is_arrival = true AND stop_id IN ({stop_ids_sql}) AND richtung IS NOT NULL
        """
    ).fetchdf()
    df["update_timestamp"] = pd.to_datetime(df["update_timestamp"], utc=True)
    return df.dropna(subset=["update_timestamp"])


def diagnose_1c(events_with_controls: list[tuple[Event, list]], disruption_hits: pd.DataFrame) -> None:
    print("\n=== 1c: Kontamination der Kontrollfenster mit anderen Störungscodes ===")
    grouped = {k: v.sort_values("update_timestamp") for k, v in disruption_hits.groupby(["stop_id", "richtung"])}

    n_windows = 0
    n_contaminated = 0
    n_contaminated_excl34 = 0
    for e, controls in events_with_controls:
        key = (e.stop_id, e.richtung)
        g = grouped.get(key)
        for c in controls:
            n_windows += 1
            if g is None:
                continue
            window_start = c + pd.Timedelta(minutes=PRE_MIN)
            window_end = c + pd.Timedelta(minutes=POST_MIN)
            in_window = g[(g["update_timestamp"] >= window_start) & (g["update_timestamp"] <= window_end)]
            if len(in_window) > 0:
                n_contaminated += 1
            if (in_window["code"] != CODE).any():
                n_contaminated_excl34 += 1

    print(f"Kontrollfenster gesamt: {n_windows:,}")
    print(f"davon mit mindestens einem Störungscode (2-69, inkl. 34) im Fenster: "
          f"{n_contaminated:,} ({n_contaminated/n_windows:.1%})")
    print(f"davon mit einem Störungscode AUSSER 34: "
          f"{n_contaminated_excl34:,} ({n_contaminated_excl34/n_windows:.1%})")
    pd.DataFrame([{
        "n_windows": n_windows, "n_contaminated_any": n_contaminated,
        "n_contaminated_excl_34": n_contaminated_excl34,
    }]).to_csv(OUTPUTS_DIR / "diagnose_1c_kontamination.csv", index=False)


# ---------------------------------------------------------------------------
# PROBLEM 1d: Sockel je Station
# ---------------------------------------------------------------------------

def diagnose_1d(events_with_controls: list[tuple[Event, list]], sv: pd.Series) -> pd.DataFrame:
    print("\n=== 1d: Sockelhöhe je Station ===")
    rows = []
    for e, controls in events_with_controls:
        event_curve = curve_for(sv, e.stop_id, e.richtung, e.t0)
        control_curves = [curve_for(sv, e.stop_id, e.richtung, c) for c in controls]
        control_mean = pd.concat(control_curves, axis=1).mean(axis=1)
        diff = event_curve - control_mean
        baseline = diff.loc[-60:-15].mean()
        rows.append({"stop_id": e.stop_id, "station_name": STATION_NAMES[e.stop_id], "baseline_sockel": baseline})
    df = pd.DataFrame(rows)
    summary = df.groupby(["stop_id", "station_name"])["baseline_sockel"].agg(["mean", "std", "count"]).reset_index()
    print(summary.round(1).to_string(index=False))
    summary.to_csv(OUTPUTS_DIR / "diagnose_1d_sockel_je_station.csv", index=False)
    return df


if __name__ == "__main__":
    from src.analysis.impulse_response import find_controls

    hits, all_events, clean_events = build_base_events()
    by_key: dict[tuple[int, str], list[Event]] = {}
    for e in all_events:
        by_key.setdefault((e.stop_id, e.richtung), []).append(e)

    rng = np.random.default_rng(RNG_SEED)
    all_controls_dist = diagnose_1b(clean_events, by_key, rng)

    rng2 = np.random.default_rng(RNG_SEED)
    events_with_controls = []
    for e in clean_events:
        controls = all_controls_dist[e.event_id][:K_CONTROLS]
        if len(controls) >= K_CONTROLS:
            events_with_controls.append((e, controls))
    print(f"\n{len(events_with_controls)} Ereignisse mit >= {K_CONTROLS} Kontrollen "
          f"(sollte mit Original übereinstimmen).")

    diagnose_1a(clean_events, events_with_controls)

    print("\nLade Störungscode-Historie (2-69) für Kontaminationsprüfung...")
    disruption_hits = load_all_disruption_hits()
    diagnose_1c(events_with_controls, disruption_hits)

    print("\nLade Zustandsvektor für Sockel-Diagnose...")
    sv = load_state_vector()
    diagnose_1d(events_with_controls, sv)

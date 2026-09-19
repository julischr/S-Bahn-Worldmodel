"""
Stufe 1 des Bauplans: Impulsantwort auf message_code 34 ("Defekt an einem
Signal") messen. Siehe outputs/auswertungsplan_stufe1.md für die Begründung
der Ereigniswahl und alle Festlegungen VOR der ersten Rechnung.

Ablauf:
  1. Ereignisse aus events_final clustern (1.2)
  2. Überlappende Ereignisse verwerfen (1.2)
  3. Kontrollfenster pro Ereignis suchen (1.3)
  4. Ereigniskurve minus Kontrollmittel, über Ereignisse gemittelt,
     tageweises Bootstrap-Konfidenzband (1.4)
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from src.utils.constants import DATA_DIR, OUTPUTS_DIR, PLOTS_DIR

EVENTS_FINAL_GLOB = str(DATA_DIR / "02_events" / "final" / "*.parquet")
STATE_VECTOR_PATH = DATA_DIR / "03_state_vector" / "state_vector_5min.parquet"

CODE = 34
CLUSTER_GAP_MIN = 15
MIN_TRIPS = 3
PRE_MIN = -60
POST_MIN = 240
K_CONTROLS = 10
TIME_TOL_MIN = 30
DATE_TOL_WEEKS = 6
N_BOOTSTRAP = 1000
RNG_SEED = 42

STAMMSTRECKE_IDS = (
    8004128, 8004129, 8004131, 8004132, 8004151, 8000262,
    8004135, 8004136, 8004134, 8004179, 8098263, 8004158,
)

# Bayerische gesetzliche Feiertage 2025 (landesweit; Mariä Himmelfahrt
# ausgeklammert, siehe Auswertungsplan).
FEIERTAGE_BAYERN_2025 = pd.to_datetime([
    "2025-01-01", "2025-01-06", "2025-04-18", "2025-04-21",
    "2025-05-01", "2025-05-29", "2025-06-09", "2025-06-19",
    "2025-10-03", "2025-11-01", "2025-12-25", "2025-12-26",
]).date


def weekday_type(ts: pd.Timestamp) -> str:
    d = ts.date()
    if d in FEIERTAGE_BAYERN_2025 or ts.weekday() == 6:
        return "So+Feiertag"
    if ts.weekday() == 5:
        return "Sa"
    if ts.weekday() == 4:
        return "Fr"
    return "Mo-Do"


@dataclass
class Event:
    event_id: int
    stop_id: int
    richtung: str
    t0: pd.Timestamp
    n_trips: int


def load_code_hits() -> pd.DataFrame:
    con = duckdb.connect()
    stop_ids_sql = ",".join(str(x) for x in STAMMSTRECKE_IDS)
    df = con.execute(
        f"""
        SELECT stop_id, richtung, trip_id, update_timestamp
        FROM read_parquet('{EVENTS_FINAL_GLOB}')
        WHERE is_arrival = true
          AND stop_id IN ({stop_ids_sql})
          AND list_contains(message_codes, {CODE})
        """
    ).fetchdf()
    df["update_timestamp"] = pd.to_datetime(df["update_timestamp"], utc=True)
    return df.dropna(subset=["update_timestamp"]).sort_values(["stop_id", "update_timestamp"])


def cluster_events(hits: pd.DataFrame) -> list[Event]:
    """1.2: Cluster Halte mit Code 34 an Station s innerhalb von 15 Minuten."""
    events: list[Event] = []
    event_id = 0
    for stop_id, g in hits.groupby("stop_id"):
        g = g.sort_values("update_timestamp").reset_index(drop=True)
        gap = g["update_timestamp"].diff().dt.total_seconds() / 60
        cluster_id = (gap > CLUSTER_GAP_MIN).cumsum()
        for _, cg in g.groupby(cluster_id):
            # Richtung des Clusters: Mehrheitsrichtung der betroffenen Fahrten;
            # nur Fahrten in dieser Mehrheitsrichtung zählen für die Mindestgröße.
            richtung_counts = cg["richtung"].value_counts(dropna=True)
            if richtung_counts.empty:
                continue
            majority_richtung = richtung_counts.idxmax()
            sub = cg[cg["richtung"] == majority_richtung]
            n_trips = sub["trip_id"].nunique()
            if n_trips < MIN_TRIPS:
                continue
            t0 = sub["update_timestamp"].min()
            events.append(Event(event_id, stop_id, majority_richtung, t0, n_trips))
            event_id += 1
    return events


def drop_overlapping(events: list[Event]) -> tuple[list[Event], int]:
    """1.2: Ereignis verwerfen, wenn im Fenster [-60,+240] ein weiteres
    Ereignis derselben Station+Richtung liegt."""
    by_key: dict[tuple[int, str], list[Event]] = {}
    for e in events:
        by_key.setdefault((e.stop_id, e.richtung), []).append(e)

    clean: list[Event] = []
    n_dropped = 0
    for key, group in by_key.items():
        group = sorted(group, key=lambda e: e.t0)
        t0s = pd.Series([e.t0 for e in group])
        for i, e in enumerate(group):
            window_start = e.t0 + pd.Timedelta(minutes=PRE_MIN)
            window_end = e.t0 + pd.Timedelta(minutes=POST_MIN)
            others = t0s.drop(i)
            overlap = ((others >= window_start) & (others <= window_end)).any()
            if overlap:
                n_dropped += 1
            else:
                clean.append(e)
    return clean, n_dropped


def find_controls(event: Event, all_events_by_key: dict, rng: np.random.Generator) -> list[pd.Timestamp]:
    """1.3: k Kontrollfenster suchen."""
    e_type = weekday_type(event.t0)
    e_minute_of_day = event.t0.hour * 60 + event.t0.minute
    key = (event.stop_id, event.richtung)
    conflicting_t0s = pd.Series([e.t0 for e in all_events_by_key.get(key, [])])

    candidates = []
    # Kandidaten: jeder Tag im Datenzeitraum, an dem Uhrzeit/Wochentagstyp passen.
    date_range = pd.date_range(event.t0.normalize() - pd.Timedelta(weeks=DATE_TOL_WEEKS),
                                event.t0.normalize() + pd.Timedelta(weeks=DATE_TOL_WEEKS),
                                freq="D", tz="UTC")
    for day in date_range:
        if weekday_type(day) != e_type:
            continue
        candidate_t0 = day + pd.Timedelta(minutes=e_minute_of_day)
        if abs((candidate_t0 - event.t0).total_seconds()) < 1:
            continue  # das Ereignis selbst
        candidates.append(candidate_t0)

    rng.shuffle(candidates)
    chosen = []
    for c in candidates:
        # Uhrzeit ± 30min ist durch die Konstruktion (gleiche minute_of_day)
        # exakt erfüllt; hier nur noch Überlappungs-Check.
        window_start = c + pd.Timedelta(minutes=PRE_MIN)
        window_end = c + pd.Timedelta(minutes=POST_MIN)
        overlap = ((conflicting_t0s >= window_start) & (conflicting_t0s <= window_end)).any()
        if not overlap:
            chosen.append(c)
        if len(chosen) >= K_CONTROLS:
            break
    return chosen


def load_state_vector() -> pd.DataFrame:
    df = pd.read_parquet(STATE_VECTOR_PATH, columns=["window", "stop_id", "richtung", "delay_mean"])
    df["window"] = pd.to_datetime(df["window"], utc=True)
    return df.set_index(["stop_id", "richtung", "window"])["delay_mean"].sort_index()


def curve_for(sv: pd.Series, stop_id: int, richtung: str, t0: pd.Timestamp) -> pd.Series:
    """delay_mean in 5-Minuten-Schritten relativ zu t0, im Fenster [-60,+240]."""
    rel_minutes = np.arange(PRE_MIN, POST_MIN + 1, 5)
    windows = [t0.floor("5min") + pd.Timedelta(minutes=int(m)) for m in rel_minutes]
    try:
        vals = sv.loc[(stop_id, richtung, windows)]
        vals.index = rel_minutes
    except KeyError:
        vals = pd.Series(index=rel_minutes, dtype=float)
        for m, w in zip(rel_minutes, windows):
            try:
                vals[m] = sv.loc[(stop_id, richtung, w)]
            except KeyError:
                vals[m] = np.nan
    return vals.reindex(rel_minutes)


def run() -> None:
    print("Lade Code-34-Treffer an den 12 Stammstreckenhalten...")
    hits = load_code_hits()
    print(f"{len(hits):,} Zeilen mit Code {CODE}")

    print("\nClustere zu Ereignissen (15-Minuten-Lücke)...")
    events = cluster_events(hits)
    print(f"{len(events)} Ereignisse mit >= {MIN_TRIPS} betroffenen Fahrten")

    print("\nVerwerfe überlappende Ereignisse...")
    clean_events, n_dropped = drop_overlapping(events)
    print(f"{n_dropped} Ereignisse verworfen (Überlappung im Fenster [{PRE_MIN},{POST_MIN}]min)")
    print(f"{len(clean_events)} saubere Ereignisse übrig")

    if not clean_events:
        print("Keine sauberen Ereignisse -- Abbruch, siehe 1.5.")
        return

    by_key: dict[tuple[int, str], list[Event]] = {}
    for e in events:
        by_key.setdefault((e.stop_id, e.richtung), []).append(e)

    print("\nLade Zustandsvektor...")
    sv = load_state_vector()

    print("\nSuche Kontrollfenster (k=10) je Ereignis...")
    rng = np.random.default_rng(RNG_SEED)
    events_with_controls = []
    for e in clean_events:
        controls = find_controls(e, by_key, rng)
        if len(controls) < K_CONTROLS:
            continue
        events_with_controls.append((e, controls))
    print(f"{len(events_with_controls)} von {len(clean_events)} Ereignissen haben {K_CONTROLS} Kontrollen gefunden")

    if not events_with_controls:
        print("Keine Ereignisse mit ausreichend Kontrollen -- Abbruch, siehe 1.5.")
        return

    print("\nBerechne Ereignis- und Kontrollkurven...")
    rel_minutes = np.arange(PRE_MIN, POST_MIN + 1, 5)
    diff_curves = []  # eine Zeile je Ereignis: Ereigniskurve - Kontrollmittel
    event_days = []
    for e, controls in events_with_controls:
        event_curve = curve_for(sv, e.stop_id, e.richtung, e.t0)
        control_curves = [curve_for(sv, e.stop_id, e.richtung, c) for c in controls]
        control_mean = pd.concat(control_curves, axis=1).mean(axis=1)
        diff_curves.append(event_curve - control_mean)
        event_days.append(e.t0.date())

    diff_df = pd.DataFrame(diff_curves)
    diff_df.index = range(len(diff_df))
    mean_curve = diff_df.mean(axis=0, skipna=True)

    print("\nBootstrap-Konfidenzband (tageweise, "
          f"{N_BOOTSTRAP} Wiederholungen)...")
    days_arr = np.array(event_days)
    unique_days = np.unique(days_arr)
    rng2 = np.random.default_rng(RNG_SEED + 1)
    boot_means = np.zeros((N_BOOTSTRAP, len(rel_minutes)))
    day_to_rows = {d: np.where(days_arr == d)[0] for d in unique_days}
    for b in range(N_BOOTSTRAP):
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
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUTS_DIR / "impulsantwort_code34.csv", index=False)
    print(f"\nGespeichert: {OUTPUTS_DIR / 'impulsantwort_code34.csv'}")
    print(f"n Ereignisse in der Auswertung: {len(events_with_controls)}, "
          f"n unique Tage: {len(unique_days)}")

    plot_result(result, len(events_with_controls), len(unique_days))


def plot_result(result: pd.DataFrame, n_events: int, n_days: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.fill_between(result["minute_relativ_t0"], result["ci_low"], result["ci_high"],
                     color="#C44E52", alpha=0.25, label="95%-Bootstrap-Band (tageweise)")
    ax.plot(result["minute_relativ_t0"], result["impulsantwort_delay_s"],
            color="#C44E52", linewidth=2, label="Impulsantwort")
    ax.set_xlabel("Minuten relativ zu t0")
    ax.set_ylabel("Δ delay_mean (Sekunden, Ereignis − Kontrollmittel)")
    ax.set_title(f"Impulsantwort auf message_code {CODE} (Defekt an einem Signal)\n"
                 f"n={n_events} Ereignisse, {n_days} Tage, Stammstrecke")
    ax.legend()
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / "impulsantwort_code34.png", dpi=150)
    plt.close(fig)
    print(f"Plot gespeichert: {PLOTS_DIR / 'impulsantwort_code34.png'}")


if __name__ == "__main__":
    run()

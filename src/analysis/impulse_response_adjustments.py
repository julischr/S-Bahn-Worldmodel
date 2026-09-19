"""
Anpassungen nach der Diagnose in impulse_response_diagnostics.py:
  - Baseline-Korrektur (Original bleibt zusätzlich erhalten)
  - Kontrollauswahl Variante B (gegen ALLE Störungscodes abgegrenzt)
  - Problem 2: Episoden-Ebene (Ereignisse über Stationen hinweg zusammenfassen)
  - Problem 3: Dauer je Ereignis, Impulsantwort nach Dauer stratifiziert

Nichts aus impulse_response.py wird überschrieben. Jede Variante bekommt
einen eigenen Dateinamen unter outputs/ bzw. outputs/plots/.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from src.analysis.impulse_response import (
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
from src.analysis.impulse_response_diagnostics import (
    ALLE_STOERUNGSCODES,
    STATION_NAMES,
    build_base_events,
    load_all_disruption_hits,
)
from src.utils.constants import DATA_DIR, OUTPUTS_DIR, PLOTS_DIR

REL_MINUTES = np.arange(PRE_MIN, POST_MIN + 1, 5)
EVENTS_FINAL_GLOB = str(DATA_DIR / "02_events" / "final" / "*.parquet")


# ---------------------------------------------------------------------------
# Gemeinsame Hilfsfunktionen
# ---------------------------------------------------------------------------

def compute_diff_curves(events_with_controls: list[tuple[Event, list]], sv: pd.Series
                         ) -> tuple[pd.DataFrame, list]:
    diff_curves = []
    event_days = []
    for e, controls in events_with_controls:
        event_curve = curve_for(sv, e.stop_id, e.richtung, e.t0)
        control_curves = [curve_for(sv, e.stop_id, e.richtung, c) for c in controls]
        control_mean = pd.concat(control_curves, axis=1).mean(axis=1)
        diff_curves.append(event_curve - control_mean)
        event_days.append(e.t0.date())
    diff_df = pd.DataFrame(diff_curves)
    diff_df.columns = REL_MINUTES
    return diff_df, event_days


def bootstrap_band(diff_df: pd.DataFrame, event_days: list, seed: int = RNG_SEED + 1
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_curve = diff_df.mean(axis=0, skipna=True).values
    days_arr = np.array(event_days)
    unique_days = np.unique(days_arr)
    rng = np.random.default_rng(seed)
    day_to_rows = {d: np.where(days_arr == d)[0] for d in unique_days}
    boot_means = np.zeros((N_BOOTSTRAP, diff_df.shape[1]))
    for b in range(N_BOOTSTRAP):
        sampled_days = rng.choice(unique_days, size=len(unique_days), replace=True)
        rows = np.concatenate([day_to_rows[d] for d in sampled_days])
        boot_means[b] = diff_df.iloc[rows].mean(axis=0, skipna=True).values
    ci_low = np.nanpercentile(boot_means, 2.5, axis=0)
    ci_high = np.nanpercentile(boot_means, 97.5, axis=0)
    return mean_curve, ci_low, ci_high


def save_curve(name: str, mean_curve, ci_low, ci_high, n_events: int, n_days: int) -> pd.DataFrame:
    df = pd.DataFrame({
        "minute_relativ_t0": REL_MINUTES,
        "impulsantwort_delay_s": mean_curve,
        "ci_low": ci_low,
        "ci_high": ci_high,
    })
    df.to_csv(OUTPUTS_DIR / f"{name}.csv", index=False)
    print(f"  n={n_events} Ereignisse, {n_days} Tage -> {OUTPUTS_DIR / f'{name}.csv'}")
    return df


# ---------------------------------------------------------------------------
# Baseline-Korrektur
# ---------------------------------------------------------------------------

def baseline_correct(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df.loc[df["minute_relativ_t0"].between(-60, -15), "impulsantwort_delay_s"].mean()
    out = df.copy()
    out["impulsantwort_delay_s"] -= baseline
    out["ci_low"] -= baseline
    out["ci_high"] -= baseline
    out.attrs["baseline_subtracted"] = baseline
    return out


def plot_original_vs_corrected(original: pd.DataFrame, corrected: pd.DataFrame, baseline: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.fill_between(original["minute_relativ_t0"], original["ci_low"], original["ci_high"],
                     color="#4C72B0", alpha=0.15)
    ax.plot(original["minute_relativ_t0"], original["impulsantwort_delay_s"],
            color="#4C72B0", linewidth=1.5, label="unkorrigiert (Original)")
    ax.fill_between(corrected["minute_relativ_t0"], corrected["ci_low"], corrected["ci_high"],
                     color="#C44E52", alpha=0.2)
    ax.plot(corrected["minute_relativ_t0"], corrected["impulsantwort_delay_s"],
            color="#C44E52", linewidth=2, label=f"baseline-korrigiert (−{baseline:.0f}s)")
    ax.set_xlabel("Minuten relativ zu t0")
    ax.set_ylabel("Δ delay_mean (Sekunden)")
    ax.set_title("Impulsantwort Code 34: unkorrigiert vs. baseline-korrigiert")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "impulsantwort_code34_baseline_korrigiert.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Kontrollauswahl Variante B: gegen ALLE Störungscodes abgegrenzt
# ---------------------------------------------------------------------------

def find_controls_strict(event: Event, all_events_by_key: dict, disruption_by_key: dict,
                          rng: np.random.Generator, k: int = K_CONTROLS) -> list[pd.Timestamp]:
    e_type = weekday_type(event.t0)
    e_minute_of_day = event.t0.hour * 60 + event.t0.minute
    key = (event.stop_id, event.richtung)
    conflicting_t0s = pd.Series([e.t0 for e in all_events_by_key.get(key, [])])
    disruptions = disruption_by_key.get(key)

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

    chosen = []
    for c in candidates:
        window_start = c + pd.Timedelta(minutes=PRE_MIN)
        window_end = c + pd.Timedelta(minutes=POST_MIN)
        overlap_code34 = ((conflicting_t0s >= window_start) & (conflicting_t0s <= window_end)).any()
        if overlap_code34:
            continue
        if disruptions is not None:
            in_window = disruptions[(disruptions["update_timestamp"] >= window_start) &
                                     (disruptions["update_timestamp"] <= window_end)]
            if len(in_window) > 0:
                continue
        chosen.append(c)
        if len(chosen) >= k:
            break
    return chosen


def run_control_variant_b(clean_events: list[Event], by_key: dict, disruption_hits: pd.DataFrame,
                           sv: pd.Series) -> None:
    print("\n=== Kontrollauswahl Variante B (gegen ALLE Störungscodes 2-69 abgegrenzt) ===")
    disruption_by_key = {k: v.sort_values("update_timestamp") for k, v in
                          disruption_hits.groupby(["stop_id", "richtung"])}
    rng = np.random.default_rng(RNG_SEED)
    events_with_controls = []
    n_found_list = []
    for e in clean_events:
        controls = find_controls_strict(e, by_key, disruption_by_key, rng, k=K_CONTROLS)
        n_found_list.append(len(controls))
        if len(controls) >= K_CONTROLS:
            events_with_controls.append((e, controls))

    dist = pd.Series(n_found_list)
    print(f"Verteilung gefundener strenger Kontrollen je Ereignis:\n{dist.describe()}")
    print(f"Ereignisse mit >= {K_CONTROLS} strengen Kontrollen: {len(events_with_controls)} "
          f"von {len(clean_events)} ({len(events_with_controls)/len(clean_events):.1%})")

    if not events_with_controls:
        print("Keine Ereignisse mit ausreichend strengen Kontrollen -- Variante B nicht auswertbar.")
        return

    diff_df, event_days = compute_diff_curves(events_with_controls, sv)
    mean_curve, ci_low, ci_high = bootstrap_band(diff_df, event_days)
    unique_days = len(set(event_days))
    print("Speichere Variante-B-Kurve...")
    save_curve("impulsantwort_code34_kontrolle_variante_b", mean_curve, ci_low, ci_high,
               len(events_with_controls), unique_days)


def plot_variant_comparison(original: pd.DataFrame, variant_b_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.plot(original["minute_relativ_t0"], original["impulsantwort_delay_s"],
            color="#4C72B0", linewidth=1.5, label="Kontrolle A (nur Code 34 ausgeschlossen, Original)")
    ax.fill_between(original["minute_relativ_t0"], original["ci_low"], original["ci_high"],
                     color="#4C72B0", alpha=0.12)
    if variant_b_path.exists():
        vb = pd.read_csv(variant_b_path)
        ax.plot(vb["minute_relativ_t0"], vb["impulsantwort_delay_s"],
                color="#55A868", linewidth=2, label="Kontrolle B (alle Störungscodes ausgeschlossen)")
        ax.fill_between(vb["minute_relativ_t0"], vb["ci_low"], vb["ci_high"],
                         color="#55A868", alpha=0.2)
    ax.set_xlabel("Minuten relativ zu t0")
    ax.set_ylabel("Δ delay_mean (Sekunden)")
    ax.set_title("Impulsantwort Code 34: Kontrollauswahl A vs. B")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "impulsantwort_code34_kontrollvarianten.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Problem 2: Episoden über Stationen hinweg
# ---------------------------------------------------------------------------

def build_episodes(all_events: list[Event]) -> list[Event]:
    df = pd.DataFrame([{"idx": i, "t0": e.t0} for i, e in enumerate(all_events)]).sort_values("t0")
    gap = df["t0"].diff().dt.total_seconds() / 60
    episode_id = (gap > 15).cumsum()
    df["episode_id"] = episode_id.values

    episodes = []
    for ep_id, g in df.groupby("episode_id"):
        member_events = [all_events[i] for i in g["idx"]]
        member_events.sort(key=lambda e: e.t0)
        first = member_events[0]
        episodes.append(Event(
            event_id=ep_id,
            stop_id=first.stop_id,
            richtung=first.richtung,
            t0=first.t0,
            n_trips=sum(e.n_trips for e in member_events),
        ))
    return episodes


def diagnose_and_run_episodes(all_events: list[Event], sv: pd.Series) -> None:
    print("\n=== Problem 2: Episoden über Stationen hinweg ===")
    df = pd.DataFrame([{"idx": i, "stop_id": e.stop_id, "t0": e.t0} for i, e in enumerate(all_events)])
    df = df.sort_values("t0")
    gap = df["t0"].diff().dt.total_seconds() / 60
    episode_id = (gap > 15).cumsum()
    df["episode_id"] = episode_id.values
    sizes = df.groupby("episode_id").size()
    n_stations_per_episode = df.groupby("episode_id")["stop_id"].nunique()

    print(f"2a: {len(all_events)} Einzel-Ereignisse fallen in {df['episode_id'].nunique()} "
          f"zeitliche Cluster (Gap>15min über ALLE Stationen kombiniert).")
    print("Verteilung Ereignisse pro Cluster:")
    print(sizes.value_counts().sort_index())
    print("\nVerteilung DISTINKTER Stationen pro Cluster:")
    print(n_stations_per_episode.value_counts().sort_index())
    sizes.to_csv(OUTPUTS_DIR / "diagnose_2a_ereignisse_pro_episode.csv")

    episodes_all = build_episodes(all_events)
    print(f"\n2b: {len(all_events)} Einzel-Ereignisse -> {len(episodes_all)} Episoden "
          f"(Reduktion um {(1 - len(episodes_all)/len(all_events)):.1%}).")

    clean_episodes, n_dropped_ep = drop_overlapping(episodes_all)
    print(f"Nach Überlappungs-Bereinigung (gleiche Repräsentativ-Station+Richtung): "
          f"{n_dropped_ep} verworfen, {len(clean_episodes)} saubere Episoden übrig.")

    by_key_ep: dict[tuple[int, str], list[Event]] = {}
    for e in episodes_all:
        by_key_ep.setdefault((e.stop_id, e.richtung), []).append(e)

    from src.analysis.impulse_response import find_controls
    rng = np.random.default_rng(RNG_SEED)
    events_with_controls = []
    for e in clean_episodes:
        controls = find_controls(e, by_key_ep, rng)
        if len(controls) >= K_CONTROLS:
            events_with_controls.append((e, controls))
    print(f"2c: {len(events_with_controls)} von {len(clean_episodes)} Episoden mit "
          f"{K_CONTROLS} Kontrollen -- Impulsantwort auf Episoden-Ebene:")

    if not events_with_controls:
        print("Keine Episoden mit ausreichend Kontrollen.")
        return

    diff_df, event_days = compute_diff_curves(events_with_controls, sv)
    mean_curve, ci_low, ci_high = bootstrap_band(diff_df, event_days)
    unique_days = len(set(event_days))
    save_curve("impulsantwort_code34_episoden_ebene", mean_curve, ci_low, ci_high,
               len(events_with_controls), unique_days)


def plot_event_vs_episode(original: pd.DataFrame, episode_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.plot(original["minute_relativ_t0"], original["impulsantwort_delay_s"],
            color="#4C72B0", linewidth=1.5, label="Ereignis-Ebene (pro Station+Richtung, Original)")
    ax.fill_between(original["minute_relativ_t0"], original["ci_low"], original["ci_high"],
                     color="#4C72B0", alpha=0.12)
    if episode_path.exists():
        ep = pd.read_csv(episode_path)
        ax.plot(ep["minute_relativ_t0"], ep["impulsantwort_delay_s"],
                color="#8172B2", linewidth=2, label="Episoden-Ebene (über Stationen zusammengefasst)")
        ax.fill_between(ep["minute_relativ_t0"], ep["ci_low"], ep["ci_high"],
                         color="#8172B2", alpha=0.2)
    ax.set_xlabel("Minuten relativ zu t0")
    ax.set_ylabel("Δ delay_mean (Sekunden)")
    ax.set_title("Impulsantwort Code 34: Ereignis- vs. Episoden-Ebene")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "impulsantwort_code34_episoden_vergleich.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Problem 3: Dauer je Ereignis
# ---------------------------------------------------------------------------

def compute_durations(hits: pd.DataFrame, clean_events: list[Event]) -> pd.DataFrame:
    hits_by_stop = {k: v.sort_values("update_timestamp") for k, v in hits.groupby("stop_id")}
    rows = []
    for e in clean_events:
        g = hits_by_stop[e.stop_id]
        cluster_end_search = g[(g["update_timestamp"] >= e.t0) &
                                (g["update_timestamp"] <= e.t0 + pd.Timedelta(hours=6))]
        gap = cluster_end_search["update_timestamp"].diff().dt.total_seconds() / 60
        gap.iloc[0] = 0
        break_idx = (gap > 15).idxmax() if (gap > 15).any() else None
        if break_idx is not None and gap.loc[break_idx] > 15:
            cluster_rows = cluster_end_search.loc[:break_idx].iloc[:-1]
        else:
            cluster_rows = cluster_end_search
        t_end = cluster_rows["update_timestamp"].max()
        duration_min = (t_end - e.t0).total_seconds() / 60
        rows.append({"event_id": e.event_id, "stop_id": e.stop_id, "t0": e.t0, "dauer_min": duration_min})
    return pd.DataFrame(rows)


def diagnose_and_run_duration(clean_events: list[Event], hits: pd.DataFrame,
                               all_events: list[Event], sv: pd.Series) -> None:
    print("\n=== Problem 3: Dauer je Ereignis ===")
    durations = compute_durations(hits, clean_events)
    print(durations["dauer_min"].describe())
    print("\nQuartile:", durations["dauer_min"].quantile([0.25, 0.5, 0.75, 0.9]).to_dict())
    durations.to_csv(OUTPUTS_DIR / "diagnose_3a_dauer_je_ereignis.csv", index=False)

    bins = [0, 30, 90, np.inf]
    labels = ["kurz (<30min)", "mittel (30-90min)", "lang (>90min)"]
    durations["kategorie"] = pd.cut(durations["dauer_min"], bins=bins, labels=labels)
    print("\nVerteilung nach Kategorie:")
    print(durations["kategorie"].value_counts())

    event_by_id = {e.event_id: e for e in clean_events}
    by_key: dict[tuple[int, str], list[Event]] = {}
    for e in all_events:
        by_key.setdefault((e.stop_id, e.richtung), []).append(e)

    from src.analysis.impulse_response import find_controls
    curves_by_cat = {}
    for cat in labels:
        cat_event_ids = durations.loc[durations["kategorie"] == cat, "event_id"]
        cat_events = [event_by_id[i] for i in cat_event_ids if i in event_by_id]
        rng = np.random.default_rng(RNG_SEED)
        events_with_controls = []
        for e in cat_events:
            controls = find_controls(e, by_key, rng)
            if len(controls) >= K_CONTROLS:
                events_with_controls.append((e, controls))
        print(f"\n{cat}: {len(cat_events)} Ereignisse, {len(events_with_controls)} mit genug Kontrollen")
        if not events_with_controls:
            continue
        diff_df, event_days = compute_diff_curves(events_with_controls, sv)
        mean_curve, ci_low, ci_high = bootstrap_band(diff_df, event_days)
        safe_name = cat.split(" ")[0]
        df_out = save_curve(f"impulsantwort_code34_dauer_{safe_name}", mean_curve, ci_low, ci_high,
                             len(events_with_controls), len(set(event_days)))
        curves_by_cat[cat] = df_out

    plot_duration_strata(curves_by_cat)


def plot_duration_strata(curves_by_cat: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    colors = {"kurz (<30min)": "#55A868", "mittel (30-90min)": "#DD8452", "lang (>90min)": "#C44E52"}
    for cat, df in curves_by_cat.items():
        ax.plot(df["minute_relativ_t0"], df["impulsantwort_delay_s"], linewidth=2,
                color=colors.get(cat), label=cat)
        ax.fill_between(df["minute_relativ_t0"], df["ci_low"], df["ci_high"],
                         color=colors.get(cat), alpha=0.15)
    ax.set_xlabel("Minuten relativ zu t0")
    ax.set_ylabel("Δ delay_mean (Sekunden)")
    ax.set_title("Impulsantwort Code 34, stratifiziert nach Ereignisdauer")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "impulsantwort_code34_dauer_stratifiziert.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    hits, all_events, clean_events = build_base_events()
    by_key: dict[tuple[int, str], list[Event]] = {}
    for e in all_events:
        by_key.setdefault((e.stop_id, e.richtung), []).append(e)

    print("\nLade Zustandsvektor...")
    sv = load_state_vector()

    original = pd.read_csv(OUTPUTS_DIR / "impulsantwort_code34.csv")

    print("\n=== Baseline-Korrektur ===")
    corrected = baseline_correct(original)
    baseline_val = original.loc[original["minute_relativ_t0"].between(-60, -15),
                                 "impulsantwort_delay_s"].mean()
    corrected.to_csv(OUTPUTS_DIR / "impulsantwort_code34_baseline_korrigiert.csv", index=False)
    print(f"Sockel [-60,-15]: {baseline_val:.1f}s -- abgezogen, Original bleibt in "
          f"impulsantwort_code34.csv erhalten.")
    plot_original_vs_corrected(original, corrected, baseline_val)

    print("\nLade Störungscode-Historie für Variante B...")
    disruption_hits = load_all_disruption_hits()
    run_control_variant_b(clean_events, by_key, disruption_hits, sv)
    plot_variant_comparison(original, OUTPUTS_DIR / "impulsantwort_code34_kontrolle_variante_b.csv")

    diagnose_and_run_episodes(all_events, sv)
    plot_event_vs_episode(original, OUTPUTS_DIR / "impulsantwort_code34_episoden_ebene.csv")

    diagnose_and_run_duration(clean_events, hits, all_events, sv)

    print("\nFertig. Alle Varianten unter outputs/ und outputs/plots/, Original unangetastet.")

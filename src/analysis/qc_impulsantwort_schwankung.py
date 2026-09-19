"""
Diagnose der Stufe-1-Impulsantwort (Code 34): Rohkurven, Schwankungsquelle,
Terzil-Schichtung, Kontrollfenster-Inhalt. Reine Diagnose, keine
Methodenänderung, nichts überschrieben -- alle Ergebnisse als eigene
Dateien. Basis: identische Rekonstruktion der 3.253-Ereignisse-Analyse
(gleicher Seed wie im Original).
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from src.analysis.impulse_response import (
    CODE,
    K_CONTROLS,
    POST_MIN,
    PRE_MIN,
    RNG_SEED,
    STAMMSTRECKE_IDS,
    Event,
    cluster_events,
    curve_for,
    drop_overlapping,
    find_controls,
    load_code_hits,
    load_state_vector,
    weekday_type,
)
from src.analysis.impulse_response_diagnostics import ALLE_STOERUNGSCODES, STATION_NAMES
from src.utils.constants import DATA_DIR, OUTPUTS_DIR, PLOTS_DIR

EVENTS_FINAL_GLOB = str(DATA_DIR / "02_events" / "final" / "*.parquet")
REL_MINUTES = np.arange(PRE_MIN, POST_MIN + 1, 5)


def reconstruct_base():
    hits = load_code_hits()
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
    return hits, all_events, clean_events, by_key, events_with_controls


def build_curve_frames(events_with_controls, sv):
    """Für jedes Ereignis: eigene Kurve, Kontroll-Mittel-Kurve, Differenz."""
    event_rows, control_rows, diff_rows, meta = [], [], [], []
    for e, controls in events_with_controls:
        ec = curve_for(sv, e.stop_id, e.richtung, e.t0)
        cc_list = [curve_for(sv, e.stop_id, e.richtung, c) for c in controls]
        cc_mean = pd.concat(cc_list, axis=1).mean(axis=1)
        event_rows.append(ec)
        control_rows.append(cc_mean)
        diff_rows.append(ec - cc_mean)
        meta.append({"event_id": e.event_id, "stop_id": e.stop_id, "richtung": e.richtung,
                     "t0": e.t0, "n_trips": e.n_trips, "t0_date": e.t0.date()})
    event_df = pd.DataFrame(event_rows); event_df.columns = REL_MINUTES
    control_df = pd.DataFrame(control_rows); control_df.columns = REL_MINUTES
    diff_df = pd.DataFrame(diff_rows); diff_df.columns = REL_MINUTES
    meta_df = pd.DataFrame(meta)
    return event_df, control_df, diff_df, meta_df


# ---------------------------------------------------------------------------
# 1. Rohkurven ohne Differenz
# ---------------------------------------------------------------------------

def teil1_rohkurven(event_df, control_df, diff_df) -> None:
    print("\n=== 1. Rohkurven ohne Differenz ===")
    mean_event = event_df.mean(axis=0, skipna=True)
    mean_control = control_df.mean(axis=0, skipna=True)
    mean_diff = diff_df.mean(axis=0, skipna=True)

    out = pd.DataFrame({
        "minute_relativ_t0": REL_MINUTES,
        "delay_mean_ereignis": mean_event.values,
        "delay_mean_kontrolle": mean_control.values,
        "differenz": mean_diff.values,
        "differenz_aus_a_minus_b": (mean_event - mean_control).values,
    })
    out.to_csv(OUTPUTS_DIR / "qc_impuls_1_rohkurven.csv", index=False)
    print(out.iloc[[0, 12, 24, 30, 40, 60]].round(1).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True, height_ratios=[2, 1])
    axes[0].plot(REL_MINUTES, mean_event, label="Ereignisfenster (absolut)", color="#C44E52")
    axes[0].plot(REL_MINUTES, mean_control, label="Kontrollfenster (absolut)", color="#4C72B0")
    axes[0].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("delay_mean (s)")
    axes[0].set_title("Rohkurven: Ereignis vs. Kontrolle (absolut)")
    axes[0].legend()

    axes[1].plot(REL_MINUTES, mean_diff, color="#55A868", label="Differenz (= Impulsantwort)")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Minuten relativ zu t0")
    axes[1].set_ylabel("Δ delay_mean (s)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "qc_impuls_1_rohkurven.png", dpi=150)
    plt.close(fig)
    print("Plot: outputs/plots/qc_impuls_1_rohkurven.png")


# ---------------------------------------------------------------------------
# 2. Woher kommt die Schwankung?
# ---------------------------------------------------------------------------

def teil2_schwankung(event_df, control_df, diff_df) -> None:
    print("\n=== 2. Woher kommt die Schwankung? ===")

    n_event = event_df.notna().sum(axis=0)
    n_control = control_df.notna().sum(axis=0)
    n_diff = diff_df.notna().sum(axis=0)
    std_event = event_df.std(axis=0, skipna=True)
    std_control = control_df.std(axis=0, skipna=True)
    std_diff = diff_df.std(axis=0, skipna=True)

    out = pd.DataFrame({
        "minute_relativ_t0": REL_MINUTES,
        "n_ereignis": n_event.values, "n_kontrolle": n_control.values, "n_diff": n_diff.values,
        "std_ereignis": std_event.values, "std_kontrolle": std_control.values, "std_diff": std_diff.values,
        "mean_diff": diff_df.mean(axis=0, skipna=True).values,
    })
    out.to_csv(OUTPUTS_DIR / "qc_impuls_2_fallzahl_und_streuung.csv", index=False)
    print(f"2a) Fallzahl je Zeitpunkt: min={n_diff.min()}, max={n_diff.max()}, "
          f"Median={n_diff.median():.0f} (von max. möglichen {len(diff_df)})")
    print(f"2c) Std der Differenz je Zeitpunkt: min={std_diff.min():.1f}s, max={std_diff.max():.1f}s, "
          f"Median={std_diff.median():.1f}s")
    se_diff = std_diff / np.sqrt(n_diff)
    print(f"    Standardfehler des Mittelwerts (std/sqrt(n)): min={se_diff.min():.1f}s, "
          f"max={se_diff.max():.1f}s -- zum Vergleich, Schwankung der mean_diff-Kurve "
          f"zwischen benachbarten 5-Min-Punkten: {diff_df.mean(axis=0, skipna=True).diff().abs().mean():.1f}s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(REL_MINUTES, out["mean_diff"], color="#55A868")
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Δ delay_mean (s)")
    axes[0].set_title("2a/2c: Impulsantwort mit Fallzahl und Streuung je Zeitpunkt")

    axes[1].plot(REL_MINUTES, out["n_diff"], color="#8172B2")
    axes[1].set_ylabel("n Ereignisse (nicht-NaN)")
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=0.8)

    axes[2].plot(REL_MINUTES, out["std_diff"], color="#DD8452", label="Std der Einzeldifferenzen")
    axes[2].plot(REL_MINUTES, se_diff, color="#C44E52", label="Standardfehler (std/√n)")
    axes[2].set_xlabel("Minuten relativ zu t0")
    axes[2].set_ylabel("Sekunden")
    axes[2].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "qc_impuls_2_fallzahl_streuung.png", dpi=150)
    plt.close(fig)
    print("Plot: outputs/plots/qc_impuls_2_fallzahl_streuung.png")

    # 2d: Autokorrelation der Residuen (mean_diff-Kurve selbst, da das die
    # eigentliche "Schwankungskurve" ist, deren Herkunft geprüft werden soll)
    print("\n2d) Autokorrelation der mean_diff-Kurve (Fahrplantakt-Test 10/20min = Lag 2/4):")
    series = out["mean_diff"].values
    series = series - series.mean()
    n = len(series)
    acf = np.correlate(series, series, mode="full")[n - 1:] / (np.arange(n, 0, -1) * series.var())
    acf_df = pd.DataFrame({"lag_schritte_a_5min": range(len(acf)), "lag_minuten": np.arange(len(acf)) * 5,
                            "autokorrelation": acf})
    acf_df.to_csv(OUTPUTS_DIR / "qc_impuls_2d_autokorrelation.csv", index=False)
    print(acf_df.iloc[:9].round(3).to_string(index=False))
    print(f"ACF bei Lag 2 (10min): {acf[2]:.3f}, Lag 4 (20min): {acf[4]:.3f}, "
          f"Lag 1 (5min): {acf[1]:.3f}, Lag 3 (15min): {acf[3]:.3f}")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.stem(acf_df["lag_minuten"][:25], acf_df["autokorrelation"][:25])
    ax.axhline(0, color="gray", linewidth=0.8)
    for lag_min in [10, 20]:
        ax.axvline(lag_min, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Lag (Minuten)")
    ax.set_ylabel("Autokorrelation")
    ax.set_title("Autokorrelation der Differenzkurve (rot gestrichelt: 10/20-Minuten-Takt)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "qc_impuls_2d_autokorrelation.png", dpi=150)
    plt.close(fig)
    print("Plot: outputs/plots/qc_impuls_2d_autokorrelation.png")


# ---------------------------------------------------------------------------
# 4. Terzile nach Peak-Höhe
# ---------------------------------------------------------------------------

def teil4_terzile(diff_df, meta_df, hits) -> None:
    print("\n=== 4. Terzile nach Peak-Höhe (Differenz bei t0) ===")
    peak = diff_df[0]
    meta_df = meta_df.copy()
    meta_df["peak_t0"] = peak.values
    meta_df["terzil"] = pd.qcut(meta_df["peak_t0"], 3, labels=["niedrig", "mittel", "hoch"])
    print(meta_df.groupby("terzil", observed=True)["peak_t0"].agg(["count", "mean", "min", "max"]).round(1))

    # Dauer je Ereignis (letzte Code-34-Meldung im Cluster minus t0), wie in der
    # Stufe-1-Nachdiagnose
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
    meta_df["dauer_min"] = meta_df["event_id"].map(durations)
    meta_df["station_name"] = meta_df["stop_id"].map(STATION_NAMES)
    meta_df["stunde"] = meta_df["t0"].dt.hour

    meta_df.to_csv(OUTPUTS_DIR / "qc_impuls_4_terzile_metadaten.csv", index=False)

    print("\nMerkmale je Terzil (Mittelwerte):")
    print(meta_df.groupby("terzil", observed=True)[["dauer_min", "n_trips", "stunde"]].mean().round(1))
    print("\nStationsverteilung je Terzil (Anteile):")
    station_ct = pd.crosstab(meta_df["station_name"], meta_df["terzil"], normalize="columns")
    print(station_ct.round(3).to_string())
    station_ct.to_csv(OUTPUTS_DIR / "qc_impuls_4_terzile_je_station.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"niedrig": "#55A868", "mittel": "#DD8452", "hoch": "#C44E52"}
    for terzil, g_idx in meta_df.groupby("terzil", observed=True).groups.items():
        curve = diff_df.loc[g_idx].mean(axis=0, skipna=True)
        ax.plot(REL_MINUTES, curve, color=colors[terzil], linewidth=2,
                label=f"{terzil} (n={len(g_idx)}, peak={meta_df.loc[g_idx,'peak_t0'].mean():.0f}s)")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Minuten relativ zu t0")
    ax.set_ylabel("Δ delay_mean (s)")
    ax.set_title("Nach Peak-Höhe geschichtet (Terzile) -- Achtung: Regression zur Mitte,\n"
                  "aussagekräftig ist die FORM, nicht die Höhe")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "qc_impuls_4_terzile.png", dpi=150)
    plt.close(fig)
    print("Plot: outputs/plots/qc_impuls_4_terzile.png")


# ---------------------------------------------------------------------------
# 5. Was ist in den Kontrollfenstern?
# ---------------------------------------------------------------------------

def teil5_kontrollinhalt(events_with_controls) -> None:
    print("\n=== 5. Was ist in den Kontrollfenstern? ===")
    con = duckdb.connect()
    stop_ids_sql = ",".join(str(x) for x in STAMMSTRECKE_IDS)
    codes_sql = ",".join(str(c) for c in ALLE_STOERUNGSCODES if c != CODE)
    disruption_hits = con.execute(
        f"""
        SELECT stop_id, richtung, update_timestamp,
               unnest(list_filter(message_codes, x -> x IN ({codes_sql}))) AS code
        FROM read_parquet('{EVENTS_FINAL_GLOB}')
        WHERE is_arrival = true AND stop_id IN ({stop_ids_sql}) AND richtung IS NOT NULL
        """
    ).fetchdf()
    disruption_hits["update_timestamp"] = pd.to_datetime(disruption_hits["update_timestamp"], utc=True)
    grouped = {k: v.sort_values("update_timestamp") for k, v in disruption_hits.groupby(["stop_id", "richtung"])}

    event_code_counts: dict[int, int] = {}
    control_code_counts: dict[int, int] = {}
    n_event_windows = 0
    n_control_windows = 0
    minute_event, minute_control = [], []
    tunnel_closure_start = pd.Timestamp("2025-10-18", tz="UTC")
    tunnel_closure_end = pd.Timestamp("2025-10-20", tz="UTC")
    n_controls_in_closure = 0
    n_controls_total = 0

    for e, controls in events_with_controls:
        key = (e.stop_id, e.richtung)
        g = grouped.get(key)
        minute_event.append(e.t0.minute)
        n_event_windows += 1
        if g is not None:
            ws, we = e.t0 + pd.Timedelta(minutes=PRE_MIN), e.t0 + pd.Timedelta(minutes=POST_MIN)
            in_win = g[(g["update_timestamp"] >= ws) & (g["update_timestamp"] <= we)]
            for c in in_win["code"]:
                event_code_counts[c] = event_code_counts.get(c, 0) + 1
        for c_t0 in controls:
            n_control_windows += 1
            n_controls_total += 1
            minute_control.append(c_t0.minute)
            if tunnel_closure_start <= c_t0 <= tunnel_closure_end:
                n_controls_in_closure += 1
            if g is not None:
                ws, we = c_t0 + pd.Timedelta(minutes=PRE_MIN), c_t0 + pd.Timedelta(minutes=POST_MIN)
                in_win = g[(g["update_timestamp"] >= ws) & (g["update_timestamp"] <= we)]
                for code in in_win["code"]:
                    control_code_counts[code] = control_code_counts.get(code, 0) + 1

    from src.analysis.stoerungsinventur import IRIS_CODES
    codes_all = sorted(set(event_code_counts) | set(control_code_counts))
    rows = []
    for code in codes_all:
        ev_c = event_code_counts.get(code, 0)
        ct_c = control_code_counts.get(code, 0)
        rows.append({
            "code": code, "bedeutung": IRIS_CODES.get(code, "UNBEKANNT"),
            "anteil_ereignisfenster": ev_c / n_event_windows,
            "anteil_kontrollfenster": ct_c / n_control_windows,
            "n_ereignisfenster": ev_c, "n_kontrollfenster": ct_c,
        })
    codetab = pd.DataFrame(rows).sort_values("anteil_ereignisfenster", ascending=False)
    codetab.to_csv(OUTPUTS_DIR / "qc_impuls_5a_codes_in_fenstern.csv", index=False)
    print("5a) Top 15 Codes in Ereignis- vs. Kontrollfenstern (Anteil der Fenster mit diesem Code):")
    print(codetab.head(15).round(3).to_string(index=False))

    print(f"\n5b) Minute-innerhalb-der-Stunde: Ereignis-t0 (n={len(minute_event)}) vs. "
          f"Kontroll-t0 (n={len(minute_control)})")
    me = pd.Series(minute_event); mc = pd.Series(minute_control)
    print(f"  Ereignis: mean={me.mean():.1f}, std={me.std():.1f}")
    print(f"  Kontrolle: mean={mc.mean():.1f}, std={mc.std():.1f}")
    print("  (per Konstruktion identisch erwartet -- Kontrollen werden mit derselben "
          "Minute-des-Tages wie ihr Ereignis erzeugt)")
    pd.DataFrame({"minute_event": me}).to_csv(OUTPUTS_DIR / "qc_impuls_5b_minuten_ereignis.csv", index=False)
    pd.DataFrame({"minute_control": mc}).to_csv(OUTPUTS_DIR / "qc_impuls_5b_minuten_kontrolle.csv", index=False)

    print(f"\n5c) Kontrollfenster mit t0 im Tunnelsperrungs-Zeitraum (18.-19.10.2025): "
          f"{n_controls_in_closure} von {n_controls_total} ({n_controls_in_closure/n_controls_total:.3%})")


if __name__ == "__main__":
    print("Rekonstruiere Basis-Analyse (Original-Fenster, gleicher Seed)...")
    hits, all_events, clean_events, by_key, events_with_controls = reconstruct_base()
    print(f"{len(events_with_controls)} Ereignisse mit Kontrollen")

    print("\nLade Zustandsvektor und baue Kurven-Frames...")
    sv = load_state_vector()
    event_df, control_df, diff_df, meta_df = build_curve_frames(events_with_controls, sv)

    teil1_rohkurven(event_df, control_df, diff_df)
    teil2_schwankung(event_df, control_df, diff_df)
    teil4_terzile(diff_df, meta_df, hits)
    teil5_kontrollinhalt(events_with_controls)

    print("\nFertig (Teile 1,2,4,5). Teil 3 (60-Minuten-Fenster) läuft separat.")

"""Abschnitt 11.4: Vertiefung des Zugebene-Plausibilitaetstests aus 11.3.

11.3 fand: die Korrelation Vorgaenger->Folgezug-Verspaetung an derselben Station ist ueber
alle Headway-Bins hinweg erstaunlich konstant (r=0,38-0,51), OHNE den erwarteten monotonen
Abfall mit laengerem Headway. Drei Zusatztests, um das einzuordnen:

11.4.1 Gemeinsame Ursache vs. direkte Uebertragung: r(delay_n, delay_n+k) fuer k in
       {1,2,3,5,10} an derselben Station/Richtung. Faellt r schnell mit k (direkte,
       kurzreichweitige Ansteckung) oder bleibt es hoch (gemeinsame, langsam variierende
       Ursache, z.B. Betriebslage des Tages)?
11.4.2 Zuwachs statt Niveau: haengt der Verspaetungs-ZUWACHS delta = delay(Station k) -
       delay(Station k-1) eines Zuges zwischen zwei benachbarten Stationen von der
       Verspaetung seines VORGAENGER-Zuges an Station k-1 ab (headway-stratifiziert)? Das ist
       ein kausalerer Test als die reine Niveau-Korrelation aus 11.3 (haelt der Vorgaenger den
       Folgezug im Tunnel auf?).
11.4.3 Gleiche Linie vs. andere Linie: erklaert die "gleiche Linie, ~20min Abstand"-Situation
       (S-Bahn-Grundtakt) einen Teil der 20-Minuten-Korrelation/-Schwingung (Abschnitt
       3.3/9.6/10.2/11.2), im fairen Vergleich zu Fremd-Linien-Paaren mit AEHNLICHEM Headway?

Nur lesender Zugriff auf data/02_events/final/, src/analysis/var/stations.py,
src/analysis/var/plausibilitaet_zugebene.py (Wiederverwendung der 11.3-Hilfsfunktionen fuer
Konsistenz). Neue Outputs unter outputs/var/plausibilitaet/ mit Praefix "vertiefung_", neue
Plots 24a/24b/24c.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.constants import DATA_DIR, PROJECT_ROOT
from .stations import STATION_ORDER_WEST_OST, STOP_ID_TO_NAME
from .plausibilitaet_zugebene import (
    EVENTS_GLOB, STAMM_IDS, BIN_EDGES, BIN_LABELS, MAX_HEADWAY_MIN,
    load_events, build_pairs, _corr_slope_with_ci,
)

OUT_DIR = PROJECT_ROOT / "outputs" / "var" / "plausibilitaet"
PLOT_DIR = PROJECT_ROOT / "outputs" / "var" / "plots"

K_VALUES = [1, 2, 3, 5, 10]
MAX_GAP_MIN = 360.0  # 6h: ueberspringt Tagesgrenzen-Luecken, siehe Docstring 11.4.1

ADJACENT_PAIRS = [
    ("Pasing", "Laim"),
    ("Hauptbahnhof_tief", "Karlsplatz_Stachus"),
    ("Isartor", "Rosenheimer_Platz"),
    ("Karlsplatz_Stachus", "Marienplatz"),
]

FINE_BIN_EDGES = [0, 5, 10, 15, 18, 22, 27, 35, MAX_HEADWAY_MIN]
FINE_BIN_LABELS = ["<5min", "5-10min", "10-15min", "15-18min", "18-22min",
                    "22-27min", "27-35min", "35-60min"]


# --------------------------------------------------------------------------------------
# 11.4.1 Gemeinsame Ursache vs. direkte Uebertragung (r als Funktion von k)
# --------------------------------------------------------------------------------------

def analyze_k_lag(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fuer k in K_VALUES: r(delay_n, delay_n+k) an derselben Station/Richtung, gepoolt ueber
    alle 12 Stationen/2 Richtungen. Zusaetzlich Median der Realzeit-Distanz zwischen Zug n und
    Zug n+k."""
    df = df.sort_values(["station_name", "richtung", "time_schedule"]).reset_index(drop=True)
    g = df.groupby(["station_name", "richtung"], sort=False)

    rows = []
    gap_rows = []
    for k in K_VALUES:
        trip_lag = g["trip_id"].shift(k)
        time_lag = g["time_schedule"].shift(k)
        delay_lag = g["delay"].shift(k)

        sub = pd.DataFrame({
            "station_name": df["station_name"],
            "richtung": df["richtung"],
            "trip_id": df["trip_id"],
            "trip_id_lag": trip_lag,
            "delay": df["delay"],
            "delay_lag": delay_lag,
            "gap_min": (df["time_schedule"] - time_lag).dt.total_seconds() / 60.0,
        }).dropna(subset=["trip_id_lag"])

        n_before = len(sub)
        sub = sub[(sub["trip_id_lag"] != sub["trip_id"]) &
                   (sub["gap_min"] > 0) & (sub["gap_min"] <= MAX_GAP_MIN)]
        n_excl = n_before - len(sub)

        res = _corr_slope_with_ci(sub["delay_lag"].to_numpy(dtype=float),
                                   sub["delay"].to_numpy(dtype=float))
        rows.append({"k": k, "n_ausgeschlossen": n_excl, **res})
        gap_rows.append({
            "k": k,
            "gap_median_min": float(sub["gap_min"].median()),
            "gap_p25_min": float(sub["gap_min"].quantile(0.25)),
            "gap_p75_min": float(sub["gap_min"].quantile(0.75)),
        })

    return pd.DataFrame(rows), pd.DataFrame(gap_rows)


def plot_r_vs_k(res: pd.DataFrame, gaps: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = res["k"].to_numpy()
    ax.errorbar(x, res["r"], yerr=[res["r"] - res["r_ci_lo"], res["r_ci_hi"] - res["r"]],
                marker="o", color="#2A6F97", capsize=4, linewidth=1.5)
    ax.set_xlabel("k (n+k-ter Zug an derselben Station/Richtung)")
    ax.set_ylabel("Pearson-Korrelation r(delay_n, delay_n+k)")
    ax.set_xticks(x)
    ax.axhline(0, color="grey", linewidth=0.7)
    ax.set_ylim(0, max(0.6, res["r_ci_hi"].max() * 1.1))

    for _, row in gaps.iterrows():
        ax.annotate(f"~{row['gap_median_min']:.0f} min",
                     xy=(row["k"], 0.02), xycoords=("data", "axes fraction"),
                     ha="center", fontsize=8, color="#555555")

    ax.set_title("Plot 24a: Korrelation Zug n <-> Zug n+k nach k (gepoolt, 12 Stationen)\n"
                 "Beschriftung unten: mediane Realzeit-Distanz zwischen Zug n und n+k "
                 "[Abschnitt 11.4.1]")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# 11.4.2 Zuwachs statt Niveau
# --------------------------------------------------------------------------------------

def load_trip_delta(upstream_id: int, downstream_id: int) -> pd.DataFrame:
    """Fuer jede trip_id mit Ankunft an BEIDEN Stationen (upstream zuerst durchfahren, ueber
    stop_sequence bestimmt, konsistent zu plausibilitaet_fahrzeit.py): delta = delay(downstream)
    - delay(upstream), sowie die Verspaetung/Zeit an der upstream-Station fuer den Join mit den
    Vorgaenger-Paaren."""
    con = duckdb.connect()
    q = f"""
    WITH ev AS (
        SELECT trip_id, stop_id, stop_sequence, time_schedule, delay, richtung
        FROM read_parquet('{EVENTS_GLOB}')
        WHERE is_arrival = true AND is_cancelled = false
          AND stop_id IN ({upstream_id}, {downstream_id})
    ),
    piv AS (
        SELECT trip_id,
            max(CASE WHEN stop_id = {upstream_id} THEN time_schedule END) AS t_up,
            max(CASE WHEN stop_id = {upstream_id} THEN stop_sequence END) AS seq_up,
            max(CASE WHEN stop_id = {upstream_id} THEN delay END) AS delay_up,
            max(CASE WHEN stop_id = {upstream_id} THEN richtung END) AS richtung_up,
            max(CASE WHEN stop_id = {downstream_id} THEN time_schedule END) AS t_down,
            max(CASE WHEN stop_id = {downstream_id} THEN stop_sequence END) AS seq_down,
            max(CASE WHEN stop_id = {downstream_id} THEN delay END) AS delay_down,
            count(*) AS n_events
        FROM ev GROUP BY trip_id
    )
    SELECT * FROM piv WHERE t_up IS NOT NULL AND t_down IS NOT NULL AND n_events = 2
    """
    df = con.execute(q).fetchdf()
    return df


def analyze_pair(name_up: str, name_down: str) -> pd.DataFrame:
    """Fuer ein geografisch benachbartes Stationspaar (name_up vor name_down in WEST_OST
    Reihenfolge): fuer beide Richtungen (Ost: upstream=name_up; West: upstream=name_down)
    delta = delay(downstream)-delay(upstream) je trip_id, gematcht mit dem Vorgaenger-Zug an
    der upstream-Station (Verspaetung + Headway, wie in 11.3 build_pairs)."""
    id_up = {v: k for k, v in STOP_ID_TO_NAME.items()}[name_up]
    id_down = {v: k for k, v in STOP_ID_TO_NAME.items()}[name_down]

    raw = load_trip_delta(id_up, id_down)
    # stop_sequence bestimmt die tatsaechliche Fahrtrichtung dieser trip_id
    raw["richtung_seq"] = np.where(raw["seq_up"] < raw["seq_down"], "Ost", "West")

    out_frames = []
    for richtung in ["Ost", "West"]:
        sub = raw[raw["richtung_seq"] == richtung].copy()
        if richtung == "Ost":
            sub["delta"] = sub["delay_down"] - sub["delay_up"]
            upstream_station, upstream_time, upstream_delay = name_up, sub["t_up"], sub["delay_up"]
        else:
            # West: Zug faehrt zuerst name_down, dann name_up -> upstream = name_down
            sub["delta"] = sub["delay_up"] - sub["delay_down"]
            upstream_station, upstream_time, upstream_delay = name_down, sub["t_down"], sub["delay_down"]

        sub["upstream_station"] = upstream_station
        sub["upstream_time"] = upstream_time
        sub["upstream_delay"] = upstream_delay
        out_frames.append(sub[["trip_id", "richtung_seq", "upstream_station",
                                 "upstream_time", "upstream_delay", "delta"]])
    trips = pd.concat(out_frames, ignore_index=True)

    # Vorgaenger-Zug je upstream-Station/Richtung: exakt wie build_pairs in 11.3.
    ev = load_events()
    ev_up = ev[ev["station_name"].isin([name_up, name_down])]
    pairs = build_pairs(ev_up)  # station_name, richtung, trip_id (=aktueller Zug an upstream),
                                  # headway_min, headway_bin, delay_prev, delay_next

    merged = trips.merge(
        pairs[["station_name", "richtung", "trip_id", "headway_min", "headway_bin", "delay_prev"]],
        left_on=["upstream_station", "richtung_seq", "trip_id"],
        right_on=["station_name", "richtung", "trip_id"],
        how="inner",
    )
    merged["stationspaar"] = f"{name_up}-{name_down}"
    return merged[["stationspaar", "richtung_seq", "trip_id", "upstream_station",
                    "headway_min", "headway_bin", "delay_prev", "delta"]]


def analyze_delta_by_headway(pairs_delta: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    group_cols = (by or []) + ["headway_bin"]
    rows = []
    for key, sub in pairs_delta.groupby(group_cols, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        res = _corr_slope_with_ci(sub["delay_prev"].to_numpy(dtype=float),
                                   sub["delta"].to_numpy(dtype=float))
        row = dict(zip(group_cols, key))
        row.update(res)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_delta_vs_headway(pooled: pd.DataFrame, out_path: Path):
    pooled = pooled.set_index("headway_bin").reindex(BIN_LABELS).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(pooled))

    ax = axes[0]
    ax.errorbar(x, pooled["r"], yerr=[pooled["r"] - pooled["r_ci_lo"], pooled["r_ci_hi"] - pooled["r"]],
                marker="o", color="#2A6F97", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(pooled["headway_bin"], rotation=20)
    ax.set_ylabel("r(delay_vorgaenger, delta_folgezug)")
    ax.set_title("Korrelation nach Headway-Bin")
    ax.axhline(0, color="grey", linewidth=0.7)

    ax2 = axes[1]
    ax2.errorbar(x, pooled["slope"], yerr=[pooled["slope"] - pooled["slope_ci_lo"], pooled["slope_ci_hi"] - pooled["slope"]],
                 marker="o", color="#C4471C", capsize=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(pooled["headway_bin"], rotation=20)
    ax2.set_ylabel("OLS-Steigung delta ~ delay_vorgaenger")
    ax2.set_title("Steigung nach Headway-Bin")
    ax2.axhline(0, color="grey", linewidth=0.7)

    fig.suptitle("Plot 24b: Verspaetungs-ZUWACHS des Folgezugs zwischen Nachbarstationen vs.\n"
                 "Verspaetung des Vorgaenger-Zugs, nach Headway-Bin (4 Stationspaare gepoolt) "
                 "[Abschnitt 11.4.2]")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# 11.4.3 Gleiche Linie vs. andere Linie
# --------------------------------------------------------------------------------------

def load_events_with_line() -> pd.DataFrame:
    con = duckdb.connect()
    q = f"""
        SELECT stop_id, richtung, trip_id, time_schedule, delay, line
        FROM read_parquet('{EVENTS_GLOB}')
        WHERE is_arrival = true AND is_cancelled = false AND richtung IS NOT NULL
          AND stop_id IN ({STAMM_IDS})
        ORDER BY stop_id, richtung, time_schedule
    """
    df = con.execute(q).fetchdf()
    df["station_name"] = df["stop_id"].map(STOP_ID_TO_NAME)
    return df


def build_pairs_with_line(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["station_name", "richtung", "time_schedule"]).reset_index(drop=True)
    g = df.groupby(["station_name", "richtung"], sort=False)
    df["trip_id_prev"] = g["trip_id"].shift(1)
    df["time_schedule_prev"] = g["time_schedule"].shift(1)
    df["delay_prev"] = g["delay"].shift(1)
    df["line_prev"] = g["line"].shift(1)

    pairs = df.dropna(subset=["trip_id_prev"]).copy()
    pairs = pairs[pairs["trip_id_prev"] != pairs["trip_id"]]
    pairs["headway_min"] = (pairs["time_schedule"] - pairs["time_schedule_prev"]).dt.total_seconds() / 60.0
    pairs = pairs.rename(columns={"delay": "delay_next", "line": "line_next"})
    pairs = pairs[(pairs["headway_min"] > 0) & (pairs["headway_min"] <= MAX_HEADWAY_MIN)]
    pairs["headway_bin"] = pd.cut(pairs["headway_min"], bins=FINE_BIN_EDGES, labels=FINE_BIN_LABELS, right=True)
    pairs["gleiche_linie"] = pairs["line_next"] == pairs["line_prev"]
    return pairs[["station_name", "richtung", "trip_id_prev", "trip_id", "headway_min",
                  "headway_bin", "delay_prev", "delay_next", "line_prev", "line_next", "gleiche_linie"]]


def analyze_line_groups(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gleiche in [True, False]:
        for b in FINE_BIN_LABELS:
            sub = pairs[(pairs["gleiche_linie"] == gleiche) & (pairs["headway_bin"] == b)]
            res = _corr_slope_with_ci(sub["delay_prev"].to_numpy(dtype=float),
                                       sub["delay_next"].to_numpy(dtype=float))
            rows.append({"gruppe": "gleiche_linie" if gleiche else "andere_linie",
                         "headway_bin": b, **res})
    return pd.DataFrame(rows)


def plot_line_groups(res: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"gleiche_linie": "#2A6F97", "andere_linie": "#C4471C"}
    for gruppe, color in colors.items():
        sub = res[res["gruppe"] == gruppe].set_index("headway_bin").reindex(FINE_BIN_LABELS).reset_index()
        x = np.arange(len(sub))
        ax.errorbar(x, sub["r"], yerr=[sub["r"] - sub["r_ci_lo"], sub["r_ci_hi"] - sub["r"]],
                     marker="o", color=color, capsize=3, label=gruppe.replace("_", " "))
    ax.axvspan(3.5, 4.5, color="#999999", alpha=0.15, label="~20min-Bin (Grundtakt-Abstand)")
    ax.set_xticks(np.arange(len(FINE_BIN_LABELS)))
    ax.set_xticklabels(FINE_BIN_LABELS, rotation=25)
    ax.set_ylabel("Pearson-Korrelation r(delay_prev, delay_next)")
    ax.axhline(0, color="grey", linewidth=0.7)
    ax.legend(fontsize=9)
    ax.set_title("Plot 24c: Vorgaenger->Folgezug-Korrelation, gleiche vs. andere Linie,\n"
                 "nach Headway-Bin (fairer Vergleich bei ~20min) [Abschnitt 11.4.3]")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 11.4.1: r(k) fuer k in", K_VALUES, "===")
    df = load_events()
    print(f"{len(df):,} Ankunfts-Ereignisse geladen.")
    res_k, gaps_k = analyze_k_lag(df)
    res_k.to_csv(OUT_DIR / "vertiefung_zugebene_r_vs_k.csv", index=False)
    gaps_k.to_csv(OUT_DIR / "vertiefung_zugebene_realzeit_abstand_k.csv", index=False)
    print(res_k.round(4).to_string(index=False))
    print(gaps_k.round(1).to_string(index=False))
    plot_r_vs_k(res_k, gaps_k, PLOT_DIR / "24a_zugebene_r_vs_k.png")

    print("\n=== 11.4.2: Zuwachs vs. Vorgaenger-Verspaetung, 4 Stationspaare ===")
    all_deltas = []
    for name_up, name_down in ADJACENT_PAIRS:
        d = analyze_pair(name_up, name_down)
        print(f"{name_up}-{name_down}: {len(d):,} gematchte Trips (beide Richtungen)")
        all_deltas.append(d)
    pairs_delta = pd.concat(all_deltas, ignore_index=True)
    pairs_delta.to_csv(OUT_DIR / "vertiefung_zuwachs_vorgaenger_rohdaten.csv.gz", index=False, compression="gzip")

    per_pair = analyze_delta_by_headway(pairs_delta, by=["stationspaar"])
    per_pair.to_csv(OUT_DIR / "vertiefung_zuwachs_vorgaenger_je_stationspaar.csv", index=False)
    print(per_pair.round(4).to_string(index=False))

    pooled_delta = analyze_delta_by_headway(pairs_delta, by=None)
    pooled_delta.to_csv(OUT_DIR / "vertiefung_zuwachs_vorgaenger_gepoolt.csv", index=False)
    print("\nGepoolt (4 Stationspaare, beide Richtungen):")
    print(pooled_delta.round(4).to_string(index=False))
    plot_delta_vs_headway(pooled_delta, PLOT_DIR / "24b_zugebene_zuwachs_vorgaenger.png")

    print("\n=== 11.4.3: gleiche Linie vs. andere Linie ===")
    df_line = load_events_with_line()
    pairs_line = build_pairs_with_line(df_line)
    pairs_line.to_csv(OUT_DIR / "vertiefung_gleiche_linie_rohdaten.csv.gz", index=False, compression="gzip")
    print("Verteilung gleiche_linie x headway_bin:")
    print(pairs_line.groupby(["gleiche_linie", "headway_bin"], observed=True).size().to_string())

    res_line = analyze_line_groups(pairs_line)
    res_line.to_csv(OUT_DIR / "vertiefung_gleiche_linie_vs_andere_linie.csv", index=False)
    print(res_line.round(4).to_string(index=False))
    plot_line_groups(res_line, PLOT_DIR / "24c_zugebene_gleiche_vs_andere_linie.png")

    print("\nAlle Outputs unter outputs/var/plausibilitaet/vertiefung_* und outputs/var/plots/24[abc]_*.png")
    return dict(res_k=res_k, gaps_k=gaps_k, pairs_delta=pairs_delta, per_pair=per_pair,
                pooled_delta=pooled_delta, pairs_line=pairs_line, res_line=res_line)


if __name__ == "__main__":
    main()

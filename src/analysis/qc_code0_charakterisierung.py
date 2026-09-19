"""
Diagnose: empirische Charakterisierung von message_code 0 (551.507
Vorkommen auf der Stammstrecke, keine bekannte Bedeutung in der
IRIS-Referenz). Ziel: Verhalten beschreiben (Auslöser- vs. Folge-/
Statuscode-Muster), OHNE die Bedeutung zu erraten.

Wiederverwendet Cluster-/Lift-/Folge-Logik aus stoerungsinventur.py.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from src.analysis.impulse_response_diagnostics import STATION_NAMES
from src.analysis.stoerungsinventur import (
    ALLE_STOERUNGSCODES,
    CLUSTER_GAP_MIN,
    IRIS_CODES,
    STAMMSTRECKE_IDS,
    build_episodes_for_code,
    load_hits,
)
from src.utils.constants import DATA_DIR, OUTPUTS_DIR, PLOTS_DIR

EVENTS_FINAL_GLOB = str(DATA_DIR / "02_events" / "final" / "*.parquet")
CODE_ZERO = 0
ISOLATION_WINDOW_MIN = 60


def run() -> None:
    print("Lade alle Codes (inkl. 0) an der Stammstrecke...")
    hits_all = load_hits(STAMMSTRECKE_IDS)
    hits_relevant = hits_all[hits_all["code"].isin(ALLE_STOERUNGSCODES + [CODE_ZERO])].copy()

    hits0 = hits_relevant[hits_relevant["code"] == CODE_ZERO]
    print(f"{len(hits0):,} Code-0-Zeilen an der Stammstrecke")

    # --- Episoden für Code 0 und alle anderen Störungscodes ---
    print("\nBaue Episoden je Code (0 + 2-69)...")
    episodes_by_code: dict[int, pd.DataFrame] = {}
    for code, g in hits_relevant.groupby("code"):
        episodes_by_code[code] = build_episodes_for_code(g)

    ep0 = episodes_by_code[CODE_ZERO]
    print(f"Code 0: {len(ep0)} Episoden, Dauer Median={ep0['dauer_min'].median():.1f}min, "
          f"Q25={ep0['dauer_min'].quantile(0.25):.1f}, Q75={ep0['dauer_min'].quantile(0.75):.1f}, "
          f"Max={ep0['dauer_min'].max():.1f}")
    ep0_out = ep0.drop(columns=["stationen"]).copy()
    ep0_out.to_csv(OUTPUTS_DIR / "qc_code0_episoden.csv", index=False)

    # --- Lift: Code 0 gegen alle anderen (Aktivitätsmatrix wie in 4c) ---
    print("\nBaue Aktivitätsmatrix (5-Minuten-Fenster) für Lift-Berechnung...")
    year_start = hits_relevant["update_timestamp"].min().floor("5min")
    year_end = hits_relevant["update_timestamp"].max().ceil("5min")
    full_index = pd.date_range(year_start, year_end, freq="5min")
    n = len(full_index)

    activity = {}
    for code, ep in episodes_by_code.items():
        if len(ep) == 0:
            activity[code] = np.zeros(n, dtype=bool)
            continue
        starts = full_index.searchsorted(ep["t_start"].dt.floor("5min").values)
        ends = full_index.searchsorted(ep["t_end"].dt.ceil("5min").values, side="right")
        ends = np.clip(ends, 0, n - 1)
        diff = np.zeros(n + 1, dtype=np.int32)
        np.add.at(diff, starts, 1)
        np.add.at(diff, ends + 1, -1)
        activity[code] = np.cumsum(diff[:n]) > 0

    act0 = activity[CODE_ZERO]
    marg0 = act0.mean()
    lift_rows = []
    for code, arr in activity.items():
        if code == CODE_ZERO:
            continue
        marg = arr.mean()
        joint = (act0 & arr).mean()
        expected = marg0 * marg
        lift = joint / expected if expected > 0 else np.nan
        lift_rows.append({"code": code, "bedeutung": IRIS_CODES.get(code, "UNBEKANNT"),
                           "gemeinsame_fenster": int((act0 & arr).sum()), "lift": lift})
    lift_df = pd.DataFrame(lift_rows).sort_values("lift", ascending=False)
    lift_df.to_csv(OUTPUTS_DIR / "qc_code0_lift.csv", index=False)
    print("\nTop 15 Codes mit höchstem Lift gegenüber Code 0 (>=20 gemeinsame Fenster):")
    print(lift_df[lift_df["gemeinsame_fenster"] >= 20].head(15).round(2).to_string(index=False))

    # --- Folge-Analyse: Code 0 vor/nach anderen Codes, je Station ---
    print("\nBaue Episodenstarts je Code+Station für Folge-Analyse...")
    ep_starts_by_station: dict[tuple[int, int], np.ndarray] = {}
    for code, g in hits_relevant.groupby("code"):
        for stop_id, gg in g.groupby("stop_id"):
            ep = build_episodes_for_code(gg)
            ep_starts_by_station[(code, stop_id)] = np.sort(ep["t_start"].values.astype("datetime64[ns]"))

    win_ns = np.timedelta64(ISOLATION_WINDOW_MIN, "m")
    other_codes = [c for c in episodes_by_code if c != CODE_ZERO]

    def folge_rate(code_a: int, code_b: int) -> tuple[float, int]:
        n_folgt = 0
        n_a_total = 0
        for stop_id in STAMMSTRECKE_IDS:
            starts_a = ep_starts_by_station.get((code_a, stop_id))
            starts_b = ep_starts_by_station.get((code_b, stop_id))
            if starts_a is None or len(starts_a) == 0:
                continue
            n_a_total += len(starts_a)
            if starts_b is None or len(starts_b) == 0:
                continue
            window_ends = starts_a + win_ns
            lo = np.searchsorted(starts_b, starts_a, side="right")
            hi = np.searchsorted(starts_b, window_ends, side="right")
            n_folgt += int(np.sum(hi > lo))
        return (n_folgt / n_a_total if n_a_total > 0 else np.nan), n_a_total

    folge_rows = []
    for code in other_codes:
        rate_0_folgt_code, n_code = folge_rate(code, CODE_ZERO)   # Code 0 folgt auf `code`
        rate_code_folgt_0, n_0 = folge_rate(CODE_ZERO, code)      # `code` folgt auf Code 0
        folge_rows.append({
            "code": code, "bedeutung": IRIS_CODES.get(code, "UNBEKANNT"),
            "anteil_0_folgt_auf_code": rate_0_folgt_code, "n_code_episoden": n_code,
            "anteil_code_folgt_auf_0": rate_code_folgt_0, "n_0_episoden": n_0,
        })
    folge_df = pd.DataFrame(folge_rows).sort_values("anteil_0_folgt_auf_code", ascending=False)
    folge_df.to_csv(OUTPUTS_DIR / "qc_code0_folgeanalyse.csv", index=False)
    print("\nTop 10: Code 0 folgt häufig auf ... (>=50 Episoden des anderen Codes):")
    print(folge_df[folge_df["n_code_episoden"] >= 50].head(10).round(3).to_string(index=False))
    print("\nTop 10: ... folgt häufig auf Code 0 (>=50 Code-0-Episoden):")
    print(folge_df[folge_df["n_0_episoden"] >= 50].sort_values("anteil_code_folgt_auf_0", ascending=False)
          .head(10).round(3).to_string(index=False))

    mean_0_folgt = folge_df.loc[folge_df["n_code_episoden"] >= 50, "anteil_0_folgt_auf_code"].mean()
    mean_folgt_0 = folge_df.loc[folge_df["n_0_episoden"] >= 50, "anteil_code_folgt_auf_0"].mean()
    print(f"\nMittelwert 'Code 0 folgt auf andere' (>=50 Episoden): {mean_0_folgt:.3f}")
    print(f"Mittelwert 'andere folgen auf Code 0' (>=50 Episoden): {mean_folgt_0:.3f}")

    # --- Zeitliche Muster ---
    print("\nZeitliche Muster (Code 0)...")
    hits0 = hits0.copy()
    hits0["hour"] = hits0["update_timestamp"].dt.hour
    hits0["weekday"] = hits0["update_timestamp"].dt.day_name()
    hits0["month"] = hits0["update_timestamp"].dt.month
    hour_dist = hits0["hour"].value_counts(normalize=True).sort_index()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_dist = hits0["weekday"].value_counts(normalize=True).reindex(weekday_order)
    month_dist = hits0["month"].value_counts(normalize=True).sort_index()
    hour_dist.to_csv(OUTPUTS_DIR / "qc_code0_stunde.csv")
    weekday_dist.to_csv(OUTPUTS_DIR / "qc_code0_wochentag.csv")
    month_dist.to_csv(OUTPUTS_DIR / "qc_code0_monat.csv")
    print("Anteil je Stunde (Auszug):")
    print(hour_dist.round(3).to_string())

    plot_zeitmuster(hour_dist, weekday_dist, month_dist)

    # --- Mittlere Verspätung: Code 0 vs. kein Störungscode ---
    print("\nMittlere Verspätung: Halte mit Code 0 vs. Halte ohne jeden Störungscode...")
    con = duckdb.connect()
    stop_ids_sql = ",".join(str(x) for x in STAMMSTRECKE_IDS)
    codes_sql = ",".join(str(c) for c in ALLE_STOERUNGSCODES)
    q_mit_0 = f"""
        SELECT avg(delay) as mean_delay, median(delay) as median_delay, count(*) as n
        FROM read_parquet('{EVENTS_FINAL_GLOB}')
        WHERE is_arrival=true AND stop_id IN ({stop_ids_sql}) AND NOT is_cancelled
          AND list_contains(message_codes, 0)
    """
    q_ohne = f"""
        SELECT avg(delay) as mean_delay, median(delay) as median_delay, count(*) as n
        FROM read_parquet('{EVENTS_FINAL_GLOB}')
        WHERE is_arrival=true AND stop_id IN ({stop_ids_sql}) AND NOT is_cancelled
          AND NOT list_contains(message_codes, 0)
          AND len(list_filter(message_codes, x -> x IN ({codes_sql}))) = 0
    """
    r_mit = con.execute(q_mit_0).fetchdf().iloc[0]
    r_ohne = con.execute(q_ohne).fetchdf().iloc[0]
    print(f"MIT Code 0:        mean={r_mit['mean_delay']:.1f}s, median={r_mit['median_delay']:.1f}s, n={r_mit['n']:,}")
    print(f"OHNE Störungscode: mean={r_ohne['mean_delay']:.1f}s, median={r_ohne['median_delay']:.1f}s, n={r_ohne['n']:,}")
    pd.DataFrame([
        {"gruppe": "mit_code_0", **r_mit.to_dict()},
        {"gruppe": "ohne_stoerungscode", **r_ohne.to_dict()},
    ]).to_csv(OUTPUTS_DIR / "qc_code0_verspaetung_vergleich.csv", index=False)


def plot_zeitmuster(hour_dist, weekday_dist, month_dist) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].bar(hour_dist.index, hour_dist.values, color="#4C72B0")
    axes[0].set_xlabel("Stunde"); axes[0].set_title("Code 0 nach Tagesstunde")
    axes[1].bar(range(len(weekday_dist)), weekday_dist.values, color="#55A868")
    axes[1].set_xticks(range(len(weekday_dist))); axes[1].set_xticklabels(weekday_dist.index, rotation=45)
    axes[1].set_title("Code 0 nach Wochentag")
    axes[2].bar(month_dist.index, month_dist.values, color="#C44E52")
    axes[2].set_xlabel("Monat"); axes[2].set_title("Code 0 nach Monat")
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / "qc_code0_zeitmuster.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()

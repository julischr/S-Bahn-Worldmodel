"""
Störungsinventur: reine Bestandsaufnahme der message_codes in events_final,
KEINE Modellierung, KEINE Impulsantwort. Liefert die Entscheidungsgrundlage
dafür, welcher Code sich für eine Impulsantwort-Analyse eignet.

Datengrundlage: events_final, is_arrival=true (eine Zeile je Halt-Ereignis),
Jahr 2025. Primärer Scope: die 12 Stammstreckenhalte. Abschnitt 1
(Bestandsaufnahme) zusätzlich einmal für alle 152 Münchner Stationen zum
Vergleich -- die Abschnitte 2-5 (Episoden, Muster, Überschneidungen, Eignung)
laufen bewusst NUR auf der Stammstrecke, das ist der primäre Analyse-Scope
der bisherigen Arbeit (Zustandsvektor, Impulsantwort) und wird hier aus
Aufwandsgründen nicht auf alle 152 Stationen ausgeweitet. Diese
Scope-Entscheidung wird hier explizit dokumentiert, nicht stillschweigend
getroffen.

"Störungscode" = IRIS-Codes 2-69 (siehe ALLE_STOERUNGSCODES in
impulse_response_diagnostics.py) -- schließt den Platzhalter 1 ("Nähere
Informationen in Kürze"), die undokumentierten 0/1000/1001 und die
Komfort-/Ausstattungshinweise 70-98 (WLAN, Sitzplatzklasse, Fahrradmitnahme
etc., keine Betriebsstörungen) aus. Diese ausgeschlossenen Codes werden in
Abschnitt 1 trotzdem mit aufgeführt (vollständige Bestandsaufnahme aller
vorkommenden Codes), aber in den Abschnitten 2-5 nicht weiter analysiert.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from src.analysis.impulse_response_diagnostics import ALLE_STOERUNGSCODES, STATION_NAMES
from src.utils.constants import DATA_DIR, OUTPUTS_DIR, PLOTS_DIR

EVENTS_FINAL_GLOB = str(DATA_DIR / "02_events" / "final" / "*.parquet")

STAMMSTRECKE_IDS = (
    8004128, 8004129, 8004131, 8004132, 8004151, 8000262,
    8004135, 8004136, 8004134, 8004179, 8098263, 8004158,
)
CLUSTER_GAP_MIN = 15
ISOLATION_WINDOW_MIN = 60
TOP_N = 20

# IRIS-Meldungscode-Referenz, Quelle: Travel::Status::DE::IRIS::Result (derf,
# github.com/derf/Travel-Status-DE-IRIS), identisch zur Tabelle in explore.py.
IRIS_CODES = {
    1: "Nähere Informationen in Kürze", 2: "Polizeieinsatz", 3: "Feuerwehreinsatz auf der Strecke",
    4: "Kurzfristiger Personalausfall", 5: "Ärztliche Versorgung eines Fahrgastes", 6: "Betätigen der Notbremse",
    7: "Unbefugte Personen auf der Strecke", 8: "Notarzteinsatz auf der Strecke", 9: "Streikauswirkungen",
    10: "Tiere auf der Strecke", 11: "Unwetter", 12: "Warten auf ein verspätetes Schiff",
    13: "Pass- und Zollkontrolle", 14: "Defekt am Bahnhof", 15: "Beeinträchtigung durch Vandalismus",
    16: "Entschärfung einer Fliegerbombe", 17: "Beschädigung einer Brücke", 18: "Umgestürzter Baum auf der Strecke",
    19: "Unfall an einem Bahnübergang", 20: "Tiere im Gleis", 21: "Warten auf Anschlussreisende",
    22: "Witterungsbedingte Beeinträchtigungen", 23: "Betriebsstabilisierung", 24: "Verspätung im Ausland",
    25: "Bereitstellung weiterer Wagen", 26: "Abhängen von Wagen", 27: "Technische Störung am Bus",
    28: "Gegenstände auf der Strecke", 29: "Ersatzverkehr mit Bus ist eingerichtet", 30: "Personalausfall im Stellwerk",
    31: "Bauarbeiten", 32: "Längere Haltezeit am Bahnhof", 33: "Defekt an der Oberleitung",
    34: "Defekt an einem Signal", 35: "Streckensperrung", 36: "Technische Störung am Zug",
    37: "Kurzfristiger Fahrzeugausfall", 38: "Defekt an der Strecke", 39: "Stau / Hohes Verkehrsaufkommen",
    40: "Defektes Stellwerk", 41: "Defekt an einem Bahnübergang", 42: "Außerplanmäßige Geschwindigkeitsbeschränkung",
    43: "Verspätung eines vorausfahrenden Zuges", 44: "Warten auf einen entgegenkommenden Zug",
    45: "Vorfahrt eines anderen Zuges", 46: "Vorfahrt eines anderen Zuges", 47: "Verspätete Bereitstellung",
    48: "Verspätung aus vorheriger Fahrt", 49: "Kurzfristiger Personalausfall", 50: "Kurzfristige Erkrankung von Personal",
    51: "Verspätetes Personal aus vorheriger Fahrt", 52: "Streik", 53: "Unwetterauswirkungen",
    54: "Verfügbarkeit der Gleise derzeit eingeschränkt", 55: "Technischer Defekt an einem anderen Zug",
    56: "Laden der Antriebsbatterie", 57: "Zusätzlicher Halt", 58: "Umleitung", 59: "Schnee und Eis",
    60: "Witterungsbedingt verminderte Geschwindigkeit", 61: "Defekte Tür", 62: "Behobener Defekt am Zug",
    63: "Technische Untersuchung am Zug", 64: "Defekt an einer Weiche", 65: "Erdrutsch", 66: "Hochwasser",
    67: "Behördliche Maßnahme", 68: "Hohes Fahrgastaufkommen", 69: "Zug verkehrt mit verminderter Geschwindigkeit",
    70: "WLAN nicht verfügbar", 71: "Eingeschränktes WLAN", 72: "Info/Entertainment nicht verfügbar",
    82: "Abweichende Wagenreihung", 85: "Ein Wagen fehlt", 88: "Keine Qualitätsmängel",
}


# ---------------------------------------------------------------------------
# Laden
# ---------------------------------------------------------------------------

def load_hits(stop_ids: tuple[int, ...]) -> pd.DataFrame:
    con = duckdb.connect()
    stop_ids_sql = ",".join(str(x) for x in stop_ids)
    df = con.execute(
        f"""
        SELECT stop_id, richtung, trip_id, update_timestamp,
               unnest(message_codes) AS code
        FROM read_parquet('{EVENTS_FINAL_GLOB}')
        WHERE is_arrival = true AND stop_id IN ({stop_ids_sql})
        """
    ).fetchdf()
    df["update_timestamp"] = pd.to_datetime(df["update_timestamp"], utc=True).dt.tz_localize(None)
    return df.dropna(subset=["update_timestamp"])


# ---------------------------------------------------------------------------
# 1. Bestandsaufnahme
# ---------------------------------------------------------------------------

def bestandsaufnahme(hits: pd.DataFrame, label: str) -> pd.DataFrame:
    grp = hits.groupby("code")
    out = grp.agg(
        n_vorkommen=("code", "size"),
        n_halte=("stop_id", "nunique"),
        n_tage=("update_timestamp", lambda s: s.dt.date.nunique()),
        erstes_auftreten=("update_timestamp", "min"),
        letztes_auftreten=("update_timestamp", "max"),
    ).reset_index()
    out["bedeutung"] = out["code"].map(IRIS_CODES)
    out["bedeutung_bekannt"] = out["bedeutung"].notna()
    out.loc[~out["bedeutung_bekannt"], "bedeutung"] = "UNBEKANNT (nicht in IRIS-Referenz)"
    out["ist_stoerungscode"] = out["code"].isin(ALLE_STOERUNGSCODES)
    out = out.sort_values("n_vorkommen", ascending=False)
    out.to_csv(OUTPUTS_DIR / f"stoerungsinventur_1_bestandsaufnahme_{label}.csv", index=False)
    print(f"\n=== 1. Bestandsaufnahme ({label}) ===")
    print(f"{len(out)} distinkte Codes, davon {out['bedeutung_bekannt'].sum()} mit bekannter Bedeutung, "
          f"{(~out['bedeutung_bekannt']).sum()} unbekannt.")
    print(out.head(25).to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# 2. Episoden je Code (zeitlich + über Stationen hinweg, 15-Minuten-Schwelle)
# ---------------------------------------------------------------------------

def build_episodes_for_code(code_hits: pd.DataFrame) -> pd.DataFrame:
    g = code_hits.sort_values("update_timestamp").reset_index(drop=True)
    gap = g["update_timestamp"].diff().dt.total_seconds() / 60
    episode_id = (gap > CLUSTER_GAP_MIN).cumsum()
    g["episode_id"] = episode_id.values

    ep = g.groupby("episode_id").agg(
        t_start=("update_timestamp", "min"),
        t_end=("update_timestamp", "max"),
        n_stationen=("stop_id", "nunique"),
        n_fahrten=("trip_id", "nunique"),
        stationen=("stop_id", lambda s: sorted(s.unique().tolist())),
    ).reset_index()
    ep["dauer_min"] = (ep["t_end"] - ep["t_start"]).dt.total_seconds() / 60
    return ep


def episoden_je_code(hits: pd.DataFrame) -> dict[int, pd.DataFrame]:
    print("\n=== 2. Episoden je Code ===")
    episodes_by_code: dict[int, pd.DataFrame] = {}
    rows = []
    for code, g in hits.groupby("code"):
        ep = build_episodes_for_code(g)
        episodes_by_code[code] = ep
        rows.append({
            "code": code,
            "bedeutung": IRIS_CODES.get(code, "UNBEKANNT"),
            "n_episoden": len(ep),
            "dauer_median_min": ep["dauer_min"].median(),
            "dauer_q25_min": ep["dauer_min"].quantile(0.25),
            "dauer_q75_min": ep["dauer_min"].quantile(0.75),
            "dauer_max_min": ep["dauer_min"].max(),
            "stationen_median": ep["n_stationen"].median(),
            "stationen_max": ep["n_stationen"].max(),
            "fahrten_median": ep["n_fahrten"].median(),
            "fahrten_summe": ep["n_fahrten"].sum(),
        })
    summary = pd.DataFrame(rows).sort_values("n_episoden", ascending=False)
    summary.to_csv(OUTPUTS_DIR / "stoerungsinventur_2_episoden_je_code.csv", index=False)
    print(summary.head(25).round(1).to_string(index=False))
    return episodes_by_code


# ---------------------------------------------------------------------------
# 3. Zeitliche Muster (Top 20)
# ---------------------------------------------------------------------------

def zeitliche_muster(hits: pd.DataFrame, top_codes: list[int]) -> None:
    print("\n=== 3. Zeitliche Muster (Top 20) ===")
    sub = hits[hits["code"].isin(top_codes)].copy()
    sub["hour"] = sub["update_timestamp"].dt.hour
    sub["weekday"] = sub["update_timestamp"].dt.day_name()
    sub["month"] = sub["update_timestamp"].dt.month

    hour_table = pd.crosstab(sub["code"], sub["hour"], normalize="index")
    hour_table.to_csv(OUTPUTS_DIR / "stoerungsinventur_3_stunde_je_code.csv")

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_table = pd.crosstab(sub["code"], sub["weekday"], normalize="index").reindex(columns=weekday_order)
    weekday_table.to_csv(OUTPUTS_DIR / "stoerungsinventur_3_wochentag_je_code.csv")

    month_table = pd.crosstab(sub["code"], sub["month"], normalize="index")
    month_table.to_csv(OUTPUTS_DIR / "stoerungsinventur_3_monat_je_code.csv")

    # Peak-Konzentration: Anteil der Meldungen in der HVZ (6-9 und 15-19 Uhr)
    hvz_hours = list(range(6, 10)) + list(range(15, 20))
    hvz_share = sub.groupby("code")["hour"].apply(lambda h: h.isin(hvz_hours).mean())
    print("\nAnteil Meldungen in Hauptverkehrszeit (6-9, 15-19 Uhr), Top 20:")
    print(hvz_share.sort_values(ascending=False).round(3).to_string())
    hvz_share.rename("hvz_anteil").to_csv(OUTPUTS_DIR / "stoerungsinventur_3_hvz_anteil.csv")

    plot_heatmaps(hour_table, month_table)


def plot_heatmaps(hour_table: pd.DataFrame, month_table: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{c} ({IRIS_CODES.get(c, '?')[:20]})" for c in hour_table.index]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    im0 = axes[0].imshow(hour_table.values, aspect="auto", cmap="YlOrRd")
    axes[0].set_yticks(range(len(labels))); axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xticks(range(len(hour_table.columns))); axes[0].set_xticklabels(hour_table.columns)
    axes[0].set_xlabel("Stunde"); axes[0].set_title("Anteil Meldungen je Stunde (zeilennormiert)")
    plt.colorbar(im0, ax=axes[0], fraction=0.03)

    im1 = axes[1].imshow(month_table.values, aspect="auto", cmap="YlGnBu")
    axes[1].set_yticks(range(len(labels))); axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xticks(range(len(month_table.columns))); axes[1].set_xticklabels(month_table.columns)
    axes[1].set_xlabel("Monat"); axes[1].set_title("Anteil Meldungen je Monat (zeilennormiert)")
    plt.colorbar(im1, ax=axes[1], fraction=0.03)

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / "stoerungsinventur_zeitmuster_top20.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Überschneidungen
# ---------------------------------------------------------------------------

def build_full_grid_activity(episodes_by_code: dict[int, pd.DataFrame], year_start, year_end) -> pd.DataFrame:
    """Für jedes 5-Minuten-Fenster: Menge der Codes, die irgendwo auf der
    Stammstrecke gerade eine aktive Episode haben. Vektorisiert über ein
    Start/Ende-Differenzarray (statt einzelner .loc-Slices je Episode --
    bei Codes mit zehntausenden Episoden sonst sehr langsam)."""
    full_index = pd.date_range(year_start, year_end, freq="5min")
    n = len(full_index)
    activity = pd.DataFrame(0, index=full_index, columns=list(episodes_by_code.keys()), dtype=np.int8)
    for code, ep in episodes_by_code.items():
        if len(ep) == 0:
            continue
        starts = full_index.searchsorted(ep["t_start"].dt.floor("5min").values)
        ends = full_index.searchsorted(ep["t_end"].dt.ceil("5min").values, side="right")
        ends = np.clip(ends, 0, n - 1)
        diff = np.zeros(n + 1, dtype=np.int32)
        np.add.at(diff, starts, 1)
        np.add.at(diff, ends + 1, -1)
        active = np.cumsum(diff[:n]) > 0
        activity[code] = active.astype(np.int8)
    return activity


def ueberschneidungen(hits: pd.DataFrame, episodes_by_code: dict[int, pd.DataFrame]) -> None:
    print("\n=== 4. Überschneidungen ===")
    year_start = hits["update_timestamp"].min().floor("5min")
    year_end = hits["update_timestamp"].max().ceil("5min")

    print("4a: Anteil 5-Minuten-Fenster mit mindestens einer Störungsmeldung, je Station...")
    full_index = pd.date_range(year_start, year_end, freq="5min")
    n_idx = len(full_index)
    per_station_rows = []
    for stop_id in STAMMSTRECKE_IDS:
        stop_hits = hits[hits["stop_id"] == stop_id]
        diff = np.zeros(n_idx + 1, dtype=np.int32)
        for code, g in stop_hits.groupby("code"):
            ep = build_episodes_for_code(g)
            if len(ep) == 0:
                continue
            starts = full_index.searchsorted(ep["t_start"].dt.floor("5min").values)
            ends = full_index.searchsorted(ep["t_end"].dt.ceil("5min").values, side="right")
            ends = np.clip(ends, 0, n_idx - 1)
            np.add.at(diff, starts, 1)
            np.add.at(diff, ends + 1, -1)
        active = np.cumsum(diff[:n_idx]) > 0
        anteil = active.mean()
        per_station_rows.append({"stop_id": stop_id, "station_name": STATION_NAMES[stop_id],
                                  "anteil_fenster_mit_stoerung": anteil})
    per_station = pd.DataFrame(per_station_rows).sort_values("anteil_fenster_mit_stoerung", ascending=False)
    print(per_station.round(3).to_string(index=False))
    per_station.to_csv(OUTPUTS_DIR / "stoerungsinventur_4a_anteil_fenster_je_station.csv", index=False)

    print("\nAufbau der netzweiten Aktivitätsmatrix (Code x 5-Minuten-Fenster)...")
    activity = build_full_grid_activity(episodes_by_code, year_start, year_end)
    any_active = (activity.sum(axis=1) > 0)
    print(f"Insgesamt (irgendein Störungscode irgendwo auf der Stammstrecke aktiv): "
          f"{any_active.mean():.1%} aller 5-Minuten-Fenster im Jahr.")
    pd.DataFrame([{"anteil_fenster_gesamt": any_active.mean()}]).to_csv(
        OUTPUTS_DIR / "stoerungsinventur_4a_anteil_fenster_gesamt.csv", index=False)

    print("\n4b: Verteilung Anzahl gleichzeitig aktiver Codes...")
    n_active = activity.sum(axis=1)
    dist = n_active.value_counts(normalize=True).sort_index()
    print(dist.head(15))
    dist.rename("anteil").to_csv(OUTPUTS_DIR / "stoerungsinventur_4b_anzahl_gleichzeitig_aktiv.csv")

    print("\n4c: Kreuztabelle Codepaare (Lift = beobachtet / erwartet bei Unabhängigkeit)...")
    codes = sorted(activity.columns)
    marg = activity.mean(axis=0)
    joint = activity.T.dot(activity) / len(activity)
    expected = pd.DataFrame(np.outer(marg, marg), index=codes, columns=codes)
    lift_arr = np.array(joint.values, dtype=float) / np.array(expected.replace(0, np.nan).values, dtype=float)
    np.fill_diagonal(lift_arr, np.nan)
    lift = pd.DataFrame(lift_arr, index=codes, columns=codes)
    lift.to_csv(OUTPUTS_DIR / "stoerungsinventur_4c_lift_matrix.csv")

    pairs = []
    for i, a in enumerate(codes):
        for b in codes[i + 1:]:
            if joint.loc[a, b] > 0:
                pairs.append({"code_a": a, "code_b": b, "lift": lift.loc[a, b],
                              "gemeinsame_fenster": int(joint.loc[a, b] * len(activity))})
    pairs_df = pd.DataFrame(pairs).sort_values("lift", ascending=False)
    pairs_df.to_csv(OUTPUTS_DIR / "stoerungsinventur_4c_top_paare.csv", index=False)
    print("Top 15 Codepaare nach Lift (>=20 gemeinsame Fenster):")
    print(pairs_df[pairs_df["gemeinsame_fenster"] >= 20].head(15).round(2).to_string(index=False))

    print("\nBaue Episodenstarts je Code+Station (für Folge- und Isolationsanalyse)...")
    ep_starts_by_station: dict[tuple[int, int], np.ndarray] = {}
    for code, g in hits.groupby("code"):
        for stop_id, gg in g.groupby("stop_id"):
            ep = build_episodes_for_code(gg)
            ep_starts_by_station[(code, stop_id)] = np.sort(ep["t_start"].values.astype("datetime64[ns]"))

    print("\n4c (Folge-Codes): welcher Code beginnt typischerweise NACH einem anderen "
          "an derselben Station? (vektorisiert per searchsorted)")
    win_ns = np.timedelta64(ISOLATION_WINDOW_MIN, "m")
    folge_rows = []
    for code_a in codes:
        for code_b in codes:
            if code_a == code_b:
                continue
            n_folgt = 0
            n_a_total = 0
            for stop_id in STAMMSTRECKE_IDS:
                starts_a = ep_starts_by_station.get((code_a, stop_id))
                starts_b = ep_starts_by_station.get((code_b, stop_id))
                if starts_a is None or starts_b is None or len(starts_a) == 0 or len(starts_b) == 0:
                    if starts_a is not None:
                        n_a_total += len(starts_a)
                    continue
                n_a_total += len(starts_a)
                window_ends = starts_a + win_ns
                lo = np.searchsorted(starts_b, starts_a, side="right")
                hi = np.searchsorted(starts_b, window_ends, side="right")
                n_folgt += int(np.sum(hi > lo))
            if n_a_total >= 20:
                folge_rows.append({"code_a": code_a, "code_b": code_b,
                                    "anteil_b_folgt_a": n_folgt / n_a_total, "n_a": n_a_total})
    folge_df = pd.DataFrame(folge_rows).sort_values("anteil_b_folgt_a", ascending=False)
    folge_df.to_csv(OUTPUTS_DIR / "stoerungsinventur_4c_folgecodes.csv", index=False)
    print("Top 15 'B folgt häufig auf A' (>=20 Episoden von A):")
    print(folge_df.head(15).round(2).to_string(index=False))

    print("\n4d: Anteil isolierter Episoden je Code (kein anderer Code im Fenster ±60min, "
          "gleiche Station) -- vektorisiert per searchsorted gegen kombinierte Zeitleiste je Station...")
    # Kombinierte Zeitleiste je Station: alle Episodenstarts aller Codes, mit Code-Label.
    combined_by_station: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for stop_id in STAMMSTRECKE_IDS:
        parts = []
        for code in codes:
            starts = ep_starts_by_station.get((code, stop_id))
            if starts is not None and len(starts):
                parts.append(pd.DataFrame({"t": starts, "code": code}))
        if parts:
            allp = pd.concat(parts).sort_values("t")
            combined_by_station[stop_id] = (allp["t"].values, allp["code"].values)
        else:
            combined_by_station[stop_id] = (np.array([], dtype="datetime64[ns]"), np.array([]))

    isolation_rows = []
    for code, ep in episodes_by_code.items():
        if len(ep) == 0:
            continue
        n_isoliert = 0
        for _, row in ep.iterrows():
            isolated = True
            win_start = np.datetime64(row["t_start"]) - win_ns
            win_end = np.datetime64(row["t_start"]) + win_ns
            for stop_id in row["stationen"]:
                times, codes_arr = combined_by_station[stop_id]
                lo = np.searchsorted(times, win_start, side="left")
                hi = np.searchsorted(times, win_end, side="right")
                if hi > lo and np.any(codes_arr[lo:hi] != code):
                    isolated = False
                    break
            if isolated:
                n_isoliert += 1
        isolation_rows.append({"code": code, "bedeutung": IRIS_CODES.get(code, "UNBEKANNT"),
                                "n_episoden": len(ep), "anteil_isoliert": n_isoliert / len(ep)})
    isolation_df = pd.DataFrame(isolation_rows).sort_values("n_episoden", ascending=False)
    isolation_df.to_csv(OUTPUTS_DIR / "stoerungsinventur_4d_isolation_je_code.csv", index=False)
    print(isolation_df.head(25).round(3).to_string(index=False))


if __name__ == "__main__":
    print("Lade Rohdaten Stammstrecke...")
    hits_stamm = load_hits(STAMMSTRECKE_IDS)

    print("Lade Rohdaten Gesamtnetz (nur für Abschnitt 1, Vergleich)...")
    import duckdb as _duckdb
    con = _duckdb.connect()
    all_stops = con.execute(
        f"SELECT DISTINCT stop_id FROM read_parquet('{EVENTS_FINAL_GLOB}') WHERE is_arrival=true"
    ).fetchdf()["stop_id"].tolist()
    hits_all = load_hits(tuple(all_stops))

    bestandsaufnahme(hits_stamm, "stammstrecke")
    bestandsaufnahme(hits_all, "gesamtnetz")

    print("\nBeschränke Abschnitte 2-5 auf Störungscodes (2-69) an der Stammstrecke...")
    hits_stoerung = hits_stamm[hits_stamm["code"].isin(ALLE_STOERUNGSCODES)].copy()

    episodes_by_code = episoden_je_code(hits_stoerung)

    top_codes = (
        hits_stoerung["code"].value_counts().head(TOP_N).index.tolist()
    )
    print(f"\nTop {TOP_N} Codes (nach Vorkommen): {top_codes}")
    zeitliche_muster(hits_stoerung, top_codes)

    ueberschneidungen(hits_stoerung, episodes_by_code)

    print("\nFertig -- Rangliste (Abschnitt 5) folgt in separatem Auswertungsschritt.")

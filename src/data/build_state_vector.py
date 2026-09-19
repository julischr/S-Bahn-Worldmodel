"""
Stufe 0.3 des Bauplans: Zustandsvektor der Münchner Stammstrecke bauen.

Raster: 5 Minuten, Einheit (zeitfenster, stop_id, richtung) -- nur die 12
Stammstreckenhalte, beide Richtungen (Ost/West aus build_events.py).

Kennzahlen je Zelle: n_zuege, delay_mean, delay_median, delay_p90,
anteil_ueber_180s, anteil_ueber_360s, n_ausfall, dwell_excess, headway_cv.

Zentrale Regel aus dem Bauplan: "Leere Fenster als NaN, nicht als 0." Ein
Fenster ohne jede Meldung (weder echter Halt noch Ausfall) heisst "unbekannt"
und muss NaN bleiben -- sonst zieht das jede spaetere Impulsantwortkurve nach
unten. Das wird hier erreicht, indem zuerst NUR ueber tatsaechlich
beobachtete Ereignisse gruppiert wird (das erzeugt nie eine "leere" Zeile)
und erst DANACH auf das volle 5-Minuten-Gitter reindexiert wird -- fehlende
Kombinationen (zeitfenster, stop_id, richtung) bleiben dabei automatisch NaN
in ALLEN Spalten, inklusive n_zuege und n_ausfall. Ein Fenster mit
Beobachtung, in dem aber alle Fahrten ausfielen, bekommt dagegen bewusst
n_zuege=0 (real beobachtet: keine gefahrene Bahn) und n_ausfall>0 -- das ist
etwas anderes als "keine Daten".

Zeitzone: das Raster wird komplett in UTC aufgespannt (time_real liegt
bereits in UTC vor) und erst ganz am Ende nach Europe/Berlin konvertiert,
um die Sommerzeitumstellung (30.3. = 23h Tag, 26.10. = 25h Tag) nicht in der
Aggregation selbst behandeln zu muessen.

headway_cv: Variationskoeffizient (std/mean) der tatsaechlichen Abstaende
zwischen aufeinanderfolgenden (nicht ausgefallenen) Zuegen an einem
Stop/einer Richtung. Braucht mindestens 2 Headway-Werte im Fenster, sonst
NaN (bei nur einem Zug im Fenster ist eine Standardabweichung nicht
definiert).
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from src.utils.constants import DATA_DIR

EVENTS_FINAL_GLOB = str(DATA_DIR / "02_events" / "final" / "*.parquet")
OUT_PATH = DATA_DIR / "03_state_vector" / "state_vector_5min.parquet"

FREQ = "5min"

STAMMSTRECKE_ORDER = [
    8004158, 8004151, 8004179, 8004128, 8004129, 8098263,
    8004132, 8004135, 8004131, 8004136, 8000262, 8004134,
]
STATION_NAMES = {
    8004158: "Pasing", 8004151: "Laim", 8004179: "Hirschgarten",
    8004128: "Donnersbergerbrücke", 8004129: "Hackerbrücke", 8098263: "Hauptbahnhof (tief)",
    8004132: "Karlsplatz (Stachus)", 8004135: "Marienplatz", 8004131: "Isartor",
    8004136: "Rosenheimer Platz", 8000262: "Ostbahnhof", 8004134: "Leuchtenbergring",
}


def load_arrivals() -> pd.DataFrame:
    con = duckdb.connect()
    core_ids_sql = ",".join(str(x) for x in STAMMSTRECKE_ORDER)
    df = con.execute(
        f"""
        SELECT stop_id, richtung, time_real, delay, is_cancelled,
               dwell_time_schedule, dwell_time_real
        FROM read_parquet('{EVENTS_FINAL_GLOB}')
        WHERE is_arrival = true
          AND stop_id IN ({core_ids_sql})
          AND richtung IS NOT NULL
        """
    ).fetchdf()
    df["time_real"] = pd.to_datetime(df["time_real"], utc=True)
    df["window"] = df["time_real"].dt.floor(FREQ)
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    real = df[~df["is_cancelled"]].copy()
    real["dwell_excess_row"] = real["dwell_time_real"] - real["dwell_time_schedule"]
    real["over_180"] = real["delay"] > 180
    real["over_360"] = real["delay"] > 360

    grp = real.groupby(["window", "stop_id", "richtung"])
    stats = grp.agg(
        n_zuege=("delay", "size"),
        delay_mean=("delay", "mean"),
        delay_median=("delay", "median"),
        delay_p90=("delay", lambda s: s.quantile(0.9)),
        anteil_ueber_180s=("over_180", "mean"),
        anteil_ueber_360s=("over_360", "mean"),
        dwell_excess=("dwell_excess_row", "mean"),
    ).reset_index()

    ausfall = (
        df[df["is_cancelled"]]
        .groupby(["window", "stop_id", "richtung"])
        .size()
        .rename("n_ausfall")
        .reset_index()
    )

    merged = stats.merge(ausfall, on=["window", "stop_id", "richtung"], how="outer")
    # n_zuege/n_ausfall: NaN durch merge nur dort, wo die jeweils ANDERE Seite
    # (real bzw. ausgefallen) im selben Fenster keine Beobachtung hatte, obwohl
    # die andere Seite eine hatte -- das ist ein beobachtetes Fenster, also 0
    # und nicht "unbekannt".
    merged["n_zuege"] = merged["n_zuege"].fillna(0).astype(int)
    merged["n_ausfall"] = merged["n_ausfall"].fillna(0).astype(int)
    return merged


def compute_headway_cv(df: pd.DataFrame) -> pd.DataFrame:
    real = df[~df["is_cancelled"]].sort_values("time_real").copy()
    results = []
    for (stop_id, richtung), g in real.groupby(["stop_id", "richtung"]):
        g = g.sort_values("time_real")
        headway_s = g["time_real"].diff().dt.total_seconds()
        tmp = pd.DataFrame({"window": g["window"], "headway_s": headway_s})
        tmp = tmp.dropna(subset=["headway_s"])
        cv = tmp.groupby("window")["headway_s"].agg(lambda s: s.std() / s.mean() if len(s) >= 2 else np.nan)
        cv = cv.rename("headway_cv").reset_index()
        cv["stop_id"] = stop_id
        cv["richtung"] = richtung
        results.append(cv)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(
        columns=["window", "stop_id", "richtung", "headway_cv"]
    )


def build_full_grid(df: pd.DataFrame) -> pd.DataFrame:
    full_index = pd.date_range(df["window"].min(), df["window"].max(), freq=FREQ, tz="UTC")
    grid = pd.MultiIndex.from_product(
        [full_index, STAMMSTRECKE_ORDER, ["Ost", "West"]],
        names=["window", "stop_id", "richtung"],
    ).to_frame(index=False)
    return grid


def run() -> None:
    print("Lade Ankünfte an den 12 Stammstreckenhalten...")
    df = load_arrivals()
    print(f"{len(df):,} Ankunfts-Ereignisse geladen "
          f"({df['is_cancelled'].sum():,} davon ausgefallen)")

    print("Aggregiere pro (Fenster, Station, Richtung)...")
    stats = aggregate(df)

    print("Berechne headway_cv...")
    headway = compute_headway_cv(df)
    stats = stats.merge(headway, on=["window", "stop_id", "richtung"], how="left")

    print("Baue vollständiges 5-Minuten-Gitter (UTC) und reindexiere...")
    grid = build_full_grid(df)
    result = grid.merge(stats, on=["window", "stop_id", "richtung"], how="left")

    n_total = len(result)
    n_observed = result["n_zuege"].notna().sum()
    print(f"{n_total:,} Zeilen im vollen Gitter, davon {n_observed:,} mit Beobachtung "
          f"({n_observed/n_total:.1%}), Rest bewusst NaN (Nachtlücken etc.)")

    result["station_name"] = result["stop_id"].map(STATION_NAMES)
    result["window_berlin"] = result["window"].dt.tz_convert("Europe/Berlin")

    cols = [
        "window", "window_berlin", "stop_id", "station_name", "richtung",
        "n_zuege", "delay_mean", "delay_median", "delay_p90",
        "anteil_ueber_180s", "anteil_ueber_360s", "n_ausfall",
        "dwell_excess", "headway_cv",
    ]
    result = result[cols].sort_values(["window", "stop_id", "richtung"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT_PATH, index=False)
    print(f"\nGespeichert: {OUT_PATH}  ({len(result):,} Zeilen)")
    print(result.describe(include="all").T)


if __name__ == "__main__":
    run()

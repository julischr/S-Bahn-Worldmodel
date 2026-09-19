"""Luecken-Behandlung und Episodenbildung (ein Betriebstag = eine Episode)."""

import numpy as np
import pandas as pd


def interpolate_short_gaps(df: pd.DataFrame, limit_steps: int = 12) -> pd.DataFrame:
    """Lineare Interpolation von Luecken bis limit_steps*5min (Default 60min), nur INNERHALB
    bestehender Daten (kein Extrapolieren an den Raendern)."""
    return df.interpolate(method="linear", limit=limit_steps, limit_area="inside", axis=0)


def build_daily_episodes(df: pd.DataFrame, min_length: int = 24) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """Zerlegt df (Index=window_berlin, nach interpolate_short_gaps) in Episoden:
    pro Kalendertag maximale zusammenhaengende Bloecke, in denen ALLE Spalten nicht-NaN sind.
    min_length: minimale Laenge einer Episode in 5-Min-Schritten (Default 24 = 2h), kuerzere werden verworfen.

    Gibt (Liste der Episoden-DataFrames, Metadaten-Tabelle) zurueck.
    """
    all_notna = df.notna().all(axis=1).to_numpy()
    dates = df.index.date
    episodes = []
    meta_rows = []
    n = len(df)
    i = 0
    while i < n:
        if not all_notna[i]:
            i += 1
            continue
        j = i
        d = dates[i]
        while j < n and all_notna[j] and dates[j] == d:
            j += 1
        length = j - i
        if length >= min_length:
            ep = df.iloc[i:j]
            episodes.append(ep)
            meta_rows.append(
                {
                    "episode_id": len(episodes) - 1,
                    "date": d,
                    "start": ep.index[0],
                    "end": ep.index[-1],
                    "n_obs": length,
                    "length_hours": length * 5 / 60.0,
                }
            )
        i = j
    meta = pd.DataFrame(meta_rows)
    return episodes, meta


def daily_anomaly_table(aux_wide: pd.DataFrame, label: str) -> pd.DataFrame:
    """Aggregiert eine Hilfsgroesse (z.B. n_zuege oder n_ausfall) je Kalendertag ueber alle Stationen,
    zur Identifikation auffaelliger Betriebstage."""
    daily_sum = aux_wide.groupby(aux_wide.index.date).sum(min_count=1).sum(axis=1)
    daily_sum.name = f"{label}_sum"
    out = daily_sum.to_frame()
    out.index.name = "date"
    return out

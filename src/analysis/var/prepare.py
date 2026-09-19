"""Laden und Pivotieren des Zustandsvektors zu parallelen Stationszeitreihen."""

from pathlib import Path

import pandas as pd

from src.utils.constants import DATA_DIR
from .stations import STOP_ID_TO_NAME, STATION_NAMES_WEST_OST

STATE_VECTOR_PATH = DATA_DIR / "03_state_vector" / "state_vector_5min.parquet"


def load_state_vector() -> pd.DataFrame:
    df = pd.read_parquet(STATE_VECTOR_PATH)
    df = df[df["stop_id"].isin(STOP_ID_TO_NAME)].copy()
    df["station_name"] = df["stop_id"].map(STOP_ID_TO_NAME)
    return df


def pivot_direction(df: pd.DataFrame, richtung: str, value_col: str = "delay_mean") -> pd.DataFrame:
    """Pivotiert eine Richtung auf ein 5-Min-Raster: Index=window_berlin, Spalten=Stationen (West->Ost)."""
    sub = df[df["richtung"] == richtung]
    wide = sub.pivot_table(index="window_berlin", columns="station_name", values=value_col, aggfunc="mean")
    wide = wide.reindex(columns=STATION_NAMES_WEST_OST)
    wide = wide.sort_index()
    # Vollstaendiges 5-Min-Raster erzwingen (Luecken im Index -> explizite NaN-Zeilen)
    full_index = pd.date_range(wide.index.min(), wide.index.max(), freq="5min", tz=wide.index.tz)
    wide = wide.reindex(full_index)
    wide.index.name = "window_berlin"
    return wide


def pivot_aux(df: pd.DataFrame, richtung: str, value_col: str) -> pd.DataFrame:
    """Wie pivot_direction, fuer Hilfsgroessen (n_zuege, n_ausfall, ...) zur Tagesdiagnose."""
    return pivot_direction(df, richtung, value_col=value_col)

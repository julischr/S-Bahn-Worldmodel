"""Zentrale Stelle fuer die Datenausfall-Bloecke (siehe src/data/find_ausfall_bloecke.py, data/ausfall_bloecke.csv).

Schalter: Umgebungsvariable S_BAHN_AUSFALL_FILTER=1 -> alle Stufen lesen data/02_events/final_bereinigt statt final
und EXCLUDED_DATES enthaelt zusaetzlich die vollen Ausfalltage. Standard (ohne Variable): altes Verhalten, nichts aendert sich.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .constants import DATA_DIR

BLOCKS_CSV = DATA_DIR / "ausfall_bloecke.csv"
DAYS_CSV = DATA_DIR / "ausfall_tage.csv"


def filter_active() -> bool:
    return os.environ.get("S_BAHN_AUSFALL_FILTER", "0") == "1"


def events_final_dir() -> Path:
    """Verzeichnis der Event-Parquets (mit Filter: bereinigte Kopie)."""
    return DATA_DIR / "02_events" / ("final_bereinigt" if filter_active() else "final")


def full_outage_days() -> list:
    d = pd.read_csv(DAYS_CSV)
    return sorted(pd.to_datetime(d.loc[d["voller_ausfalltag"], "bt"]).dt.date)


def extend_excluded(dates):
    """Alte EXCLUDED_DATES (date-Array) + volle Ausfalltage, nur wenn der Filter aktiv ist."""
    if not filter_active():
        return dates
    return sorted(set(dates) | set(full_outage_days()))


def load_blocks(scope: str = "alle") -> pd.DataFrame:
    b = pd.read_csv(BLOCKS_CSV, parse_dates=["von_lokal", "bis_lokal"])
    return b[b["scope"] == scope] if scope else b


class BlockIndex:
    """Schneller Test, ob ein Zeitfenster [t+lo, t+hi] (UTC-Sekunden) einen Ausfall-Block (scope 'alle') der Station/Richtung
    oder einen Netzblock (stop_id -1) beruehrt."""

    def __init__(self):
        import numpy as np
        b = load_blocks("alle")
        b["s"] = b["von_lokal"].dt.tz_localize("Europe/Berlin", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC").astype("int64") // 10**9 if False else \
            pd.to_datetime(b["von_lokal"]).dt.tz_localize("Europe/Berlin", ambiguous=True, nonexistent="shift_forward").dt.tz_convert("UTC").map(lambda x: int(x.timestamp()))
        b["e"] = b["s"] + 3600
        self.np = np
        self.net = self._merge(b[b["stop_id"] == -1][["s", "e"]].to_numpy())
        self.per = {}
        for (sid, r), g in b[b["stop_id"] != -1].groupby(["stop_id", b["richtung"].fillna("NA")]):
            self.per[(int(sid), r)] = self._merge(g[["s", "e"]].to_numpy())

    def _merge(self, a):
        np = self.np
        if len(a) == 0:
            return np.zeros((0, 2), dtype="int64")
        a = a[np.argsort(a[:, 0])]
        out = [list(a[0])]
        for s, e in a[1:]:
            if s <= out[-1][1]:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        return np.array(out, dtype="int64")

    def hit(self, stop_id, richtung, t_s, lo_min=-60, hi_min=240):
        np = self.np
        t_s = np.asarray(t_s, dtype="int64")
        out = np.zeros(len(t_s), bool)
        keys = list(zip(np.asarray(stop_id).tolist(), [("NA" if (r is None or r != r) else r) for r in richtung]))
        arrs = {}
        for k in set(keys):
            arrs[k] = self.per.get((int(k[0]), k[1]), np.zeros((0, 2), dtype="int64"))
        for i, k in enumerate(keys):
            for iv in (arrs[k], self.net):
                if len(iv) == 0:
                    continue
                a, b = t_s[i] + lo_min * 60, t_s[i] + hi_min * 60
                j = np.searchsorted(iv[:, 0], b, side="left")      # Bloecke mit Start < b
                if j > 0 and iv[j - 1, 1] > a:
                    out[i] = True
                    break
        return out

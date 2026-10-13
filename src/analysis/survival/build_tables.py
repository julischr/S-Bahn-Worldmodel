"""Stufe 3: baut Episoden- und Placebo-Tabelle mit allen Erholungsmassen.
Aufruf: uv run python -m src.analysis.survival.build_tables"""

import numpy as np
import pandas as pd

from . import data as D
from . import recovery as R


def main():
    tr = pd.read_parquet(D.ZW / "zuege.parquet") if (D.ZW / "zuege.parquet").exists() else D.build_trains()
    tr.to_parquet(D.ZW / "zuege.parquet")
    hits = D.load_hits()
    ep, h = D.build_episodes(hits)
    ep = D.add_stoerungslast(ep, hits)
    fp = R.Footprints(h)
    series = R.build_series(tr)

    ep["t0_s"] = D.ns(ep["t0"]) // R.NS
    m = R.measure(ep[["stop_id", "richtung", "t0_s"]], series, fp, ep["episode_id"].to_numpy())
    epm = pd.concat([ep, m], axis=1)
    for tag, anch in (("placebo", False), ("placebo2", True)):
        pl = R.make_placebo(ep, fp, series, anchored=anch)
        mp = R.measure(pl[["stop_id", "richtung", "t0_s"]], series, fp, np.full(len(pl), -1))
        plm = pd.concat([pl, mp], axis=1)
        loc = pd.to_datetime(plm["t0_s"], unit="s", utc=True).dt.tz_convert(D.TZ)
        plm["hvz"] = loc.dt.hour.isin(D.HVZ_HOURS)
        plm = plm.merge(ep[["episode_id", "opday"]].rename(columns={"episode_id": "parent_episode",
                                                                    "opday": "parent_opday"}), on="parent_episode")
        plm.to_parquet(D.ZW / f"{tag}.parquet")
        print(tag, len(plm))
    epm.to_parquet(D.ZW / "episoden.parquet")
    print(len(epm))


if __name__ == "__main__":
    main()

"""IRF-Kurve Pasing->Leuchtenbergring (Ost) bzw. Leuchtenbergring->Pasing (West) bei FESTEM Lag (alt: Ost p=5, West p=17),
damit Datenwirkung und Lag-Wahl getrennt werden. Aufruf: python -m src.analysis.var.run_sensitivity_irf <tag>
(mit/ohne S_BAHN_AUSFALL_FILTER=1); schreibt outputs/var/sens_irf_kurve_<tag>_{ost,west}.csv"""
import sys

import numpy as np
import pandas as pd

from .run_all import OUT_DIR, run_direction
from .stations import STATION_NAMES_OST_WEST, STATION_NAMES_WEST_OST

tag = sys.argv[1]
OUT_DIR.mkdir(parents=True, exist_ok=True)
for name, order, p in (("ost", STATION_NAMES_WEST_OST, 5), ("west", STATION_NAMES_OST_WEST, 17)):
    r = run_direction("Ost" if name == "ost" else "West", order, p_range=range(p, p + 1), n_boot=200)
    th = r["theta_geo"]                      # (H+1, i=Ziel, j=Schock)
    K = th.shape[1]
    cur = pd.DataFrame(dict(minuten=np.arange(th.shape[0]) * 5, antwort_s=th[:, K - 1, 0]))
    cur.to_csv(OUT_DIR / f"sens_irf_kurve_{tag}_{name}.csv", index=False)
    print(name, p, "Episoden", len(r["episodes"]), "Peak", cur.loc[cur.antwort_s.abs().idxmax()].to_dict(), flush=True)

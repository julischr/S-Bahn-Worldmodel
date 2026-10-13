"""Konsistenzpruefung Simulator <-> Trainingsmerkmale (Teacher Forcing, Plan Abschnitt 5).

Der Simulator wird mit den ECHTEN naechsten Ankuenften gefuettert (forced=True); die dabei
berechneten Merkmale muessen den Trainingsmerkmalen (features.py) entsprechen. Wuerde der
Simulator Information benutzen, die das Training nicht hat (oder umgekehrt), zeigt sich das hier.
"""
import sys

import numpy as np
import pandas as pd

from .data import OUT, load_trains
from .features import FEATURES, load_dataset
from .sim import NS_MIN, load_day, simulate


def main(days):
    trains = load_trains()
    ds = load_dataset()
    rows = []
    for d in days:
        d = pd.Timestamp(d).date()
        day = load_day(trains, d)
        res = simulate(day, None, R=1, forced=True, collect=True)
        c = res.collected
        c["T"] = (c["t"].astype("int64") + day.m0) * NS_MIN
        c["rank"] = c["rank"].astype(int)
        ref = ds[ds["opday"] == d]
        m = c.merge(ref, on=["zug", "rank", "T"], suffixes=("_sim", "_trn"), how="outer", indicator=True)
        both = m[m["_merge"] == "both"]
        rows.append(dict(tag=str(d), n_sim=len(c), n_train=len(ref), n_both=len(both),
                         nur_sim=int((m["_merge"] == "left_only").sum()), nur_train=int((m["_merge"] == "right_only").sum())))
        for f in [x for x in FEATURES if x != "rank"]:
            a, b = both[f + "_sim"].to_numpy(float), both[f + "_trn"].to_numpy(float)
            same = (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) < 1e-4)
            rows[-1]["diff_" + f] = int((~same).sum())
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(t.T.to_string())
    t.to_csv(OUT / "check_forced.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or ["2025-10-07", "2025-11-15", "2025-12-03"])

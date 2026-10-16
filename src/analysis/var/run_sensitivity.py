"""Sensitivitaetslauf VAR (Beide Richtungen) fuer die Korrektur Datenausfall; schreibt nur Kernkennzahlen.
Aufruf (im Sandbox-Projekt mit S_BAHN_AUSFALL_FILTER=1): python -m src.analysis.var.run_sensitivity
Richtung Ost: Reihenfolge West->Ost, n_boot 500 (wie Hauptlauf); West: Ost->West, n_boot 200 (wie outputs/var/west)."""
import json

import numpy as np
import pandas as pd

from .run_all import EXCLUDED_DATES, OUT_DIR, run_direction, save_outputs_ost
from .stations import STATION_NAMES_OST_WEST, STATION_NAMES_WEST_OST


def kern(r):
    cols = r["station_order"]
    K = len(cols)
    ttp = pd.DataFrame(r["time_to_peak"], index=cols, columns=cols)
    amp = pd.DataFrame(r["amplitude"], index=cols, columns=cols)
    hl = pd.DataFrame(r["half_life"], index=cols, columns=cols)
    f = pd.DataFrame(r["fevd_h"], index=cols, columns=cols)
    fremd = 1 - np.diag(f.to_numpy())          # Anteil Fremdvarianz je Station (Zeile = Zielstation)
    return dict(richtung=r["richtung"], chosen_p=r["chosen_p"], aic_p=r["aic_p"], n_episoden=len(r["episodes"]), T_obs=int(r["var_result"].T),
                max_eig=float(np.max(np.abs(r["eig"]))),
                peak_min_erste_zu_letzte=float(ttp.iloc[0, K - 1]), amp_erste_zu_letzte=float(amp.iloc[0, K - 1]),
                halbwertszeit_selbst_min={c: float(hl.loc[c, c]) for c in cols},
                fremdvarianz_je_station={c: float(v) for c, v in zip(cols, fremd)},
                fremdvarianz_mittel=float(fremd.mean()), netto_kopplung={c: float(v) for c, v in r["net"].items()}), ttp, amp, f


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, order, nb in (("ost", STATION_NAMES_WEST_OST, 500), ("west", STATION_NAMES_OST_WEST, 200)):
        r = run_direction("Ost" if name == "ost" else "West", order, n_boot=nb)
        k, ttp, amp, f = kern(r)
        res[name] = k
        ttp.to_csv(OUT_DIR / f"sens_irf_zeit_bis_peak_{name}.csv"); amp.to_csv(OUT_DIR / f"sens_irf_amplitude_{name}.csv")
        f.to_csv(OUT_DIR / f"sens_fevd_{name}.csv"); r["net"].to_csv(OUT_DIR / f"sens_netto_kopplung_{name}.csv", header=["netto"])
        print(name, json.dumps({a: b for a, b in k.items() if not isinstance(b, dict)}, default=float), flush=True)
    res["ausgeschlossene_tage"] = [str(d) for d in EXCLUDED_DATES]
    json.dump(res, open(OUT_DIR / "sens_kern.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

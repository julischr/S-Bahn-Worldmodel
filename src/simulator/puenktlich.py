"""Tag ohne jeden Eingriff mit pünktlichem Eintritt (eintritt="puenktlich") und Pruefung, wie oft die
dabei entstehenden Merkmalswerte in den Trainingsdaten vorkommen (Extrapolationsrisiko).

Aufruf: uv run python -m src.simulator.puenktlich 2025-11-12 2025-11-16 --runs 50
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .data import OUT, load_trains
from .features import FEATURES, load_dataset
from .metrics import overall
from .model_io import Models
from .sim import Day, simulate
from .train import split

CELL_FEATS = ["hvz", "d_k", "net_mean15", "act_gap_pred"]


def cell_key(df: pd.DataFrame) -> np.ndarray:
    hvz = df["hvz"].to_numpy().astype(int)
    dk = np.clip(np.rint(df["d_k"].to_numpy()), -3, 15).astype(int) + 3        # 0..18
    nm = np.clip(np.rint(np.nan_to_num(df["net_mean15"].to_numpy(), nan=-2)), -2, 12).astype(int) + 2  # 0..14
    ag = np.clip(np.nan_to_num(df["act_gap_pred"].to_numpy(), nan=9), 1, 9).astype(int)             # 1..9
    return ((hvz * 19 + dk) * 15 + nm) * 10 + ag


def seltenheit(sim_rows: pd.DataFrame, ref_rows: pd.DataFrame, train: pd.DataFrame) -> dict:
    """Anteil der Entscheidungszeilen in Zellen (hvz, d_k, net_mean15, act_gap_pred), die im Training
    n = 0, < 30 bzw. < 300 Zeilen haben; dazu Randverteilung gegen Training-Perzentile."""
    cnt = pd.Series(cell_key(train)).value_counts()
    out = {}
    for name, df in (("sim", sim_rows), ("referenz_echte_tage", ref_rows)):
        c = cnt.reindex(cell_key(df)).fillna(0).to_numpy()
        out[name] = dict(n=len(df), anteil_zelle_0=float((c == 0).mean()), anteil_zelle_unter_30=float((c < 30).mean()),
                         anteil_zelle_unter_300=float((c < 300).mean()), median_trainingszeilen_je_zelle=float(np.median(c)))
    marg = []
    for f in ["d_k", "d_pred_k", "net_mean15", "act_gap_pred", "cancel_net30", "cancel_sd30", "net_n15"]:
        lo, hi = np.nanpercentile(train[f], [1, 99])
        for name, df in (("sim", sim_rows), ("referenz_echte_tage", ref_rows)):
            v = df[f].to_numpy(float)
            v = v[np.isfinite(v)]
            marg.append(dict(merkmal=f, quelle=name, train_p01=lo, train_p99=hi, anteil_unter_p01=float((v < lo).mean()),
                             anteil_ueber_p99=float((v > hi).mean()), mittel=float(v.mean())))
    # kombinierte Lage: HVZ bei fast leerem Netz (Mittelverspaetung < 1 min)
    hv = {}
    for name, df in (("training", train), ("sim", sim_rows), ("referenz_echte_tage", ref_rows)):
        m = (df["hvz"] == 1)
        hv[name] = float(((df["net_mean15"] < 1.0) & m).sum() / max(m.sum(), 1))
    out["hvz_und_netzmittel_unter_1min_anteil_an_hvz_zeilen"] = hv
    return out, pd.DataFrame(marg)


def profil(res, day) -> pd.DataFrame:
    o = overall(res)
    n_pos = day.plen.astype(int).sum() - day.N          # Positionen nach dem ersten Halt je Lauf
    return pd.DataFrame(dict(station_rang=np.repeat(np.arange(12), 2), richtung=["Ost", "West"] * 12,
                             mittel_verspaetung=(o["prof_sum"].sum(0) / np.maximum(o["prof_n"].sum(0), 1)))), \
        dict(ausfallquote_modelliert=float(o["n_canc"].mean() / max(o["n_mod"], 1)), n_modelliert=int(o["n_mod"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tage", nargs="+")
    ap.add_argument("--runs", type=int, default=50)
    a = ap.parse_args()
    trains = load_trains(); M = Models()
    ds = load_dataset(); tr = split(ds)["train"]
    rows_p, prof_all, ex_all, ms, selt = [], [], [], [], {}
    for d in a.tage:
        d = pd.Timestamp(d).date()
        day = Day(trains[trains["opday"] == d], d)
        pr = {}
        for mode in ("echt", "puenktlich"):
            res = simulate(day, M, R=a.runs, seed=5, eintritt=mode, collect=(mode == "puenktlich"))
            p, q = profil(res, day)
            p["modus"] = mode; p["tag"] = str(d); prof_all.append(p)
            ms.append(dict(tag=str(d), modus=mode, **q, mittel_verspaetung_alle_ankuenfte=float(p["mittel_verspaetung"].mean()), n_clamp=res.meta["n_clamp"]))
            if mode == "puenktlich":
                sim_rows = res.collected
        real_rows = ds[ds["opday"] == d]
        sel, marg = seltenheit(sim_rows, real_rows, tr)
        marg["tag"] = str(d); ex_all.append(marg)
        for k, v in sel.items():
            rows_p.append(dict(tag=str(d), **({"quelle": k, **v} if isinstance(v, dict) and "n" in v else {"quelle": k, **v})))
    pd.DataFrame(ms).to_csv(OUT / "puenktlich_ausfallquote.csv", index=False)
    pd.concat(prof_all).to_csv(OUT / "puenktlich_profil.csv", index=False)
    pd.concat(ex_all).to_csv(OUT / "puenktlich_seltenheit_rand.csv", index=False)
    pd.DataFrame(rows_p).to_csv(OUT / "puenktlich_seltenheit.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(pd.DataFrame(ms)); print(pd.DataFrame(rows_p)); print(pd.concat(ex_all).round(3).to_string())


if __name__ == "__main__":
    main()

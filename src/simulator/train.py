"""Stufe 4, Schritt 2: LightGBM-Modelle (Ausfall binaer, Verspaetungszuwachs 19 Klassen).

Aufteilung nach Betriebstag-Monat (Plan Abschnitt 1): Training Jan-Jul, Validierung Aug
(Hyperparameter, Early Stopping), Kalibrierung Sep, Test Okt-Dez (hier nicht benutzt).
"""

from __future__ import annotations

import itertools
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import ZW
from .features import CAT_FEATURES, CLASS_HI, CLASS_LO, FEATURES, N_CLASS, load_dataset

MODELS = ZW.parent / "modelle"
GRID = [dict(num_leaves=nl, min_data_in_leaf=md) for nl, md in itertools.product([31, 127], [100, 1000])]
BASE = dict(learning_rate=0.08, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            lambda_l2=5.0, max_bin=127, num_threads=10, verbose=-1, seed=7)
MAX_ROUNDS = 600
MAX_PLAN_RUN = 15


def split(d: pd.DataFrame) -> dict:
    d = d[d["plan_run"] <= MAX_PLAN_RUN]
    return dict(train=d[d["month"] <= 7], val=d[d["month"] == 8], cal=d[d["month"] == 9],
                test=d[d["month"] >= 10])


def fit(kind: str, parts: dict, log) -> tuple[lgb.Booster, dict, list]:
    if kind == "cancel":
        tr, va = parts["train"], parts["val"]
        ytr, yva = tr["cancel_next"], va["cancel_next"]
        obj = dict(objective="binary", metric="binary_logloss")
    else:
        tr, va = parts["train"][parts["train"]["cancel_next"] == 0], parts["val"][parts["val"]["cancel_next"] == 0]
        ytr, yva = tr["dclass"], va["dclass"]
        obj = dict(objective="multiclass", num_class=N_CLASS, metric="multi_logloss")
    dtr = lgb.Dataset(tr[FEATURES], ytr, categorical_feature=CAT_FEATURES, free_raw_data=False)
    dva = lgb.Dataset(va[FEATURES], yva, reference=dtr, categorical_feature=CAT_FEATURES)
    results = []
    best = None
    for g in GRID:
        t0 = time.time()
        params = {**BASE, **obj, **g}
        m = lgb.train(params, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        score = m.best_score["valid_0"][obj["metric"]]
        row = dict(**g, best_iter=m.best_iteration, val_logloss=score, sec=round(time.time() - t0))
        results.append(row)
        log(f"{kind} {row}")
        if best is None or score < best[0]:
            best = (score, m, params)
    m = best[1]
    return m, best[2], results


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    logf = open(MODELS / "train_log.txt", "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    d = load_dataset()
    n0 = len(d)
    parts = split(d)
    log(f"Zeilen gesamt {n0}, nach plan_run<={MAX_PLAN_RUN}: {sum(len(v) for v in parts.values())}; "
        + ", ".join(f"{k}={len(v)}" for k, v in parts.items()))
    log("cancel_next-Rate: " + ", ".join(f"{k}={v['cancel_next'].mean():.5f} (n_cancel={int(v['cancel_next'].sum())})" for k, v in parts.items()))

    summary = {}
    for kind in ("cancel", "class"):
        m, params, results = fit(kind, parts, log)
        m.save_model(str(MODELS / f"lgbm_{kind}.txt"), num_iteration=m.best_iteration)
        pd.DataFrame(results).to_csv(MODELS / f"grid_{kind}.csv", index=False)
        summary[kind] = dict(params={k: v for k, v in params.items()}, best_iter=m.best_iteration)
        log(f"{kind}: bestes Modell {summary[kind]}")
    # Randverteilungen der Klassen (Trainingsdaten, Jan-Jul)
    tr = parts["train"]
    tr = tr[tr["cancel_next"] == 0]
    lo = (tr.loc[tr["delta"] <= CLASS_LO, "delta"] - CLASS_LO).to_numpy()   # <= 0
    hi = (tr.loc[tr["delta"] >= CLASS_HI, "delta"] - CLASS_HI).to_numpy()   # >= 0
    np.savez(MODELS / "rand_verteilung.npz", lo=lo, hi=hi)
    summary["rand"] = dict(n_lo=len(lo), n_hi=len(hi), lo_mean=float(lo.mean()), hi_mean=float(hi.mean()),
                           lo_min=float(lo.min()), hi_max=float(hi.max()))
    json.dump(summary, open(MODELS / "summary.json", "w"), indent=1, default=str)
    log("fertig")


if __name__ == "__main__":
    main()

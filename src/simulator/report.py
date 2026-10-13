"""Erzeugt outputs/simulator/tabellen.md: alle zentralen Tabellen als Markdown (Quelle fuer
BEFUNDE_SIMULATOR.md und UEBERGABE_KI.md). Liest nur CSV/JSON aus outputs/simulator/."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import OUT


def fmt(v, nd=3):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "–"
        if np.isinf(v):
            return "n.e."
        if abs(v) >= 1000:
            return f"{v:,.0f}".replace(",", ".")
        return f"{v:.{nd}f}".replace(".", ",")
    if isinstance(v, (int, np.integer)):
        return f"{v:,}".replace(",", ".")
    return str(v)


def md(df: pd.DataFrame, nd=3, cols=None, rename=None) -> str:
    if cols:
        df = df[cols]
    if rename:
        df = df.rename(columns=rename)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "|".join(["---"] * len(df.columns)) + "|\n"
    body = "".join("| " + " | ".join(fmt(v, nd) for v in r) + " |\n" for r in df.itertuples(index=False))
    return head + body


def read(name):
    p = OUT / name
    return pd.read_csv(p) if p.exists() else None


def main(tag="basis", sztag="szenarien"):
    parts = {}
    for lab in ("roh", "kalibriert"):
        t = read(f"einschritt_verspaetung_{lab}.csv")
        if t is not None:
            parts[f"einschritt_verspaetung_{lab}"] = md(t, 4, ["gruppe", "modell", "n", "mae", "rmse", "bias", "crps", "logloss"])
        t = read(f"einschritt_ausfall_{lab}.csv")
        if t is not None:
            parts[f"einschritt_ausfall_{lab}"] = md(t, 6, ["gruppe", "modell", "n", "n_cancel", "rate", "brier", "brier_skill_vs_V2", "logloss", "auc"])
        t = read(f"reliability_ausfall_{lab}.csv")
        if t is not None:
            parts[f"reli_ausfall_{lab}"] = md(t, 5)
        t = read(f"reliability_zuwachs_{lab}.csv")
        if t is not None:
            parts[f"reli_zuwachs_{lab}"] = md(t, 4)
        t = read(f"reliability_klassen_{lab}.csv")
        if t is not None:
            parts[f"reli_klassen_{lab}"] = md(t, 5)
    for n in ("shap_ausfall", "shap_zuwachs"):
        t = read(n + ".csv")
        if t is not None:
            parts[n] = md(t.head(15), 4)
    t = read("modelle/grid_cancel.csv") if False else None
    for k in ("cancel", "class"):
        p = OUT / "modelle" / f"grid_{k}.csv"
        if p.exists():
            parts[f"grid_{k}"] = md(pd.read_csv(p), 5)
    for k in ("a", "b", "c", "d", "e"):
        t = read(f"pruefstein_{k}_{tag}.csv")
        if t is not None:
            parts[f"pruefstein_{k}"] = md(t, 3)
    for n in (f"plausibilitaet_{tag}.csv", f"profil_verspaetung_{tag}.csv", "check_forced.csv", "sperrung_faelle.csv", f"sperrung_test_{sztag}.csv"):
        t = read(n)
        if t is not None:
            parts[n.replace(".csv", "")] = md(t, 3)
    t = read(f"szenario_einzel_{sztag}.csv")
    if t is not None:
        parts["szenario_einzel"] = md(t, 2)
    p = OUT / f"szenario_zusammen_{sztag}.csv"
    if p.exists():
        parts["szenario_zusammen"] = md(pd.read_csv(p), 2)
    with open(OUT / "tabellen.md", "w") as f:
        for k, v in parts.items():
            f.write(f"\n<!-- {k} -->\n### {k}\n\n{v}\n")
    json.dump({k: len(v) for k, v in parts.items()}, open(OUT / "tabellen_index.json", "w"))
    return parts


if __name__ == "__main__":
    main()

"""Fasst szenario_einzel_*.csv je Variante (Art, N, HVZ) zusammen (Mittel ueber die Beispielzuege, Spanne)."""
import sys
import pandas as pd
from .data import OUT

def main(tag="szenarien"):
    e = pd.read_csv(OUT / f"szenario_einzel_{tag}.csv")
    cols = ["betroffen_pro_lauf_mittel", "betroffen_q05", "betroffen_q95", "zugaenderung_mittel", "zugaenderung_median",
            "zugaenderung_q10", "zugaenderung_q90", "zugaenderung_max", "gesamt_mehrverspaetung_zugmin_mittel",
            "gesamt_zugmin_q05", "gesamt_zugmin_q95", "zusaetzliche_ausfaelle_mittel", "zus_ausf_q05", "zus_ausf_q95",
            "spitze_diff_min", "abklingzeit_ki_min", "abklingzeit_unter_0_1min", "eigener_zug_aenderung_ende_mittel"]
    g = e.groupby(["art", "minuten", "hvz"])
    m = g[cols].mean().add_suffix("_mittel_ueber_zuege")
    lo = g[["betroffen_pro_lauf_mittel", "gesamt_mehrverspaetung_zugmin_mittel", "zusaetzliche_ausfaelle_mittel", "abklingzeit_ki_min"]].min().add_suffix("_min_zug")
    hi = g[["betroffen_pro_lauf_mittel", "gesamt_mehrverspaetung_zugmin_mittel", "zusaetzliche_ausfaelle_mittel", "abklingzeit_ki_min"]].max().add_suffix("_max_zug")
    out = pd.concat([g.size().rename("n_sim"), m, lo, hi], axis=1).reset_index()
    out.to_csv(OUT / f"szenario_zusammen_{tag}.csv", index=False)
    print(out.T.to_string())

if __name__ == "__main__":
    main(*sys.argv[1:])

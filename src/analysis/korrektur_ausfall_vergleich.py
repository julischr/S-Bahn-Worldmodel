"""Vergleich alt (outputs/ im Hauptprojekt) gegen neu (Sandbox ../S-Bahn_bereinigt mit Ausfall-Bloecken entfernt).
Schreibt outputs/korrektur_ausfall/vergleich_*.csv."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.constants import OUTPUTS_DIR, PROJECT_ROOT

OLD = OUTPUTS_DIR
NEW = PROJECT_ROOT.parent / "S-Bahn_bereinigt" / "outputs"
OUT = OUTPUTS_DIR / "korrektur_ausfall"
OUT.mkdir(exist_ok=True)


def beg():
    a, b = pd.read_csv(OLD / "beg_validierung_2025.csv"), pd.read_csv(NEW / "beg_validierung_2025.csv")
    t = a.merge(b, on="variante", suffixes=("_alt", "_neu"))
    t = t[["variante", "n_real_alt", "n_real_neu", "quote_alt", "quote_neu", "eigene_ausfallquote_alt", "eigene_ausfallquote_neu"]]
    t["luecke_zu_BEG_pp_alt"] = (0.879 - t["quote_alt"]) * 100
    t["luecke_zu_BEG_pp_neu"] = (0.879 - t["quote_neu"]) * 100
    t.to_csv(OUT / "vergleich_beg.csv", index=False)
    return t


def stage1():
    a, b = pd.read_csv(OLD / "impulsantwort_code34.csv"), pd.read_csv(NEW / "impulsantwort_code34.csv")
    rows = []
    for nm, d in (("alt", a), ("neu", b)):
        s = d[(d.minute_relativ_t0 >= -60) & (d.minute_relativ_t0 <= -15)]
        k = d.loc[d.impulsantwort_delay_s.idxmax()]
        rows.append(dict(version=nm, sockel_m60_m15_s=s.impulsantwort_delay_s.mean(), wert_t0_s=float(d.loc[d.minute_relativ_t0 == 0, "impulsantwort_delay_s"].iloc[0]),
                         spitze_s=k.impulsantwort_delay_s, spitze_bei_min=k.minute_relativ_t0,
                         wert_p60_s=float(d.loc[d.minute_relativ_t0 == 60, "impulsantwort_delay_s"].iloc[0]) if (d.minute_relativ_t0 == 60).any() else np.nan,
                         sockel_ki_lo=s.ci_low.mean(), sockel_ki_hi=s.ci_high.mean()))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "vergleich_stage1_code34.csv", index=False)
    m = a.merge(b, on="minute_relativ_t0", suffixes=("_alt", "_neu"))
    m.to_csv(OUT / "vergleich_stage1_code34_kurve.csv", index=False)
    return t


def survival():
    out = {}
    a, b = pd.read_csv(OLD / "survival" / "km_ergebnisse.csv"), pd.read_csv(NEW / "survival" / "km_ergebnisse.csv")
    sel = lambda d, pl: d[(d.gruppe.astype(str) == "alle") & (d.ebene == "st") & (d.schwelle == 180) & (d.zensur == "std") & (d.ursprung == "R0") & (d.placebo == pl)]
    rows = []
    for pl in ("placebo", "placebo2"):
        for nm, d in (("alt", a), ("neu", b)):
            r = sel(d, pl)
            if len(r):
                r = r.iloc[0]
                rows.append(dict(placebo=pl, version=nm, n_episoden=r.n_episoden, n_placebo=r.n_placebo, median_stoerung=r["median"], ki_lo=r.median_lo, ki_hi=r.median_hi,
                                 median_placebo=r.median_placebo, diff_median=r.diff_median, diff_lo=r.diff_median_lo, diff_hi=r.diff_median_hi,
                                 anteil_zensiert=r.anteil_zensiert, ohne_wirkung=r.ohne_wirkung, ohne_wirkung_placebo=r.ohne_wirkung_placebo))
    pd.DataFrame(rows).to_csv(OUT / "vergleich_survival_A1_A2.csv", index=False)
    out["A1A2"] = pd.DataFrame(rows)
    ka, kb = pd.read_csv(OLD / "survival" / "kipppunkt_klassen_median.csv"), pd.read_csv(NEW / "survival" / "kipppunkt_klassen_median.csv")
    k = ka.merge(kb, on=["quelle", "hvz", "klasse"], suffixes=("_alt", "_neu"))
    k = k[k.hvz.astype(str) == "alle"][["quelle", "klasse", "n_alt", "n_neu", "median_alt", "median_neu", "median_lo_alt", "median_hi_alt", "median_lo_neu", "median_hi_neu", "anteil_zensiert_alt", "anteil_zensiert_neu"]]
    k.to_csv(OUT / "vergleich_survival_A6.csv", index=False)
    out["A6"] = k
    if (NEW / "survival" / "ausfallquote_schwereklassen.csv").exists():
        qa, qb = pd.read_csv(OLD / "survival" / "ausfallquote_schwereklassen.csv"), pd.read_csv(NEW / "survival" / "ausfallquote_schwereklassen.csv")
        qa.to_csv(OUT / "vergleich_survival_plot7_alt.csv", index=False); qb.to_csv(OUT / "vergleich_survival_plot7_neu.csv", index=False)
    for f in ("zuege_info.csv", "cox_fallzahlen.csv"):
        pass
    return out


def zugebene():
    a, b = pd.read_csv(OLD / "var/plausibilitaet/vertiefung_zuwachs_vorgaenger_gepoolt.csv"), pd.read_csv(NEW / "var/plausibilitaet/vertiefung_zuwachs_vorgaenger_gepoolt.csv")
    k = a.columns[0]
    t = a.merge(b, on=k, suffixes=("_alt", "_neu"))
    t.to_csv(OUT / "vergleich_zugebene_11_4_2.csv", index=False)
    c, d = pd.read_csv(OLD / "var/plausibilitaet/ausfallkanal_zellen.csv"), pd.read_csv(NEW / "var/plausibilitaet/ausfallkanal_zellen.csv")
    u = c.merge(d, on=["headway_bin", "delay_prev_bin"], suffixes=("_alt", "_neu"))
    u.to_csv(OUT / "vergleich_ausfallkanal_11_8.csv", index=False)
    return t, u


def var():
    n = json.load(open(NEW / "var" / "sens_kern.json"))
    rows = []
    for name, sub in (("ost", ""), ("west", "west")):
        base = OLD / "var" / sub
        ttp = pd.read_csv(base / "irf_zeit_bis_peak_min.csv", index_col=0)
        fv = pd.read_csv(base / "fevd.csv", index_col=0)
        nk = pd.read_csv(base / "netto_kopplung.csv", index_col=0).iloc[:, 0]
        hl = pd.read_csv(base / "irf_halbwertszeit_min.csv", index_col=0)
        cols = list(ttp.columns)
        fremd = 1 - np.diag(fv.to_numpy())
        meta = json.load(open(base / "run_meta.json"))
        amp = pd.read_csv(base / "irf_amplitude.csv", index_col=0)
        rows.append(dict(richtung=name, version="alt", chosen_p=meta["chosen_p"], n_episoden=meta["n_episodes"], T_obs=meta["T_obs"], peak_min_erste_zu_letzte=ttp.iloc[0, -1],
                         amp_erste_zu_letzte=amp.iloc[0, -1], fremdvarianz_mittel=fremd.mean(), halbwertszeit_selbst_mittel=np.mean(np.diag(hl.to_numpy())), max_eig=meta["max_abs_eigenvalue"]))
        k = n[name]
        rows.append(dict(richtung=name, version="neu", chosen_p=k["chosen_p"], n_episoden=k["n_episoden"], T_obs=k["T_obs"], peak_min_erste_zu_letzte=k["peak_min_erste_zu_letzte"],
                         amp_erste_zu_letzte=k["amp_erste_zu_letzte"], fremdvarianz_mittel=k["fremdvarianz_mittel"], halbwertszeit_selbst_mittel=np.mean(list(k["halbwertszeit_selbst_min"].values())),
                         max_eig=k["max_eig"]))
        nn = pd.Series(k["netto_kopplung"])
        t = pd.DataFrame(dict(netto_alt=nk, netto_neu=nn)).assign(richtung=name)
        t.to_csv(OUT / f"vergleich_var_netto_kopplung_{name}.csv")
        pd.DataFrame(dict(fremd_alt=pd.Series(fremd, index=cols), fremd_neu=pd.Series(k["fremdvarianz_je_station"]))).to_csv(OUT / f"vergleich_var_fremdvarianz_{name}.csv")
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "vergleich_var_kern.csv", index=False)
    return t


if __name__ == "__main__":
    import sys
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    for f in sys.argv[1:] or ["beg", "stage1", "survival", "zugebene", "var"]:
        print("=====", f)
        r = globals()[f]()
        if isinstance(r, dict):
            for k, v in r.items():
                print(k); print(v if not isinstance(v, tuple) else v[0]); 
        else:
            print(r if not isinstance(r, tuple) else r[0])

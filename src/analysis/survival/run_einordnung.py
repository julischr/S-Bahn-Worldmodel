"""Schritt 7: Einordnung. Beschreibende Stratifizierung nach Meldedauer (wie Stufe 1:
<30 / 30-90 / >90 min; ex post bekannt, nur beschreibend) und Zusammenfassungen."""
import numpy as np, pandas as pd
from . import data as D, km as K

ep = pd.read_parquet(D.ZW / "episoden.parquet")
days = np.array(sorted(ep.opday_nr.unique())); Wd = K.make_day_weights(len(days), 1000, 5)
ep["dauer"] = pd.cut(ep.meldedauer_min, [-1, 30, 90, 1e9], labels=["kurz <30", "mittel 30-90", "lang >90"])
rows = []
for lvl in ("st", "ln"):
    for nofolge in ("std", "nofolge"):
        tc, ec = f"t_R0_180_{lvl}_{nofolge}", f"e_R0_180_{lvl}_{nofolge}"
        for k, s in ep.dropna(subset=[tc]).groupby("dauer", observed=True):
            b = K.km_bootstrap(s[tc], s[ec], np.searchsorted(days, s.opday_nr), Wd)
            r = dict(ebene=lvl, zensur=nofolge, dauer=k, n=len(s), anteil_zensiert=1 - s[ec].mean())
            for i, nm in enumerate(("q25", "median", "q75")):
                r[nm], r[nm + "_lo"], r[nm + "_hi"] = K.fmt_q(b["q_point"][i], b["q_boot"][:, i])
            rows.append(r)
out = pd.DataFrame(rows); out.to_csv(D.OUT / "einordnung_dauerstrata.csv", index=False)
print(out.round(2).to_string())
# Richtung: Streckenebene ohne Folgezensur nach Richtung
rows = []
for rich, s in ep.dropna(subset=["t_R0_180_ln_nofolge"]).groupby("richtung"):
    b = K.km_bootstrap(s["t_R0_180_ln_nofolge"], s["e_R0_180_ln_nofolge"], np.searchsorted(days, s.opday_nr), Wd)
    r = dict(richtung=rich, n=len(s), anteil_zensiert=1 - s["e_R0_180_ln_nofolge"].mean())
    for i, nm in enumerate(("q25", "median", "q75")):
        r[nm], r[nm + "_lo"], r[nm + "_hi"] = K.fmt_q(b["q_point"][i], b["q_boot"][:, i])
    rows.append(r)
o2 = pd.DataFrame(rows); o2.to_csv(D.OUT / "einordnung_strecke_richtung.csv", index=False); print(o2.round(2).to_string())
print(ep.lag_erste_meldung_min.describe(percentiles=[.5,.9]).round(1).to_dict())

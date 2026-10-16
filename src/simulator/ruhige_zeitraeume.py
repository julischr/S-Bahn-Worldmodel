"""Findet die ruhigsten Zeitraeume auf der Stammstrecke (Echtdaten, ohne Sperrtage und ohne Datenausfall-Tage).

Datenausfall: Bloecke (5 min), in denen alle Ankuenfte exakt 0 min haben und nichts ausfaellt (kein Echtzeitfeed,
z. B. ~20 Tage im September 2025) werden samt +-30 min ausgeschlossen. Gleitende Fenster von 60 min (Ortszeit,
05-23 Uhr); Kennzahl = mittlere Verspaetung aller Ankuenfte (12 Stationen, beide Richtungen), Anteil >= 3 min,
Zahl ausgefallener Halte. Fenster werden nach Ruhe sortiert, ueberlappende Fenster zusammengefasst."""
import numpy as np, pandas as pd
from .data import OUT, TZ, load_trains, HVZ_HOURS

W = 12  # Bloecke je Fenster (60 min)
t = load_trains()
t["blk"] = (t["sched"] // (300 * 10**9)).astype("int64")
run = t[~t["cancelled"]]
g = run.groupby("blk").agg(n=("delay_min", "size"), s=("delay_min", "sum"), late=("delay_min", lambda x: (x >= 3).sum()),
                           zero=("delay_min", lambda x: (x == 0).sum()))
g["canc"] = t[t["cancelled"]].groupby("blk").size().reindex(g.index).fillna(0)
g = g.reindex(range(g.index.min(), g.index.max() + 1)).fillna(0)
out = ((g["n"] >= 20) & (g["zero"] == g["n"]) & (g["canc"] == 0)).astype(float)
bad = out.rolling(13, center=True, min_periods=1).max() > 0
loc = pd.to_datetime(g.index.to_numpy() * 300, unit="s", utc=True).tz_convert(TZ)
day = (loc.hour >= 5) & (loc.hour < 23)
valid = (g["n"] >= 20) & ~bad & day
R = g[["n", "s", "late", "canc"]].rolling(W).sum()
full = valid.rolling(W).sum() == W
w = pd.DataFrame(dict(n=R["n"], mean=R["s"] / R["n"], late=R["late"] / R["n"], canc=R["canc"]))[full]
w["start_blk"] = w.index - W + 1
w = w.sort_values("mean")
picked, rows = [], []
for blk, r in w.iterrows():
    a = r["start_blk"]
    if any(abs(a - p) < W for p in picked):
        continue
    picked.append(a)
    la = pd.to_datetime(a * 300, unit="s", utc=True).tz_convert(TZ)
    lb = pd.to_datetime((a + W) * 300, unit="s", utc=True).tz_convert(TZ)
    hv = np.mean([h in HVZ_HOURS for h in pd.to_datetime(np.arange(a, a + W) * 300, unit="s", utc=True).tz_convert(TZ).hour])
    rows.append(dict(datum=str(la.date()), wochentag=la.strftime("%a"), von=la.strftime("%H:%M"), bis=lb.strftime("%H:%M"),
                     mittel_verspaetung_min=round(float(r["mean"]), 2), anteil_ab_3min=round(float(r["late"]), 3),
                     ausgefallene_halte=int(r["canc"]), ankuenfte=int(r["n"]), hvz_anteil=round(float(hv), 2)))
res = pd.DataFrame(rows)
res.head(40).to_csv(OUT / "ruhige_zeitraeume.csv", index=False)
print("Datenausfall-Bloecke:", int(out.sum()), "gueltige Fenster:", len(w))
print(res.head(25).to_string())
print("beste HVZ-Fenster (hvz_anteil>=0.9):"); print(res[res.hvz_anteil >= 0.9].head(10).to_string())

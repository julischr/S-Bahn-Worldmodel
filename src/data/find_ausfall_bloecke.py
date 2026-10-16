"""Systematische Suche nach Datenausfall-Bloecken in data/02_events/final (alle Stationen, beide Richtungen, ganzes Jahr 2025).

Hintergrund: In Ausfallzeiten liefert der Echtzeitfeed nichts; die Zeilen sind reine Fahrplanzeilen
(update_timestamp leer, time_real = time_schedule, delay = 0, keine Ausfaelle, keine message_codes).
Das sieht wie "perfekte Puenktlichkeit" aus, ist aber keine Beobachtung.

Regeln (Block = Station x Richtung x lokale Stunde, nur Ankuenfte; Betriebstag = lokal - 3 h):
  A kein_update      : Anteil update_timestamp leer >= 0,5 bei n >= 2 (Normalbetrieb: Median 0,04 %).
  B netz_nur_null    : im ganzen Netz in dieser Stunde delay = 0 bei >= 95 % der nicht ausgefallenen Ankuenfte, n >= 200
                       (Feed haengt: Update-Zeitstempel vorhanden, Verspaetung aber ueberall 0) -> alle Stationen der Stunde.
  C station_nur_null : Lauf aufeinanderfolgender Ankuenfte einer Station/Richtung mit delay = 0, keine Ausfaelle,
                       >= 10 Ankuenfte und >= 60 min Dauer, ausserhalb A/B.
  D codes_fehlen     : im ganzen Netz in dieser Stunde keine einzige message_code bei n >= 300 (nur fuer code-basierte
                       Auswertungen relevant: scope = nur_codes).
  E konstante Verspaetung (delay identisch != 0, n >= 8 je Block): nur gemeldet, nicht ausgeschlossen (siehe Bericht).
Ergebnis: data/ausfall_bloecke.csv (Stunden-Bloecke; scope alle|nur_codes), data/ausfall_tage.csv (Tage, an denen
>= 90 % der Stammstrecken-Stundenbloecke 05-23 Uhr betroffen sind), data/ausfall_suche_info.json.
"""
import json

import duckdb
import numpy as np
import pandas as pd

from src.utils.constants import DATA_DIR

G = str(DATA_DIR / "02_events" / "final" / "*.parquet")
STAMM = [8004128, 8004129, 8004131, 8004132, 8004151, 8000262, 8004135, 8004136, 8004134, 8004179, 8098263, 8004158]
TZ = "Europe/Berlin"


def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    b = con.execute(f"""
      SELECT stop_id, richtung, date_trunc('hour', time_schedule AT TIME ZONE '{TZ}') AS h,
             count(*) n, sum((update_timestamp IS NULL)::int) nu,
             sum((delay=0 AND NOT is_cancelled)::int) nz, sum((NOT is_cancelled)::int) nrun, sum(is_cancelled::int) nc,
             sum((len(message_codes)>0)::int) ncode, count(DISTINCT delay) ndel
      FROM read_parquet('{G}') WHERE is_arrival GROUP BY 1,2,3""").fetchdf()
    b["h"] = pd.to_datetime(b["h"])
    # A
    A = b[(b["n"] >= 2) & (b["nu"] / b["n"] >= 0.5)][["stop_id", "richtung", "h", "n"]].assign(art="A_kein_update", scope="alle")
    # B, D (Netzstunden)
    net = b.groupby("h").agg(n=("n", "sum"), nz=("nz", "sum"), nrun=("nrun", "sum"), nc=("nc", "sum"), ncode=("ncode", "sum")).reset_index()
    Bh = net[(net["n"] >= 200) & (net["nz"] / net["nrun"].clip(lower=1) >= 0.95)]
    Dh = net[(net["n"] >= 300) & (net["ncode"] == 0)]
    Ba = Bh.assign(stop_id=-1, richtung="alle", art="B_netz_nur_null", scope="alle")[["stop_id", "richtung", "h", "n", "art", "scope"]]
    Da = Dh.assign(stop_id=-1, richtung="alle", art="D_codes_fehlen", scope="nur_codes")[["stop_id", "richtung", "h", "n", "art", "scope"]]
    # C (Laeufe)
    c = con.execute(f"""
      WITH r AS (SELECT stop_id, richtung, time_schedule t, (delay=0 AND NOT is_cancelled) z, is_cancelled c,
                        (update_timestamp IS NULL) nu
                 FROM read_parquet('{G}') WHERE is_arrival),
      s AS (SELECT *, sum((NOT z)::int) OVER (PARTITION BY stop_id, richtung ORDER BY t ROWS UNBOUNDED PRECEDING) run FROM r)
      SELECT stop_id, richtung, run, min(t) t0, max(t) t1, count(*) n, sum(nu::int) nu FROM s WHERE z
      GROUP BY 1,2,3 HAVING count(*) >= 10 AND epoch(max(t)) - epoch(min(t)) >= 3600""").fetchdf()
    c = c[c["nu"] / c["n"] < 0.5]       # nicht schon durch A erfasst
    rows = []
    for r in c.itertuples():
        for h in pd.date_range(pd.Timestamp(r.t0).tz_convert(TZ).tz_localize(None).floor("h"),
                               pd.Timestamp(r.t1).tz_convert(TZ).tz_localize(None).floor("h"), freq="h"):
            rows.append((r.stop_id, r.richtung, h, r.n, "C_station_nur_null", "alle"))
    Ca = pd.DataFrame(rows, columns=["stop_id", "richtung", "h", "n", "art", "scope"])
    # nur C-Bloecke ausserhalb von B
    if len(Ca):
        Ca = Ca[~Ca["h"].isin(Bh["h"])]
    allb = pd.concat([A, Ba, Da, Ca], ignore_index=True)
    allb["von_lokal"] = allb["h"]
    allb["bis_lokal"] = allb["h"] + pd.Timedelta(hours=1)
    allb["betriebstag"] = (allb["h"] - pd.Timedelta(hours=3)).dt.date
    allb = allb.drop(columns=["h"]).sort_values(["art", "betriebstag", "stop_id", "richtung", "von_lokal"])
    allb.to_csv(DATA_DIR / "ausfall_bloecke.csv", index=False)
    # Tage: Anteil betroffener Stammstrecken-Stunden 05-23 Uhr
    sh = b[b["stop_id"].isin(STAMM)].copy()
    sh["bt"] = (sh["h"] - pd.Timedelta(hours=3)).dt.date
    sh = sh[(sh["h"].dt.hour >= 5) & (sh["h"].dt.hour < 23)]
    key = set(zip(allb.loc[allb.scope == "alle", "stop_id"], allb.loc[allb.scope == "alle", "richtung"], allb.loc[allb.scope == "alle", "von_lokal"]))
    hb = set(allb.loc[(allb.art == "B_netz_nur_null"), "von_lokal"])
    sh["bad"] = [((s, r, h) in key) or (h in hb) for s, r, h in zip(sh["stop_id"], sh["richtung"], sh["h"])]
    d = sh.groupby("bt").agg(anteil_betroffen=("bad", "mean"), n_bloecke=("bad", "size")).reset_index()
    d["voller_ausfalltag"] = d["anteil_betroffen"] >= 0.9
    d.to_csv(DATA_DIR / "ausfall_tage.csv", index=False)
    # E nur melden
    e = b[(b["n"] >= 8) & (b["ndel"] == 1) & (b["nz"] == 0) & (b["nc"] == 0)]
    info = dict(n_bloecke_gesamt=len(b), A=len(A), B_stunden=len(Bh), D_stunden=len(Dh), C_laeufe=len(c), C_blockstunden=len(Ca),
                E_konstante_nichtnull_bloecke_nur_gemeldet=len(e), E_beispiele=e.head(5).astype(str).to_dict("records"),
                voller_ausfalltage=int(d["voller_ausfalltag"].sum()), tage_mit_teilausfall=int(((d.anteil_betroffen > 0.02) & ~d.voller_ausfalltag).sum()))
    json.dump(info, open(DATA_DIR / "ausfall_suche_info.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in info.items() if k != "E_beispiele"}, indent=1))
    print(allb.groupby(["art", "scope"]).size())


if __name__ == "__main__":
    main()

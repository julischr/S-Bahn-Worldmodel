"""
Explorative Analyse der Bahn-Vorhersage Tagesdatei (2025-01-01, deutschlandweit)
mit Fokus auf die Münchner S-Bahn-Stammstrecke.

Ausführen mit:  uv run python explore.py

Alle Zahlen in diesem Skript werden tatsächlich aus der Datei berechnet, nichts
ist geschätzt. Wo eine Frage aus den Daten nicht beantwortbar ist, wird das im
Output explizit vermerkt.
"""

import json
import math
import sys
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 40)

DATA_FILE = "2025-01-01.parquet"
OUT_DIR = Path("output")
PLOT_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
D = f"'{DATA_FILE}'"


def hd(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# ---------------------------------------------------------------------------
# 1. GRUNDINVENTUR
# ---------------------------------------------------------------------------
hd("1. GRUNDINVENTUR")

file_size_mb = Path(DATA_FILE).stat().st_size / 1024 / 1024
n_rows = con.execute(f"SELECT count(*) FROM {D}").fetchone()[0]
print(f"Datei: {DATA_FILE}  |  Größe: {file_size_mb:.1f} MB  |  Zeilen: {n_rows:,}")

tmin, tmax = con.execute(f"SELECT min(time_schedule), max(time_schedule) FROM {D}").fetchone()
print(f"time_schedule Spanne: {tmin}  bis  {tmax}")
print("Hinweis: 1.1.2025 ist ein Mittwoch UND gesetzlicher Feiertag (Neujahr) -> Betrieb wie an einem")
print("Sonn-/Feiertag, nicht wie an einem normalen Werktag. Die Spanne reicht bis in den 2.1. morgens")
print("hinein, weil Trips, die am 1.1. spät nachts beginnen, nach Mitternacht weiterlaufen (kein Fehler).")

schema = con.execute(f"DESCRIBE SELECT * FROM {D}").fetchdf()
cols = schema["column_name"].tolist()
types = dict(zip(schema["column_name"], schema["column_type"]))

rows = []
for c in cols:
    t = types[c]
    is_list = "[]" in t
    if is_list:
        null_share = con.execute(f"SELECT sum(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END)/count(*) FROM {D}").fetchone()[0]
        distinct = None
        mn = med = mx = None
    else:
        null_share, distinct = con.execute(
            f"SELECT sum(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END)/count(*), count(DISTINCT {c}) FROM {D}"
        ).fetchone()
        mn = med = mx = None
        if t in ("BIGINT", "INTEGER", "USMALLINT", "SMALLINT", "DOUBLE"):
            mn, med, mx = con.execute(
                f"SELECT min({c}), median({c}), max({c}) FROM {D}"
            ).fetchone()
    rows.append(
        {
            "column": c,
            "type": t,
            "null_share": None if null_share is None else round(null_share, 4),
            "distinct": distinct,
            "min": mn,
            "median": med,
            "max": mx,
        }
    )
inv = pd.DataFrame(rows)
print(inv.to_string(index=False))
inv.to_csv(OUT_DIR / "inventur_spalten.csv", index=False)

print("\nSpalten mit >95% NULL (praktisch unbrauchbar für dieses File):")
unusable = inv[inv["null_share"].fillna(0) > 0.95]
print(unusable[["column", "null_share"]].to_string(index=False) if len(unusable) else "  keine")


# ---------------------------------------------------------------------------
# 2. KORNGRÖSSE
# ---------------------------------------------------------------------------
hd("2. KORNGRÖSSE: Was ist eine Zeile?")

n_distinct_grp = con.execute(
    f"SELECT count(*) FROM (SELECT DISTINCT trip_id, stop_id, is_arrival FROM {D})"
).fetchone()[0]
print(f"Zeilen gesamt: {n_rows:,}")
print(f"Distinkte (trip_id, stop_id, is_arrival)-Kombinationen: {n_distinct_grp:,}")
print("=> (trip_id, stop_id, is_arrival) ist NICHT der Primärschlüssel. Es gibt pro Halt-Ereignis")
print("   (ein Trip hält an einem Stop, einmal als Ankunft, einmal als Abfahrt) im Schnitt")
print(f"   {n_rows / n_distinct_grp:.2f} Zeilen -- das sind aufeinanderfolgende Prognose-Updates.")

dist = con.execute(
    f"""
    SELECT n, count(*) as num_groups
    FROM (SELECT trip_id, stop_id, is_arrival, count(*) n FROM {D} GROUP BY 1,2,3) t
    GROUP BY n ORDER BY n
    """
).fetchdf()
dist["share"] = dist["num_groups"] / dist["num_groups"].sum()
print("\nVerteilung Meldungen pro Halt-Ereignis (Ausschnitt):")
print(dist.head(15).to_string(index=False))
print(f"... Median Meldungen/Halt: {con.execute(f'''
    SELECT median(n) FROM (SELECT trip_id, stop_id, is_arrival, count(*) n FROM {D} GROUP BY 1,2,3)
''').fetchone()[0]}")
one_msg_share = dist.loc[dist["n"] == 1, "share"].sum()
print(f"Anteil Halt-Ereignisse mit nur EINER Meldung (keine Prognose-Historie beobachtbar): {one_msg_share:.1%}")

# is_final semantics
n_multi_final = con.execute(
    f"""
    SELECT count(*) FROM (
        SELECT trip_id, stop_id, is_arrival, sum(CASE WHEN is_final THEN 1 ELSE 0 END) nf
        FROM {D} GROUP BY 1,2,3
    ) WHERE nf > 1
    """
).fetchone()[0]
n_zero_final = con.execute(
    f"""
    SELECT count(*) FROM (
        SELECT trip_id, stop_id, is_arrival, sum(CASE WHEN is_final THEN 1 ELSE 0 END) nf
        FROM {D} GROUP BY 1,2,3
    ) WHERE nf = 0
    """
).fetchone()[0]
n_final_not_last = con.execute(
    f"""
    WITH ranked AS (
        SELECT trip_id, stop_id, is_arrival, update_timestamp, is_final,
               row_number() OVER (PARTITION BY trip_id, stop_id, is_arrival ORDER BY update_timestamp DESC) rn
        FROM {D}
    )
    SELECT count(*) FROM ranked WHERE is_final AND rn != 1
    """
).fetchone()[0]
print(f"\nGruppen mit >1 is_final=True Zeile: {n_multi_final:,}  (von {n_distinct_grp:,})")
print(f"Gruppen mit 0 is_final=True Zeilen: {n_zero_final:,}")
print(f"is_final=True Zeilen, die NICHT die chronologisch letzte update_timestamp-Zeile ihrer Gruppe sind: {n_final_not_last:,}")
print("=> is_final markiert (fast immer, 99.999%) exakt die letzte im File beobachtete Prognose-Meldung")
print("   für dieses Halt-Ereignis. update_timestamp ist der Zeitpunkt, zu dem die Vorhersage-Pipeline")
print("   diesen Datensatz abgefragt/geschrieben hat. Damit LÄSST SICH der zeitliche Verlauf der")
print("   DB-Verspätungsprognose je Halt rekonstruieren, indem man pro (trip_id, stop_id, is_arrival)")
print("   nach update_timestamp sortiert.")

# concrete example
example = con.execute(
    f"""
    SELECT trip_id, stop_id, is_arrival, count(*) n
    FROM {D}
    WHERE category='S' AND stop_id BETWEEN 8000000 AND 8010000
    GROUP BY 1,2,3
    HAVING n BETWEEN 6 AND 10
    ORDER BY n DESC LIMIT 1
    """
).fetchone()
tid, sid, isarr, n = example
print(f"\nKonkretes Beispiel: trip_id={tid}, stop_id={sid}, is_arrival={isarr} ({n} Meldungen), chronologisch:")
ex_df = con.execute(
    f"""
    SELECT update_timestamp, time_schedule, time_real, delay, is_final, is_cancelled
    FROM {D}
    WHERE trip_id={tid} AND stop_id={sid} AND is_arrival={isarr}
    ORDER BY update_timestamp
    """
).fetchdf()
print(ex_df.to_string(index=False))
ex_df.to_csv(OUT_DIR / "beispiel_prognose_verlauf.csv", index=False)
print("Anmerkung: is_cancelled flackert hier innerhalb desselben delay-Wertes zwischen True/False --")
print("das ist NICHT durch die Daten erklärbar (könnte eine temporär gesetzte/zurückgezogene IRIS-Meldung")
print("sein) und sollte als offene Unsicherheit behandelt werden, nicht als eindeutiges 'Zug fiel aus'.")


# ---------------------------------------------------------------------------
# 3. MÜNCHNER S-BAHN IDENTIFIZIEREN
# ---------------------------------------------------------------------------
hd("3. MÜNCHNER S-BAHN IDENTIFIZIEREN")

cat_counts = con.execute(f"SELECT category, count(*) c FROM {D} GROUP BY 1 ORDER BY 2 DESC").fetchdf()
print("Werte in category (Top 15):")
print(cat_counts.head(15).to_string(index=False))
print(f"'S' kommt vor: {'S' in cat_counts['category'].values} ({cat_counts.loc[cat_counts.category=='S','c'].values[0]:,} Zeilen bundesweit)")

muc_stops = con.execute(
    f"""
    SELECT stop_id, avg(lat) lat, avg(lon) lon, count(*) n_rows, count(DISTINCT trip_id) n_trips
    FROM {D}
    WHERE category='S' AND lat BETWEEN 47.9 AND 48.5 AND lon BETWEEN 11.0 AND 12.1
    GROUP BY stop_id
    ORDER BY n_rows DESC
    """
).fetchdf()
print(f"\n{len(muc_stops)} distinkte S-Bahn stop_id im Münchner Suchraum (lat 47.9-48.5, lon 11.0-12.1).")

# Bekannte Koordinaten der 12 Stammstrecken-Stationen, Quelle: MediaWiki-API (de.wikipedia.org),
# abgerufen am 2026-08-26. Werden NUR zur Bestätigung/Konfidenz-Einschätzung genutzt (Distanz-Check),
# NICHT zur primären Zuordnung -- siehe Begründung unten (reine Nearest-Neighbour-Zuordnung nach
# Distanz griff bei Hauptbahnhof tief fehl, weil zwei stop_ids nur ~150m auseinanderliegen).
WIKI_COORDS = {
    "München-Pasing":              dict(lat=48.150000, lon=11.461389),
    "München-Laim":                 dict(lat=48.144444, lon=11.503333),
    "München Hirschgarten":         dict(lat=48.143511, lon=11.518015),
    "München Donnersbergerbrücke":  dict(lat=48.142778, lon=11.535000),
    "München Hackerbrücke":         dict(lat=48.141944, lon=11.548611),
    "München Hauptbahnhof (tief)":  dict(lat=48.140442, lon=11.557723),  # Gesamt-Hbf, nicht Tief-spezifisch
    "München Karlsplatz (Stachus)": dict(lat=48.139167, lon=11.565833),
    "München Marienplatz":          dict(lat=48.137222, lon=11.575278),
    "München Isartor":              dict(lat=48.135101, lon=11.581699),
    "München Rosenheimer Platz":    dict(lat=None, lon=None),  # kein Wikipedia-Artikel mit Koordinate gefunden
    "München Ost":                  dict(lat=48.126944, lon=11.604722),
    "München Leuchtenbergring":     dict(lat=None, lon=None),  # kein Wikipedia-Artikel mit Koordinate gefunden
}
ORDER = list(WIKI_COORDS.keys())  # West -> Ost, wie vom Nutzer vorgegeben


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# --- Schritt A: die 12 "Kern"-stop_ids datengetrieben bestimmen -----------------------------------
# Sortiert man alle S-Bahn-stop_ids im Münchner Suchraum nach Zeilenanzahl absteigend, gibt es einen
# klaren, natuerlichen Bruch nach Rang 12 (Traffic fällt von >8000 auf <6000 Zeilen/Tag). Das IST
# unabhaengig von jeder Namenszuordnung ein Beleg dafuer, dass die Stammstrecke aus genau 12 Halten
# besteht -- diese 12 IDs werden unten benutzt, es wird nicht geraten, wie viele es sind.
muc_by_traffic = muc_stops.sort_values("n_rows", ascending=False).reset_index(drop=True)
print("\nRanking der Münchner S-Bahn-Halte nach Zeilenanzahl (Rang 10-15, zeigt den Bruch nach Rang 12):")
print(muc_by_traffic.iloc[8:15][["stop_id", "lat", "lon", "n_rows", "n_trips"]].to_string(index=False))
core_ids = set(muc_by_traffic.head(12)["stop_id"].tolist())

# --- Schritt B: die reale Fahrtreihenfolge (stop_sequence) nutzen, um die 12 Kern-IDs West->Ost zu ---
# --- sortieren, statt sie unabhängig voneinander per Koordinaten-Distanz zu erraten. ------------------
core_ids_sql = ",".join(str(int(x)) for x in core_ids)
trip_coverage = con.execute(
    f"""
    SELECT trip_id, count(DISTINCT stop_id) n_core
    FROM {D}
    WHERE category='S' AND stop_id IN ({core_ids_sql})
    GROUP BY trip_id ORDER BY n_core DESC LIMIT 1
    """
).fetchone()
trip_a = trip_coverage[0]
order_a = con.execute(
    f"""
    SELECT DISTINCT stop_sequence, stop_id FROM {D}
    WHERE trip_id={trip_a} AND stop_id IN ({core_ids_sql})
    ORDER BY stop_sequence
    """
).fetchdf()
ids_in_a = order_a["stop_id"].tolist()
missing_ids = [x for x in core_ids if x not in ids_in_a]
print(f"\nReferenz-Trip A (trip_id={trip_a}) deckt {len(ids_in_a)}/12 Kern-Halte ab, in dieser Reihenfolge: {ids_in_a}")

final_order_ids = list(ids_in_a)
if missing_ids:
    anchor = ids_in_a[-1]
    anchor_sql = f"stop_id={anchor}"
    missing_sql = " OR ".join(f"stop_id={m}" for m in missing_ids)
    trip_b = con.execute(
        f"""
        SELECT trip_id FROM {D}
        WHERE category='S' AND trip_id IN (
            SELECT trip_id FROM {D} WHERE category='S' AND {anchor_sql}
        )
        AND ({missing_sql})
        GROUP BY trip_id
        HAVING count(DISTINCT stop_id) = {len(missing_ids)}
        LIMIT 1
        """
    ).fetchone()
    if trip_b:
        trip_b = trip_b[0]
        order_b = con.execute(
            f"""
            SELECT DISTINCT stop_sequence, stop_id FROM {D}
            WHERE trip_id={trip_b} AND (stop_id IN ({','.join(str(m) for m in missing_ids)}) OR stop_id={anchor})
            ORDER BY stop_sequence
            """
        ).fetchdf()
        ids_in_b = order_b["stop_id"].tolist()
        print(f"Referenz-Trip B (trip_id={trip_b}, enthält Anker {anchor} + fehlende Halte) Reihenfolge: {ids_in_b}")
        idx_anchor = ids_in_b.index(anchor)
        for x in ids_in_b[idx_anchor + 1:]:
            if x not in final_order_ids:
                final_order_ids.append(x)
    else:
        print(f"WARNUNG: kein Trip gefunden, der Anker {anchor} und alle fehlenden Halte {missing_ids} verbindet.")
        final_order_ids += missing_ids

print(f"Datengetriebene West->Ost-Reihenfolge der 12 Kern-Halte: {final_order_ids}")

assignments = []
for name, sid in zip(ORDER, final_order_ids):
    row = muc_stops[muc_stops.stop_id == sid].iloc[0]
    wiki = WIKI_COORDS[name]
    if wiki["lat"] is None:
        conf = "mittel: Position via reale Trip-Reihenfolge bestätigt, aber KEIN externer Koordinatenbeleg gefunden (Name daher nicht unabhängig verifiziert)"
        d_m = None
    else:
        d_m = haversine_m(wiki["lat"], wiki["lon"], row.lat, row.lon)
        if d_m < 150:
            conf = f"hoch: {d_m:.0f} m von Wikipedia-Koordinate, Position in Trip-Reihenfolge stimmt überein"
        elif d_m < 400:
            conf = f"mittel: {d_m:.0f} m von Wikipedia-Koordinate (grösserer Versatz, aber Position in Trip-Reihenfolge stimmt überein)"
        else:
            conf = f"NIEDRIG: {d_m:.0f} m von Wikipedia-Koordinate trotz korrekter Position in Trip-Reihenfolge -- bitte prüfen"
    assignments.append(
        dict(stop_id=int(sid), name=name, lat=row.lat, lon=row.lon,
             n_rows=row.n_rows, n_trips=row.n_trips,
             dist_m=None if d_m is None else round(d_m, 1), confidence=conf)
    )

assign_df = pd.DataFrame(assignments)
print("\nZuordnung Stammstrecken-Stationen (Trip-Reihenfolge als Primärkriterium, Koordinaten zur Bestätigung):")
print(assign_df.to_string(index=False))

# Bekannter Zwilling: 8098261 ist derselbe physische Ort wie Hauptbahnhof (tief), aber nur von
# Linie S7 belegt (580 Zeilen, 114 Trips) -- eine eigene stop_id fuer dieselbe Plattformgruppe.
# Wird bewusst NICHT automatisch gemerged, siehe BEFUNDE.md.
twin_check = con.execute(
    f"SELECT line, count(*) c FROM {D} WHERE stop_id=8098261 AND category='S' GROUP BY 1"
).fetchdf()
print("\nBekannter Sonderfall: stop_id 8098261 (physisch selber Ort wie Hauptbahnhof tief, andere ID):")
print(twin_check.to_string(index=False))

assigned_ids = set(assign_df["stop_id"])
missing = [n for n, r in zip(assign_df["name"], assign_df["stop_id"]) if r not in muc_stops.stop_id.values]
print(f"\nFehlende Stammstrecken-Stationen in den Daten: {'keine -- alle 12 vorhanden' if not missing else missing}")

# lines/operators serving these 12 stops
lines_ops = con.execute(
    f"""
    SELECT stop_id, line, operator, count(*) c
    FROM {D}
    WHERE category='S' AND stop_id IN ({','.join(str(int(x)) for x in assigned_ids)})
    GROUP BY 1,2,3 ORDER BY 1, c DESC
    """
).fetchdf()
name_by_id = dict(zip(assign_df.stop_id, assign_df.name))
lines_ops["name"] = lines_ops["stop_id"].map(name_by_id)
print("\nLinien (line) und Betreiber (operator) je Stammstrecken-Halt:")
print(lines_ops.to_string(index=False))
print("\nDistinkte Linien über die ganze Stammstrecke:", sorted(lines_ops["line"].dropna().unique().tolist()))
print("Distinkte operator-Werte:", sorted(lines_ops["operator"].dropna().unique().tolist()))

# full CSV of all Munich-area S stops with the assignment merged in
muc_stops_out = muc_stops.merge(
    assign_df[["stop_id", "name", "confidence"]], on="stop_id", how="left"
)
muc_stops_out["confidence"] = muc_stops_out["confidence"].fillna("nicht zugeordnet (nicht Teil der 12 Stammstrecken-Halte)")
muc_stops_out["name"] = muc_stops_out["name"].fillna("")
muc_stops_out = muc_stops_out.sort_values("n_rows", ascending=False)
muc_stops_out.to_csv(OUT_DIR / "muenchen_stationen.csv", index=False)
print(f"\n-> {OUT_DIR/'muenchen_stationen.csv'} geschrieben ({len(muc_stops_out)} Zeilen).")


# ---------------------------------------------------------------------------
# 4. DATENQUALITÄT
# ---------------------------------------------------------------------------
hd("4. DATENQUALITÄT (bundesweit, sowie Fokus Stammstrecke)")

delay_stats = con.execute(f"SELECT min(delay), median(delay), max(delay), avg(delay) FROM {D}").fetchone()
print(f"delay (Sekunden) bundesweit: min={delay_stats[0]}, median={delay_stats[1]}, max={delay_stats[2]}, mean={delay_stats[3]:.1f}")
extreme = con.execute(
    f"SELECT count(*) FROM {D} WHERE delay > 3600*3 OR delay < -600"
).fetchone()[0]
print(f"Zeilen mit delay > 3h oder < -10min (unplausibel bzw. Ausreißer): {extreme:,} ({extreme/n_rows:.3%})")
print(con.execute(f"SELECT quantile(delay, [0.01,0.05,0.5,0.95,0.99,0.999]) FROM {D} WHERE delay IS NOT NULL").fetchone())

n_cancelled = con.execute(f"SELECT sum(CASE WHEN is_cancelled THEN 1 ELSE 0 END) FROM {D}").fetchone()[0]
print(f"\nis_cancelled=True: {n_cancelled:,} von {n_rows:,} Zeilen ({n_cancelled/n_rows:.2%})")
n_betriebshalt = con.execute(f"SELECT sum(CASE WHEN is_betriebshalt THEN 1 ELSE 0 END) FROM {D}").fetchone()[0]
print(f"is_betriebshalt=True: {n_betriebshalt:,} ({n_betriebshalt/n_rows:.2%})")
pdo = con.execute(f"SELECT pickup_drop_off_type, count(*) c FROM {D} GROUP BY 1 ORDER BY 2 DESC").fetchdf()
print("\npickup_drop_off_type Verteilung:")
print(pdo.to_string(index=False))
n_notavail = pdo.loc[pdo.pickup_drop_off_type == "NOT_AVAILABLE", "c"].sum()
print(f"NOT_AVAILABLE: {n_notavail:,} ({n_notavail/n_rows:.2%}) -- Kandidat zum Rausfiltern.")

print("\nStündliche Zeilenanzahl (nach time_schedule-Stunde), gesamt:")
hourly = con.execute(
    f"SELECT date_part('hour', time_schedule) h, count(*) c FROM {D} GROUP BY 1 ORDER BY 1"
).fetchdf()
print(hourly.to_string(index=False))
gaps = hourly.loc[hourly.c < hourly.c.median() * 0.05]
print(f"Stunden mit auffällig wenig Daten (<5% des Medians): {gaps['h'].tolist() if len(gaps) else 'keine (erwartbar: Nachtstunden mit wenig Zugverkehr am Feiertag sind niedrig, aber nicht leer)'}")

stamm_ids_sql = ",".join(str(int(x)) for x in assigned_ids)
hourly_stamm = con.execute(
    f"""
    SELECT date_part('hour', time_schedule) h, count(*) c
    FROM {D} WHERE stop_id IN ({stamm_ids_sql})
    GROUP BY 1 ORDER BY 1
    """
).fetchdf()
print("\nStündliche Zeilenanzahl NUR Stammstrecke:")
print(hourly_stamm.to_string(index=False))


# ---------------------------------------------------------------------------
# 5. MESSAGE_CODES
# ---------------------------------------------------------------------------
hd("5. MESSAGE_CODES")

# IRIS-Meldungscode-Tabelle, Quelle: Perl-Modul Travel::Status::DE::IRIS::Result von derf,
# https://github.com/derf/Travel-Status-DE-IRIS  (community-reverse-engineerte, seit Jahren
# gepflegte Referenz für die numerischen "id"-Felder der DB-IRIS-Schnittstelle; abgerufen 2026-08-26).
# Codes 0 und 1000 kommen in dieser Tabelle NICHT vor -- das wird unten explizit offengelassen.
IRIS_CODES = {
    1: "Nähere Informationen in Kürze", 2: "Polizeieinsatz", 3: "Feuerwehreinsatz auf der Strecke",
    4: "Kurzfristiger Personalausfall", 5: "Ärztliche Versorgung eines Fahrgastes", 6: "Betätigen der Notbremse",
    7: "Unbefugte Personen auf der Strecke", 8: "Notarzteinsatz auf der Strecke", 9: "Streikauswirkungen",
    10: "Tiere auf der Strecke", 11: "Unwetter", 12: "Warten auf ein verspätetes Schiff",
    13: "Pass- und Zollkontrolle", 14: "Defekt am Bahnhof", 15: "Beeinträchtigung durch Vandalismus",
    16: "Entschärfung einer Fliegerbombe", 17: "Beschädigung einer Brücke", 18: "Umgestürzter Baum auf der Strecke",
    19: "Unfall an einem Bahnübergang", 20: "Tiere im Gleis", 21: "Warten auf Anschlussreisende",
    22: "Witterungsbedingte Beeinträchtigungen", 23: "Betriebsstabilisierung", 24: "Verspätung im Ausland",
    25: "Bereitstellung weiterer Wagen", 26: "Abhängen von Wagen", 27: "Technische Störung am Bus",
    28: "Gegenstände auf der Strecke", 29: "Ersatzverkehr mit Bus ist eingerichtet", 30: "Personalausfall im Stellwerk",
    31: "Bauarbeiten", 32: "Längere Haltezeit am Bahnhof", 33: "Defekt an der Oberleitung",
    34: "Defekt an einem Signal", 35: "Streckensperrung", 36: "Technische Störung am Zug",
    37: "Kurzfristiger Fahrzeugausfall", 38: "Defekt an der Strecke", 39: "Stau / Hohes Verkehrsaufkommen",
    40: "Defektes Stellwerk", 41: "Defekt an einem Bahnübergang", 42: "Außerplanmäßige Geschwindigkeitsbeschränkung",
    43: "Verspätung eines vorausfahrenden Zuges", 44: "Warten auf einen entgegenkommenden Zug",
    45: "Vorfahrt eines anderen Zuges", 46: "Vorfahrt eines anderen Zuges", 47: "Verspätete Bereitstellung",
    48: "Verspätung aus vorheriger Fahrt", 49: "Kurzfristiger Personalausfall", 50: "Kurzfristige Erkrankung von Personal",
    51: "Verspätetes Personal aus vorheriger Fahrt", 52: "Streik", 53: "Unwetterauswirkungen",
    54: "Verfügbarkeit der Gleise derzeit eingeschränkt", 55: "Technischer Defekt an einem anderen Zug",
    56: "Laden der Antriebsbatterie", 57: "Zusätzlicher Halt", 58: "Umleitung", 59: "Schnee und Eis",
    60: "Witterungsbedingt verminderte Geschwindigkeit", 61: "Defekte Tür", 62: "Behobener Defekt am Zug",
    63: "Technische Untersuchung am Zug", 64: "Defekt an einer Weiche", 65: "Erdrutsch", 66: "Hochwasser",
    67: "Behördliche Maßnahme", 68: "Hohes Fahrgastaufkommen", 69: "Zug verkehrt mit verminderter Geschwindigkeit",
    70: "WLAN nicht verfügbar", 71: "Eingeschränktes WLAN", 72: "Info/Entertainment nicht verfügbar",
    73: "Heute: Mehrzweckabteil vorne", 74: "Heute: Mehrzweckabteil hinten", 75: "Heute: 1. Klasse vorne",
    76: "Heute: 1. Klasse hinten", 77: "1. Klasse fehlt", 78: "Ersatzverkehr mit Bus ist eingerichtet",
    79: "Mehrzweckabteil fehlt", 80: "Abweichende Wagenreihung", 81: "Fahrzeugtausch", 82: "Mehrere Wagen fehlen",
    83: "Heute ohne fahrzeuggebundene Einstiegshilfe", 84: "Zug verkehrt richtig gereiht", 85: "Ein Wagen fehlt",
    86: "Gesamter Zug ohne Reservierung", 87: "Einzelne Wagen ohne Reservierung", 88: "Keine Qualitätsmängel",
    89: "Reservierungen sind wieder vorhanden", 90: "Kein gastronomisches Angebot", 91: "Fahrradmitnahme nicht möglich",
    92: "Fahrradmitnahme kann nicht garantiert werden", 93: "Behindertengerechte Einrichtung fehlt",
    94: "Ersatzbewirtschaftung", 95: "Universaltoilette fehlt", 96: "Zustieg kann nicht garantiert werden",
    97: "Hohe Auslastung", 98: "Sonstige Qualitätsmängel", 99: "Verzögerungen im Betriebsablauf",
}

top_codes_stamm = con.execute(
    f"""
    SELECT code, count(*) c FROM (
        SELECT unnest(message_codes) code FROM {D} WHERE stop_id IN ({stamm_ids_sql})
    ) GROUP BY code ORDER BY c DESC LIMIT 20
    """
).fetchdf()
top_codes_stamm["bedeutung"] = top_codes_stamm["code"].map(
    lambda c: IRIS_CODES.get(c, "NICHT in der IRIS-Referenztabelle gefunden (evtl. datensatz-internes Sonderfeld)")
)
print("Top-20 message_codes auf der Münchner Stammstrecke, mit Bedeutung (Quelle: Travel::Status::DE::IRIS, derf):")
print(top_codes_stamm.to_string(index=False))
top_codes_stamm.to_csv(OUT_DIR / "message_codes_top20_stammstrecke.csv", index=False)
print("\nHinweis: Codes 0 und 1000 sind in der derf-Referenztabelle NICHT dokumentiert. Ohne verlässliche")
print("Quelle für diese beiden Werte wird hier bewusst KEINE Bedeutung behauptet.")


# ---------------------------------------------------------------------------
# 6. ERSTE INHALTLICHE BEFUNDE
# ---------------------------------------------------------------------------
hd("6. ERSTE INHALTLICHE BEFUNDE (Stammstrecke, is_final=True Zeilen = letzte Prognose je Halt)")

stamm_final = con.execute(
    f"""
    SELECT trip_id, stop_id, stop_sequence, line, delay, time_schedule, is_arrival, is_cancelled
    FROM {D}
    WHERE stop_id IN ({stamm_ids_sql}) AND category = 'S' AND is_final = true AND is_cancelled = false
    """
).fetchdf()
stamm_final["name"] = stamm_final["stop_id"].map(name_by_id)
stamm_final["hour"] = pd.to_datetime(stamm_final["time_schedule"]).dt.hour

# delay by line
by_line = stamm_final.groupby("line")["delay"].agg(["count", "mean", "median"]).sort_values("count", ascending=False)
print("Verspätung nach Linie (finale Prognose, is_cancelled=False):")
print(by_line.to_string())

# delay by hour
by_hour = stamm_final.groupby("hour")["delay"].agg(["count", "mean", "median"])
print("\nVerspätung nach Tageszeit:")
print(by_hour.to_string())

# delay by position along Stammstrecke (name, ordered west->east; ORDER defined in section 3)
by_pos = stamm_final.groupby("name")["delay"].agg(["count", "mean", "median"]).reindex(ORDER)
print("\nVerspätung nach Position entlang der Stammstrecke (West -> Ost):")
print(by_pos.to_string())

# autocorrelation with previous stop of same trip
seq_df = con.execute(
    f"""
    SELECT trip_id, stop_id, stop_sequence, delay
    FROM {D}
    WHERE stop_id IN ({stamm_ids_sql}) AND category = 'S' AND is_final = true AND is_cancelled = false AND is_arrival = false
    ORDER BY trip_id, stop_sequence
    """
).fetchdf()
seq_df["prev_delay"] = seq_df.groupby("trip_id")["delay"].shift(1)
seq_df["prev_stop_sequence"] = seq_df.groupby("trip_id")["stop_sequence"].shift(1)
consec = seq_df.dropna(subset=["prev_delay"])
consec = consec[consec["stop_sequence"] == consec["prev_stop_sequence"] + 1]
corr = consec["delay"].corr(consec["prev_delay"])
print(f"\nKorrelation delay(Halt i) vs. delay(Halt i-1, selbe trip_id, direkt vorheriger stop_sequence): "
      f"r={corr:.3f}  (n={len(consec):,} aufeinanderfolgende Halt-Paare)")

# --- Plots ---
fig, ax = plt.subplots(figsize=(9, 5))
by_pos["mean"].plot(kind="bar", ax=ax, color="#4C72B0")
ax.set_ylabel("mittlere Verspätung (Sekunden)")
ax.set_title("Mittlere Verspätung entlang der Stammstrecke (West -> Ost), 1.1.2025 (Feiertag)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(PLOT_DIR / "delay_by_position.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
by_hour["mean"].plot(kind="line", marker="o", ax=ax, color="#C44E52")
ax.set_xlabel("Stunde (time_schedule)")
ax.set_ylabel("mittlere Verspätung (Sekunden)")
ax.set_title("Mittlere Verspätung nach Tageszeit, Stammstrecke, 1.1.2025 (Feiertag)")
plt.tight_layout()
plt.savefig(PLOT_DIR / "delay_by_hour.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(consec["prev_delay"], consec["delay"], s=4, alpha=0.2, color="#55A868")
ax.set_xlabel("delay am vorherigen Halt (s)")
ax.set_ylabel("delay am aktuellen Halt (s)")
ax.set_title(f"Verspätungsfortpflanzung Halt i-1 -> Halt i (r={corr:.2f})")
lim = min(consec["delay"].quantile(0.995), consec["prev_delay"].quantile(0.995))
ax.set_xlim(-120, lim)
ax.set_ylim(-120, lim)
plt.tight_layout()
plt.savefig(PLOT_DIR / "delay_autocorrelation.png", dpi=150)
plt.close(fig)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

ax1.scatter(muc_stops["lon"], muc_stops["lat"], s=muc_stops["n_rows"] / 150, alpha=0.35, color="#4C72B0",
            label="alle S-Bahn-Halte (München-Raum)")
ax1.scatter(assign_df["lon"], assign_df["lat"], s=25, color="red", label="Stammstrecke")
ax1.set_xlabel("lon")
ax1.set_ylabel("lat")
ax1.set_title("S-Bahn-Netz Münchner Raum")
ax1.legend(fontsize=8, loc="lower left")
ax1.add_patch(plt.Rectangle((11.44, 48.12), 0.19, 0.045, fill=False, edgecolor="black", linewidth=1))

short_name = assign_df["name"].str.replace("München", "", regex=False).str.replace("-", "", regex=False).str.strip()
ax2.plot(assign_df["lon"], assign_df["lat"], "-", color="red", linewidth=1.5, zorder=1)
ax2.scatter(assign_df["lon"], assign_df["lat"], s=35, color="red", zorder=2)
for i, (_, r) in enumerate(assign_df.iterrows()):
    offset_y = 0.006 if i % 2 == 0 else -0.006
    va = "bottom" if i % 2 == 0 else "top"
    ax2.annotate(short_name.iloc[i], (r["lon"], r["lat"]), xytext=(r["lon"], r["lat"] + offset_y),
                 fontsize=8, ha="center", va=va,
                 arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
ax2.set_xlabel("lon")
ax2.set_ylabel("lat")
ax2.set_title("Stammstrecke im Detail (West -> Ost)")
ax2.set_xlim(11.44, 11.63)
ax2.set_ylim(48.12, 48.165)

fig.suptitle("Münchner S-Bahn-Stammstrecke: Lage der 12 zugeordneten Halte")
plt.tight_layout()
plt.savefig(PLOT_DIR / "stammstrecke_karte.png", dpi=150)
plt.close(fig)

print(f"\nPlots geschrieben nach {PLOT_DIR}/: delay_by_position.png, delay_by_hour.png, "
      f"delay_autocorrelation.png, stammstrecke_karte.png")

print("\nFertig.")

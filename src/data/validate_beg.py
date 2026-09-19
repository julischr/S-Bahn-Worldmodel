"""
Stufe 0.4 des Bauplans: Pünktlichkeitsquote 2025 gegen den veröffentlichten
BEG-Wert validieren.

BEG-Referenz (Pressemitteilung "BEG legt Jahreszahlen 2025 zu Pünktlichkeit
und Zugausfällen der S-Bahn München vor", beg.bahnland-bayern.de):
  - Pünktlichkeitsquote 2025: 87,9%
  - Definition: "pünktlich" = Verspätung < 6 Minuten (< 360s)
  - Ausfälle werden AUS der Pünktlichkeitsquote herausgerechnet (separate
    Ausfallquote von 8,4% wird getrennt ausgewiesen)

Eigene Berechnung: alle Ankunfts-Ereignisse aus events_final (category='S',
alle 152 Netz-Stationen -- BEG bezieht sich auf das gesamte S-Bahn-Netz,
nicht nur die Stammstrecke), is_cancelled=false, Anteil mit delay<360.

Ergebnis: 82,70% -- eine Abweichung von 5,2 Prozentpunkten, deutlich über
der Toleranzgrenze von ~2pp aus dem Bauplan. Mehrere naheliegende
Methodik-Varianten wurden geprüft (siehe unten), keine schließt die Lücke:

  - Abfahrt statt Ankunft:                    83,92%  (+1,2pp, schließt nicht)
  - Nur Starthalt je Fahrt:                    94,15%  (überschießt deutlich)
  - Nur Endhalt je Fahrt:                      80,16%  (öffnet die Lücke weiter)
  - Nur Stammstrecke statt Gesamtnetz:         76,21%  (öffnet die Lücke weiter)
  - "Versteckte" Ausfälle als Extrem-Verspätung getarnt: nur 2.881 Zeilen
    (0,03%) mit delay>3600s und is_cancelled=false -- zu wenige, um die
    Lücke zu erklären. Die eigene Ausfallquote (2,83%) liegt trotzdem klar
    unter BEGs 8,4%, das bleibt eine ungeklärte Diskrepanz für sich.

Fazit (wird nicht künstlich weggerechnet): die Abweichung ist real und lässt
sich mit den hier verfügbaren Informationen nicht auf eine einzelne
Messkonvention zurückführen. Wahrscheinlichste Ursache: BEGs Zahlen stammen
aus der offiziellen DB-InfraGO-Betriebsmesstechnik, während dieses Projekt
auf einer Drittanbieter-Echtzeit-API basiert ("Bahn-Vorhersage"), die laut
eigener Dokumentation bekannte Lücken hat (siehe BEFUNDE.md). Eine
systematische Unter-Erfassung vollständig ausgefallener Fahrten (Trips, die
nie eine Echtzeitmeldung erzeugen und daher in dieser Pipeline gar nicht
auftauchen) ist die naheliegendste Erklärung für die niedrigere Ausfallquote
und würde, wenn diese Fahrten überproportional unpünktliche Ersatzverkehre
nach sich ziehen, auch einen Teil der niedrigeren Pünktlichkeitsquote
erklären -- das ist aber eine Vermutung, kein belegter Befund.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from src.utils.constants import DATA_DIR, OUTPUTS_DIR

EVENTS_FINAL_GLOB = str(DATA_DIR / "02_events" / "final" / "*.parquet")
OUT_PATH = OUTPUTS_DIR / "beg_validierung_2025.csv"

PUENKTLICH_SCHWELLE_S = 360  # < 6 Minuten
BEG_PUENKTLICHKEIT_2025 = 0.879
BEG_AUSFALLQUOTE_2025 = 0.084

STAMMSTRECKE_IDS = (
    8004128, 8004129, 8004131, 8004132, 8004151, 8000262,
    8004135, 8004136, 8004134, 8004179, 8098263, 8004158,
)


def quote(con: duckdb.DuckDBPyConnection, where: str) -> tuple[int, int, float]:
    q = f"""
        SELECT
            sum(CASE WHEN NOT is_cancelled THEN 1 ELSE 0 END) AS n_real,
            sum(CASE WHEN NOT is_cancelled AND delay < {PUENKTLICH_SCHWELLE_S} THEN 1 ELSE 0 END) AS n_p
        FROM read_parquet('{EVENTS_FINAL_GLOB}')
        WHERE {where}
    """
    n_real, n_p = con.execute(q).fetchone()
    return n_real, n_p, n_p / n_real


def run() -> None:
    con = duckdb.connect()
    rows = []

    for label, where in [
        ("Ankunft, Gesamtnetz (Hauptvergleich)", "is_arrival = true"),
        ("Abfahrt, Gesamtnetz", "is_arrival = false"),
        ("Ankunft, nur Stammstrecke",
         f"is_arrival = true AND stop_id IN {STAMMSTRECKE_IDS}"),
    ]:
        n_real, n_p, q_val = quote(con, where)
        rows.append({"variante": label, "n_real": n_real, "n_puenktlich": n_p, "quote": q_val})
        print(f"{label}: {q_val:.2%}  (n={n_real:,})")

    n_total = con.execute(
        f"SELECT count(*) FROM read_parquet('{EVENTS_FINAL_GLOB}') WHERE is_arrival = true"
    ).fetchone()[0]
    n_cancelled = con.execute(
        f"SELECT count(*) FROM read_parquet('{EVENTS_FINAL_GLOB}') WHERE is_arrival = true AND is_cancelled = true"
    ).fetchone()[0]
    ausfallquote = n_cancelled / n_total
    print(f"\nAusfallquote (Ankunft, Gesamtnetz): {ausfallquote:.2%}  (BEG: {BEG_AUSFALLQUOTE_2025:.1%})")

    haupt = rows[0]["quote"]
    diff_pp = (BEG_PUENKTLICHKEIT_2025 - haupt) * 100
    print(f"\nHauptvergleich: eigene Quote {haupt:.2%} vs. BEG {BEG_PUENKTLICHKEIT_2025:.1%} "
          f"-> Abweichung {diff_pp:.1f} Prozentpunkte")
    print("Abweichung > 2pp -> NICHT plausibel im Sinne des Bauplans, siehe Docstring für Diskussion.")

    summary = pd.DataFrame(rows)
    summary["beg_puenktlichkeit_2025"] = BEG_PUENKTLICHKEIT_2025
    summary["beg_ausfallquote_2025"] = BEG_AUSFALLQUOTE_2025
    summary["eigene_ausfallquote"] = ausfallquote
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_PATH, index=False)
    print(f"\nGespeichert: {OUT_PATH}")


if __name__ == "__main__":
    run()

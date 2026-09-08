# Befunde: Bahn-Vorhersage-Tagesdatei 2025-01-01, Fokus Münchner S-Bahn-Stammstrecke

Alle Zahlen in diesem Dokument stammen aus dem tatsächlichen Lauf von `explore.py`
gegen `data/raw/2025-01-01.parquet` (Log: `outputs/run_log.txt`). Nichts ist geschätzt.

**Wichtigste Einschränkung vorab:** Der 1.1. ist Neujahr, ein gesetzlicher Feiertag.
Der Betrieb entspricht einem Sonn-/Feiertagsfahrplan (reduzierter Takt, andere
Lastspitzen). Für alles, was mit *normalem* Werktagsbetrieb zu tun hat -- Pendlerspitzen,
Regel-Verspätungsniveau, Kapazitätsgrenzen der Stammstrecke im HVZ -- braucht es
mindestens einen Dienstag/Mittwoch/Donnerstag aus `data/raw/2025.tar`, der kein Feiertag und
keine Ferienzeit ist.

## Was drin ist

Die Datei enthält 3.581.875 Zeilen für den 1.1.2025 (deutschlandweit, 41,9 MB), mit
`time_schedule` von 2025-01-01 01:00 bis 2025-01-02 14:26 -- die Überlappung in den 2.1.
hinein ist normal (Trips, die am Silvesterabend/Neujahrsnacht beginnen, laufen über
Mitternacht weiter) und kein Datenfehler.

Keine Spalte ist "praktisch leer" (>95% NULL). `message_codes` ist zu 69% NULL (die
meisten Halte haben keine Zusatzmeldung), `update_timestamp` zu 1,45% NULL,
`dwell_time_schedule`/`dwell_time_real` zu 5,8%/7,8% NULL -- alles plausibel und nutzbar.

**Die Kerngröße (Aufgabe 2) ist die wichtigste Erkenntnis für das Projekt:** Eine Zeile ist
NICHT ein Halt-Ereignis, sondern eine einzelne Prognose-Meldung zu einem Halt-Ereignis.
`(trip_id, stop_id, is_arrival)` hat im Schnitt 3,6 Zeilen (Median 3), mit einer langen
Verteilung bis zu >70 Meldungen für einzelne Halte. `update_timestamp` ist der Zeitpunkt
der jeweiligen Abfrage/Prognose-Aktualisierung, `is_final` markiert (in 994.672 von
994.680 Gruppen, also 99,999%) exakt die chronologisch letzte beobachtete Meldung. Damit
lässt sich der volle Prognoseverlauf je Halt rekonstruieren -- genau die Grundlage, die
für ein Verspätungsfortpflanzungsmodell gebraucht wird. Ein konkretes Beispiel
(trip_id=3626237496988271079, stop_id=8004134, Ankunft) mit 10 Meldungen liegt in
`outputs/beispiel_prognose_verlauf.csv`: die Prognose wandert dort von "pünktlich" über
120s, 180s, 300s... bis 480s (final) -- ein sauberer Fall von schrittweise wachsender
Verspätungsprognose. 25,6% aller Halt-Ereignisse haben aber nur EINE Meldung -- für diese
lässt sich kein Prognoseverlauf beobachten, nur der Endzustand.

**Münchner Stammstrecke (Aufgabe 3):** Alle 12 vom Nutzer genannten Stationen sind in den
Daten vorhanden -- keine fehlt. Die Zuordnung wurde nicht rein über Koordinaten-Distanz
gemacht (das schlug bei Hauptbahnhof tief fehl, siehe Fallstricke unten), sondern
datengetrieben: die 12 stop_ids mit dem höchsten Verkehrsaufkommen im Münchner Suchraum
zeigen einen klaren, natürlichen Bruch (Rang 12: 8216 Zeilen/933 Trips, Rang 13: 5670
Zeilen/333 Trips) -- das bestätigt unabhängig von jeder Namenszuordnung, dass es genau 12
Kernhalte gibt. Ein einzelner echter Trip (trip_id=3120438276051845293) durchfährt alle 12
in der Reihenfolge Pasing -> Laim -> Hirschgarten -> Donnersbergerbrücke -> Hackerbrücke
-> Hauptbahnhof(tief) -> Karlsplatz/Stachus -> Marienplatz -> Isartor -> Rosenheimer Platz
-> Ostbahnhof -> Leuchtenbergring. Diese Reihenfolge plus Wikipedia-Koordinaten (wo
verfügbar) ergeben die Zuordnung in `muenchen_stationen.csv`. Zehn der zwölf Stationen sind
mit "hoch" oder "mittel" (Abweichung <250m von einer unabhängigen Wikipedia-Koordinate)
belegt; für Rosenheimer Platz und Leuchtenbergring wurde keine unabhängige externe
Koordinate gefunden -- ihre Position in der echten Fahrtreihenfolge ist zwar eindeutig
bestätigt, der Name selbst aber nicht extern verifiziert. Bedient werden die Stationen von
S1-S8 (alle mit operator=800725), an Hauptbahnhof (tief) zusätzlich ein separates
stop_id 8098261 nur für S7 (580 Zeilen) -- siehe Fallstricke.

**Datenqualität (Aufgabe 4):** delay liegt (bundesweit) zwischen -87.180s und +22.860s,
Median 60s, Mittelwert 176s. Nur 0,017% der Zeilen haben delay > 3h oder < -10min -- die
Ausreißer sind selten, aber real vorhanden (nicht gefiltert). is_cancelled=True in 2,04%
der Zeilen. is_betriebshalt=True nur in 0,02% (659 Zeilen) -- praktisch vernachlässigbar.
pickup_drop_off_type=NOT_AVAILABLE in 0,44% (15.851 Zeilen) -- ein sinnvoller, kleiner
Filterkandidat. Keine Stunde des Tages ist ohne Daten, auch nachts nicht (0-4 Uhr klar
niedriger, aber nie null) -- weder bundesweit noch auf der Stammstrecke.

**message_codes (Aufgabe 5):** Auf der Stammstrecke ist Code 7 ("Unbefugte Personen auf
der Strecke") mit Abstand am häufigsten (119.053 Vorkommen), gefolgt von 36 ("Technische
Störung am Zug", 31.244) und 5 ("Ärztliche Versorgung eines Fahrgastes", 19.763). Die
Zuordnungstabelle stammt aus dem Quellcode des Perl-Moduls
`Travel::Status::DE::IRIS::Result` (github.com/derf/Travel-Status-DE-IRIS), einer seit
Jahren gepflegten, community-reverse-engineerten Referenz für die numerischen IRIS-
Meldungscodes. Die volle Top-20-Tabelle liegt in
`outputs/message_codes_top20_stammstrecke.csv`. **Codes 0 und 1000 kommen in dieser Referenz
nicht vor** (0: 14.357 Vorkommen, 1000: 318) -- dafür wurde keine verlässliche Quelle
gefunden, ihre Bedeutung bleibt bewusst offen.

**Erste inhaltliche Befunde (Aufgabe 6):** Die mittlere Verspätung liegt auf allen acht
S-Linien der Stammstrecke bei 230-350s (Median 120-180s) -- am 1.1. also ähnlich über alle
Linien, keine auffällig schlechtere Linie. Nach Tageszeit zeigt sich ein deutlicher Peak
zwischen 8 und 12 Uhr (Mittelwert bis 1169s um 9 Uhr) sowie ein kleinerer um 20 Uhr (521s)
-- plausibel als Silvester-Nachwirkung/Sonderverkehr an einem Feiertag, aber OHNE
Werktagsvergleich nicht von "normalem" Vormittagsverkehr unterscheidbar. Entlang der
Strecke (West->Ost) liegt die mittlere Verspätung recht gleichmäßig zwischen 240s und 325s,
mit dem höchsten Wert an Hauptbahnhof (tief) und fallend Richtung Leuchtenbergring -- am
ehesten ein Hinweis darauf, dass sich Verspätungen im Tunnelkern stauen/sammeln, aber bei
nur einem (Feiertags-)Tag statistisch nicht belastbar. Die Korrelation der Verspätung
zwischen aufeinanderfolgenden Halten derselben Fahrt ist mit r=0,96 (n=8.770 Halt-Paare)
sehr hoch -- Verspätung an einem Halt ist ein extrem starker Prädiktor für die Verspätung
am nächsten Halt derselben Fahrt. Der Kartenplot (`stammstrecke_karte.png`) zeigt den
Tunnelverlauf klar als durchgehenden Bogen in den Koordinaten.

## Was überrascht hat

1. **Reine Koordinaten-Nearest-Neighbour-Zuordnung ist bei dieser Station zu riskant.**
   Für Hauptbahnhof (tief) liegen zwei stop_ids nur ~150m auseinander (8098263, alle
   Linien außer S7, ~14.000 Zeilen/933 Trips; 8098261, NUR S7, 580 Zeilen/114 Trips) --
   und die Wikipedia-Koordinate des Gesamtbahnhofs liegt dem NIEDRIG-Traffic-Kandidaten
   sogar geringfügig näher (142m) als dem richtigen (221m). Eine naive
   Distanz-Zuordnung hätte hier den falschen, fast bedeutungslosen Knoten gewählt. Erst
   der Abgleich mit der echten Fahrtreihenfolge hat das aufgedeckt.
2. **is_cancelled flackert.** Im Beispiel-Prognoseverlauf wechselt is_cancelled bei
   gleichbleibendem delay-Wert zwischen True und False hin und her (siehe
   `beispiel_prognose_verlauf.csv`, ursprüngliche Recherche). Das lässt sich aus den
   Daten allein nicht erklären -- vermutlich eine temporär gesetzte und wieder
   zurückgezogene IRIS-Meldung, aber das ist eine Vermutung, keine belegte Aussage.
3. **`trip_headsign` ist keine Text-Spalte, sondern ein stop_id-Verweis** (BIGINT,
   Werte wie 8000262 = München Ost). Ohne eine Stop-Namens-Tabelle bleibt das Fahrtziel
   nur als Zahl lesbar -- ein Punkt, an dem man leicht auf die falsche Fährte kommt,
   wenn man "headsign" für einen Klartext-Anzeigetext hält.
4. **line="4" bei category="S".** Eine Handvoll Zeilen (operator=800725, also derselbe
   Betreiber wie alle S-Bahnen) tragen als `line`-Wert die blanke Zahl "4" statt "S4".
   Vermutlich ein Normalisierungsfehler in der Quelle für einzelne frühe/späte Kurse,
   aber nicht weiter aufgeklärt.

## Die drei größten Fallstricke für die weitere Arbeit

1. **Ein Tag ist kein Werktag.** Fast jeder inhaltliche Befund oben (Verspätung nach
   Linie/Tageszeit/Position, Autokorrelation) basiert auf dem 1.1., einem Feiertag mit
   Sonderfahrplan und vermutlich atypischem Fahrgastverhalten. Für alles, was allgemein
   über den Stammstreckenbetrieb aussagen soll, muss mindestens ein normaler Dienstag
   bis Donnerstag außerhalb der Schulferien aus `data/raw/2025.tar` dazukommen -- am besten
   mehrere, um Tag-zu-Tag-Schwankung von echtem Muster zu trennen.
2. **Granularität zuerst entscheiden, dann filtern.** Wer naiv über alle 3,58 Mio. Zeilen
   aggregiert, mischt Prognose-Historie und Endzustand und zählt manche Halt-Ereignisse
   3-70x. Für Verspätungsfortpflanzung sollte praktisch immer zuerst auf `is_final=true`
   gefiltert werden (bzw. für Prognosequalität explizit die Zeitreihe über
   `update_timestamp` genutzt werden) -- sonst verzerrt die schiere Menge an
   Zwischen-Meldungen jede Statistik. Zusätzlich: `is_cancelled=false` und ggf.
   `pickup_drop_off_type != 'NOT_AVAILABLE'` filtern.
3. **stop_id ist nicht 1:1 mit "Station".** Hauptbahnhof (tief) existiert als zwei
   verschiedene stop_ids (8098263 und 8098261) für denselben physischen Ort, je nachdem
   welche Linie fährt. Wer pro Station aggregieren will, muss solche Aliase erst
   identifizieren und zusammenführen -- sonst fehlt z.B. der S7-Verkehr an Hauptbahnhof
   tief in jeder Analyse, die nur auf 8098263 filtert. Es ist nicht auszuschließen, dass
   es an anderen Stationen ähnliche, hier nicht gefundene Aliase gibt.

## Offene Fragen, die die Daten NICHT beantworten

- Bedeutung der message_codes 0 und 1000 (keine verlässliche Quelle gefunden).
- Warum is_cancelled bei gleichem delay-Wert wechselt.
- Ob line="4" ein Datenfehler oder eine bewusste Sonderkennzeichnung ist.
- Ob es außer dem Hauptbahnhof-Fall weitere stop_id-Aliase für dieselbe physische Station
  gibt -- dafür wurde keine systematische Prüfung über die gesamte Stammstrecke gemacht,
  nur der eine auffällige Fall wurde bemerkt.

## Dateien

- `explore.py` -- reproduzierbares Skript (`uv run python explore.py`)
- `outputs/muenchen_stationen.csv` -- alle 149 Münchner S-Bahn-Halte mit Zuordnung und
  Konfidenz-Einschätzung
- `outputs/message_codes_top20_stammstrecke.csv`
- `outputs/beispiel_prognose_verlauf.csv`
- `outputs/inventur_spalten.csv`
- `outputs/run_log.txt` -- vollständige Konsolenausgabe des Skriptlaufs
- `outputs/plots/` -- delay_by_position.png, delay_by_hour.png, delay_autocorrelation.png,
  stammstrecke_karte.png

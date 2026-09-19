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
- `src/data/windowing.py` -- Pipeline zur Aggregation in 5-min-Matrizen
- `tests/sanity_check.py` -- Validierung von Zeitindex, Imputation und Nacht-Lücken
- `data/processed/state_matrix_5min.parquet` -- Berechnete Verspätungsmatrix
- `data/processed/count_matrix_5min.parquet` -- Zugfrequenz pro 5-Minuten-Intervall

## Architektur-Befunde: Preprocessing & Imputation (Status: Pipeline implementiert)

Beim Überführen der asynchronen Roh-Updates in eine äquidistante 5-Minuten-Zustandsmatrix (`windowing.py`) wurden folgende kritische Datenrealitäten für das spätere ML-Training (VAR/XGBoost) gelöst:

* **Unhashable Spalten & Duplikate:** Die Rohdaten enthalten Listen/NumPy-Arrays (wie `message_codes`). Ein systemweites `drop_duplicates()` wirft hier sofort `TypeError` (unhashable type). Diese Spalten müssen konsequent abgeworfen werden, *bevor* Zeilen-Duplikate entfernt werden.
* **Der "stumme" Zeitsprung (Gruppierungs-Falle):** Pandas `groupby(pd.Grouper(freq='5min'))` lässt Intervalle, in denen systemweit kein Zug fährt (z. B. nachts zwischen 02:00 und 04:30 Uhr), im Index komplett weg. Für ein Vektorautoregressions-Modell (VAR) ist das tödlich, da es die Enden als benachbarte Zeitschritte (t und t+1) interpretiert. Der Index *muss* nachträglich per `reindex(pd.date_range(...))` lückenlos aufgespannt werden, um die Nachtlücken als leere Zeilen zu materialisieren.
* **Kontrollierte Imputation & Episoden:** Kurzfristige Lücken (bis 30 Min) werden via Forward-Fill (`ffill(limit=6)`) überbrückt, da sich der Streckenzustand (Verspätungen) in der Zwischenzeit physikalisch nicht in Luft auflöst. Bei echten Nachtpausen (Lücken > 30 Min) greift das Limit, und die Zeilen fallen auf `NaN`.
* **Architektur-Entscheidung:** Diese massiven `NaN`-Blöcke in der Nacht werden nicht mit Nullen aufgefüllt. Stattdessen dienen sie dem nachgelagerten ML-Skript als natürliche Sollbruchstellen, um den kontinuierlichen Datenstrom in unabhängige "Tages-Episoden" (Episodic Training) zu zerschneiden.

## Stufe 0 (Bauplan): Status abgeschlossen

### 0.1 Filterlauf

`src/data/filter_muenchen.py` liest `2025.tar` (19 GB, unkomprimiert) einmalig Tag für Tag und filtert auf die Münchner Stationsliste. Alle 365 Tage vorhanden, keiner fehlt (`outputs/filter_log.csv`). Ausgabe: 12 Monatsdateien unter `data/01_muenchen/` (1,4 GB, 143 Mio. Zeilen).

**Stationsliste -- zwei Fehlversuche, dann datengetrieben korrekt:**
1. Erster Versuch (`stops.parquet`, bundesweite DB-Referenztabelle, Name+Bbox-Filter): 278 Kandidaten -- viel zu viele, massenhaft Duplikate derselben physischen Station aus unterschiedlichen Datenquellen (z. B. "Baierbrunn", "Grafrath", "Starnberg" je doppelt/dreifach), Bbox zu großzügig.
2. Zweiter Versuch, methodisch besser (analog zur 12-Stammstrecken-Identifikation in `explore.py`): eine stop_id zählt nur, wenn dort 2025 mindestens eine echte S-Bahn-Fahrt hielt (`category='S'`, `operator='800725'`). Ergebnis: 141 Stationen -- deutlich sauberer, aber immer noch zu wenig. Ursache: die ERSTE Bbox (beim allerersten Rohdaten-Filterlauf) hatte die Obergrenze bei Längengrad 11,85 gesetzt und damit den kompletten Erdinger Außenast (Grafing, Ebersberg, Kirchseeon, Markt Schwaben, Erding, Altenerding, Aufhausen, Eglharting, Ottenhofen, St. Koloman -- alle bei Längengrad 11,86-11,97) schon beim 20GB-Rohdatenfilter abgeschnitten. Diese Stationen konnten datengetrieben nie wieder auftauchen, weil ihre Rohzeilen nie extrahiert wurden.
3. Bbox korrigiert (Längengrad-Obergrenze auf 12,05), kompletter 0.1-Filterlauf wiederholt, Stationsliste neu abgeleitet: **152 Stationen.**

**Validiert gegen Wikipedia** ("Liste der Stationen der S-Bahn München", 150 aktive Stationen laut Artikeltext, nach Abzug von 4 stillgelegten + 7 geplanten Stationen aus deren 161-Zeilen-Rohtabelle -- 161-11=150 exakt). Die verbleibende Differenz 152 vs. 150 ist vollständig erklärt: unsere Daten führen den Hauptbahnhof-Komplex als 4 stop_ids (unterschiedliche Bahnsteiggruppen: tief, Gl.27-36, Gl.5-10, allgemein), Wikipedia als 3 benannte Einträge ("Hauptbahnhof", "Hauptbahnhof – Holzkirchner Bf.", "Hauptbahnhof – Starnberger Bf.") -- eine reale Zählkonvention-Differenz am selben Gebäude, kein Datenfehler. Finale Liste: `outputs/muenchen_stationen_namen.csv`.

### 0.2 Ereignistabellen

`src/data/build_events.py` baut aus `data/01_muenchen/` zwei Tabellen (`data/02_events/final/`, `data/02_events/progression/`, zusammen 1,8 GB):

- Beide gefiltert auf `category='S'` (data/01_muenchen enthält an denselben Stationen auch Regional-/Fernverkehr), `is_betriebshalt=false`, `pickup_drop_off_type != 'NOT_AVAILABLE'`, `delay >= -1200`. `is_cancelled` bleibt als Spalte erhalten (nicht gefiltert).
- Negativ-delay-Histogramm (`outputs/plots/delay_histogram_negativ.png`) zeigt für die Münchner S-Bahn allein einen glatten Abfall statt der scharfen Doppel-Häufung aus dem bundesweiten Befund weiter oben (dort bis -87.180s durch Datumsüberträge) -- nur 790 von 124 Mio. Zeilen (0,0006%) liegen unter -1200s.
- `events_final`: 19.105.708 Zeilen, exakt eine je `(trip_id, stop_id, is_arrival)` (auf Duplikate geprüft).
- `events_progression`: ~124 Mio. Zeilen, voller Prognoseverlauf über `update_timestamp`.
- Neue Spalte `richtung` (Ost/West), aus der `stop_sequence`-Reihenfolge an den 12 Stammstreckenhalten pro Fahrt abgeleitet: 335.860 von 406.494 Fahrten mit Stammstreckenkontakt erhalten eine Richtung (167.980 West / 167.880 Ost -- fast perfekt symmetrisch), der Rest bleibt NaN (nur ein Kernhalt-Kontakt, Richtung nicht bestimmbar).
- Zeitstempel bleiben in UTC (Sommerzeit-Behandlung erst in 0.3).

### 0.3 Zustandsvektor

`src/data/build_state_vector.py` baut `data/03_state_vector/state_vector_5min.parquet`: 2.523.144 Zeilen (12 Stationen × 2 Richtungen × 105.120 Jahres-5-Minuten-Fenster, UTC-Raster). 9 Kennzahlen: `n_zuege`, `delay_mean`, `delay_median`, `delay_p90`, `anteil_ueber_180s`, `anteil_ueber_360s`, `n_ausfall`, `dwell_excess`, `headway_cv`.

- 72,8% der Zellen mit Beobachtung, Rest bewusst `NaN` (nicht 0) -- Nachtlücken. Umgesetzt, indem zuerst nur über beobachtete Ereignisse gruppiert und ERST DANACH auf das volle Gitter reindexiert wird, statt umgekehrt zu füllen.
- Feinere Unterscheidung funktioniert: 15.216 Fenster haben `n_zuege=0` UND `n_ausfall>0` (beobachtete Totalstörung, alle Fahrten im Fenster ausgefallen) -- getrennt von echten Datenlücken (beide NaN).
- `headway_cv` nur in ~60% der besetzten Zellen berechenbar (braucht ≥2 Züge im selben 5-Minuten-Fenster).

### 0.4 BEG-Validierung -- Abweichung dokumentiert, nicht aufgelöst

`src/data/validate_beg.py`. BEG-Referenz (Pressemitteilung "Jahreszahlen 2025", beg.bahnland-bayern.de): 87,9% Pünktlichkeit (Schwelle <6min, Ausfälle separat mit 8,4% ausgewiesen und aus der Pünktlichkeitsquote herausgerechnet).

Eigene Berechnung (Gesamtnetz, alle 152 Stationen, Ankünfte, `is_cancelled=false`, delay<360s): **82,70%** -- Abweichung **5,2 Prozentpunkte**, deutlich über der ~2pp-Toleranz aus dem Bauplan.

Geprüfte Alternativ-Methodiken, keine schließt die Lücke:
| Variante | Quote |
|---|---|
| Abfahrt statt Ankunft | 83,92% |
| Nur Starthalt je Fahrt | 94,15% (überschießt) |
| Nur Endhalt je Fahrt | 80,16% (öffnet die Lücke weiter) |
| Nur Stammstrecke statt Gesamtnetz | 76,21% (öffnet die Lücke weiter) |

Auch die eigene Ausfallquote (2,83%) liegt klar unter BEGs 8,4% -- fast dieselbe Differenz wie beim Pünktlichkeitswert (5,6pp vs. 5,2pp), was auf denselben Grundmechanismus hindeuten *könnte*. Geprüft und verworfen: "versteckte" Ausfälle, die als Extrem-Verspätung statt als `is_cancelled=true` geführt werden -- nur 2.881 Zeilen (0,03%) mit delay>3600s bei is_cancelled=false, zu wenige, um die Lücke zu erklären.

**Fazit, nicht künstlich weggerechnet:** die wahrscheinlichste Erklärung ist ein grundsätzlicher Unterschied in der Datenquelle -- BEGs Zahlen stammen vermutlich aus der offiziellen DB-InfraGO-Betriebsmesstechnik, dieses Projekt aus einer Drittanbieter-Echtzeit-API ("Bahn-Vorhersage"), die laut eigener Dokumentation bekannte Lücken hat. Eine systematische Unter-Erfassung vollständig ausgefallener Fahrten (die nie eine Echtzeitmeldung erzeugen und daher in dieser Pipeline gar nicht auftauchen) ist naheliegend, aber nicht belegt -- reine Vermutung.

**Abnahme Stufe 0 (Bauplan-Kriterien):** Zustandsvektor liegt als Datei vor ✓, BEG-Vergleich dokumentiert ✓ (mit ungeklärter Restabweichung), Liste fehlender Tage existiert ✓ (leer -- alle 365 Tage vorhanden).

## Stufe 1: Impulsantwort auf message_code 34 ("Defekt an einem Signal")

Auswertungsplan vorab schriftlich festgelegt: `outputs/auswertungsplan_stufe1.md`.
Abweichung vom Bauplan-Vorschlag ("häufigster Ursachencode"): der tatsächlich
häufigste Code (43, "Verspätung eines vorausfahrenden Zuges") ist ein
generischer Folge-Code ohne diskreten Beginn und daher für eine
Impulsantwort ungeeignet. Stattdessen Code 34 (Signalstörung, viert-
häufigster Code, 281.153 Vorkommen) -- passt methodisch (klarer, extern
verursachter Beginn) und deckt sich mit dem BEG-Jahresbericht 2025, der
Signal-/Sicherungstechnik als häufigste netzweite Verspätungsursache nennt.

**Pipeline** (`src/analysis/impulse_response.py`):
- 147.695 Code-34-Zeilen an den 12 Stammstreckenhalten (aus `events_final`).
- Clustering (15-Minuten-Lücke, Mehrheitsrichtung, Mindestgröße 3 Fahrten):
  5.478 Ereignisse.
- Bereinigung um Überlappungen im Fenster [-60,+240]min: 1.902 verworfen,
  **3.576 saubere Ereignisse** übrig.
- Kontrollfenster (k=10, gleicher Wochentagstyp/Uhrzeit ±30min/Datum
  ±6 Wochen/gleiche Station+Richtung/keine Überlappung): für 3.253 von 3.576
  Ereignissen gefunden (an 206 unterschiedlichen Tagen).
- Zielgröße `delay_mean` direkt aus `state_vector_5min.parquet`, Differenz
  Ereignis−Kontrollmittel, tageweises Bootstrap-Konfidenzband (1000 Ziehungen).

**Ergebnis** (`outputs/impulsantwort_code34.csv`, `outputs/plots/impulsantwort_code34.png`):
klare Kurve, komplett über der Nulllinie -- kein Abbruch nach 1.5 nötig.
Spitze von **+152s** (95%-CI 122-186s) genau bei t0, danach Abklingen über
gut zwei Stunden auf ein erhöhtes Plateau von **+30 bis +40s**, das auch nach
den vollen 240 Minuten nicht auf null zurückkehrt -- die Stammstrecke
erholt sich nicht innerhalb des Beobachtungsfensters vollständig von einer
Signalstörung.

**Auffälligkeit, offen dokumentiert statt versteckt:** schon bei t0-60min
liegt die Kurve bei ca. +75 bis +90s statt bei 0 -- also bereits VOR dem
offiziellen Ereignisbeginn erhöht. Zwei mögliche Erklärungen, keine davon
abschließend geprüft: (1) `update_timestamp` (Basis für t0) markiert den
Meldezeitpunkt, nicht den tatsächlichen Störungsbeginn, der demnach schon
etwas früher liegen dürfte (vom Bauplan selbst so erwartet, siehe 1.2);
(2) die Kontroll-Matching-Kriterien (Wochentagstyp, Uhrzeit, ±6 Wochen)
gleichen zwar Tagesrhythmus und Saison aus, aber nicht, dass Tage mit
Signalstörungen im Schnitt ohnehin etwas störungsanfälliger sein könnten.

### Nachdiagnose auf Anfrage: drei Probleme geprüft, Ursache eingegrenzt

Nach Durchsicht des Ergebnisses wurden drei Probleme benannt und der Reihe
nach geprüft: der Sockel vor t0, die Ereignisdefinition (Vermutung:
Mehrfachzählung über Stationen), und die Impuls-Annahme selbst bei
potenziell langen Signaldefekten. Code: `src/analysis/impulse_response_diagnostics.py`
(Diagnose) und `src/analysis/impulse_response_adjustments.py` (Anpassungen).
**Nichts am Original (`outputs/impulsantwort_code34.csv`,
`outputs/plots/impulsantwort_code34.png`) wurde verändert** -- alle
Varianten liegen als eigene Dateien daneben.

**Problem 1 -- Diagnose:**
- 1a: Stunde, Wochentagstyp, Station und Richtung sind zwischen Ereignis-
  und Kontrollfenstern durch die Matching-Konstruktion EXAKT identisch
  verteilt (Differenz 0,0pp überall) -- das ist NICHT die Ursache. Der
  Monat streut geringfügig (bis 4,9pp, z.B. November über-, September
  unterrepräsentiert bei den Ereignissen) -- ein möglicher kleiner Beitrag,
  aber zu klein für einen 80s-Effekt.
- 1b: Im Schnitt wurden 23,3 Kontrollkandidaten gefunden (Suche bei 30
  gedeckelt), Median 30 (Suche meist ausgeschöpft). 323 von 3576 sauberen
  Ereignissen (9,0%) hatten weniger als die geforderten k=10 und wurden
  komplett aus der Analyse ausgeschlossen -- nicht mit weniger Kontrollen
  gerechnet, nicht aufgefüllt.
- 1c: Kontrollfenster waren zu 82,2% mit irgendeinem anderen Störungscode
  (2-69, außer 34) belastet -- die Ereignisfenster selbst zu 99,2%. Beide
  Werte sind hoch, weil auf einem dicht befahrenen Netz IRGENDeine
  Störungsmeldung in einem 5-Stunden-Fenster praktisch die Norm ist, nicht
  die Ausnahme. Der Unterschied (99,2% vs. 82,2%) ist real, aber zu klein,
  um einen 80s-Sockel allein zu erklären.
- 1d: Die Sockelhöhe unterscheidet sich je Station (69,9s Hirschgarten bis
  127,1s Pasing; die beiden "Rand"-Stationen der Stammstrecke -- Pasing und
  Leuchtenbergring, wo Fahrten aus dem Zulaufnetz in den Tunnel ein-/
  austreten -- liegen am höchsten). Streuung pro Station ist aber sehr groß
  (std 200-340s), also ein Trend, keine scharfe Erklärung.

**Problem 1 -- Anpassung:**
- Baseline-Korrektur (`outputs/impulsantwort_code34_baseline_korrigiert.csv/png`):
  Sockel [-60,-15] = 80,7s abgezogen. Die korrigierte Kurve startet
  planmäßig bei 0, hat aber ein NEUES Problem: sie fällt ab ca. +40min
  unter null und bleibt bis +240min bei -40 bis -51s (CI schließt 0
  überwiegend aus). Eine reine additive Sockel-Korrektur ist damit das
  falsche Modell -- sie verschiebt das Problem nur, statt es zu lösen (die
  eigentliche Ursache ist strukturell, siehe Problem 3).
- Kontrolle Variante B (`outputs/impulsantwort_code34_kontrolle_variante_b.csv/png`,
  gegen ALLE Störungscodes 2-69 statt nur 34 abgegrenzt): nur 769 von 3576
  Ereignissen (21,5%) fanden noch 10 vollständig "saubere" Kontrollen --
  echte störungsfreie Fenster sind auf diesem Netz selten. Die resultierende
  Kurve liegt komplett auf einem viel höheren, unruhigeren Niveau (~200-320s
  statt ~30-150s), weil "komplett störungsfrei" ein untypisches, nicht ein
  repräsentatives Referenzniveau ist. Fazit: Variante B löst Problem 1
  NICHT -- sie tauscht eine Verzerrung gegen eine andere und verkleinert
  die Stichprobe drastisch.

**Problem 2 -- Diagnose:** 5.478 Einzel-Ereignisse (vor Überlappungs-
Bereinigung) fallen, über ALLE Stationen hinweg nach t0-Nähe (15min)
geclustert, in nur **930 zeitliche Cluster** -- eine Reduktion um 83%. Die
meisten Cluster betreffen mehrere Stationen gleichzeitig (Median >1,
Maximalfall 20 Einzel-Ereignisse über bis zu 12 Stationen in einem
Cluster). Die Vermutung war richtig: ein realer Signaldefekt erzeugt durch
das stationsweise Clustering im Schnitt ~4-5 "Ereignisse".

**Problem 2 -- Anpassung:** Auf Episoden-Ebene (Repräsentativ-Station = erste
Meldung, gleiche Überlappungs-Bereinigung wie im Original) bleiben 811
saubere Episoden, davon 795 mit genug Kontrollen (`outputs/impulsantwort_code34_episoden_ebene.csv/png`).
Die Kurve ist in Form und Größenordnung praktisch identisch zum Original
(Spitze 140,7s vs. 151,8s, gleicher Abklingverlauf, größere Bänder wegen
kleinerer Stichprobe). **Wichtige Korrektur für den Bericht:** die
eigentliche Zahl unabhängiger Störungen ist ~800, nicht 3.253 -- die
Mehrfachzählung hat die Kurvenform nicht sichtbar verzerrt, aber die
gemeldete Fallzahl war irreführend hoch.

**Problem 3 -- Diagnose:** Dauer je Ereignis (letzte Code-34-Meldung im
Cluster minus t0) ist stark rechtsschief: Median 18,2min, Mittel 34,0min,
75%-Quartil 42,1min, 90%-Quartil 88,1min, Maximum bis zur 6h-Suchgrenze.
Verteilung: 2.353 kurz (<30min, 66%), 874 mittel (30-90min, 24%),
349 lang (>90min, 10%).

**Problem 3 -- Anpassung** (`outputs/impulsantwort_code34_dauer_{kurz,mittel,lang}.csv`,
`outputs/plots/impulsantwort_code34_dauer_stratifiziert.png`): klarer,
sauberer Dosis-Wirkungs-Zusammenhang. Kurze Ereignisse zeigen eine
plausible Impulsform (Spitze ~122s bei t0, Abklingen nahe Sockel bis
~100-150min). Mittlere Ereignisse sind schon vor t0 höher (~80s) und
klingen langsamer ab (~45s Restniveau bei +240min). Lange Ereignisse sind
bereits bei t0-60min auf ~139s und bleiben bis weit über +150min stark
erhöht (>50s, teils >100s) -- kein Abklingen im beobachteten Fenster.

**Antwort auf 1.5/3c, ohne Beschönigung:** Der Sockel ist NICHT durch einen
einzelnen Fehler erklärt, sondern durch eine Mischung aus (a) einem
station-abhängigen Trend (1d), (b) einem harten strukturellen Befund: lange
Ereignisse (10% der Fälle) ziehen sowohl den Sockel als auch den
Abkling-Schwanz der gepoolten Kurve deutlich nach oben, weil sie selbst vor
t0 schon erhöht sind und sich im Beobachtungsfenster nicht erholen. Selbst
in der saubersten Teilgruppe (kurz) bleibt ein Rest-Sockel von ~66s bei
t0-60min -- das ist NICHT vollständig aufgeklärt und wird hier auch nicht
als aufgeklärt behauptet.

**Hält die Impuls-Annahme für Code 34?** Nur teilweise. Für kurze Ereignisse
(66% der Fälle) ist "Störung -> Spitze -> Abklingen auf Ausgangsniveau
innerhalb von ~2h" eine vertretbare Beschreibung. Für mittlere und
besonders lange Ereignisse (34% der Fälle, aber mit deutlich größerer
Amplitude) ist die Annahme eines abklingenden Impulses nicht haltbar -- das
sieht eher nach einem anhaltenden Störzustand aus, der sich nicht auf eine
"Halbwertszeit" reduzieren lässt. Eine einzelne gepoolte Impulsantwort-Kurve
für Code 34 (wie im ursprünglichen Ergebnis) vermischt damit zwei
unterschiedliche Dynamiken. Empfehlung für den weiteren Bericht: die nach
Dauer stratifizierten Kurven als Hauptergebnis für Code 34 zeigen, die
gepoolte Kurve explizit als Mischung kennzeichnen, und "Halbwertszeit" nur
für die kurze Teilgruppe verwenden, nicht für Code 34 insgesamt.

**Abweichung vom Auswertungsplan, dokumentiert statt rückwirkend
geändert:** der ursprüngliche Plan (`outputs/auswertungsplan_stufe1.md`)
sah eine einzelne Kurve für Code 34 auf Ereignis-Ebene vor. Die
Dauer-Stratifizierung und die Episoden-Ebene sind Erweiterungen dieser
Nachdiagnose, keine rückwirkende Änderung des ursprünglichen Plans oder
Ergebnisses -- beide bleiben unverändert bestehen.

## Störungsinventur: Bestandsaufnahme statt Modellierung

Auf Anfrage vor der Fortsetzung der Impulsantwort-Arbeit: eine reine
Inventur aller `message_codes` in `events_final`, um die Wahl des
Auslöse-Codes datengetrieben statt intuitiv zu treffen. Keine Modellierung,
keine Impulsantwort in diesem Abschnitt. Code: `src/analysis/stoerungsinventur.py`.
Primärer Scope: die 12 Stammstreckenhalte, Jahr 2025; Abschnitt 1
(Bestandsaufnahme) zusätzlich einmal für alle 152 Münchner Stationen zum
Vergleich (Rangfolge und relative Größenordnungen praktisch identisch, nur
absolute Zahlen ca. 2,5x höher). Abschnitte 2-5 laufen bewusst NUR auf der
Stammstrecke -- explizite Scope-Entscheidung, nicht auf alle 152 Stationen
ausgeweitet.

"Störungscode" = IRIS-Codes 2-69 (schließt den Platzhalter 1, die
undokumentierten Codes 0/1000/1001 und die Komfort-/Ausstattungshinweise
70-98 aus, siehe Docstring des Skripts für die Begründung).

### 1. Bestandsaufnahme (`stoerungsinventur_1_bestandsaufnahme_{stammstrecke,gesamtnetz}.csv`)

80 distinkte Codes insgesamt, davon 64 mit bekannter Bedeutung (IRIS-
Referenz derf/Travel-Status-DE-IRIS) und **16 klar als unbekannt markiert**
statt geraten (u.a. 0, 1000, 1001 -- vgl. bereits in der Tagesanalyse vom
1.1. aufgefallen). Häufigste Codes auf der Stammstrecke: 43 (935.676,
"Verspätung eines vorausfahrenden Zuges"), 0 (551.507, unbekannt), 34
(281.153, Signaldefekt), 85 (166.509, "Ein Wagen fehlt" -- kein
Störungscode im engeren Sinn), 48 (125.572, "Verspätung aus vorheriger
Fahrt"). Reihenfolge und Größenordnung sind am Gesamtnetz (152 Stationen)
praktisch identisch, nur mit ca. 2,5x höheren Absolutzahlen.

### 2. Episoden statt Einzelmeldungen (`stoerungsinventur_2_episoden_je_code.csv`)

Codes 43, 48, 44 haben die meisten Episoden (2.968 / 3.482 / 2.386), aber
auch die längsten und unregelmäßigsten Dauern (Code 43: Median 25,6min,
IQR 4,3-97,2min, Maximum 1.285min = 21h) -- ein weiterer Beleg dafür, dass
43/44/48 keine diskreten Ereignisse, sondern anhaltende Begleitzustände
sind. Signaldefekte (34) liegen bei 1.610 Episoden, Median 14,6min, IQR
0,5-36,1min -- die aus der vorherigen Impulsantwort-Diagnose bekannte
Rechtsschiefe.

### 3. Zeitliche Muster (`stoerungsinventur_3_*.csv`, `plots/stoerungsinventur_zeitmuster_top20.png`)

Deutliche HVZ-Konzentration bei einigen Codes (33 "Defekt Oberleitung"
61,0%, 55 "Techn. Defekt anderer Zug" 59,2%, 5 "Ärztliche Versorgung"
56,2% der Meldungen in der Hauptverkehrszeit), andere sind gleichmäßiger
verteilt (40 "Defektes Stellwerk" 35,8%, 8 "Notarzteinsatz" 38,4%).
Auffällige Einzelmuster in der Heatmap: Code 31 (Bauarbeiten) hat einen
extremen Ausschlag im November (ein einzelnes großes Bauprojekt, nicht
saisonal im engeren Sinn), Code 23 (Betriebsstabilisierung) konzentriert
sich stark auf Juni/Juli. Code 55 zeigt einen sehr scharfen Ausschlag um
6 Uhr morgens -- bei nur 418 Episoden möglicherweise ein Artefakt weniger
Tage statt ein echtes Muster; nicht weiter untersucht.

### 4. Überschneidungen -- der wichtigste Teil

**4a:** 76,9% aller 5-Minuten-Fenster im Jahr haben irgendwo auf der
Stammstrecke mindestens eine aktive Störungsmeldung (Codes 2-69). Je
Station zwischen 51,5% (Leuchtenbergring, östliches Streckenende) und
66,2% (Donnersbergerbrücke). Das bestätigt quantitativ, was in der
Impulsantwort-Diagnose schon als Kontaminationsrate auffiel: ein wirklich
"ruhiges" Fenster ist auf diesem Netz die Ausnahme, nicht die Regel.

**4b:** Verteilung gleichzeitig aktiver Codes: 23,1% der Fenster haben
keinen aktiven Störungscode, aber 44,6% haben 2 oder mehr GLEICHZEITIG
aktive verschiedene Codes, in seltenen Fällen bis zu 14 (`stoerungsinventur_4b_anzahl_gleichzeitig_aktiv.csv`).

**4c:** Kreuztabelle (`stoerungsinventur_4c_lift_matrix.csv`,
`_top_paare.csv`) und Folge-Analyse (`_folgecodes.csv`). Stärkster,
statistisch robuster Befund (großes n): **Code 44 ("Warten auf einen
entgegenkommenden Zug") wird in 66% von 25.887 Fällen innerhalb von 60min
von Code 48 ("Verspätung aus vorheriger Fahrt") gefolgt** -- ein klarer
Beleg für eine Folgekette 44→48. Weitere robuste Folge-Paare (n≥500):
33→43 (68%, n=1.875), 54→48 (71%, n=508). Das bestätigt unabhängig die
frühere Einschätzung, Code 43 (und jetzt auch 44, 48) als generische
Folge-/Symptomcodes zu behandeln, nicht als Auslöser.

**4d:** Isolationsraten (`stoerungsinventur_4d_isolation_je_code.csv`)
sind für ALLE Top-25-Codes extrem niedrig (0,0-9,5%, die meisten unter
2%) -- unter der strengen Definition "kein anderer Störungscode im
Fenster ±60min an derselben Station" gibt es auf der Stammstrecke praktisch
keine wirklich isolierten Episoden, für keinen Code. Das ist konsistent
mit 4a (77% Grundlast) und relativiert die "3.576 saubere Ereignisse" aus
der ursprünglichen Impulsantwort-Analyse: die dortige "Sauberkeit" bezog
sich nur auf Überlappungen MIT DEMSELBEN Code (34), nicht mit allen
anderen -- exakt die Einschränkung, die in der Nachdiagnose zu Problem 1
bereits aufgedeckt wurde. Relativ am wenigsten schlecht: Code 43 (9,5%,
aber selbst ein Folgecode, siehe 4c) und Code 7 (2,8%).

### 5. Eignung als Impuls -- Rangliste (`stoerungsinventur_5_eignung_ranking.csv`)

Bewertung der Top-20-Codes über fünf Kriterien (Fallzahl, kurze Dauer,
einheitliche Dauer, Isolation, kein Folgecode), jeweils als Rang
gemittelt -- transparent statt Black-Box-Score. Wichtiger Vorbehalt: die
Isolationsraten liegen für alle Codes im niedrigen einstelligen
Prozentbereich (siehe 4d) -- die Rangliste bewertet relative, nicht
absolute Eignung.

| Rang | Code | Bedeutung | Episoden | Dauer (Median) | Hauptargument |
|---|---|---|---|---|---|
| 1 | **36** | Technische Störung am Zug | 2.061 | 4,3min | Beste Gesamtbilanz: viertgrößte Fallzahl, kürzeste Median-Dauer der Top-Codes, moderate Isolation, kein dominanter Folgecode |
| 2 | **31** | Bauarbeiten | 855 | 10,0min | Große Fallzahl, plausibel extern/planbar verursacht, gute relative Isolation |
| 3 | 8 | Notarzteinsatz auf der Strecke | 117 | 5,0min | Sehr gute Isolation und kaum Folgecode, aber Fallzahl zu klein für robuste Bootstrap-Bänder |
| 4 | 23 | Betriebsstabilisierung | 567 | 2,0min | Kürzeste Dauer, aber Bedeutung selbst deutet auf eine Reaktion/Maßnahme hin, nicht auf eine Ursache -- Vorsicht geboten |
| 5 | 34 | Defekt an einem Signal | 1.610 | 14,6min | Bereits ausführlich analysiert (siehe oben); mittleres Ranking, weil Dauerverteilung breiter ist als bei 36 |
| ... | 43, 44, 45, 48 | (Folge-Codes) | groß | lang, unregelmäßig | Ganz unten -- durch 4c als Folgecodes identifiziert, für Impulsantwort ungeeignet |

**Empfehlung:** **Code 36 ("Technische Störung am Zug")** als primärer
Kandidat für die nächste Impulsantwort-Analyse -- beste Kombination aus
Fallzahl, kurzer Dauer und relativer Isolation, plausibel eine externe
Ursache (Fahrzeugdefekt) statt ein Folgezustand. **Code 31 (Bauarbeiten)**
als zweiter, methodisch interessanter Kandidat (planbare, ggf. sogar vorab
bekannte Start-/Endzeiten, was eine externe Validierung der
Ereignisdefinition erlauben könnte). Code 34 bleibt als bereits
durchgerechnetes Beispiel wertvoll, ist aber laut dieser Inventur nicht der
sauberste verfügbare Kandidat. Von den zahlenmäßig größten Codes (43, 44,
45, 48) wird ausdrücklich abgeraten -- sie sind Folgecodes, keine Auslöser.

**Nicht abschließend geklärt:** warum genau Codes 0 und 1000/1001 auftreten
(weiterhin ohne belastbare Quelle), und ob Code 55's HVZ-Ausschlag ein
echtes Muster oder ein Artefakt weniger Tage ist.

## QS-Prüfung Stufe 0: vier Kontrollen vor dem Übergang zu Stufe 1

Auf Anfrage vor der Fortsetzung: vier gezielte Prüfungen der Stufe-0-
Pipeline (Zustandsvektor + Ereignistabellen), rein diagnostisch, nichts
repariert. Ergebnis vorweg: **keine der vier Prüfungen hat einen Bug in der
Kern-Pipeline gefunden.** Zwei überraschende Nebenbefunde (Datenlücke am
18./19.10., minutengenaue delay-Quantisierung) wurden dabei aufgedeckt und
unten dokumentiert.

### 1. Bilanz-Abgleich (die härteste Prüfung)

`sum(n_zuege)` über den kompletten state_vector = 3.433.449, `sum(n_ausfall)`
= 113.363, zusammen **3.546.812**. Das entspricht exakt der Zahl der
Ankunfts-Zeilen in `events_final` an den 12 Stammstreckenhalten MIT
`richtung IS NOT NULL` (3.546.812) -- und die Aufteilung real/ausgefallen
stimmt ebenfalls exakt überein (3.433.449 / 113.363). **Innerhalb der
richtung-bekannten Population gibt es keinen einzigen verlorenen oder
doppelt gezählten Halt.**

Außerhalb dieser Population: **58.704 Halte (1,63% von 3.605.516
Ankünften)** fallen durch `richtung IS NULL` komplett aus dem
state_vector -- weder in n_zuege noch in n_ausfall gezählt. Das ist ein
stiller, aber jetzt exakt bezifferter Verlust.

Wichtige Korrektur zur ursprünglich befürchteten Größenordnung: die 70.634
"Fahrten ohne Richtung" aus der 0.2-Statistik sind eine **Fahrten**-Zahl,
keine **Halte**-Zahl. Geprüft: jede einzelne dieser 70.634 Fahrten berührt
**exakt eine** Stammstreckenstation (deshalb ist keine Richtung bestimmbar
-- man braucht mindestens zwei Stationen, um eine Reihenfolge abzuleiten).
Davon haben 58.704 an dieser einen Station eine Ankunftszeile, die übrigen
11.930 nur eine Abfahrt (voraussichtlich Fahrtbeginn an dieser Station).
Der tatsächliche Datenverlust auf Halte-Ebene ist damit **1,63%**, nicht
17%. Nicht weiter untersucht: ob diese einzel-Stationen-Kontakte
überwiegend Ein-/Ausfädler an den Streckenenden (Pasing/Leuchtenbergring)
sind oder gleichmäßig verteilt -- für die Größenordnung des Verlusts nicht
entscheidend.

### 2. Sommerzeit

Fenster- und n_zuege-Vergleich für 2025-03-30 (Frühjahr) und 2025-10-26
(Herbst) gegen die jeweils benachbarten Sonntage:

| Datum | Typ | Zeilen im state_vector (12 Stationen x 2 Richtungen) |
|---|---|---|
| 23.03. | normaler Sonntag | 6.912 (= 288 Fenster x 24) |
| **30.03.** | **Umstellung Frühjahr** | **6.624 (= 276 Fenster x 24 -- exakt 23h)** |
| 06.04. | normaler Sonntag | 6.912 |
| 19.10. | normaler Sonntag* | 6.912 |
| **26.10.** | **Umstellung Herbst** | **7.200 (= 300 Fenster x 24 -- exakt 25h)** |
| 02.11. | normaler Sonntag | 6.912 |

Beide Umstellungstage zeigen exakt die erwartete 23h- bzw. 25h-Fensterzahl.
**Kein fehlendes oder doppeltes Stunden-Fenster.** Die UTC-Raster-Strategie
aus 0.3 funktioniert wie geplant: die Sommerzeitumstellung entsteht
automatisch korrekt beim Umrechnen von UTC nach Europe/Berlin, ohne
Sonderbehandlung im Code.

**Nebenbefund, kein Sommerzeit-Bug:** 19.10. (*) fiel zunächst durch
ungewöhnlich niedrige `n_zuege`-Summe auf (255 statt ~8.000-10.000). Ursache
geprüft und gefunden: **9 von 12 Stammstreckenhalten fehlen am 18. UND
19.10.2025 fast vollständig** in den Rohdaten (`data/01_muenchen`) --
nur die beiden Tunnelportale Pasing/Donnersbergerbrücke und Ostbahnhof
haben an diesem Wochenende normale S-Bahn-Zeilen, der komplette
Tunnelkern (Hirschgarten bis Leuchtenbergring inkl. Hauptbahnhof tief) hat
praktisch null category='S'-Zeilen. Das Muster (Portale + ein Endpunkt
normal, gesamter Tunnel leer) sieht nach einer echten
Wochenend-Tunnelsperrung aus, nicht nach einem Pipeline-Fehler -- wird hier
aber nur als Verdacht dokumentiert, nicht als bestätigter Fakt (keine
externe Quelle geprüft). **Wichtig für die Praxis:** der ursprüngliche
Stufe-0.1-Vollständigkeitscheck (`outputs/filter_log.csv`) prüft nur
Gesamt-Zeilenzahlen pro Tag über alle 152 Stationen -- ein Ausfall einzelner
Stationen bei sonst normalem Netzverkehr wird dadurch NICHT erkannt. Das
ist eine Lücke im bisherigen Prüfverfahren, keine im state_vector selbst.

### 3. Mitternachtsgrenze

`outputs/plots/qc_mitternacht_uebergang.png`: mittlere `n_zuege` je
5-Minuten-Fenster, 22-02 Uhr Berlin-Zeit, gemittelt über alle
Dienstag-Donnerstage des Jahres. Der Übergang um Mitternacht ist **glatt**
-- kein Sprung, kein Einbruch, keine Verdopplung genau an der Grenze. Der
sichtbare Rückgang gegen 2 Uhr ist der erwartete, graduelle Übergang in die
Nachtruhe. Ergänzend bereits aus Stufe 0.2 bekannt: `events_final` hat
global null Duplikate auf `(trip_id, stop_id, is_arrival)` -- das schließt
Doppelzählung an Tagesdatei-Grenzen strukturell aus, unabhängig von der
Uhrzeit.

### 4. Eine Fahrt von Hand

Drei Fahrten manuell nachvollzogen (alle 12 Kernhalte, Rohzeile gegen
state_vector-Zelle):

- **Dienstag 8 Uhr** (trip_id 8348699748597743297, 10.6.2025): 9 Halte,
  Verspätung wächst von 0s (Pasing) auf 300s (Isartor) -- plausibler
  Verspätungsaufbau. Alle 9 Zellen im state_vector gefunden, `n_zuege`
  zwischen 1 und 3 (je nachdem wie viele andere Züge im selben 5-Min-Fenster
  waren), `delay_mean` in jedem Fall konsistent mit der eigenen Verspätung
  plus ggf. anderer Züge im Fenster.
- **Nacht 1 Uhr** (trip_id 8362856096841624644, 11.6.2025): 9 Halte,
  durchgehend pünktlich bis leicht verspätet (0-60s). Alle 9 Zellen
  gefunden und konsistent.
- **Störungstag** (trip_id 6408338996652544951, 10.11.2025, dem Tag mit den
  meisten Code-34-Meldungen im Jahr): alle 12 Kernhalte durchfahren, Verspätung
  konstant bei 32-38 Minuten (1920-2280s) über die gesamte Fahrt -- exakt
  das Bild eines "feststeckenden" Zuges nach einer Signalstörung. Alle 12
  Zellen im state_vector gefunden, `n_zuege` 1-4, `delay_mean` in jedem Fall
  plausibel (z.B. Leuchtenbergring: eigene Verspätung 2220s bei `n_zuege=4`
  und `delay_mean=1005s` -- die anderen 3 Züge im Fenster waren im Schnitt
  deutlich weniger verspätet, was den Mittelwert unter den Einzelwert
  dieser Fahrt drückt).

**Alle drei Fahrten in exakt den erwarteten Fenstern, keine Abweichung.**

## BEG-Abweichung: zwei von drei offenen Fragen jetzt (fast) geklärt

### a) Einheiten-Fehlpaarung: Halt vs. Fahrt -- **löst den Großteil der Ausfall-Diskrepanz**

Ausfallquote auf Fahrt-Ebene (eine Fahrt zählt als ausgefallen, wenn
mindestens ein Halt ausfällt), Gesamtnetz, alle Ankünfte 2025:
**8,16%** (36.634 von 449.075 Fahrten) -- gegenüber 2,83% auf Halt-Ebene.

BEGs veröffentlichte Ausfallquote 2025: **8,4%**. Differenz auf Fahrt-Ebene:
**0,24 Prozentpunkte** -- praktisch identisch, innerhalb jeder plausiblen
Fehlertoleranz. Die ursprünglich beunruhigende 5,6pp-Lücke war zu einem
großen Teil ein Einheitenproblem (Halt- statt Fahrt-Anteil), keine
inhaltliche Abweichung der Daten.

### b) Schwellenwert `<360` vs. `<=360` -- **löst einen Großteil der Pünktlichkeits-Diskrepanz, aber mit Vorbehalt**

`delay` ist in den Rohdaten **komplett minutengenau quantisiert** (nur
Vielfache von 60s kommen vor: 0, 60, 120, ... nie z.B. 45 oder 90). Bei
genau 360s (= 6 Minuten) liegen dadurch 398.963 Zeilen (4,30% aller
nicht-ausgefallenen Ankünfte) -- ein großer, aber im Kontext der
Nachbar-Buckets (300s: 543.591, 420s: 284.118) NICHT auffällig hoher Wert,
sondern Teil einer glatten, insgesamt abfallenden Verteilung.

Wegen der groben Quantisierung macht die Schwellen-Wahl trotzdem einen
großen Unterschied:
- `delay < 360` (strikt "unter 6 Minuten"): **82,70%**
- `delay <= 360` (inklusive "6 Minuten oder weniger"): **86,93%**
- Differenz: **4,24 Prozentpunkte**

Mit der inklusiven Variante schrumpft die Lücke zu BEGs 87,9% von 5,2pp auf
**0,97pp** -- damit fast vollständig erklärt.

**Aber, offen und nicht schöngerechnet:** die BEG-Pressemitteilung
formuliert explizit **"weniger als sechs Minuten"** (wörtlich geprüft,
nicht "bis zu sechs Minuten") -- das spricht für die STRIKTE Definition
(`<360`), die aber die GRÖSSERE Lücke ergibt (5,2pp), nicht die kleinere.
Es besteht also ein Widerspruch zwischen BEGs eigener Wortwahl und der
Variante, die numerisch näher an ihrem Ergebnis liegt. Eine mögliche,
NICHT bestätigte Erklärung: BEGs Rohmessung könnte auf ungerundeten
(sekundengenauen) Verspätungen beruhen und erst danach auf ganze Minuten
gerundet werden, mit einer anderen Rundungsregel als in diesem Datensatz --
das würde erklären, warum die inklusive Variante auf UNSEREN bereits
minutengerundeten Daten näher an BEGs (anders gerundetem) Ergebnis liegt.
Reine Hypothese, nicht geprüft.

### Quelle verifiziert

BEG-Pressemitteilung "BEG legt Jahreszahlen 2025 zu Pünktlichkeit und
Zugausfällen der S-Bahn München vor", beg.bahnland-bayern.de,
veröffentlicht 1. April 2026. Bezieht sich explizit und ausschließlich auf
die **S-Bahn München** (nicht auf bayernweite Regionalzüge -- dazu gibt es
eine separate, hier nicht verwendete Pressemitteilung desselben Tages) und
auf das **volle Kalenderjahr 2025**. Zitat: "Im Schnitt waren 87,9 Prozent
aller ihrer Züge pünktlich." Keine Seitenzahl vorhanden (Web-Pressemitteilung,
keine PDF-Publikation mit Seitenangabe).

### Verbleibender, ungeklärter Rest

Nach a) und b): Ausfallquote so gut wie vollständig erklärt (0,24pp Rest).
Pünktlichkeitsquote von 5,2pp auf ~1pp Rest reduziert, ABER mit dem oben
genannten Wortlaut-Widerspruch, der nicht sauber aufgelöst ist. Dieser
letzte ~1pp-Rest (bei strikter Lesart weiterhin 5,2pp) bleibt offen und wird
hier nicht als geschlossen behauptet.

## Drei unabhängige Nachdiagnosen

Rein diagnostisch, keine neue Modellierung, nichts an bestehenden Dateien
überschrieben.

### 1. Sommerzeit im Kontroll-Matching -- Verdacht bestätigt, Effekt klein

`src/analysis/qc_sommerzeit_matching.py`. Die Original-Ereignisse und
-Kontrollen wurden mit demselben Seed exakt rekonstruiert (3.253 Ereignisse,
stimmt mit dem Original überein). Von 32.530 Ereignis-Kontroll-Paaren
überspannen **3.451 (10,6%)** tatsächlich eine Sommerzeitumstellung
(unterschiedlicher Berlin-UTC-Offset zwischen Ereignis- und
Kontroll-Zeitpunkt).

Diese Paare verhalten sich messbar anders: mittlere paarweise
Sockeldifferenz [-60,-15] bei Paaren OHNE Umstellung 90,7s, bei Paaren MIT
Umstellung nur 53,9s (`outputs/qc_sommerzeit_paare.csv`,
`outputs/qc_sommerzeit_sockel_vergleich.csv`). Der Verdacht ist damit
bestätigt: das UTC-Uhrzeit-Matching führt bei einer Minderheit der Paare zu
einem echten, aber nicht dominanten Effekt.

Lokalzeit-Variante gerechnet (`outputs/impulsantwort_code34_kontrolle_lokalzeit.csv`,
Original `outputs/impulsantwort_code34.csv` unverändert): Sockel [-60,-15]
sinkt von 80,7s (UTC-Matching) auf **77,0s (Lokalzeit-Matching)** -- eine
Reduktion um 3,7s bzw. **4,6% des Sockels**. Die Umstellung auf Lokalzeit
ist methodisch korrekter und sollte für künftige Läufe übernommen werden,
klärt aber den ganz überwiegenden Teil (>95%) des ursprünglichen
Sockel-Rätsels aus der Stufe-1-Nachdiagnose NICHT auf -- die dort
identifizierte Dauer-Abhängigkeit (kurze vs. lange Ereignisse) bleibt die
mit Abstand größere Erklärung.

### 2. BEG-Lücke über Halt- vs. Fahrt-Gewichtung -- Hypothese widerlegt

`outputs/qc_beg_variante_{a,b,c}_*.csv`. Alle drei Varianten gegen 87,9%
BEG-Referenz und gegen die ursprüngliche Alle-Halte-Quote (82,70%,
Lücke 5,20pp):

| Variante | Quote | Lücke zu BEG |
|---|---|---|
| a) Zufälliger Halt/Fahrt, Gesamtnetz (100 Ziehungen, Seed 0-99) | 83,91% (±0,04%) | 3,99pp |
| b) Mittlerer Halt/Fahrt (nach stop_sequence) | 80,24% | 7,66pp |
| c) Zufälliger Halt/Fahrt, nur Stammstrecke (100 Ziehungen) | 77,74% (±0,03%) | 10,16pp |
| Referenz: alle Halte (Original) | 82,70% | 5,20pp |

**Die Hypothese "BEG misst einmal pro Fahrt, nicht pro Halt" hält der
Prüfung nicht stand.** Nur Variante a) verbessert die Lücke leicht (5,20pp
auf 3,99pp), schließt sie aber nicht. Die Varianten b) und c) sind sogar
DEUTLICH schlechter als die ursprüngliche Alle-Halte-Rechnung -- eine
Ein-Halt-pro-Fahrt-Gewichtung ist damit klar keine Erklärung für die
verbleibende Pünktlichkeits-Lücke, unabhängig davon wie der eine Halt
gewählt wird. Die Streuung über 100 Ziehungen ist in allen Fällen minimal
(<0,05 Prozentpunkte) -- die Wahl des Zufalls-Seeds ist nicht die Ursache
der Unterschiede zwischen den Varianten, nur die Auswahlstrategie selbst
(zufällig/mittig/Stammstrecke) macht den Unterschied.

### 3. Code 0 empirisch charakterisiert -- klares Bild: Folge-/Statuscode, kein Auslöser

`src/analysis/qc_code0_charakterisierung.py`. 551.507 Zeilen, 47.973
Episoden-Cluster an der Stammstrecke (Median-Dauer 15,6min, Q25-Q75
0,8-36,0min, Maximum 957,7min -- Größenordnung ähnlich anderen Codes,
für sich genommen nicht auffällig).

**Folge-Analyse** (`outputs/qc_code0_folgeanalyse.csv`), der aussagekräftigste
Befund: Code 0 startet in **51,9%** der Fälle innerhalb von 60 Minuten NACH
einer Episode eines anderen Störungscodes (Mittel über alle Codes mit
≥50 Episoden) -- u.a. 86,4% nach Code 63 (Technische Untersuchung am Zug),
70,6% nach Code 68 (Hohes Fahrgastaufkommen), 56,6% nach Code 7 (Unbefugte
Personen). Umgekehrt folgen andere Codes nur in **6,0%** der Fälle auf eine
Code-0-Episode. Diese **fast 9-fache Asymmetrie** ist ein starkes,
konsistentes Muster: Code 0 tritt weit überwiegend als Folge anderer
Ereignisse auf, nicht als deren Auslöser.

**Verspätung** (`outputs/qc_code0_verspaetung_vergleich.csv`): Halte MIT
Code 0 haben eine mittlere Verspätung von **461,2s** (Median 300s,
n=215.343), Halte OHNE jeden Störungscode nur **215,2s** (Median 180s,
n=118.504) -- gut doppelt so hoch. Das passt zum Bild eines Codes, der
auftritt, NACHDEM sich bereits eine spürbare Verspätung aufgebaut hat,
nicht als deren erste Ursache.

**Lift** (`outputs/qc_code0_lift.csv`): schwache bis moderate
Ko-Vorkommen-Überhöhung mit witterungs- und fahrzeugbezogenen Codes (11
Unwetter lift=2,93, 60 witterungsbedingt verminderte Geschwindigkeit
lift=2,84, 68 Hohes Fahrgastaufkommen lift=2,57, 18 umgestürzter Baum
lift=2,34) -- die absoluten Fallzahlen sind hier aber klein (20-260
gemeinsame Fenster), das ist ein Hinweis, kein belastbarer Beweis für einen
thematischen Zusammenhang.

**Zeitmuster** (`outputs/plots/qc_code0_zeitmuster.png`,
`outputs/qc_code0_{stunde,wochentag,monat}.csv`): Tagesverlauf mit
Vormittags-/Nachmittagsspitze (10 und 15 Uhr), deutlich seltener an
Wochenenden -- unauffällig, ähnlich anderen Störungscodes. **Auffällig,
nicht weiter untersucht:** eine starke zeitliche Konzentration auf das
vierte Quartal -- Oktober/November/Dezember zusammen tragen ca. 77% aller
Code-0-Vorkommen des Jahres, während die Monate Januar bis September
zusammen nur ca. 23% beisteuern. Mögliche Erklärungen (z.B. eine im Jahr
2025 neu eingeführte oder geänderte Codierung) wurden NICHT geprüft --
reine Beobachtung.

**Antwort auf die Ausgangsfrage, ohne die Bedeutung zu erraten:** Code 0
verhält sich nach allen vier unabhängigen Kriterien (Folge-Rate,
Verspätungsniveau, und konsistent über verschiedene Vorläufer-Codes hinweg)
wie ein **Folge-/Statuscode**, nicht wie ein Auslöser. Er tritt auf,
nachdem andere, benannte Störungen bereits begonnen haben und wenn bereits
überdurchschnittliche Verspätung vorliegt -- am ehesten vereinbar mit einer
Art generischem "Verspätungssituation dauert an"-Statusvermerk oder einer
Sammelkategorie für nicht anderweitig codierte Nachfolgemeldungen. Für die
Auswahl eines Impulsantwort-Auslösers (Stufe 1) bestätigt das die bereits
in der Störungsinventur getroffene Einschätzung: Code 0 ist wie 43/44/48
für diesen Zweck ungeeignet.

## Nachdiagnose: Rohkurven, Schwankung und Robustheit der Code-34-Impulsantwort

Rein diagnostisch, keine Methodenänderung, nichts überschrieben. Basis:
identische Rekonstruktion der 3.253-Ereignisse-Analyse (gleicher Seed,
Original-Fenster [-60,+240]). Skripte: `src/analysis/qc_impulsantwort_schwankung.py`
(Teile 1,2,4,5), `src/analysis/qc_impuls_kurzfenster.py` (Teil 3).

### 1. Rohkurven ohne Differenz

`outputs/qc_impuls_1_rohkurven.csv`, `outputs/plots/qc_impuls_1_rohkurven.png`.
Das **Kontrollfenster ist flach und stabil** (238-260s über das gesamte
Fenster, keine erkennbare Struktur um t0). Das **Ereignisfenster hat die
gesamte Form**: 314-334s in der Vorlaufzeit, Spitze bei 397s exakt bei t0,
Abklingen auf ein Plateau von 275-295s. Fast die komplette Wackligkeit der
Differenzkurve stammt aus dem Ereignisfenster, nicht aus der Kontrolle --
die Kontrollkurve trägt kaum Rauschen bei. Beide Kurven liegen aber auf
einem hohen "Sockel" weit über null (Kontrolle ~245s, Ereignis ~280-320s
abseits des Peaks) -- das ist die bereits bekannte, allgemein hohe
Grundverspätung des Netzes, nicht spezifisch für Code 34.

### 2. Woher kommt die Schwankung?

`outputs/qc_impuls_2_fallzahl_und_streuung.csv`, zwei Plots.

**2a** Fallzahl je Zeitpunkt: zwischen 2.517 und 2.878 von 3.253 möglichen
(77-88%), **kein dramatischer Einbruch** an einer bestimmten Stelle --
die Fallzahl nimmt zum Fensterende hin leicht ab (mehr Nachtstunden-NaNs
bei den spät liegenden Offsets), aber ohne Sprung.

**2b** Code-Stelle: `mean_curve = diff_df.mean(axis=0, skipna=True)`
(`impulse_response.py`, Zeile 247) und identisch beim Bootstrap (Zeile 259).
NaN wird also **je Zeitpunkt einzeln übersprungen**, nicht das ganze
Ereignis verworfen -- ein Ereignis mit NaN bei t=+120min trägt trotzdem an
allen anderen Zeitpunkten bei, an denen es einen Wert hat.

**2c, die eigentliche Antwort:** Std der Einzeldifferenzen je Zeitpunkt
liegt bei 240-478s (Median 276s) -- das ist **enorm groß** im Vergleich zu
den beobachteten Differenzen selbst (30-150s). Der Standardfehler des
Mittelwerts (std/√n) liegt dadurch bei 4,6-9,2s. Zum Vergleich: die
tatsächliche Schwankung der gemittelten Kurve von einem 5-Minuten-Punkt zum
nächsten beträgt im Mittel 7,6s -- **das liegt exakt in derselben
Größenordnung wie der Standardfehler.** Damit ist die sichtbare
"Zackigkeit" der Kurve zu einem großen Teil ganz gewöhnliches
Stichproben-Rauschen bei enorm variablen Einzelverläufen, kein verstecktes
Artefakt. 3.253 Ereignisse reichen für die MITTLERE Kurve (Standardfehler
klein), aber die EINZELNEN Verläufe streuen so stark, dass Punkt-zu-Punkt-
Wackeln der Mittelwertkurve normal und erwartbar ist.

**2d** Autokorrelation: die Rohkurve selbst ist stark und glatt
autokorreliert (0,94 bei 5min, 0,87 bei 10min, 0,74 bei 20min) -- aber das
ist überwiegend der Abkling-Trend, keine Periodizität (`outputs/qc_impuls_2d_autokorrelation.csv`).
Nach Entfernen des Trends (gleitendes Mittel, Fenster 25min) zeigt das
**Residuum** bei Lag 10min einen auffälligen Wert von **-0,35**, bei Lag
20min -0,16 (`outputs/qc_impuls_2d_autokorrelation_residuen.csv`) -- das ist
KEINE positive, verstärkende Periodizität im Sinne eines wiederkehrenden
Peaks alle 10/20 Minuten, sondern eher ein alternierendes Muster. Wichtiger
Vorbehalt, nicht verschwiegen: ein gleitendes Mittel als Entrendungsmethode
erzeugt selbst mechanisch eine gewisse negative Autokorrelation bei kurzen
Lags, auch bei reinem Rauschen. Der Befund ist damit ein **Hinweis, kein
Beweis** für eine Fahrplantakt-Struktur -- nicht abschließend geklärt.

### 3. Fenster auf [-60,+60] verkürzt

`outputs/qc_impuls_3_kurzfenster_60min.csv/png`. Von 5.478 Ereignissen
werden bei diesem kürzeren Fenster nur noch **914 wegen Überlappung
verworfen** (statt 1.902 bei [-60,+240]) -- **4.564 saubere Ereignisse**
bleiben übrig (statt 3.576), davon **4.404 mit ausreichend Kontrollen**
(statt 3.253).

Ergebnis: **Spitze bei t0 = 149,0s**, praktisch identisch zum
Original-Fenster (151,8s). **Sockel [-60,-15] liegt bei 67-88s**, ebenfalls
praktisch identisch zum Original (~80s). Die deutlich größere Stichprobe
bestätigt Peak und Sockel als robuste Eigenschaften der Daten -- sie sind
kein Artefakt der Fensterlänge oder der dadurch bedingten Ereignisauswahl.

### 4. Terzile nach Peak-Höhe

**Caveat vorab, wie gefordert:** die Schichtung nach der beobachteten Höhe
bei t0 erzeugt selbst einen Gipfelunterschied durch Regression zur Mitte
-- Ereignisse im "niedrig"-Terzil haben per Definition (auch durch reines
Rauschen) im Mittel einen niedrigeren Peak, unabhängig vom "wahren" Effekt.
Aussagekräftig ist die FORM, nicht die Höhe.

`outputs/qc_impuls_4_terzile_metadaten.csv`, `_je_station.csv`, Plot.
Terzile: niedrig (n=960, Peak -135s, teils negativ), mittel (n=959, Peak
53s), hoch (n=959, Peak 538s, bis 3.594s). Formunterschied klar sichtbar:
"hoch" klingt über ~60-100min auf ein Plateau von 60-90s ab, "mittel"
bleibt bei 20-40s, "niedrig" pendelt um null. **Auffällig: selbst das
niedrig-Terzil hat vor t0 einen positiven Sockel (~35-55s bei -60 bis
-20min)** -- der Sockel ist also nicht nur ein Artefakt der extremen,
langen Ereignisse, sondern in irgendeiner Form über das gesamte Spektrum
vorhanden.

Korrelation mit Merkmalen: klar monoton mit **Dauer** (28,4 -> 33,9 ->
41,5min) und **Zahl betroffener Fahrten** (13,2 -> 14,0 -> 18,2) je Terzil
-- deckt sich mit der Dauer-Stratifizierung aus der Stufe-1-Nachdiagnose.
KEINE erkennbare Korrelation mit Tagesstunde (12,6/12,4/12,1) oder Station
(Anteile pro Station variieren zwischen den Terzilen nur um 1-2
Prozentpunkte, keine Station dominiert ein Terzil).

### 5. Was ist in den Kontrollfenstern?

**5a** `outputs/qc_impuls_5a_codes_in_fenstern.csv`: für JEDEN der Top-15
anderen Störungscodes ist der Anteil betroffener Fenster im
**Ereignisfenster systematisch höher** als im Kontrollfenster (z.B. Code 43
in 32,0% der Ereignisfenster vs. 25,9% der Kontrollfenster; Code 64 6,3%
vs. 3,4%; Code 48 5,2% vs. 3,5%). Das ist ein konsistentes, durchgängiges
Muster, keine Einzelfälle -- Ereignis-Zeiträume sind generell "unruhiger"
als vergleichbare Kontroll-Zeiträume, nicht nur wegen Code 34 selbst.

**5b** Minute-innerhalb-der-Stunde: Ereignis (Mittel 23,2, Std 15,0) und
Kontrolle (Mittel 23,2, Std 15,0) sind **exakt identisch** --
erwartungsgemäß, weil Kontrollen per Konstruktion dieselbe Minute-des-Tages
wie ihr Ereignis erhalten. Kein Verzerrungsgrund hier, aber auch keine
unabhängige Bestätigung -- das Ergebnis ist tautologisch durch die
Matching-Regel vorgegeben, nicht empirisch neu geprüft.

**5c** Nur **99 von 32.530 Kontrollfenstern (0,30%)** fallen in den
Tunnelsperrungs-Zeitraum 18.-19.10.2025 -- vernachlässigbar, keine
relevante Verzerrungsquelle für die Gesamtkurve.

### Antwort auf die Ausgangsfrage

**Woher kommt die Schwankung?** Größtenteils gewöhnliches
Stichproben-Rauschen: die Einzelverläufe streuen enorm (Std 240-480s), der
daraus resultierende Standardfehler der Mittelwertkurve (~5-9s) erklärt die
beobachtete Punkt-zu-Punkt-Wackeligkeit (~7,6s) fast vollständig. Ein
schwacher, methodisch nicht eindeutiger Hinweis auf eine Taktstruktur bei
10 Minuten besteht (negative Autokorrelation der entrendeten Residuen),
ist aber nicht als bewiesen zu behandeln.

**Woher kommt der Sockel?** Nicht vollständig neu erklärt gegenüber der
vorherigen Nachdiagnose, aber zusätzlich erhärtet: (a) er ist robust
gegenüber der Fensterlänge (identisch bei [-60,+60] und [-60,+240]), (b) er
ist nicht nur ein Artefakt der langen Ereignisse -- selbst das
niedrigste Peak-Terzil zeigt einen positiven Sockel, (c) Ereignisfenster
sind generell mit mehr anderen Störungscodes belastet als Kontrollfenster
-- ein generelles "Ereigniszeiten sind unruhiger"-Muster, keine einzelne
Ursache.

**Verstehen wir jetzt, woher Schwankung und Sockel kommen?** Die Schwankung:
größtenteils ja -- sie ist mit der beobachteten Varianz und Fallzahl
statistisch erwartbar, kein Hinweis auf einen Fehler. Der Sockel: teilweise
-- er ist jetzt als robust, breit angelegtes und nicht auf einzelne
Extremereignisse zurückführbares Phänomen bestätigt, aber die letzte Ursache
(warum Code-34-Zeiträume systematisch "unruhiger" sind als vergleichbare
Kontrollzeiträume) bleibt offen. Was fehlt: eine Erklärung, WARUM sich
Code-34-Ereignisse bevorzugt in bereits angespannten Netzzuständen
ereignen -- das würde eine Untersuchung auf Ursache-Ebene erfordern
(z.B. Zusammenhang mit Zugdichte oder vorlaufender Netzlast), die hier
nicht Teil des Auftrags war.

## Impulsantwort für seltenere, gravierende Codes (5, 8, 2) vs. Code 34

Codeauswahl aus der Störungsinventur: **Code 5** ("Ärztliche Versorgung
eines Fahrgastes", 522 Episoden lt. Inventur -- größere Fallzahl, daher
Hauptfall), **Code 8** ("Notarzteinsatz auf der Strecke", 117 Episoden --
Vergleich), **Code 2** ("Polizeieinsatz", 603 Episoden -- der angefragte
Personenschaden-/Polizeicode, zusätzlich). Gleiche Grundpipeline wie Code
34, Fenster [-60,+120] statt [-60,+240] (kleinere Fallzahl je Code). Skripte:
`src/analysis/impulse_response_neue_codes.py`, `src/analysis/run_impuls_neue_codes.py`.
Nach Ereignis-Clustering (identische 15-Minuten-Logik wie bei Code 34, pro
Station, nicht über Stationen hinweg zusammengefasst -- deshalb höhere
Rohzahlen als in der Inventur): Code 5: 1.324 Ereignisse -> 1.175 sauber
-> **1.175 mit Kontrollen**. Code 8: 437 -> 352 -> **352**. Code 2: 1.611
-> 1.361 -> **1.349**.

### Artefakt-Tests zuerst (vor jeder Interpretation)

**Test 1 -- Terzile nach Wert bei -35min** (`outputs/plots/qc_neucode_5_4_terzil_m35.png`,
Code 8 analog): erzeugt erwartungsgemäß einen mechanischen Ausschlag GENAU
am Sortierpunkt -35min (das ist per Konstruktion so, kein Fehler). Der
entscheidende Befund: **die drei Terzil-Kurven laufen bei t0 wieder
zusammen** (alle drei bei ~220-250s) statt einen neuen, mitwandernden
Ausschlag zu zeigen. Das spricht GEGEN die Befürchtung, der Peak bei t0 sei
nur ein Artefakt der Sortierung nach Peak-Höhe -- ein Regression-zur-Mitte-
Artefakt würde sich an jedem beliebigen Sortierpunkt zeigen und dort
bleiben, nicht bei einem ANDEREN Punkt (t0) wieder verschwinden.
Nebenbefund: die "hoch bei -35min"-Gruppe bleibt auch nach der Konvergenz
im Median höher (Plateau ~50-80s ab +50min statt ~0-20s) -- ein echter,
nicht-artefaktueller Zusammenhang zwischen früher Vorbelastung und
späterer Erholungsdauer.

**Test 2 -- Zufällige Dreiteilung** (`outputs/plots/qc_neucode_5_5_zufallsteilung.png`):
alle drei Zufallsgruppen sind **deckungsgleich** über das gesamte Fenster,
inklusive des gemeinsamen Peaks bei t0 (~230-300s je nach Gruppe, im Rahmen
von Stichprobenrauschen bei n~380-405). Kein Sortierkriterium -> keine
Gruppenunterschiede. Der Null-Test besteht.

**Beide Artefakt-Tests bestehen.** Die nachfolgende Interpretation des
Peaks bei t0 ist damit nicht durch das bei Code 34 gefundene
Sortierartefakt-Risiko in Frage gestellt.

### Ergebnisse Hauptfall: Code 5

`outputs/impulsantwort_code5.csv`, `outputs/plots/qc_neucode_5_1_impulsantwort.png`
(mit Fallzahl-Subplot) und `_2_rohkurven.png`. **Deutlich sauberere Kurve
als Code 34:**

| Kennzahl | Code 34 (Original) | Code 5 |
|---|---|---|
| Spitze bei t0 | 151,8s | **254,5s** (Max bei +5min: 263s) |
| Sockel [-60,-15] | ~80s | **~40s** |
| Rückkehr Richtung 0 | nicht vollständig (30-40s Rest bei +240) | **fast vollständig** (~20s ab +50min, CI berührt 0) |
| Kontrollfenster-Rauschen | gering | **noch geringer** (245-266s, sehr eng) |

Die Rohkurven zeigen: das Kontrollfenster ist noch flacher/ruhiger als bei
Code 34, das Ereignisfenster liefert den kompletten, sauberen Ausschlag
(303s -> 522s -> zurück auf ~265-300s).

**Verteilung der Einzelwerte bei t0** (`outputs/plots/qc_neucode_5_11_histogramm.png`,
n=956, nach NaN-Ausschluss): rechtsschief, Median 138s deutlich unter dem
Mittel 252s (Std 400s, Spannweite -960s bis +2.565s) -- die im Mittelwert
sichtbare Spitze wird von einer Minderheit sehr stark betroffener
Einzelfälle mitgetragen, der "typische" (Median-)Fall ist deutlich
moderater als der Mittelwert suggeriert.

**Terzile nach Ereignisdauer/Fahrtenzahl** (`outputs/plots/qc_neucode_5_7_terzil_dauer.png`,
`_6_terzil_fahrten.png`): dieselbe Dosis-Wirkungs-Beziehung wie bei Code 34
-- längere/größere Ereignisse zeigen höhere Peaks und langsameres Abklingen.
**HVZ vs. Rest** (`_8_hvz.png`): ähnliche Form, HVZ-Ereignisse mit etwas
höherer Spitze, kein grundlegend anderer Verlauf. **Je Station**
(`_9_je_station.png`, 12 Panels): Grundform an allen 12 Stationen ähnlich,
Fallzahl je Panel naturgemäß klein (Median ~90-100 Ereignisse je Station),
keine Station mit qualitativ abweichendem Muster.

### Vergleich: Code 8 (Notarzteinsatz)

`outputs/impulsantwort_code8.csv`, `outputs/plots/qc_neucode_8_1_impulsantwort.png`.
Mit nur 352 Ereignissen an 31 Tagen ist die Kurve deutlich verrauschter
(Konfidenzband bis zu 0-500s breit). **Auffälliger Formunterschied:** kein
scharfer Peak bei t0, sondern ein **verzögerter, langsamerer Anstieg** auf
ein Maximum von ~150-200s erst bei +15 bis +20min. Plausible, nicht
bewiesene Erklärung: "Notarzteinsatz AUF DER STRECKE" (also nicht am
Bahnsteig) dürfte einen längeren Vorlauf bis zum tatsächlichen Halt und zur
sichtbaren Verspätung haben als "Ärztliche Versorgung eines Fahrgastes"
(vermutlich meist am oder nahe einem Bahnsteig). Wegen der kleinen
Fallzahl und breiten Konfidenzintervalle ist diese Formaussage mit
Vorsicht zu behandeln.

### Zusätzlich: Code 2 (Polizeieinsatz)

`outputs/impulsantwort_code2.csv`. 1.349 Ereignisse, 174 Tage -- ähnlich
sauber wie Code 5: scharfer Peak bei t0 (212s), niedriger Sockel (~55-78s),
schnelles Abklingen auf ~15-20s ab +50-60min.

### Direktvergleich (Plot 10)

`outputs/plots/qc_neucode_10_vergleich_code34.png`. Alle vier Kurven in
einem Bild (Code 34 auf 0-120min beschnitten für Vergleichbarkeit):

- **Code 5 und Code 2** zeigen die höchsten, schärfsten Spitzen (254s bzw.
  212s) bei gleichzeitig NIEDRIGEREM Vorlauf-Sockel als Code 34 (~40-55s
  vs. ~80s) und einer deutlich vollständigeren Rückkehr Richtung null.
- **Code 34** hat die niedrigste Spitze (152s), aber den höchsten Sockel
  und das unvollständigste Abklingen (Plateau ~30-40s bleibt bestehen).
- **Code 8** fällt aus dem Muster: niedrigere, verzögerte, sehr breit
  gestreute Antwort -- am ehesten der kleinen Fallzahl und/oder einer
  echten Vorlaufzeit-Besonderheit zuzuschreiben, nicht sicher unterscheidbar.

### Antwort auf die Ausgangsfrage

**Sieht der gravierende Code anders aus als Code 34?** Ja, für Code 5 und
Code 2 eindeutig: **höhere Amplitude, niedrigerer Vorlauf-Sockel, sauberere
und vollständigere Erholung** -- das Bild eines echten, kurzen externen
Schocks, nicht eines anhaltenden Störzustands wie bei Code 34 (das dort
schon als Mischung aus kurzen und langen Ereignissen identifiziert wurde).
Code 8 ist die Ausnahme und zeigt eine andere, verzögerte Form -- ob das
ein echter Unterschied oder ein Fallzahl-Artefakt ist, bleibt bei n=352
unklar.

**Hält der Artefakt-Test?** Ja. Beide Kontrolltests (verschobene Terzile,
Zufallsteilung) bestätigen, dass der Peak bei t0 für Code 5 kein
Sortierartefakt ist -- die Terzil-Kurven konvergieren bei t0 statt dort
einen neuen Ausschlag zu erzeugen, und die Zufallsteilung zeigt keine
Gruppenunterschiede. Das erhöht das Vertrauen auch in die ursprüngliche
Code-34-Terzilanalyse (Stufe-1-Nachdiagnose), die denselben Sortiermodus
verwendet hatte, aber dort nicht gegen dieses Artefakt getestet worden war.
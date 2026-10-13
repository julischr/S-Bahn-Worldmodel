# Auswertungsplan Stufe 3 — Survival-Analyse der Erholungszeit

Festgelegt VOR der ersten Erholungsrechnung. Nicht rückwirkend ändern; Abweichungen
werden unten im Abschnitt "Abweichungen" mit Datum/Grund nachgetragen und in
BEFUNDE_SURVIVAL.md wiederholt. Stufe 1, VAR und Plausibilitätstests bleiben unangetastet
(nur lesender Zugriff auf `data/02_events/final/`, `run_all.EXCLUDED_DATES`,
`var/stations.py`).

Stand der Datenkenntnis beim Festlegen: Struktur der Parquet-Dateien, Stufe-1-Logik und
Inventur-Befunde gelesen. Keine Erholungszeit, keine Schwere, keine Kurve berechnet.
(Nur Zeilenzahlen: 3,6 Mio Ankünfte auf den 12 Stationen, 115 Tsd. ausgefallene Halte.)

## 0. Forschungsfragen

1. Wie lange dauert es nach einer Störung bis zum Normalbetrieb?
2. Wovon hängt das ab (Code, Ort, Uhrzeit, Schwere)?
3. Wächst die Erholungszeit überproportional mit der Schwere ("Kipppunkt")?

## 1. Zugebene

- Quelle `data/02_events/final/*.parquet`; `is_arrival = true`, `richtung IS NOT NULL`,
  12 Stammstreckenstationen (`var/stations.py`), beide Richtungen. `is_cancelled`-Zeilen
  bleiben drin. Keine 5-Minuten-Fenster.
- **Kopplung:** zwei trip_ids bilden einen Verband, wenn sie an >= 2 gemeinsamen Stationen
  (gleiche Richtung) exakt dieselbe `time_schedule` haben (Headway = 0, BEFUNDE_VAR 11.6.1).
  Transitiver Abschluss (Union-Find). Ein Verband = ein Zug je Station:
  `time_schedule` = gemeinsamer Wert, `delay` = Maximum der nicht ausgefallenen Mitglieder,
  `ausgefallen` nur, wenn ALLE Mitglieder ausgefallen sind.
- **Betriebstag** = lokale Zeit (Europe/Berlin) minus 3 h, als Datum. Ausgeschlossen:
  Betriebstage in `run_all.EXCLUDED_DATES` (Züge, Episoden, Placebo-Kandidaten).
- Pünktlich: `delay < 180 s` bzw. `< 360 s` und nicht ausgefallen. Ausgefallen = nicht
  pünktlich.

## 2. Störungsereignisse (Episoden)

**Auslösercodes:** Codes 2–69 (Störungscodes der Inventur, BEFUNDE.md "Störungsinventur")
ohne die Folgecodes 43, 44, 45, 46, 48 (46 = gleiche Bedeutung wie 45, kommt nicht vor).
Code 0, 1, 70+ nicht. Begründung je Code und Fallzahl: `auslosercodes.csv`.
Code 23 (Betriebsstabilisierung) bleibt drin (laut Inventur eher Maßnahme als Ursache;
Vorbehalt wird bei Befunden zu 23 genannt). Gruppen mit < 50 Episoden → "selten".

**Episodenbildung (wie Stufe 1, aber ohne Mindestgröße):**
1. Je (Code, Station): Meldungen (Ankünfte, `update_timestamp` nicht null) chronologisch;
   Lücke > 15 min → neuer Stationscluster.
2. Stationscluster desselben Codes über Stationen hinweg zu Episoden verketten, wenn der
   Abstand der Cluster-t0 <= 15 min ist.
3. t0 = frühester `update_timestamp`; Meldestation = Station der ersten Meldung
   (Gleichstand: kleinere stop_id); Richtung = Mehrheitsrichtung aller Meldungen
   (Gleichstand: Richtung der ersten Meldung).
Keine Mindestzahl Fahrten (Stufe 1: >= 3). Sensitivität: nur Episoden mit >= 3 Fahrten.

**Je Episode:** Code, Meldestation, Richtung, Uhrzeit (Ortszeit), HVZ (Stunde 6–9 oder
15–19, wie Inventur), Wochentagstyp (Mo–Do/Fr/Sa/So+Feiertag, Feiertage wie Stufe 1,
am Betriebstag), Monat, Fahrten gesamt (`n_fahrten`), Fahrten in den ersten 15 min
(`n_fahrten_15`), Meldedauer (letzte − erste Meldung; nur beschreibend, weil im
Nachhinein bekannt), Störungslast, Vorbelastung, Schwere, Ausfallquote.

- **Störungslast** = Zahl verschiedener ANDERER Auslösercodes mit einer Meldung
  (beliebige Station/Richtung der Stammstrecke) in [t0−30, t0].
- **Vorbelastung** = mittlere Verspätung (min) der Züge am Meldestation/Richtung mit
  `time_real` in [t0−30, t0] (nur Beobachtetes vor t0; ausgefallene Züge ausgeschlossen).
- **Schwere** = maximale Verspätung (min) der nicht ausgefallenen Züge am Meldestation/
  Richtung mit `time_schedule` in [t0, t0+15).
- **Ausfallquote** = Anteil ausgefallener Züge am Meldestation/Richtung mit
  `time_schedule` in [t0+15, t0+135).

## 3. Erholung

Züge an Station s, Richtung d, chronologisch nach `time_schedule`. Erholt = 3 Züge in
Folge pünktlich, innerhalb desselben Betriebstags. Erholungszeit T = Planzeit des ersten
der drei − Ursprung. Zwei Schwellen (180/360 s), zwei Ebenen:
(a) Meldestation; (b) Strecke = Zeitpunkt, an dem die LETZTE der 12 Stationen (Richtung
d) erholt ist. Stationen ohne weitere Züge im Betriebstag zählen nicht; ist eine Station
zensiert, ist die Streckenzeit zensiert bei max(T_s bzw. Zensierzeit_s) über die Stationen.

**Zwei Ursprünge:**
- **R0:** Ursprung t0, alle Züge mit `time_schedule >= t0`. Für Kaplan-Meier,
  Placebo-Vergleich, Histogramm (Schritte 3, 4).
- **R15 (Landmark):** Ursprung t0+15 min, Züge mit `time_schedule >= t0+15`. Für alles mit
  Schwere als Kovariate/Klasse (Schritte 5, 6). Grund: Schwere ist aus Zügen
  [t0, t0+15) gemessen, die Erholung darf diese Züge nicht mehr enthalten, sonst sind
  Schwere und Ergebnis mechanisch gekoppelt. Fälle mit Folgeepisode vor t0+15 am selben
  Footprint entfallen in R15 (gezählt und berichtet).

**Zensierung:** (i) Betriebstagsende = Zensierung bei Planzeit des letzten Zuges des
Betriebstags; (ii) eine ANDERE Auslöserepisode beginnt (erste Meldung) am selben
(Station, Richtung) [Ebene a] bzw. in derselben Richtung an irgendeiner Station [Ebene b],
bevor die Erholung abgeschlossen ist — "abgeschlossen" = Planzeit des dritten pünktlichen
Zuges < Beginn der Folgeepisode. Zensiert, nicht verworfen. **Sensitivität S-ohne-Folge:**
Zensierung (ii) abgeschaltet (nur Betriebstagsende).
Hinweis (Vorab-Annahme): Zensierung (ii) ist wahrscheinlich informativ (nicht erholtes
Netz erzeugt neue Störungen); deshalb wird sie als annahmeabhängig berichtet.

**"Ohne Wirkung"** = die ersten 3 Züge ab Ursprung sind pünktlich (T = Wartezeit auf den
ersten Zug). Anteil je Code und Schwelle, mit Placebo-Anteil daneben.

## 4. Placebo

Je Episode K = 5 Placebo-Zeitpunkte (Seed 20250): gleiche Station, Richtung,
Wochentagstyp, Ortsstunde von t0; Betriebstag aus ±42 Tagen (nicht der eigene); Zeitpunkt
gleichverteilt in der Stunde; kein ausgeschlossener Betriebstag; **keine Meldung eines
Auslösercodes an (Station, Richtung) in [τ−60, τ+240]**. Bis zu 60 Versuche je Zeitpunkt;
Ausfälle werden gezählt. Placebo erbt den Code der Episode (code-spezifische Placebo-
Kurve). Dieselbe Erholungsmessung (R0, R15, Schwere, Vorbelastung, Ausfallquote), dieselbe
Zensierregel (ii) (am Footprint kommt in den ersten 240 min per Konstruktion keine; danach
möglich). **Bekannte Asymmetrie:** Placebo sind per Konstruktion störungsarme Zeitpunkte
(Selektion auf ruhige Zeit); Ereignisse werden zusätzlich durch Folgeepisoden zensiert.
Der Placebo-Vergleich ist daher eine Untergrenze für "ohne Störung", keine saubere
Gegenfaktische. Wird in BEFUNDE als Annahmeabhängigkeit geführt.

## 5. Kaplan-Meier (Schritt 4)

- Kurven je Gruppe (Code >= 50 Episoden einzeln, Rest "selten"), je Schwelle, je Ebene,
  Ereignis vs. code-spezifisches Placebo. Median, Quartile (25/75 %).
- Konfidenzintervalle: tageweiser Bootstrap (Betriebstage der Episoden mit Zurücklegen,
  Placebo hängen an ihrer Episode), B = 1000, 95 %-Perzentil. Nicht erreichte Quantile
  → "n.e." (nicht erreicht).
- Log-Rank: (1) Mehrgruppentest zwischen Codes (R0, Station, 180 s); (2) je Code gegen
  Placebo. Weil Episoden eines Tages korreliert sind, zusätzlich ein Cox-Wald-Test mit
  clusterrobusten SE (Betriebstag) für denselben Kontrast; Entscheidungsgröße ist der
  clusterrobuste p-Wert, Holm-Korrektur über die Code-vs-Placebo-Tests.

## 6. Cox (Schritt 5)

- Ergebnis: Erholung (R15, Station, 180 s als Haupt; Strecke und 360 s als Nebenmodell).
  HR > 1 = schnellere Erholung.
- Kovariaten: Code (Referenz: häufigste Gruppe), Meldestation (Referenz: Station mit den
  meisten Episoden, im Output genannt), Richtung, HVZ, Wochentagstyp,
  Störungslast, Vorbelastung (je 5 min), Schwere (je 5 min), `log2(n_fahrten_15)`.
  **Interpretation von "Zahl betroffener Fahrten":** nur Fahrten mit Meldung in den
  ersten 15 min (kein Vorgriff auf die spätere Episodenlänge).
- Efron-Ties, clusterrobuste SE nach Betriebstag. Vollständige Fälle; Ausfall durch
  fehlende Kovariaten wird gezählt.
- PH-Annahme: Grambsch-Therneau-Test auf Schoenfeld-Residuen (Zeittransformation: Rang),
  je Kovariate + global. Verletzt = globaler p < 0,01 (bei großem n wird jede Kleinigkeit
  signifikant — deshalb zusätzlich Bewertung der Größenordnung: Quotient der HR im
  frühen vs. späten Zeitfenster, Median-Split der Zeit). Falls verletzt: (a) Modell
  stratifiziert nach Code, (b) Log-Normal- und Weibull-AFT (Zeitverhältnisse), beide berichtet.

## 7. Kipppunkt (Schritt 6)

- Klassen Schwere: < 3, 3–5, 5–10, 10–20, > 20 min. Je Klasse und HVZ ja/nein Median
  Erholungszeit (R15, Station, 180 s) mit Tages-Bootstrap-KI, Ereignis UND Placebo in
  derselben Schwereklasse (Placebo-Pseudo-Schwere aus ihren eigenen ersten 15 min). Das ist
  die Kontrolle gegen Regression zur Mitte: dieselbe Auswahlregel wird auf Zeitpunkte ohne
  Störung angewendet.
- Spline: Restricted Cubic Spline (4 Knoten bei den Quantilen 5/35/65/95 % der Schwere,
  Schwere bei 120 min gedeckelt) im Cox- und im Log-Normal-AFT-Modell. Test gegen linear:
  robuster Wald-Test der nichtlinearen Terme (primär) und partieller Likelihood-Quotient.
  Interaktion Schwere × HVZ: robuster Wald-Test.
- **Entscheidungsregel "überproportional":** (i) robuster Wald-Test der Nichtlinearität
  p < 0,01 im AFT-Modell (Zeitskala) UND (ii) Anstieg der Median-Erholungszeit pro Minute
  Schwere zwischen Klassenmitten der oberen Klassen (10–20 → > 20 min) größer als der
  zwischen unteren (3–5 → 5–10 min), Verhältnis mit tageweisem Bootstrap-95%-KI, Untergrenze
  > 1; (iii) Placebo-Kurven zeigen dieses Muster nicht. Sonst: "linear/ nicht
  überproportional belegt". Nur (i) → "Nichtlinearität, Richtung offen".
- Ausfallquote nach t0 je Schwereklasse (Ereignis vs. Placebo), Wilson-KI mit
  Tageswiederholung (Bootstrap).

## 8. Einordnung (Schritt 7)

Abgleich der Median-/Quantil-Erholungszeiten (Strecke) mit VAR-Halbwertszeit (Ost ~60,
West ~120 min) und der Dauer-Stratifizierung aus Stufe 1 (kurz < 30 / mittel 30–90 /
lang > 90 min Meldedauer; hier nur als beschreibende Zusatzstratifizierung, weil
Meldedauer ex post bekannt ist). Reiner Plausibilitätsabgleich, keine neue Hypothese.

## 9. Belastbarkeitsregeln für BEFUNDE_SURVIVAL.md

- **(a) belastbar:** Gruppe >= 50 Episoden; Richtung des Effekts gleich in Sensitivitäten
  (360 s, S-ohne-Folge, >= 3 Fahrten); clusterrobuste Inferenz.
- **(b) annahmeabhängig:** hängt an Placebo-Konstruktion, Zensierung (ii), Kopplungsregel,
  Schwellenwahl, PH-Annahme.
- **(c) offen.**

## 10. Plots (je ein Zweck)

1 KM je Code + Placebo (180 s, Station); 2 dito Strecke; 3 Schwellenvergleich 180 vs.
360; 4 Histogramm der Erholungszeiten inkl. "ohne Wirkung"; 5 Forest-Plot HR;
6 Erholungszeit je Schwereklasse nach HVZ; 7 Ausfallquote je Schwereklasse.

## Abweichungen (nachträglich dokumentiert)

1. **Störungslast — Versuch verworfen, Plan gilt unverändert:** Nach einer ersten Rechnung
   mit Median 35 verschiedenen Codes im Fenster habe ich eine strengere Definition
   erwogen (zusätzlich `time_real` im Fenster), weil `update_timestamp` bei 31 % der
   Auslöser-Meldungen > 30 min nach `time_real` liegt (nachträglich gesetzte Codes). Die
   35 waren aber ein Einheitenfehler im eigenen Code (ns vs. µs), nicht ein Datenbefund.
   Nach Korrektur Median 2 Codes; die Planformulierung bleibt, die Verzögerungs-
   Eigenschaft (31 % der Meldungen > 30 min nach Zugzeit; Diagnose
   `lag_erste_meldung_min` je Episode für die ERSTE Meldung) ist eine Einschränkung
   für alle t0-basierten Zeitfenster und wird in BEFUNDE_SURVIVAL.md unter (b) geführt.
2. **Placebo-Variante P2 zusätzlich (nach dem ersten Bau der Tabellen, vor jeder Auswertung
   mit KM/Cox/Tests):** P1 = Plan (gleichverteilter Zeitpunkt in der Ortsstunde) bleibt die
   vorab festgelegte Hauptvariante für alle Tests. Beim ersten Tabellenbau sah ich zwei
   Dinge: (i) Fehler im eigenen Code: Placebo-Tage außerhalb des Datenzeitraums (Episoden
   nahe Jahresanfang/-ende, ±42 Tage) → Bedingung "Betriebstag hat Züge an
   (Station, Richtung)" ergänzt (Fehlerbehebung); (ii) strukturell: echte t0 liegen per
   Konstruktion kurz nach einem Zug (Median-Verzögerung der ersten Meldung 1,9 min), P1-
   Zeitpunkte liegen gleichverteilt im Zugtakt, die Wartezeit bis zum ersten Zug nach dem
   Ursprung ist dadurch für Ereignisse und Placebo systematisch verschieden (Phasen-
   Artefakt). **P2** = time_real eines zufälligen Zuges derselben Station/Richtung in der
   Ortsstunde + 120 s, sonst identische Regeln. P2 wird in den Kurven gestrichelt gezeigt und
   in den Sensitivitäten berichtet; Tests werden für P1 (Plan) UND P2 ausgewiesen. Dabei
   gesehen (Rohzahlen, keine Wirkung ausgewertet): 27–30 % der angeforderten Placebo-
   Zeitpunkte finden keinen störungsfreien Gegenpart in ±42 Tagen (P1 19.569 von 71.705;
   P2 21.843); Episoden in dicht gestörten Zeiten haben daher seltener ein Placebo.
3. **Clusterrobuster Test je Code vs. Placebo (Rechenzeit):** Plan: Cox-Wald mit robusten SE.
   PHReg brauchte ~55 s je Test bei 58.000 Zeilen (rund 500 Tests nötig). Ersatz: clusterrobuster
   Log-Rank-Score-Test (Score-Residuen, Cluster = Betriebstag) plus Ein-Schritt-Näherung der
   HR (`hr_vs_placebo`). Gleicher Kontrast, andere Schätzung; der Gesamttest zwischen Codes
   nutzt weiter Cox (Breslow-Ties) mit robustem Wald.
4. **Cox Strecke und Cox stratifiziert:** liefen, aber die robuste Kovarianz ist dort nicht
   bestimmbar (KI = NaN; Strecke nur 963 Ereignisse bei 49 Parametern; stratifiziert: gleiche
   Ursache). Berichtet werden nur Punktschätzer, keine Inferenz.
5. **Kipppunkt-Entscheidungsregel Teil (ii):** Klassenmitte = empirischer Median der Schwere
   in der Klasse (im Plan nicht festgelegt); Steigung = Median-Differenz / Differenz dieser
   Klassenmitten.

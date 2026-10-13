# Übergabe an andere KI: Stufe 4 — gelernter Simulator der Münchner S-Bahn-Stammstrecke

Dieses Dokument ist in sich vollständig (kein Zugriff auf Code, Plots oder Dateien nötig). Alle Zahlen stammen aus gerechneten Läufen; Fallzahl und Einheit stehen dabei. „Verspätung“ = Ankunftsverspätung in Minuten (Rohdaten in Sekunden, ganzminütig). Zeitreihe: Betriebstage 2025; Test = Oktober–Dezember 2025.

**Kurzfassung.** Ein LightGBM-Modell sagt je Zug und Station vorher, (a) ob der Zug an der nächsten Station ausfällt und (b) wie sich seine Verspätung ändert (19 Klassen). Ein Simulator zieht daraus Zug für Zug einen ganzen Tag (100 Läufe parallel, Minutentakt). Ein-Schritt-Prüfung: besteht gegen beide Vergleichsmodelle. Rollout-Prüfsteine (5): a bestanden, b bestanden, c, d, e **nicht bestanden**. Szenarien liefern kleine Effekte; die experimentelle Sperrung ist falsch skaliert.

## 1. Aufbau

**Daten.** `data/02_events/final/*.parquet` (Monate 2025-01 … 2025-12), nur Ankünfte (`is_arrival`) an den 12 Stammstreckenstationen, Richtung bekannt. Reihenfolge West→Ost: 0 Pasing, 1 Laim, 2 Hirschgarten, 3 Donnersbergerbrücke, 4 Hackerbrücke, 5 Hauptbahnhof tief, 6 Karlsplatz (Stachus), 7 Marienplatz, 8 Isartor, 9 Rosenheimer Platz, 10 Ostbahnhof, 11 Leuchtenbergring. Richtung „Ost“ = aufsteigender Rang. Nur beobachtete Werte (`delay`, `time_real`, `is_cancelled`), keine DB-Prognosen.

| Schritt | Zeilen |
|---|---|
| Ankünfte roh (12 Stationen, Richtung bekannt) | 3.546.812 |
| nach Zusammenfassen gekoppelter Verbände (gleiche Planzeit an ≥ 2 gemeinsamen Stationen = ein Zug) | 3.187.273 Zug-Station-Zeilen |
| nach Entfernen der Sperrtage (`run_all.EXCLUDED_DATES`: 2025-01-11/12, 05-10/11, 06-07…06-12, 10-18/19, 2026-01-01) | 3.169.666 |
| nach Pfadnormalisierung (s. u.) | 3.140.439 Zeilen, 295.621 Züge |
| Übergänge Station k → k+1 (Zug fuhr an k, k+1 liegt im Pfad) | 2.751.924 |
| davon mit Ausfall an k+1 | 4.873 (0,18 %) |
| Übergänge mit geplanter Fahrzeit > 15 min (ausgeschlossen) | 9 |

Pfadnormalisierung (nötig, damit Training und Simulator dieselbe Welt sehen): je Zug nur das längste zusammenhängende Pfadsegment (benachbarte Stationen); Ausfall ist absorbierend (ab dem ersten Ausfall nach dem ersten gefahrenen Halt gilt der Rest als ausgefallen). Entfernt: 29.227 Zeilen (Lücken), 6.408 Zeilen mit „Wiederaufnahme der Fahrt nach Ausfall“ (2.027 Züge) wurden als ausgefallen gesetzt.

**Aufteilung nach Betriebstag-Monat.** Training Jan–Jul: 1.576.674 Übergänge (2.643 Ausfälle, Rate 0,168 %); Validierung August (Early Stopping, Hyperparameter): 229.623 (356; 0,155 %); Kalibrierung September: 267.957 (101; 0,038 %); Test Okt–Dez: 677.661 (1.773; 0,262 %). Der Test wurde nie zum Anpassen benutzt (siehe Abschnitt 9 zu einer Entscheidung, die nach Sicht der Testzahlen fiel).

**Zielgrößen.** (a) `cancel_next` (Ausfall an k+1). (b) Zuwachs Δ = Verspätung(k+1) − Verspätung(k) in ganzen Minuten, nur wenn der Zug an k+1 fährt, in 19 Klassen: Klasse 0 = Δ ≤ −3, Klasse j = Δ = j − 3 für j = 1…17 (−2 … +14), Klasse 18 = Δ ≥ +15. Randverteilungen (aus Training): Δ ≤ −3: n = 4.003, Mittel der Überschreitung (Δ+3) −1,51; Δ ≥ 15: n = 606, mittlere Überschreitung (Δ−15) +9,59, Extreme −157 … +153. Im Simulator werden Randklassen aus diesen empirischen Listen nachgezogen.

**Merkmale (17; Stichtag T = Ankunft des Zuges an k; nur Ereignisse mit Realzeit < T, die gleiche Minute ist nicht sichtbar). Keine Störungscodes.**

| Merkmal | Definition | Einheit |
|---|---|---|
| d_k | eigene Verspätung an k | min |
| d_km1 | eigene Verspätung an k−1 (NaN am Eintritt) | min |
| rank | Station k (0…11) | – |
| richtung | 0 Ost, 1 West | – |
| line | Linie S1…S8, sonst „other“ (kategorisch) | – |
| hour | Ortszeit von T | h |
| hvz | 6–9 Uhr oder 15–19 Uhr | 0/1 |
| wtype | 0 Mo–Do, 1 Fr, 2 Sa, 3 So/Feiertag | – |
| plan_gap_pred | geplanter Abstand zum Vorgänger an k (Planzeit-Differenz; kann negativ sein bei vertauschter Reihenfolge) | min |
| act_gap_pred | tatsächlicher Abstand zum Vorgänger an k (≥ 1) | min |
| d_pred_k | Verspätung des Vorgängers an k | min |
| d_pred_k1 | Verspätung des Vorgängers an k+1, nur wenn er dort Realzeit < T angekommen ist, sonst NaN | min |
| net_mean15 | mittlere Verspätung aller Ankünfte (12 Stationen, beide Richtungen) mit Realzeit in [T−15, T) | min |
| net_n15 | Zahl dieser Ankünfte (Ergänzung zum Auftrag) | – |
| cancel_sd30 | ausgefallene Halte an derselben Station/Richtung, „Bekanntzeit“ in [T−30, T) | Zahl |
| cancel_net30 | dasselbe auf der ganzen Stammstrecke | Zahl |
| plan_run | geplante Fahrzeit k → k+1 (Fahrplanwissen, Ergänzung) | min |

Vorgänger = der Zug, der an (Station, Richtung) zuletzt vor T angekommen ist (Realminute < T, nicht ausgefallen); bei gleicher Minute entscheidet Planzeit, dann Zug-ID. Bekanntzeit eines ausgefallenen Haltes = max(Planzeit, Realzeit der letzten Ankunft desselben Zuges davor); ohne frühere Ankunft = Planzeit. Fachlich: ein Ausfall ist erst bekannt, wenn der Zug seinen letzten gefahrenen Halt erreicht hat.

**Leckprüfung (Teacher Forcing).** Der Simulator wurde mit den echten nächsten Ankünften betrieben und seine Merkmale mit den Trainingsmerkmalen verglichen (3 Testtage). Ergebnis: identisch für d_k, d_km1, richtung, line, hour, hvz, wtype, plan_run und beide Ausfallzähler (bis auf 4 Zeilen); Abweichungen bei Vorgänger- und Netzmerkmalen nur dort, wo die Kausalitätsklammer (Ankunft an k+1 ≥ 1 min nach k) eine echte Ankunft um eine Minute verschiebt:

{{check_forced}}

**LightGBM.** Gemeinsam: Lernrate 0,08, feature_fraction 0,8, bagging_fraction 0,8 (bagging_freq 1), lambda_l2 5, max_bin 127, seed 7; Gitter num_leaves ∈ {31, 127} × min_data_in_leaf ∈ {100, 1000}, Early Stopping (30 Runden, max. 600) auf August, Auswahl nach Log-Loss. Ausfall (binary): gewählt 31 Blätter, min_data 100, 255 Bäume. Zuwachs (multiclass 19): gewählt 127 Blätter, min_data 100, 143 Runden. Gitter-Ergebnisse (Val-Log-Loss August):

{{grid_cancel}}

{{grid_class}}

## 2. Ein-Schritt-Prüfung (Test Okt–Dez 2025)

Vergleichsmodelle (angepasst auf Training Jan–Jul): V1 = Verspätung bleibt gleich (Ausfall: globale Rate 0,168 %); V1b = Persistenz + globale Δ-Verteilung (nur für Log-Loss/CRPS); V2 = empirische Δ-Verteilung bzw. Ausfallrate je (Station, Richtung, Stunde), mit Glättung (5 Pseudobeobachtungen aus der globalen Verteilung). MAE/RMSE/Bias der Verspätung an k+1 in min; Punktprognose des Modells = d_k + Median (bzw. Mittel) der Klassenverteilung; CRPS in min über die 18 Klassenschwellen (beobachtetes Δ auf [−3, 15] begrenzt); Log-Loss in nats. n = Übergänge mit Fahrt an k+1.

**2a. Verspätung, Rohwahrscheinlichkeiten (diese Variante nutzt der Simulator):**

{{einschritt_verspaetung_roh}}

**2b. Ausfall, Rohwahrscheinlichkeiten** (n = alle Test-Übergänge; Brier-Skill = 1 − Brier / Brier(V2)):

{{einschritt_ausfall_roh}}

**2c. Mit Temperaturskalierung (T = 0,906, auf September gefittet):**

{{einschritt_verspaetung_kalibriert}}

(Ausfall unverändert, da die Platt-Kalibrierung abgelehnt wurde — Tabelle identisch mit 2b.) **Urteil:** Das Modell schlägt V1, V1b und V2 in MAE (0,409 gegen 0,579 für V2-Median, 0,627 für V2-Mittel), CRPS (0,306 gegen 0,432), Log-Loss (0,872 gegen 1,176) und Brier (0,002131 gegen 0,002574), in beiden Richtungen. Die Abbruchregel („Modell schlägt V2 nicht → nicht weiterbauen“) greift nicht. Die Median-Prognose hat einen kleinen negativen Bias (−0,06 min), die Mittelprognose praktisch keinen (−0,002 min).

## 3. Kalibrierung

Reliability-Tabellen (Test; Bins nach vorhergesagter Wahrscheinlichkeit; obs_lo/hi = 95-%-Wilson-Intervall).

**Ausfall (roh):**

{{reli_ausfall_roh}}

**Zuwachs, kumulative Ereignisse (roh)** (Δ ≤ 0 = Klassen 0–3; Δ ≥ +2 = Klassen 5–18; Δ ≥ +5 = Klassen 8–18):

{{reli_zuwachs_roh}}

**Mittlere vorhergesagte gegen beobachtete Häufigkeit je Klasse (roh):**

{{reli_klassen_roh}}

ECE (Test, roh): Ausfall 0,00045; Zuwachs Δ ≤ 0: 0,0052; Δ ≥ +2: 0,0025; Δ ≥ +5: 0,00018. Nach Temperaturskalierung: 0,0144 / 0,0074 / 0,0021 (also schlechter). Nachkalibrierung nach Plan (Fit auf September; Entscheidung per Tageshälften: ungerade Tage fitten, gerade Tage prüfen): Temperatur 0,918 (ungerade) bzw. 0,906 (alle Tage); Log-Loss gerade Tage roh 0,23982 → kalibriert 0,23785 (Verbesserung, daher „angenommen“). Platt-Kalibrierung Ausfall: Brier gerade Tage roh 0,000302 → 0,000305, Log-Loss 0,001792 → 0,001793, also nicht angenommen (52 Ausfälle in dieser Hälfte; Kalibriermonat hat nur 101 Ausfälle, Rate 0,038 % gegenüber 0,262 % im Test, also untypisch). Beobachtung: Roh-Ausfallwahrscheinlichkeiten werden im Test bei 0,5–10 % leicht unterschätzt (z. B. vorhergesagt 7,0 %, beobachtet 10,0 %), weil die Ausfallrate im Test höher ist als im Training.

## 4. SHAP (TreeExplainer; 40.000 bzw. 20.000 zufällige Testzeilen)

Ausfallmodell, mittleres |SHAP| in log-Odds:

{{shap_ausfall}}

Zuwachsmodell, Summe über die 19 Klassen des mittleren |SHAP| (log-Odds), dazu die Werte für die Klassen Δ = 0 und Δ = +2:

{{shap_zuwachs}}

Lesart: Das Ausfallmodell hängt am stärksten an der Zahl der Ausfälle der letzten 30 min auf der Strecke (cancel_net30). Das Zuwachsmodell nutzt am meisten die Verspätung des Vorgängers an k+1, die eigene Verspätung, die Station und die geplante Fahrzeit (plan_run, wichtig für Δ = 0).

## 5. Rollout

- Prüfsteintage: 30 aus den Testmonaten gezogen (Seed 4, nach Wochentagstyp geschichtet, ohne Sperrtage): 2025-10-05, 10-06, 10-15, 10-22, 10-25, 10-30, 11-04, 11-09, 11-13, 11-14, 11-15, 11-16, 11-17, 11-19, 11-20, 11-21, 11-27, 11-28, 11-29, 12-01, 12-04, 12-09, 12-10, 12-11, 12-12, 12-16, 12-25, 12-27, 12-28, 12-31. Je Tag 100 Läufe (gemeinsamer Seed 11) = 3.000 simulierte Tagesläufe; 20 Läufe je Tag für Prüfstein e.
- Mechanik: Minutentakt; Eintritt am ersten laut Daten gefahrenen Halt mit echter Verspätung; Ausfall absorbierend; Ziehung per Inversion mit je (Lauf, Zug, Pfadposition) festen Zufallszahlen (common random numbers).
- Ausfallquote (modellierte Positionen = Halte nach dem Eintritt; 230.800 Positionen/Lauf-Tag-Summe): Echt 2,55 %, Sim 2,03 %.
- Kausalitätsklammern: 152.314 über alle 3.000 Tagesläufe (ca. 0,6 % der Übergänge; Echt: 0,64 % der Übergänge haben Fahrzeit ≤ 0 min).

**Physikalisch unmögliche bzw. unplausible Zustände (nur gezählt):**

{{plausibilitaet_basis}}

Spalten: `ueberholungen` = Zahl der Paare, die an k hintereinander ankommen und an k+1 in vertauschter Reihenfolge (strikt nach Ankunftsminute); `paare` = Zahl betrachteter benachbarter Paare; `gleiche_minute` = aufeinanderfolgende Ankünfte an einer Station in derselben Minute; `abstand_hoechstens_1min_quote` = Anteil Abstände ≤ 1 min. Sim-Summen sind über 100 Läufe, Echt über je einen Tag (Summen daher nicht vergleichbar, nur Quoten). Sim hat 2,5-mal so viele Überholungen, 4,4-mal so viele gleiche Minuten und doppelt so viele Abstände ≤ 1 min.

Mittlere Verspätung je Station/Richtung (Alle Ankünfte der 30 Tage; Sim = Mittel über Läufe):

{{profil_verspaetung_basis}}

## 6. Prüfsteine (Simulation gegen Wirklichkeit, 30 Testtage)

Zusammenfassung mit Plan-Toleranzen:

{{steine}}

Konfidenzintervalle: Bootstrap über Betriebstage (1.000 Ziehungen); „Sim-Lauf q025/q975“ = Streuung über die 100 Läufe.

**(a) Ankunft am Pfadende** (Züge, die laut Daten bis zu ihrem letzten Halt fuhren und mindestens 2 Halte haben; Sim-Band aus den Läufen, in denen der Zug nicht ausfiel). Abdeckung = Anteil echter Werte in [q05 (untere Ordnungsstatistik), q95 (obere)]:

{{pruefstein_a}}

**(b) Verspätungsaufbau Eintritt → Ende, Züge mit Rang 0 und 11 im Pfad, die am Streckenanfang eintraten und am Ende fuhren** (min):

{{pruefstein_b}}

**(c) Zuwachs des Folgezugs (Verspätung am Ende − am Eintritt), nach Vorgängerverspätung am Eintritt (≤ 2, 2–5, 5–10, > 10 min) und geplantem Abstand zum Vorgänger am Eintrittshalt (Tabelle 11.7-Definition, ganze Kernstrecke).** Vorgänger = vorheriger gefahrener Zug nach Planzeit am Eintrittshalt. Die Zelleneinteilung kommt aus den Echtdaten (Vorgängerverspätung am Eintritt ist Replay); n_sim_pro_lauf ist die mittlere Zahl je Lauf. Zelle gilt als geprüft bei n_Echt ≥ 200; `zelle_ok` = |Diff| ≤ 0,3 min (1 = ok, 0 = nicht ok, leer = nicht geprüft). Referenz aus BEFUNDE_VAR 11.7 (ganze Strecke, ganzes Jahr): z. B. Zeile „<5 min“: 2,79 → 3,87 → 4,01 → 4,54 min.

{{pruefstein_c}}

**(d) Ausfallquote des Folgezugs nach Vorgängerverspätung und Abstand (Tabelle 11.8-Definition), nur an modellierten Übergängen** (Folgezug fuhr an der Vorstation; Vorgänger = unmittelbarer Vorgänger im Plan an derselben Station, nicht ausgefallen, Abstand 0 < Δ ≤ 60 min). `diff_pp` = Sim − Echt in Prozentpunkten; Toleranz max(0,3 pp; 40 % der Echtrate); geprüft bei n_Echt ≥ 500.

{{pruefstein_d}}

**(e) Erholungszeit je Schwereklasse.** Messung exakt wie Stufe 3 (`recovery.measure`: Erholung = drei aufeinanderfolgende Züge an der Meldestation < 3 min Verspätung, ab t0+15 min, Kaplan-Meier-Median; Schwere = max. Verspätung in den ersten 15 min; Standard-Zensierung durch Folgeepisoden) auf den echten Episodenzeitpunkten der 30 Tage (1.332 Episoden; 965 auswertbar je Lauf), (i) auf echten Daten dieser Tage, (ii) auf 20 simulierten Läufen je Tag (Schwere und Erholung aus den simulierten Zügen). Mediane in min; „n.e.“ = nicht erreicht; Sim: Mittel der Lauf-Mediane und Spanne (2,5/97,5 %-Perzentil der 20 Lauf-Mediane, „–“ wenn Läufe n.e. liefern); `klasse_ok` nach Regel (|Sim − Echt| ≤ max(25 % Echt; 5 min) oder Echt-Median in Lauf-Spanne; bei Echt „n.e.“: > 50 % der Läufe n.e.).

{{pruefstein_e}}

Vergleich mit Plot 6 der Survival-Stufe (Median-Erholungszeit ab t0+15 min, Störungen, ganzes Jahr, n = 1.358 / 1.844 / 4.228 / 2.451 / 608): 15,2 / 23,4 / 58,8 / 121,1 / 236,7 min (obere Klasse mit KI 124–n.e.). Die Testtage (kleinere Stichprobe, andere Monate) liefern 32,0 / 32,6 / 88,2 / 132,4 / n.e.; Sim 24,5 / 39,6 / 86,7 / 199,3 / 223,4. Alle drei zeigen monotones Ansteigen mit der Schwere. Starke Zensierung: Echt 39 % / 51 % / 68 % / 80 % / 91 %; Sim 39 % / 54 % / 69 % / 83 % / 90 %.

## 7. Szenarien

Aufbau: Tage 2025-11-12 und 2025-12-03 (Mittwoche). HVZ: Eintritt 7:00–9:00 bzw. 16:00–18:30 Uhr; Nicht-HVZ: 10:00–14:00 Uhr. Je Tag/Zeitfenster 2 Beispielzüge (bevorzugt Linie S3, Richtung Ost, Eintritt in Pasing, Verspätung am Eintritt ≤ 3 min, ≥ 8 Halte). Fenster: Simulation startet 60 min vor Eingriff mit echtem Zustand und endet 6 h danach. Je Szenario 100 Läufe; Basis und Szenario mit denselben Zufallszahlen. Definitionen: „+N“: Der Zug betritt die Strecke mit absolut N min Verspätung (die Änderung gegenüber der echten Verspätung steht in `eigener Zug Δ am Ende`). „betroffen“: Änderung der Verspätung am Pfadende ≥ 1 min im jeweiligen Lauf (andere Züge als X). `Δ je betr. Zug`: je Zug über die Läufe gemittelte Änderung am Pfadende, für Züge mit |Mittel| ≥ 0,25 min; Verteilung über diese Züge (Mittel, Median, q10, q90, max). Gesamt-Mehrverspätung = Summe der Änderungen am Pfadende aller anderen Züge je Lauf (Zug-Minuten). Zusätzliche Ausfälle = Zahl simulierter Ausfälle (Szenario − Basis) an modellierten Positionen anderer Züge. Abklingzeit (KI-Regel): ab dem Spitzenblock der Mittel-Differenz über 10-min-Blöcke der letzte Block, dessen 95-%-Intervall (über die Läufe) die 0 nicht einschließt, plus Blocklänge; Abklingzeit < 0,1: erster Block ab der Spitze, nach dem 3 Blöcke in Folge |mittlere Differenz| < 0,1 min. Alle Werte der Tabelle sind Mittel über die beiden Tage × je 2 Züge (n_sim = 4 Szenariosimulationen je Variante); Min/Max zeigt die Spanne zwischen den 4.

{{szenario_kompakt}}

Einzelergebnisse je Simulation (alle Spalten):

{{szenario_einzel}}

**Befund.** Verspätung eines Zuges wird im Modell kaum weitergegeben: bei +5/+10/+20 min Eintrittsverspätung ändert sich die Summe der Endverspätungen aller anderen Züge im Mittel um 0,1–6,5 Zug-Minuten (außerhalb HVZ) bzw. 3,4–12,0 (HVZ); die Zugwerte streuen stark (Minima der Zug-Minuten bis −10, Maxima bis +41 bei q95). Zusätzliche Ausfälle sind im Mittel nicht von 0 verschieden (Mittel zwischen −0,9 und +0,1 für Verspätung, +1,8/+0,6 für gestrichene Fahrt). Der Rauschpegel (zufälliges Umkippen der Ziehungen nach kleinen Änderungen der Wahrscheinlichkeiten) ist so groß wie der Effekt bei +5 min. Die Abklingzeiten (Mittel 72–228 min nach KI-Regel, 38–88 min nach der 0,1-min-Regel) sind unsicher, weil sie auf dem Rauschen aufsetzen.

**Sperrung (experimentell).** Regel: Station s von t1 bis t2 nicht passierbar; Züge, deren Ankunft an s in [t1, t2) läge, kommen frühestens bei t2 + Fahrzeit an, danach im Abstand ≥ 2 min; Rest Modell. Wartende Züge blockieren in der Simulation die Vorstation nicht. Datensuche: 18.698 Lücken > 15 min an Station/Richtung tagsüber (5–23 Uhr) im ganzen Jahr; die meisten an Randstationen/ Randzeiten; nur 1.693 mit ≥ 3 Zügen an der Vorstation in der Lücke; im Testzeitraum mit ≥ 8 Zügen und ≥ 20 min: 6 Fälle (nach Duplikatentfernung je Tag/Station):

{{sperrung_faelle}}

Test (100 Läufe; Fenster t1−60 min … t2+150 min; Kennzahlen für Züge, die in den 60 min ab t2 an s bzw. der nächsten Station ankommen): Zahl der Züge, mittlere Verspätung, Minuten bis zur ersten Ankunft nach t2:

{{sperrung_test_szenarien}}

**Urteil:** In den echten Fällen lag die mittlere Verspätung nach der Lücke bei 2,8–8,0 min (ein Fall 20,5). Der Simulator mit Sperrungsregel liefert 23–55 min, der Basislauf ohne Sperrung 2,5–7,6 min (der Basislauf bildet die echten Fälle gut ab, weil er die echten Eintritte nutzt und die Lücke in den Daten bei Beginn des Fensters bereits teilweise bekannt ist). Außerdem lässt die Simulation 29–30 statt 20–24 Züge in den 60 min nach t2 ankommen (alle gewarteten Züge fahren, in den Daten fielen Züge aus oder wendeten). Die Regel ist demnach in diesen Fällen falsch; Modell und Regel kennen Wenden/Ausfälle/Rückstau nicht.

## 8. Plots als Text

Alle Plots in `outputs/simulator/plots/`.

1. **01_reliability** (4 Panels, Test): Achsen vorhergesagte Wahrscheinlichkeit (x) gegen beobachtete Häufigkeit (y) mit 95-%-Intervallen, Linie = Diagonale, durchgezogen roh, gestrichelt temperaturskaliert. Panel 1 (Ausfall, log-log): Punkte (vorhergesagt → beobachtet) 0,00016 → 0,00014; 0,0007 → 0,00087; 0,0014 → 0,0016; 0,0031 → 0,0041; 0,0070 → 0,0085; 0,014 → 0,0144; 0,031 → 0,036; 0,070 → 0,100; 0,141 → 0,184; 0,435 → 0,392. Roh nahe der Diagonale, leichte Unterschätzung im Bereich 0,5–20 %, Überschätzung ganz oben. Panels 2–4 (Δ ≤ 0, Δ ≥ +2, Δ ≥ +5): Punkte liegen auf der Diagonale; z. B. Δ ≤ 0: 0,029 → 0,029; 0,148 → 0,157; 0,350 → 0,349; 0,653 → 0,656; 0,857 → 0,866; 0,971 → 0,971 (Tabellen oben). Die temperaturskalierte Linie liegt sichtbar etwas weiter weg.
2. **02_shap**: zwei Balkendiagramme der Top-15 (Zahlen in Abschnitt 4).
3. **03_pruefstein_a_b**: links Streudiagramm echte Verspätung am Pfadende (x, −5…40 min) gegen Sim-Median (y) für 6.000 Züge mit 90-%-Bändern für 300 Züge; Punkte um die Diagonale, Bänder bei hohen Verspätungen breiter; einzelne Ausreißer (Sim ≈ 0 bei echten 30–40 min). Mitte: Histogramm Median(Sim) − Echt, zentriert bei ≈ 0, Spitze bei 0 bis +1, Schwanz bis −15; Bias −0,11 min. Rechts: Histogramm Aufbau Eintritt→Ende: Echt Mittel 3,31, Sim (Lauf-Mittel je Zug) 3,69; die Sim-Verteilung ist schmaler (fehlt der lange rechte Schwanz bis +19 min und die Masse bei 0 bis −5 min der Echtdaten).
4. **04_pruefstein_c_d**: gruppierte Balken je Zelle (Abstandsbin × Vorgängerklasse, 20 Zellen), grau Echt, rot Sim. Links (c): Sim in allen Zellen mit Echt-Daten höher; Zeile „<5 min“ Echt 3,29 / 3,85 / 4,09 / 4,67, Sim 3,59 / 4,19 / 4,62 / 4,83 (Tabelle (c)). Rechts (d, %): Zeile „<5 min“ Echt 0,68 / 2,26 / 3,47 / 4,73, Sim 0,99 / 1,73 / 2,50 / 3,54; Zellen mit langen Abständen Echt 0 (n klein), Sim 0,5–1 %.
5. **05_pruefstein_e**: Balken der Median-Erholungszeit je Schwereklasse, grau Echt, rot Sim mit Spanne: Zahlen in der Tabelle (e); Klassen 10–20 und > 20 Sim deutlich über Echt bzw. Echt nicht erreicht.
6. **06_profil**: je Richtung mittlere Verspätung gegen Stationsrang; Echt und Sim fast deckungsgleich, Sim 0,05–0,4 min höher (Zahlen in der Tabelle in Abschnitt 5); Ost steigt von 4,3 (Pasing) auf 6,2/6,5 (Leuchtenbergring), West von 2,95 (Leuchtenbergring) auf 6,3/6,6 (Pasing).
7. **07_szenario_verspaetung / 07_szenario_ausfall**: mittlere Verspätungsdifferenz Szenario − Basis aller anderen Züge (min) über die Minuten nach dem Eingriff (10-min-Blöcke), links HVZ, rechts Nicht-HVZ, eine Kurve je N (5, 10, 20). Werte liegen im Bereich ±0,3 min, Spitze bis etwa +0,3 min (N = 20, HVZ) in den ersten 1–2 Stunden, danach Streuung um 0; Spitze der Mittel-Differenz (Tabelle): 0,04 / 0,13–0,20 / 0,20–0,29 min bei N = 5 / 10 / 20.
8. **08_bildfahrplan_basis / 08_bildfahrplan_verspaetung10**: Zeit-Weg-Diagramm Richtung Ost am 2025-11-12 07:55–08:50, x = Uhrzeit, y = Stationen (äquidistant, Pasing unten, Leuchtenbergring oben). Graue dünne Linien = Plan, schwarz = echte Fahrten, rot = Sim-Median, rosa = 90-%-Band über 100 Läufe, blau gestrichelt (nur im Szenario) = Basislauf-Median. Echte Fahrten laufen meist innerhalb des Bandes; im Szenario (S3 +10 min um 08:00 in Pasing) verschiebt sich der betroffene Zug (rot) gegenüber dem Basislauf um ca. 10 min, Folgezüge kaum.

## 9. Abweichungen vom Plan, Fehler und Annahmen

**Abweichungen** (Plan bleibt unverändert in `plan.md`):
1. *Kalibrierung:* Plan-Regel nahm die Temperaturskalierung (T = 0,906) an. Nach Sicht der Testzahlen (schlechter: Log-Loss 0,8786 gegen 0,8722) verwendet der Simulator die Rohwahrscheinlichkeiten. Diese Entscheidung ist nicht unabhängig vom Test; der Test-Rollout lief nur mit Rohwerten.
2. *Hyperparameter-Gitter:* wie geplant; `max_bin` 127, Early Stopping 30.
3. *Pfadnormalisierung* (nicht im Plan, siehe Abschnitt 1) — nötig für Konsistenz Training/Simulator.
4. *Bekanntzeit von Ausfällen* nach dem Teacher-Forcing-Test geändert (Plan: Realzeit der Vorstation; neu: Realzeit des letzten gefahrenen Haltes), vor dem endgültigen Training.
5. *Eintritt:* Plan nannte Pasing/Leuchtenbergring; Züge treten auch in Laim, Hirschgarten, Ostbahnhof u. a. ein; Eintritt = erster gefahrener Halt.
6. *Läufe:* Prüfsteine 100 Läufe (Plan: z. B. 200; Prüfstein e nur 20 Läufe/Tag); Szenarien 100 Läufe und 2 Züge je Variante (Plan: z. B. 200 Läufe, „mehrere“ Züge). Der erste Szenarioanlauf mit 200 Läufen und 3 Zügen lief über 5 Stunden ohne Ergebnis und wurde abgebrochen; Ursache nicht geklärt.
7. *„+N min“* als absolute Eintrittsverspätung des Zuges (nicht additiv).
8. *Prüfstein e:* Plan sagte „im Bootstrap-KI der Sim“; umgesetzt als Spanne (2,5–97,5 %) der 20 Lauf-Mediane. Die Echt-Vergleichsgröße sind die Testtage (nicht das Jahr).
9. *Zeitdiskret* in Minuten statt streng ereignisgesteuert (alle Daten ganzminütig; gleiche Minute ist füreinander unsichtbar).
10. Zusätze: Merkmale `net_n15`, `plan_run`; Baseline V1b; zusätzliche Stratifizierungen.
11. Parallelisierung über Tage mit mehreren Prozessen war langsamer als ein Prozess mit 10 Threads; es lief 1 Prozess × 10 Threads.

**Fehler und Behebung.** (i) LightGBM fand unter macOS keine OpenMP-Bibliothek (kein Homebrew): `libomp.dylib` aus scikit-learn in `lightgbm/lib/` kopiert, install-name und rpath gesetzt, neu signiert (nur in der .venv). (ii) `delay` ist in Sekunden, nicht Minuten (früh erkannt). (iii) Bei Ausfällen sind `delay` und `time_real` nicht leer (Müllwerte), daher bei Ausfall ignoriert. (iv) Erstes Training lief auf nicht normalisierten Daten; nach dem Teacher-Forcing-Test (Abweichungen bei Netzfenstern und Ausfallzählern wegen Lücken/Wiederaufnahmen, wegen verspätet bekannter Ausfälle) Daten normalisiert, Bekanntzeit korrigiert, alles neu trainiert (erstes Modell verworfen). (v) Merge-Schlüssel-Fehler im Prüfskript behoben. (vi) Prüfstein-d-Zellen mit 0 Ausfällen in beiden Richtungen sind wegen kleiner n nicht auswertbar.

**Annahmen.** Zug = Verband; Ausfall absorbierend; Vorgänger = zuletzt vor T angekommener Zug an der Station; Ausfall erst nach Ankunft am letzten gefahrenen Halt bekannt; Eintritt/Replay mit echter Verspätung; Randklassen aus Trainings-Randverteilung; keine Fahrplanänderungen im Test berücksichtigt (insbesondere Dezember); Prüfsteintage zufällig, einmal gezogen.

## 10. Offene Fragen und Unsicherheiten

- Warum liegt der Zuwachs im Rollout in allen 9 Zellen über Echt (+0,12…+0,78 min)? Station-spezifische Ein-Schritt-Fehler der mittleren Δ-Vorhersage (±0,1–0,2 min, z. B. West Rang 5: Modell −0,36, Echt −0,16) und Verschiebung Training → Test sind plausible, aber nicht getrennt untersucht.
- Ob die Ausfallzähler (`cancel_net30` ist wichtigstes Ausfallmerkmal) im Rollout eine künstliche Rückkopplung erzeugen (Sim-Ausfälle erhöhen später den Zähler). Der Befund, dass die Sim-Ausfallquote insgesamt niedriger ist (2,03 % gegen 2,55 %), spricht nicht für eine Ausuferung, wurde aber nicht gezielt geprüft; an einem einzelnen Tag (2025-11-12, 40 Läufe, Zwischenversuch mit einem Notmodell) zeigten sich 3-mal so viele Ausfälle wie in den Daten (Zwischenversuch mit dem endgültigen Modell, 40 Läufe: 279 gegen 94 Ausfälle an modellierten Positionen; im 30-Tage-Lauf nicht wiederzufinden, Ursache dieser Tagesabweichung nicht untersucht).
- Ursache der 2,5-fach zu vielen Überholungen und 4-fach zu vieler gleicher Ankunftsminuten (vermutlich fehlende Abstandskopplung; nicht getestet).
- Ob Prüfstein e bei mehr Läufen/Tagen anders ausfällt: Klassen 10–20 und > 20 sind zu 80–91 % zensiert; Echt-Median in der oberen Klasse nicht definiert.
- Verallgemeinerung der Szenarien: 4 Simulationen je Variante, Rauschpegel ≈ Effekt bei +5 min.
- Die experimentelle Sperrung ist nur an 6 Fällen getestet und wird dort stark verfehlt.
- Wie sich die Ergebnisse mit anderem Trainingszeitraum (bis September) oder ohne Ausfallzähler verhalten.

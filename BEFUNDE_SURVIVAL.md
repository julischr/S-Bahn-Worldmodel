# BEFUNDE Stufe 3 — Survival-Analyse der Erholungszeit nach Störungen (Stammstrecke München, 2025)

Plan (vorab festgelegt, Abweichungen am Ende dokumentiert): `outputs/survival/auswertungsplan.md`.
Code: `src/analysis/survival/`. Alle Tabellen/Plots: `outputs/survival/`.
Stufe 1, VAR und Plausibilitätstests wurden nicht verändert.

## Zusammenfassung in fünf Sätzen

1. Nach einer Störungsmeldung an der Meldestation (12 Stammstreckenhalte, 14.205 auswertbare Fälle) dauert es im Median **rund 60 Minuten (KI 55–68)**, bis dort drei Züge in Folge weniger als 3 Minuten verspätet fahren; an ruhigen Vergleichszeitpunkten sind es etwa 9–12 Minuten.
2. Nur 14 % der Meldungen sind „ohne Wirkung" (die ersten drei Züge sind pünktlich), an ruhigen Vergleichszeitpunkten sind es 47–49 %.
3. Wie lange die Erholung dauert, hängt vor allem davon ab, **wie verspätet es schon vorher und in den ersten 15 Minuten ist** und in welche Richtung (Richtung West ist langsamer), und weit weniger vom Störungscode: Nach Berücksichtigung dieser Größen unterscheidet sich die Mehrzahl der Codes nicht von „Technische Störung am Zug" (Code 36).
4. Auch an ruhigen Zeitpunkten, die zufällig gerade verspätet sind, dauert die Erholung ähnlich lang wie nach einer Störung mit gleicher Verspätung; ein störungsspezifisches „Kippen" ließ sich nicht zeigen: Die Erholungszeit wächst mit der Schwere zwar nichtlinear, aber nicht überproportional (flacher werdend), während die Ausfallquote bei Verspätungen über 20 Minuten deutlich ansteigt (8,6 % nach Störung, 10,6 % im Placebo).
5. Alle Zahlen hängen stark daran, dass 61 % der Störungsfälle durch eine **neue** Störung unterbrochen („zensiert") werden und dass die Placebo-Zeitpunkte per Konstruktion ruhigere Zeiten sind; die Ganze-Strecke-Erholung (alle 12 Halte gleichzeitig) tritt innerhalb eines Betriebstags nur selten ein und ist kaum belastbar auswertbar.

## Datenbasis und Fallzahlen

| Schritt | Zahl |
|---|---|
| Ankünfte, 12 Stationen, Richtung bekannt | 3.546.812 |
| nach Zusammenfassen gekoppelter Verbände (36.696 Verbände aus 73.393 trip_ids; Regel: gleiche Planzeit an ≥ 2 gemeinsamen Stationen) | 3.187.273 Züge |
| davon auf ausgeschlossenen Betriebstagen (`run_all.EXCLUDED_DATES`) | 17.607 entfernt → **3.169.666 Züge**, 97.658 ausgefallene Halte (bleiben drin) |
| Auslöser-Meldungen (Codes 2–69 ohne 43/44/45/46/48) | 1.232.931 |
| **Episoden** (Stationscluster ≤ 15 min, über Stationen verkettet ≤ 15 min, ohne Mindestgröße) | **14.341** (Ost 7.254, West 7.087); 29 Codes ≥ 50 Episoden einzeln, 335–337 in „selten" |
| Placebo P1 (Plan) / P2 (zugverankert) | 52.136 / 49.862 von je 71.705 angeforderten (27–30 % ohne störungsfreien Gegenpart in ±42 Tagen) |

Codeliste mit Begründung: `auslosercodes.csv`. Code 23 (Betriebsstabilisierung) ist drin, laut Inventur eher Maßnahme als Ursache.
Auswertbar je Analyse (fehlende Züge nach t0 u. ä.): KM Meldestation 14.205, Strecke 14.265; Cox 10.504 (Strecke 9.245), da R15 Fälle mit Folgeepisode in den ersten 15 min und Fälle ohne Zug in den ersten 15 min ausschließt.

## (a) Belastbar

Kriterium: Gruppe ≥ 50 Episoden, Richtung gleich in den Sensitivitäten (360 s, ohne Folgezensierung, ≥ 3 Fahrten, Placebo P2), clusterrobuste/tageweise Inferenz.

**A1. Nach einer Auslöser-Meldung dauert die Erholung an der Meldestation länger als an ruhigen Vergleichszeitpunkten.** (`km_ergebnisse.csv`, Plot 1, 3)
- Schwelle 180 s, R0, Meldestation, alle Auslöser (n = 14.205): Median 60,7 min (54,6–68,1), Quartile 16,0 / 194,3; Placebo P1 11,5 (n = 48.354), P2 9,0. Differenz 49,2 min (42,4–57,0); P2: 51,7 (45,2–59,6). Robuster Log-Rank p < 1e-6.
- Gleiche Richtung in allen Sensitivitäten: ohne Folgezensierung Differenz 44,4 (38,0–51,5); ≥ 3 Fahrten (n = 5.235) 69,6 (60,7–84,3); Schwelle 360 s: Median 9,4 (8,6–10,4) vs. 3,1, Differenz 6,4 (5,5–7,3).
- Je Code: Differenz > 0 in 30 von 30 Gruppen (P1 und P2, 180 s). Holm-signifikant (robust, p < 0,05) in 28/30 (P1) bzw. 30/30 (P2). Nicht signifikant: Code 8 Notarzteinsatz (n = 119) und Code 54 (n = 61), also die kleinsten Gruppen.
- Die Streckenebene zeigt dieselbe Richtung (Störung länger als Placebo), ist aber annahmeabhängig (siehe B3).

**A2. „Ohne Wirkung" ist die Ausnahme.** 13,9 % der Meldungen (Schwelle 180 s; 43,0 % bei 360 s) gegenüber 47,3 % (P1) bzw. 48,8 % (P2) im Placebo (R0, Meldestation). Je Code zwischen 6 % (Code 3, n = 178) und 18 % (Codes 25, 42, 32).

**A3. Vorbelastung und Verspätung in den ersten 15 min bestimmen die weitere Erholung stark** (Cox, R15, Meldestation, 180 s, n = 10.504, 4.337 Ereignisse, clusterrobust nach Betriebstag; HR < 1 = langsamere Erholung): Vorbelastung je +5 min Verspätung HR 0,45 (0,40–0,51); Schwere je +5 min HR 0,73 (0,70–0,77). Gleiche Richtung bei 360 s (0,63 / 0,79) und ohne Folgezensierung (0,62 / 0,87).

**A4. Richtung West erholt sich langsamer als Ost** (Cox 180 s: HR 0,59 [0,55–0,64]; AFT: Zeitverhältnis 2,1 [1,9–2,3]; ohne Folgezensierung HR 0,52). Auf der Streckenebene ohne Folgezensierung: Median Ost 185 (167–206) vs. West 650 min (547–754), n = 7.230 / 7.035. **Aber:** bei 360 s ist der Unterschied praktisch weg (HR 0,94 [0,89–1,00]); er gilt für die strenge Pünktlichkeitsschwelle.

**A5. Wochenende erholt schneller** (Cox: Sa HR 1,19 [1,04–1,37], So/Feiertag 1,27 [1,11–1,47], jeweils vs. Mo–Do; AFT-Zeitverhältnis 0,74 / 0,66).

**A6. Der Zusammenhang Schwere → Erholung ist auch an Placebo-Zeitpunkten gleicher Verspätung da** (Plot 6, `kipppunkt_klassen_median.csv`). Median-Erholungszeit ab t0+15 min, Störung vs. Placebo P1: < 3 min: 15,2 vs. 2,0 (n = 1.358 / 21.267); 3–5: 23,4 vs. 19,0; 5–10: 58,8 vs. 48,0; 10–20: 121,1 vs. 90,7; > 20: 236,7 (124–n.e.) vs. 244,6 (216–266) (n = 608 / 1.027). Der Störungs-Aufschlag schrumpft mit wachsender Schwere. Das ist die Regression-zur-Mitte-Kontrolle: dieselbe Auswahlregel auf Zeitpunkte ohne Störung angewendet.

## (b) Hängt von Annahmen ab

**B1. Informative Zensierung.** 61,3 % der Episoden-Verläufe an der Meldestation enden, weil eine *andere* Auslöser-Episode am selben Halt/Richtung beginnt (nur 9,9 % ohne diese Regel; Placebo 14,7 %). Ein nicht erholtes Netz erzeugt neue Meldungen, die Zensierung ist daher wahrscheinlich informativ. Das KM-Ergebnis (Median 60,7) ändert sich ohne die Regel nur mäßig (56,0, KI 50,2–62,7), aber Schwere und Cox hängen von dieser Regel ab (Schwere-HR 0,73 mit vs. 0,87 ohne Regel; HVZ-HR 0,99 vs. 0,95). In den hohen Schwereklassen sind 77–85 % zensiert; die Mediane dort (und die KIs, teils „n.e.") sind entsprechend unsicher.

**B2. Placebo-Konstruktion.** Placebo-Zeitpunkte sind per Konstruktion frei von Auslöser-Meldungen am Halt in [−60, +240] min, also ruhigere Zeiten; dazu fehlt für 27–30 % der Episoden ein Gegenpart (eher dicht gestörte Zeiten). Der Vergleich ist eine Untergrenze, keine saubere Gegenfaktische. P1 (Plan) und das nachträglich ergänzte P2 (zugverankert, wegen Phasen-Artefakt zum Zugtakt) liefern ähnliche Ergebnisse (Differenz 49 vs. 52 min).

**B3. Ganze Strecke (alle 12 Halte).** Mit Folgezensierung 91,3 % zensiert (Median 313 min, KI 227–577; Placebo 93,5; n = 14.265); ohne: 34 % zensiert, Median 326 (295–364) vs. Placebo 101. Selbst das Placebo braucht ~1,5 h. Das liegt an der strengen Definition (jeder der 12 Halte muss gleichzeitig dreimal pünktlich sein). Das Cox-Modell auf der Streckenebene hat nur 963 Ereignisse und keine bestimmbare robuste Kovarianz; es wird nicht interpretiert.

**B4. Proportional-Hazards-Annahme ist verletzt** (Grambsch-Therneau, Hauptmodell: global χ² = 270, df = 49, p ≈ 2e-32; vor allem Vorbelastung χ² = 111, Schwere χ² = 28, Freitag). Größenordnung (Split bei 11,2 min): Vorbelastung HR 0,32 früh vs. 0,57 spät; Schwere 0,66 vs. 0,80, die Effekte wirken früh stärker. Stratifizieren nach Code ändert die Punktschätzer praktisch nicht (Vorbelastung 0,447, Schwere 0,731), und die Verletzung bleibt (χ² = 241, df = 20); das stratifizierte Modell hat keine bestimmbare robuste KI. AFT log-normal (AIC 42.849 vs. Weibull 43.377; Gradient am Optimum ≈ 2e-4): Zeitverhältnis je +5 min Schwere 1,50 (1,39–1,62), je +5 min Vorbelastung 2,05 (1,51–2,78). Die geschätzte Streuung ist groß (σ ≈ 4,8) bei 59 % Zensierung; die log-normale Form ist eine Annahme. **HR sind daher gemittelte Effekte über die Zeit, keine konstanten.**

**B5. Codeunterschiede.** Roh: Codes unterscheiden sich (k-Gruppen-Log-Rank χ² = 219, df = 28, robuster Wald χ² = 190, beide p < 1e-6; 62 von 406 Paaren Holm-signifikant). Langsamere Mediane (R0, 180 s): Code 3 Feuerwehr 132 min (n = 178), 57 Zusätzlicher Halt 120, 31 Bauarbeiten 114, 40 Stellwerk 114, 34 Signal 92 (n = 1.907); schnellere: 25 (26), 32 (32), 42 (34), 21 (35), 61 (37). Adjustiert (Cox, Ref. Code 36): nur Code 31 deutlich langsamer (HR 0,62 [0,52–0,76]), Code 40 0,72 (0,57–0,92), Code 34 0,86 (0,75–0,99); schneller 55 (1,26) und 67 (1,30), beides ohne Mehrfachtest-Korrektur bei 29 Vergleichen nicht belastbar. Die übrigen 24 Gruppen sind nicht von Code 36 unterscheidbar. Die rohen Unterschiede sind demnach größtenteils Unterschiede in Schwere/Vorbelastung, nicht im Code.

**B6. Meldestation.** Gegen Ostbahnhof (Ref.) etwas langsamer: Isartor 0,71, Karlsplatz 0,71, Rosenheimer Platz 0,81, Marienplatz 0,80, Pasing 0,87; schneller Laim 1,22, Hirschgarten 1,21, Leuchtenbergring 1,17. Konfundiert mit Richtung/Takt.

**B7. Kipppunkt, formal.** Plan-Regel für „überproportional": (i) robuster Wald-Test der Nichtlinearität p < 0,01 (AFT) UND (ii) Steigungsverhältnis oben/unten > 1 (Untergrenze des KI) UND (iii) nicht im Placebo.
- (i) erfüllt: Nichtlinearität Schwere Cox χ² = 39,6 (df 2), AFT χ² = 75,7, jeweils p < 1e-6; LR p < 1e-6.
- (ii) **nicht erfüllt:** Steigung der Median-Erholungszeit je Minute Schwere oben (10–20 → > 20) / unten (3–5 → 5–10): Störung 0,82 (KI −0,04…1,03), außerhalb HVZ 0,27 (−0,03…0,63), HVZ 0,47 (−0,13…0,81); Placebo P1 1,22 (0,95…1,54). Die Kurve wird bei Störungen flacher, nicht steiler. Im Cox-Partialeffekt (relativ zu 3 min) fällt log-HR in 4-min-Schritten um 0,33, 0,27, 0,15 (außerhalb HVZ) bzw. 0,46, 0,31, 0,15 (HVZ), also abnehmende Zuwächse.
- **Ergebnis laut Regel: „Nichtlinearität, aber kein belegtes überproportionales Wachstum der Erholungszeit".** Mit 77–85 % Zensierung in den oberen Klassen kann das nicht ausgeschlossen werden (KI-Obergrenzen „n.e.").

**B8. HVZ.** Haupteffekt HVZ im Cox ≈ 0 (HR 0,99 [0,93–1,07]; bei 360 s 1,11 [1,05–1,17]). Die Interaktion Schwere × HVZ ist signifikant (Cox χ² = 24,6, df 3; AFT χ² = 29,5; p < 1e-4): in der HVZ erholen sich schwere Fälle langsamer (Klasse 10–20: Median 158 [121–179] vs. 97 [71–122] min außerhalb; > 20: 237 vs. 126). Das gilt aber genauso für Placebo-Zeitpunkte (10–20: 125 vs. 58 min), also ist es kein störungsspezifischer Effekt, sondern das Netz in der HVZ insgesamt.

**B9. Ausfallquote nach t0** (Plot 7, `ausfallquote_schwereklassen.csv`; ausgefallene Halte in [t0+15, t0+135) an Meldestation/Richtung; Fallzahlen Störung 1.597 / 2.270 / 5.449 / 3.451 / 875): Störung 1,8 % / 2,4 % / 3,2 % / 4,5 % / 8,6 %; Placebo P1 0,2 % / 0,8 % / 1,4 % / 3,1 % / 10,6 %. Die Quote steigt in beiden stark mit der Schwere und verdoppelt sich zwischen den Klassen 10–20 und > 20 (Störung 1,9×, Placebo 3,4×), also ein nichtlinearer Anstieg, der aber an Placebo-Zeitpunkten gleicher Verspätung ebenfalls auftritt. Bei kleiner Schwere liegt die Störungsquote über dem Placebo, bei > 20 min nicht (KI überlappen knapp).

## (c) Offen

- **Zeitpunkt t0.** `update_timestamp` ist bei 31 % der Auslöser-Meldungen > 30 min nach der Zugzeit (nachträglich gesetzte Codes). Für die *erste* Meldung je Episode: Median 1,9 min, 90 %-Quantil 48 min, Extremwerte ±1.050 min. Die wahre Störungszeit kann deutlich vor t0 liegen; die Erholungszeiten gelten ab *Meldung*, nicht ab Störung.
- Ob die Codeunterschiede (Bauarbeiten langsamer) echt sind oder Schwere/Ort nicht vollständig fangen.
- Ob sich die Streckenerholung unter weniger strenger Definition (z. B. Anteil erholter Halte) anders verhält.
- Wie stark die informative Zensierung die Schwere-Effekte verzerrt (Schwere-HR 0,73 mit vs. 0,87 ohne Regel; keine Methode für abhängige Zensierung angewandt).
- Ob ein Kipppunkt jenseits von 20 min existiert: nur 608 Fälle in der Klasse, 85 % zensiert.
- Folgecodes (43/44/45/48) wurden bewusst nicht als Auslöser verwendet; ihr Beitrag zur Erholungsdauer ist hier nicht untersucht.
- Die Auswahl der Parameter (15-min-Fenster, Klassen, 3 Züge) wurde nicht variiert, außer Schwelle 180/360 s.

## Einordnung (Schritt 7) — nur Plausibilitätsabgleich

- **VAR-Halbwertszeit (~60 min Ost, ~120 min West):** Die Erholungszeit an der Meldestation (Median 56–61 min, Quartile 15 / ~195) liegt in derselben Größenordnung wie die Ost-Halbwertszeit; die Richtungsordnung (West langsamer) stimmt bei 180 s überein (HR 0,59), bei 360 s nicht (0,94). Die Streckenebene (Median 185 Ost / 650 West ohne Folgezensierung) ist die Zeit bis *alle* 12 Halte stabil pünktlich sind, keine Halbwertszeit; die Größenordnungen sind nicht direkt vergleichbar, nur die Richtung (West langsamer). „Passt zusammen" nur qualitativ.
- **Stufe-1-Dauerstratifizierung** (Meldedauer kurz < 30 / mittel 30–90 / lang > 90 min; ex post bekannt, nur beschreibend; `einordnung_dauerstrata.csv`): Median Meldestation 58,0 (52–65, n = 10.477) / 60,3 (54–75, n = 2.796) / 92,5 (73–112, n = 932). Kurz und mittel sind hier nicht unterscheidbar, nur „lang" ist deutlich länger. Stufe 1 fand einen klareren Dosis-Wirkungs-Zusammenhang in der Verspätungskurve; die Erholungszeit nach der Drei-Züge-Regel bildet das nur für lange Meldungen ab. Auf Streckenebene ist keine Abstufung erkennbar (Mediane 203–390, KIs sehr breit, 91 % zensiert).

## Abweichungen vom Plan (Details im Plan, Abschnitt „Abweichungen")

1. Störungslast: Versuch einer strengeren Definition verworfen (Auslöser war ein Einheitenfehler im eigenen Code), Plan gilt.
2. Placebo P2 zusätzlich (Phasen-Artefakt); Fehlerbehebung: Placebo-Tage außerhalb des Datenzeitraums.
3. Clusterrobuster Log-Rank-Score-Test statt Cox-Wald je Code (Rechenzeit).
4. Cox Strecke und stratifiziert ohne bestimmbare KI.
5. Klassenmitte in der Kipppunkt-Regel präzisiert.

Zahlen mit „p < 1e-6" sind numerisch kleiner als die Ausgabegenauigkeit der Tabellen.

## Plots

`outputs/survival/plots/`: 01 KM je Code (Meldestation, 180 s) · 02 dito Strecke · 03 Schwellen 180 vs. 360 · 04 Histogramm inkl. „ohne Wirkung" · 05 Forest-Plot Cox · 06 Erholungszeit je Schwereklasse nach HVZ · 07 Ausfallquote je Schwereklasse.


## Korrektur Datenausfall (nachträglich; alter Text oben unverändert)

**Ursache und Methode:** siehe `outputs/korrektur_ausfall/KORREKTUR_DATENAUSFALL.md`. Placebo-Zeitpunkte wurden zufällig gezogen und mussten „keine Auslöser-Meldung in [−60, +240] min“ erfüllen; in Datenausfallzeiten gibt es per Konstruktion keine Meldungen und alle Verspätungen sind 0, deshalb lagen viele Placebos dort (Erholung sofort). Neu: bereinigte Daten, außerdem Episoden und Placebos verworfen, deren Fenster [−60, +240] min einen Ausfall-Block berührt (97 von 14.336 Episoden; 5.587 von 46.123 P1, 1.263 von 41.194 P2). Neuer Stand (Meldestation, 180 s, R0, alle Auslöser):

| Befund | alt | neu |
|---|---|---|
| A1 Median Störung | 60,7 min (54,6–68,1), n = 14.205 | 61,4 (55,6–69,1), n = 14.105 |
| A1 Median Placebo P1 / P2 | 11,5 / 9,0 min | **43,6 / 37,0 min** (n = 37.890 / 39.251) |
| A1 Differenz Störung − Placebo P1 | 49,2 (42,4–57,0) | **17,8 (11,9–24,4)** |
| A1 Differenz Störung − Placebo P2 | 51,7 (45,2–59,6) | **24,4 (19,2–30,7)** |
| A2 „ohne Wirkung“ Störung | 13,9 % | 13,9 % |
| A2 „ohne Wirkung“ Placebo P1 / P2 | 47,3 / 48,8 % | **22,7 / 22,1 %** |
| A3–A5 Cox, Schwere-HR je +5 min | 0,732 | 0,733 |
| Vorbelastung-HR / West-HR | 0,449 / 0,594 | 0,441 / 0,592 |
| Sa-HR / So+Feiertag-HR | 1,194 / 1,274 | 1,202 / 1,285 |

A6 (Median Erholungszeit ab t0+15 min, Klassen < 3 / 3–5 / 5–10 / 10–20 / > 20 min):

| | Störung alt | Störung neu | Placebo P1 alt | Placebo P1 neu |
|---|---|---|---|---|
| Median (min) | 15,2 / 23,4 / 58,8 / 121,1 / 236,7 | 15,2 / 23,4 / 59,1 / 121,6 / 236,7 | 2,0 / 19,0 / 48,0 / 90,7 / 244,6 | 10,8 / 20,2 / 50,5 / 93,4 / 236,2 |
| n | 1.358 / 1.844 / 4.228 / 2.451 / 608 | 1.340 / 1.824 / 4.208 / 2.442 / 603 | 21.267 / 6.457 / 10.303 / 4.219 / 1.027 | 6.819 / 7.152 / 12.300 / 5.087 / 1.221 |

**Was das ändert:** (1) A1 und A2 bleiben qualitativ bestehen (Störung → längere Erholung, seltener „ohne Wirkung“), aber **die Effektgröße war stark überschätzt**: Differenz Faktor 2–3 kleiner, Abstand bei „ohne Wirkung“ 9 statt 33 Prozentpunkte. Die Aussage „60 Minuten gegen etwa 9–12“ in der Zusammenfassung ist so nicht haltbar; richtig ist etwa 61 gegen 37–44 Minuten. (2) Die Störungswerte selbst und alle Cox-Befunde ändern sich praktisch nicht. (3) A6: Die Placebo-Klasse < 3 min bestand zu zwei Dritteln aus Ausfallzeiten (n 21.267 → 6.819, Median 2,0 → 10,8 min). Der alte Befund „Der Störungs-Aufschlag schrumpft mit wachsender Schwere“ stützte sich u. a. auf diese Klasse: neuer Aufschlag (Störung − Placebo) 4,4 / 3,2 / 8,6 / 28,2 / 0,5 min, also **nicht monoton schrumpfend**, sondern bei Schwere 10–20 am größten und bei > 20 praktisch null (KIs breit, bei > 20 min 85 % zensiert). Die Folgerung „kein Kipppunkt, flacher werdend“ (B7) und die Regression-zur-Mitte-Kontrolle sind damit neu zu bewerten, nicht neu gerechnet. (4) Die Placebo-Konstruktion bleibt fragil: Nur Zeilen zu entfernen (ohne Fensterfilter) lieferte eine Differenz von 8,0 (1,9–15,0), mit Fensterfilter 17,8.

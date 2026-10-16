# BEFUNDE Stufe 4 — gelernter Simulator der Stammstrecke München (Test Okt–Dez 2025)

Plan (vorab, unverändert): `outputs/simulator/plan.md`. Code: `src/simulator/`. Alle Tabellen/Plots: `outputs/simulator/`. Ausführliche Textfassung für andere KI-Modelle: `outputs/simulator/UEBERGABE_KI.md`. Stufe 1, VAR, Plausibilitätstests und Survival wurden nicht verändert (nur gelesen).

## Zusammenfassung in fünf Sätzen

1. Ich habe ein Modell trainiert, das für jeden Zug an jeder der 12 Stammstreckenstationen vorhersagt, wie sich seine Verspätung bis zur nächsten Station ändert und ob er dort ausfällt, und daraus einen Simulator gebaut, der einen ganzen Betriebstag Zug für Zug „würfelt“, wobei die Züge einander über Abstand, Verspätung der Vorgänger und die Lage auf der Strecke beeinflussen.
2. Der einzelne Schritt wird deutlich besser vorhergesagt als mit einfachen Vergleichsregeln (Verspätungsfehler 0,41 min gegen 0,58 min bei der besten Vergleichsregel; Brier-Score für Ausfälle 17 % besser) und die Wahrscheinlichkeiten sind gut kalibriert.
3. Über ganze, nicht zum Training benutzte Tage gelingt es dem Simulator, die Verspätung am Streckenende zu treffen (93,6 % der echten Werte im 90-%-Band, Fehler des Medians 1,7 min gegen 2,9 min bei „Verspätung bleibt gleich“) und den Verspätungsaufbau über die Strecke (3,7 gegen 3,3 min); damit sind 2 von 5 Prüfsteinen bestanden.
4. Drei Prüfsteine sind **nicht bestanden**: der Simulator überschätzt den Verspätungszuwachs des Folgezugs in allen 9 ausgewerteten Zellen um 0,1–0,8 min, bildet die Zunahme der Ausfallquote mit der Verspätung des Vorgängers zu flach ab, und lässt Verspätungen nach Störungen schwerer Stärke deutlich länger anhalten als in den Daten; außerdem entstehen 2,5-mal so viele Überholungen und 4-mal so viele gleichzeitige Ankünfte wie in der Wirklichkeit.
5. Szenarien („Zug kommt mit +10 min“, „Fahrt entfällt“) liefern kleine, verrauschte Folgeeffekte (wenige Zug-Minuten Mehrverspätung), was zur Beobachtung passt, dass Züge ihre Verspätung mitnehmen statt sie weiterzugeben; die experimentelle Sperrung überschätzt die Verspätungen nach Wiederöffnung um das 3- bis 15-Fache und ist nicht brauchbar.

## Prüfstein-Tabelle (bestanden / nicht bestanden)

{{steine}}

Stichprobe: 30 zufällig gezogene Testtage (Seed 4, nach Wochentagstyp geschichtet, ohne Sperrtage), je 100 Läufe (Prüfstein e: 20 Läufe/Tag). Details und vollständige Zellentabellen unten und in `UEBERGABE_KI.md`.

## (a) Was der Simulator kann und wo er geprüft ist

**Datenbasis.** 3.546.812 Ankünfte an 12 Stationen; nach Zusammenfassen gekoppelter Verbände und Entfernen der Sperrtage 3.169.666 Zug-Station-Zeilen (wie Stufe 3); nach Pfadnormalisierung (siehe Abweichungen) 3.140.439 Zeilen von 295.621 Zügen. Daraus 2.751.924 Übergänge (Zug fuhr an k, Station k+1 existiert). Ausfall im Übergang: 4.873 (0,18 %). Aufteilung: Training Jan–Jul 1.576.674, Validierung Aug 229.623, Kalibrierung Sep 267.957, Test Okt–Dez 677.661 Übergänge.

**Modelle.** LightGBM, Ausfall binär (255 Bäume), Zuwachs 19 Klassen (−3 … +15 min, Ränder zusammengefasst; 143 Runden, 127 Blätter, min. 100 Zeilen je Blatt). 17 Merkmale, keine Störungscodes (Liste in `UEBERGABE_KI.md`). Wichtigste Merkmale (SHAP): Ausfall — Zahl der Ausfälle der letzten 30 min auf der Strecke (0,79), Verspätung an k−1 (0,47), Station (0,34); Zuwachs — Verspätung des Vorgängers an k+1, eigene Verspätung, Station, geplante Fahrzeit.

**Ein-Schritt-Prüfung (Test, n = 675.888 bzw. 677.661).** Verspätungsfehler (MAE) 0,409 min (Vergleich V1 „bleibt gleich“ 0,602; V2 Gruppenverteilung je Station/Richtung/Stunde 0,579); CRPS 0,306 (V2 0,432); Log-Loss 0,872 (V2 1,176). Ausfall: Brier 0,002131 (V2 0,002574, Skill +17,2 %; Ost +5,6 %, West +22,1 %), AUC 0,958. **Das Modell schlägt beide Vergleichsmodelle in allen Kennzahlen, auch je Richtung**; die Abbruchregel greift nicht.

**Kalibrierung.** Rohwahrscheinlichkeiten sind gut kalibriert (Test: ECE Ausfall 0,00045; ECE für „Zuwachs ≤ 0“ 0,0052, „≥ +2“ 0,0025, „≥ +5“ 0,0002). Die Ausfallwahrscheinlichkeit wird in den mittleren Klassen etwas unterschätzt (vorhergesagt 7,0 %, beobachtet 10,0 % im Bereich 5–10 %). Nachkalibrierung nach Plan-Regel (September): Temperatur 0,906 für die Zuwachsklassen angenommen, Platt für Ausfall abgelehnt. **Auf dem Test war die Temperaturskalierung schlechter** (Log-Loss 0,8786 statt 0,8722; ECE „≤ 0“ 0,0144 statt 0,0052). Beides ist berichtet; der Simulator verwendet die Rohwahrscheinlichkeiten (Abweichung, siehe unten).

**Rollout-Prüfsteine.**
- (a) Ankunft am Pfadende: Abdeckung des 90-%-Bandes 93,6 %, des 50-%-Bandes 65,4 %; Bias des Medians −0,11 min; MAE 1,71 min gegen 2,87 min (Persistenz „Eintrittsverspätung“). Ost/West und HVZ/Nicht-HVZ ähnlich (Abdeckung 93,3–93,9 %; MAE HVZ 1,88 gegen 3,25).
- (b) Verspätungsaufbau Rang 0↔11: Echt 3,31 min, Sim 3,69 min (+0,38; KI +0,14…+0,58). Der Referenzwert aus BEFUNDE_VAR 11.7 (≈ 2,8 min bei pünktlichem Vorgänger) liegt in derselben Größenordnung.
- Reproduktion der Richtung der Befunde 11.7/11.8 (Zuwachs und Ausfallquote steigen mit der Vorgängerverspätung) ist qualitativ vorhanden, quantitativ nicht ausreichend (c, d).

{{pruefstein_a}}

{{pruefstein_b}}

## (b) Wo der Simulator versagt oder an Annahmen hängt

- **Überschätzt den Zuwachs (c, b).** Alle 9 Zellen von Prüfstein c liegen im Sim über Echt (+0,12 … +0,78 min). Ein-Schritt-Prüfung zeigt: die Gesamt-Bias der mittleren Zuwachsvorhersage ist null (Modell 0,264, Echt 0,265 min), aber je Station/Richtung treten Abweichungen von ±0,1–0,2 min auf (z. B. West Rang 5: −0,36 gegen −0,16), die sich über 11 Etappen aufsummieren. Ob das an der Verschiebung Training (Jan–Jul) → Test (Okt–Dez, u. a. Fahrplanwechsel im Dezember) liegt, wurde **nicht** untersucht.
- **Ausfallkanal zu flach (d).** Sim-Quote in der Zelle „Abstand <5 min, Vorgänger ≤ 2 min“ 1,0 % (Echt 0,7 %), bei „>10 min“ 3,5 % (Echt 4,7 %). Insgesamt 2,03 % modellierte Positionen ausgefallen im Sim gegen 2,55 % in den Daten. Ausfälle am Eintritt werden aus den Daten übernommen und sind nicht modelliert; Wiederaufnahme der Fahrt nach Ausfall ist im Simulator nicht möglich.
- **Erholung nach Störungen (e).** Für Schwere 10–20 min Median 199 min (Sim, Spanne 160–n.e.) gegen 132 min (Echt, KI 100–n.e.); in der Klasse > 20 min erreicht der echte Median nie, im Sim in 45 % der Läufe nicht. Die oberen Klassen sind zu 80–90 % zensiert, Aussagen dort sind unsicher (Fallzahl 293 bzw. 75 Episoden).
- **Physikalische Plausibilität (nicht erzwungen, nur gezählt).** Überholungen benachbarter Züge zwischen zwei Stationen: Sim 2,13 % gegen Echt 0,85 % der Paare; gleiche Ankunftsminute aufeinanderfolgender Züge an einer Station: 5,04 % gegen 1,14 %; Abstand ≤ 1 min: 18,2 % gegen 9,0 %. Vermutete Ursache (nicht geprüft): die Ziehungen verschiedener Züge sind bedingt unabhängig, es gibt keine Abstandsregel. Kausalitätsklammern (Ankunft an k+1 frühestens 1 min nach k): 152.314 von über 20 Mio. simulierten Übergängen (0,6 %), vergleichbar mit dem Anteil in den Daten (0,64 % der Übergänge haben eine Fahrzeit ≤ 0 min).
- **Eintritt = Replay.** Was vor der Stammstrecke geschieht, wird nicht simuliert; Eintritt ist der erste Halt des Zuges, der laut Daten gefahren wurde (nicht nur Pasing/Leuchtenbergring).
- **Fahrgäste, Fahrzeugumläufe, Gleisbelegung, Störungsursachen sind nicht im Modell.** Effekte über vollere Folgezüge nach einem Ausfall sind unbekannt.
- **Test auf einem einzigen Zeitraum** (Okt–Dez 2025, 30 Tage) mit einmalig gewählter Stichprobe.

## (c) Szenarien (100 Läufe, gleiche Zufallszahlen für Basis und Szenario)

Zwei Mittwoche (2025-11-12, 2025-12-03), je HVZ und Nicht-HVZ, je 2 Beispielzüge der Linie S3 Richtung Ost am Eintritt Pasing (pünktlich, ≤ 3 min), Fenster 60 min vor bis 6 h nach dem Eingriff. „+N“ heißt: der Zug betritt die Strecke mit N min (absolut); „betroffen“ = Änderung der Verspätung am Pfadende ≥ 1 min im Lauf; Abklingzeit siehe `UEBERGABE_KI.md` Abschnitt 7.

{{szenario_kompakt}}

Lesart: Selbst bei +20 min ändern sich andere Züge im Mittel nur um etwa +0,4 min (HVZ), die Gesamt-Mehrverspätung aller anderen Züge liegt bei 6,5 Zug-Minuten (außerhalb HVZ) bis 12,0 (HVZ) bei einem Lauf-Streubereich, der bei +5/+10 min die 0 einschließt; zusätzliche Ausfälle sind im Mittel nicht von 0 unterscheidbar. Der Rauschpegel (Anzahl „betroffener“ Züge bei +5 min: 7,6 nicht-HVZ) ist so groß wie der Effekt, weil schon kleine Wahrscheinlichkeitsänderungen die gezogenen Klassen zufällig umkippen lassen. Mit nur 2 Zügen je Variante ist die Streuung zwischen Zügen (Spalten „min./max. Zug“) groß. Das **Ausfall-Szenario** ist von der Größe her kaum von Rauschen zu trennen (zusätzliche Ausfälle im Mittel +1,8 außerhalb HVZ bzw. +0,6 in der HVZ bei Streuung q05…q95 von etwa −6…+15, Gesamt-Mehrverspätung +2,3 bzw. +0,9 Zug-Min.). Hinweis: Fahrgastzahlen sind nicht in den Daten, Effekte über vollere Folgezüge kann das Modell nicht kennen.

**Sperrung (experimentell).** 6 echte Fälle in Okt–Dez (> 20 min kein Zug an der Station, ≥ 8 Züge an der Vorstation in der Lücke):

{{sperrung_test_szenarien}}

Echte mittlere Verspätung der Züge in den 60 min nach Wiederöffnung: 2,8–8,0 min (ein Fall 20,5). Sim mit Sperrungsregel: 23–55 min; Basislauf ohne Sperrung: 2,5–7,6 min. Die Regel („alle warten, keiner fällt aus, 2 min Abstand beim Abfluss“) ist in den echten Fällen falsch; die realen Fälle enthalten offenbar Ausfälle/Wenden, die hier nicht vorkommen, und das Modell hat solche Lagen kaum gesehen. **Nicht verwenden.**

## Darstellung und Bedienung

Bildfahrplan `outputs/simulator/plots/08_bildfahrplan_basis.png` und `..._verspaetung10.png` (Richtung Ost, 07:55–08:50, 2025-11-12; Plan grau, echt schwarz, Sim-Median rot, 90-%-Band rosa, Basislauf gestrichelt). Weitere Plots: `01_reliability`, `02_shap`, `03_pruefstein_a_b`, `04_pruefstein_c_d`, `05_pruefstein_e`, `06_profil`, `07_szenario_*`. Funktion: `from src.simulator.api import simulate_day` bzw. CLI `uv run python -m src.simulator.api 2025-11-12 --art verspaetung --linie S3 --uhrzeit 08:00 --minuten 10 --plot x.png`.

## Offen

- Ursachen der Fehlschläge c/d/e wurden nicht experimentell getrennt (z. B. Modell ohne Ausfallzähler, Training bis September, Abstandsregel).
- Ob die Tests mit 200 statt 100 Läufen oder anderen Tagen anders ausfielen.
- Szenarien nur an 2 Tagen × 2 Zügen je Variante; Verallgemeinerung offen.
- Ob eine Abstandsregel die Überholungen senkt, ohne die bestandenen Prüfsteine zu verschlechtern.

## Abweichungen vom Plan (Auszug; vollständig in `UEBERGABE_KI.md` Abschnitt 9)

1. Simulator verwendet Rohwahrscheinlichkeiten statt der nach Plan-Regel gewählten Temperaturskalierung (Entscheidung nach Sicht auf Testergebnisse der Ein-Schritt-Prüfung).
2. Szenarien mit 100 statt 200 Läufen, 2 statt 3 Zügen je Variante; erster Anlauf mit 200 Läufen nach über 5 h ohne Ergebnis abgebrochen (Ursache nicht geklärt).
3. Pfadnormalisierung (längstes zusammenhängendes Segment, Ausfall absorbierend): 29.227 Zeilen entfernt, 6.408 Wiederaufnahmen verworfen (2.027 Züge).
4. Eintritt am ersten gefahrenen Halt statt nur Pasing/Leuchtenbergring.


## Korrektur Datenausfall (nachträglich; alter Text oben unverändert)

Im Echtzeitfeed fehlen 21 volle Tage (v. a. September 2025) und viele Teilblöcke (Zeilen mit delay = 0, keine Ausfälle, keine Codes; Details `outputs/korrektur_ausfall/KORREKTUR_DATENAUSFALL.md`). Folgen für Stufe 4: (1) Der **Kalibrierungsmonat September war überwiegend Datenausfall** (Ausfallrate 0,038 % statt ca. 0,2 %): die Temperaturskalierung (T = 0,906) wurde auf zum Teil leeren Daten gefittet; das erklärt, warum sie auf dem Test schlechter war. Die Begründung „untypischer Monat“ ist damit unvollständig. Der Simulator nutzt Rohwahrscheinlichkeiten (weder davon noch von Aug/Jan–Jul stark betroffen). (2) Das Training (Jan–Jul) enthält Ausfallzeilen (2025-01-11/12/13, 01-30, 03-19/20, 04-09 u. a., ca. 2 % der Zeilen (alle Stationen)); der Testzeitraum Okt–Dez ca. 1,2 % (v. a. 2025-10-03 und Teilblöcke). Die Prüfsteintage wurden nicht auf Ausfallblöcke geprüft. (3) Der Vorgängerbefund „ruhigste Stunden“ (`ruhige_zeitraeume.csv`) wurde bereits ohne Ausfallblöcke gesucht. **Nicht neu gerechnet:** Training, Ein-Schritt-Prüfung und Prüfsteine mit bereinigten Daten; mit `S_BAHN_AUSFALL_FILTER=1` ist das möglich (Simulator liest dann ebenfalls `final_bereinigt`). Bis dahin sind die Zahlen dieses Berichts als nicht bereinigt zu lesen.

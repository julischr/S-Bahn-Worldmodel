# Plan Stufe 4 — gelernter Simulator der Stammstrecke (vorab festgelegt)

Stand: vor dem ersten Training. Abweichungen werden unten im Abschnitt „Abweichungen“ **angehängt**, dieser Plan wird nicht rückwirkend geändert.
Code: `src/simulator/`, Ausgaben: `outputs/simulator/`, Befunde: `BEFUNDE_SIMULATOR.md`, Übergabe: `outputs/simulator/UEBERGABE_KI.md`.
Stufe 1, VAR, Plausibilitätstests und Survival werden nur gelesen (Importe von `EXCLUDED_DATES`, Stationsliste, Funktionen der Survival-Erholungsmessung), nie verändert.

## 1. Datensatz

- Quelle: `data/02_events/final/*.parquet`, nur `is_arrival = true`, 12 Stammstreckenstationen (`src/analysis/var/stations.py`), `richtung` bekannt. Nur beobachtete Werte (`delay`, `time_real`, `is_cancelled`), keine DB-Prognosen (`events_progression` wird nicht gelesen).
- Einheit „Zug“: gekoppelte Verbände (gleiche Planzeit an ≥ 2 gemeinsamen Stationen, Regel wie Stufe 3) = ein Zug. Betriebstage aus `run_all.EXCLUDED_DATES` entfernt. Erwartung: 3.169.666 Zug-Station-Zeilen (Stufe-3-Zahl).
- `delay` in den Rohdaten ist in Sekunden; hier in ganzen Minuten.
- Pfad eines Zuges = seine geplanten Stammstreckenhalte in Fahrtrichtung. Übergang = Paar benachbarter Halte (Rang-Differenz genau 1) k → k+1, bei dem der Zug an k nicht ausgefallen ist.
- Zielgrößen: (a) `cancel_next` = Zug fällt an k+1 aus; (b) `dclass` = Verspätungszuwachs delay(k+1) − delay(k) in Minuten, nur wenn der Zug an k+1 fährt; Klassen −3 … +15 in Minutenschritten, Ränder zusammengefasst (≤ −3, ≥ +15), also 19 Klassen.
- Zeitaufteilung: Training Jan–Jul 2025, **Validierungsmonat August** (Early Stopping, Hyperparameter), **Kalibrierungsmonat September** (Nachkalibrierung), Test Okt–Dez 2025 (nie zum Anpassen verwendet). Begründung: der Auftrag nennt „Training Jan–Sep“ und „Hyperparameter nur auf einem Validierungsmonat innerhalb des Trainingszeitraums“; Aug/Sep liegen im Trainingszeitraum, werden aber nicht für die Baumanpassung benutzt, damit Kalibrierung ehrlich ist (kein Refit).

## 2. Merkmale (Stichtag: Ankunft des Zuges an k, Zeit T; nur Ereignisse mit Realzeit < T)

Alle Merkmale zur Entscheidung „was passiert zwischen k und k+1“. Auf Minutenebene: Ereignisse in derselben Minute T sind füreinander **nicht** sichtbar (strikt `<`).

| Merkmal | Definition |
|---|---|
| d_k | eigene Verspätung an k (min) |
| d_km1 | eigene Verspätung an k−1 (NaN am Eintritt) |
| rank, richtung, line | Station (0 Pasing … 11 Leuchtenbergring), Richtung, Linie (S1–S8, sonst „other“) |
| hour, hvz, wtype | Ortszeit von T in Stunden, HVZ (6–9, 15–19 Uhr), Wochentagstyp (Mo–Do, Fr, Sa, So/Feiertag) |
| plan_gap_pred | geplanter Abstand zum Vorgänger an k (Planzeit-Differenz, min; kann negativ sein bei Reihenfolgetausch) |
| act_gap_pred | tatsächlicher Abstand zum Vorgänger an k (min) |
| d_pred_k | Verspätung des Vorgängers an k |
| d_pred_k1 | Verspätung des Vorgängers an k+1, NUR wenn er dort Realzeit < T angekommen ist, sonst NaN |
| net_mean15 | mittlere Verspätung aller Ankünfte (12 Stationen, beide Richtungen) mit Realzeit in [T−15, T) |
| net_n15 | Zahl dieser Ankünfte (Verkehrsdichte; **Ergänzung**) |
| cancel_sd30, cancel_net30 | Zahl ausgefallener Halte (Station+Richtung bzw. gesamte Stammstrecke) mit „Bekanntzeit“ in [T−30, T) |
| plan_run | geplante Fahrzeit k → k+1 (min; **Ergänzung**, Fahrplanwissen, kein Leck) |

Vorgänger = der Zug, der an (Station, Richtung) zuletzt vor T angekommen ist (nicht ausgefallen, Realminute < T). Das ist im Simulator ohne Zusatzannahme identisch berechenbar. Bekanntzeit eines Ausfalls an Station s = max(Planzeit s, Realzeit der Ankunft des Zuges an der Vorstation), wenn er an der Vorstation fuhr, sonst Planzeit s. **Keine Störungscodes** als Merkmal.

## 3. Modell

- LightGBM, `multiclass` (19 Klassen) für dclass, `binary` für cancel_next; Linie als kategorisch.
- Kleines festes Gitter (num_leaves ∈ {31, 127} × min_data_in_leaf ∈ {100, 1000}, Lernrate 0,08, feature_fraction 0,8, bagging 0,8, lambda_l2 = 5), Auswahl nach Log-Loss auf August, Early Stopping auf August (max. 600 Runden).
- Kalibrierung auf September: Reliability-Tabellen vorher; Nachkalibrierung (Cancel: isotonisch; Klassen: Temperaturskalierung) nur wenn sie Log-Loss/ECE auf September verbessert; auf Test beides berichtet.
- SHAP (TreeExplainer) auf Stichprobe der Testdaten (20.000 Zeilen), Top-15 je Modell.

## 4. Ein-Schritt-Prüfung (Test Okt–Dez)

Vergleich: (V1) Verspätung bleibt gleich (Δ = 0; Ausfall: globale Trainingsrate), (V2) mittlerer Zuwachs bzw. empirische Verteilung je Station/Richtung/Stunde (Trainingsdaten; Ausfall: Rate je Station/Richtung/Stunde). Zusatz V1b: Persistenz mit globaler empirischer Δ-Verteilung (für Log-Loss, da Punktmasse unendliche Log-Loss hat).
Kennzahlen: MAE der Verspätung an k+1 (Punktprognose = Median der Verteilung; V2: Median bzw. Mittel je Gruppe, beides berichtet), Log-Loss, mittlere RPS/CRPS der Δ-Verteilung (über Klassen, Einheit min), Brier-Score und Log-Loss für Ausfall; je Richtung und gesamt.
**Abbruchregel:** Schlägt das Modell V2 nicht bei MAE, CRPS und Brier (gesamt), wird nicht weitergebaut, bevor die Ursache gefunden ist.

## 5. Simulator

- Zeitdiskret in Minutenschritten (alle Rohdaten sind ganzminütig), **alle Läufe eines Tages parallel** (Vektoren über Läufe × Züge). Je Minutenschritt werden alle Ankünfte dieser Minute aus Merkmalen gebildet, die nur Zustand < T benutzen, dann je Ereignis Ausfall und Δ gezogen. Das entspricht der ereignisgesteuerten Reihenfolge mit minutengenauer Auflösung.
- Eintritt: jeder Zug beginnt am **ersten Halt seines Pfades, an dem er laut Daten fuhr**, mit seiner tatsächlichen Ankunftsverspätung (Replay). Das sind nicht nur Pasing/Leuchtenbergring (viele Züge kommen in Laim, Hirschgarten, Ostbahnhof usw. hinein); wird dokumentiert. Zeigt die Spalte für einen Zug von Anfang an Ausfall (nie gefahren), bleibt er ausgefallen (Replay, keine Modellierung; zählt für Ausfall-Merkmale).
- Fährt ein Zug in den Daten nach einer Lücke im Pfad (nicht benachbarte Halte), wird der Pfad nach der Lücke abgeschnitten; Zahl wird berichtet.
- Ausfall an k+1 ist absorbierend (Zug endet), Ausfälle bekannt zur Bekanntzeit (s. o.).
- Common random numbers: je (Lauf, Zug, Pfadposition) feste Gleichverteilte (U_cancel, U_klasse, U_rand), identisch in Basis und Szenario; Ziehung per Inversion der Verteilung.
- Randklassen: ≥ +15 bzw. ≤ −3 werden aus der empirischen Randverteilung der Trainingsdaten nachgezogen.
- Kausalität: Ankunftszeit an k+1 mindestens 1 min nach Ankunft an k (Klammer wird gezählt und berichtet). Reihenfolge/Abstände werden **nicht** erzwungen, nur gezählt (Überholungen: Reihenfolge der Ankunftsminuten an k+1 gegen k vertauscht; „zu enger Abstand“: Abstand aufeinanderfolgender Ankünfte an einer Station 0 min, im Vergleich zur Echtquote).
- Testtage: 30 zufällig gezogene Testtage (Seed 4, nach Wochentagstyp geschichtet; ohne `EXCLUDED_DATES`), je 100 Läufe für die Prüfsteine (Ziel „z. B. 200“ wird aus Rechenzeitgründen für Szenarien erreicht, nicht für alle Prüfsteintage; ggf. Abweichung dokumentiert).

## 6. Prüfsteine und Toleranzen (Ergebnis je Prüfstein: bestanden / nicht bestanden)

Alle auf den simulierten Testtagen gegen die Echtdaten desselben Tages. Run-Streuung = Bandbreite über die Läufe, Konfidenzintervalle über Betriebstage (Bootstrap, 1000 Ziehungen).

- **(a) Ankunft am Streckenende.** Zug = in den Echtdaten am Pfadende gefahren. Kriterien, **alle** müssen gelten: (a1) Anteil der echten Werte im 90-%-Band [q05, q95] der Simulation (Läufe, in denen der Zug nicht ausfällt) in [0,85; 0,95]; (a2) |mittlere Abweichung Median(Sim) − Echt| ≤ 0,5 min; (a3) MAE(Median Sim) ≤ MAE(Persistenz „Verspätung am Eintritt“).
- **(b) Verspätungsaufbau über den Tunnel.** Züge mit Pfad Pasing ↔ Leuchtenbergring (Rang 0 und 11). Mittelwert delay(Ende) − delay(Eintritt). Bestanden, wenn |Sim − Echt| ≤ 0,5 min (Referenz BEFUNDE_VAR 11.7: ≈ 2,8 min bei pünktlichem Vorgänger, 4,5 min bei > 10 min).
- **(c) Zuwachs des Folgezugs (Tabelle 11.7, ganze Kernstrecke).** Zellen Vorgängerverspätung am Eintritt (≤ 2, 2–5, 5–10, > 10 min) × geplanter Abstand (< 5, 5–10, 10–15, 15–20, 20–60 min) wie in 11.6/11.7 definiert. Bestanden, wenn in ≥ 75 % der Zellen mit n_Echt ≥ 200 |Sim − Echt| ≤ 0,3 min **und** je Zeile mit gültigen Endzellen der Mittelwert der Klasse > 10 min größer ist als der der Klasse ≤ 2 min (Vorzeichen des Gradienten wie in Echt).
- **(d) Ausfallquote nach Vorgänger und Abstand (Tabelle 11.8).** Hauptauswertung nur an **modellierten Übergängen** (Zug fuhr an k, Ausfall an k+1), weil Ausfälle am Eintritt aus den Daten kopiert werden; die volle 11.8-Definition wird zusätzlich berichtet. Bestanden, wenn in ≥ 75 % der Zellen mit n_Echt ≥ 500 |Sim − Echt| ≤ max(0,3 Prozentpunkte; 40 % der Echtrate) **und** die Rate in der Zeile „< 5 min“ mit der Vorgängerklasse monoton steigt (≤ 2 < 2–5 < 5–10 < > 10 min) wie in Echt.
- **(e) Erholungszeit je Schwereklasse.** Dieselbe Messung wie Survival (`recovery.measure`, R15, 180 s, Meldestation, Standardzensierung) an den echten Episodenzeitpunkten der Testtage, angewendet (i) auf Echtdaten, (ii) auf jeden simulierten Lauf (Schwere und Erholung aus simulierten Zügen). Schwereklassen < 3, 3–5, 5–10, 10–20, > 20 min. Bestanden, wenn für Klassen mit n ≥ 50 (Echt) mindestens 4 von 5 Klassen |Median Sim − Median Echt| ≤ max(25 % Echt; 5 min) erfüllen oder der Echt-Median im Bootstrap-KI der Sim liegt, **und** die Mediane mit der Schwere monoton steigen. Vergleichsgröße Plot 6 (BEFUNDE_SURVIVAL A6) wird mitgeführt (Jahr, andere Tage, daher nur qualitativ).

Zusätzlich berichtet (kein Bestehenskriterium): Überholungs- und Abstandsquoten Sim gegen Echt, Zahl der Kausalitätsklammern, Ausfallquote insgesamt Sim gegen Echt.

## 7. Szenarien

Gleiche Tage, gleiche Zufallszahlen für Basis und Szenario; Ausgaben je Zug (Verspätungsdifferenz Szenario − Basis), zusätzliche Ausfälle, Zeit bis zum „kein Unterschied mehr“ (Mittel der Differenz über die Läufe, 3 Betriebs-Zeitfenster in Folge < 0,5 min bzw. Differenzen in der Gesamtheit nicht mehr von 0 unterscheidbar nach Bootstrap; genaue Regel im Code dokumentiert).
- (a) Zug X betritt mit +N min (N = 5, 10, 20), HVZ und Nicht-HVZ, mehrere Beispielzüge je Variante (S3 Ost Pasing u. a.).
- (b) Fahrt X gestrichen. Hinweis: keine Fahrgastzahlen in den Daten.
- (c) EXPERIMENTELL Sperrung: Station s von t1 bis t2 nicht passierbar, Züge warten an der Vorstation (Ankunft an s frühestens t2 + Fahrzeit, Folgezüge im Mindestabstand 2 min), danach Modell. Test an echten Fällen > 15 min ohne Zug an einer Station (Suche in den Daten).

## 8. Darstellung

Bildfahrplan (Zeit-Weg, Plan / Echt / Sim-Band) für einen Beispieltag, Differenz-Plots je Szenario, `simulate(datum, szenario)` und CLI in `src/simulator/api.py`.

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

| tag | n_sim | n_train | n_both | nur_sim | nur_train | diff_d_k | diff_d_km1 | diff_richtung | diff_line | diff_hour | diff_hvz | diff_wtype | diff_plan_gap_pred | diff_act_gap_pred | diff_d_pred_k | diff_d_pred_k1 | diff_net_mean15 | diff_net_n15 | diff_cancel_sd30 | diff_cancel_net30 | diff_plan_run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-10-07 | 8.331 | 8.331 | 8.275 | 56 | 56 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 32 | 78 | 32 | 46 | 896 | 890 | 0 | 0 | 0 |
| 2025-11-15 | 2.431 | 2.431 | 2.421 | 10 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 31 | 20 | 19 | 48 | 48 | 0 | 4 | 0 |
| 2025-12-03 | 9.352 | 9.352 | 9.267 | 85 | 85 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 61 | 109 | 56 | 74 | 1.405 | 1.361 | 0 | 0 | 0 |

**LightGBM.** Gemeinsam: Lernrate 0,08, feature_fraction 0,8, bagging_fraction 0,8 (bagging_freq 1), lambda_l2 5, max_bin 127, seed 7; Gitter num_leaves ∈ {31, 127} × min_data_in_leaf ∈ {100, 1000}, Early Stopping (30 Runden, max. 600) auf August, Auswahl nach Log-Loss. Ausfall (binary): gewählt 31 Blätter, min_data 100, 255 Bäume. Zuwachs (multiclass 19): gewählt 127 Blätter, min_data 100, 143 Runden. Gitter-Ergebnisse (Val-Log-Loss August):

| num_leaves | min_data_in_leaf | best_iter | val_logloss | sec |
|---|---|---|---|---|
| 31 | 100 | 255 | 0,00716 | 9 |
| 31 | 1.000 | 205 | 0,00723 | 8 |
| 127 | 100 | 125 | 0,00724 | 9 |
| 127 | 1.000 | 104 | 0,00718 | 9 |

| num_leaves | min_data_in_leaf | best_iter | val_logloss | sec |
|---|---|---|---|---|
| 31 | 100 | 502 | 0,91222 | 323 |
| 31 | 1.000 | 489 | 0,91456 | 312 |
| 127 | 100 | 143 | 0,91092 | 175 |
| 127 | 1.000 | 143 | 0,91325 | 189 |

## 2. Ein-Schritt-Prüfung (Test Okt–Dez 2025)

Vergleichsmodelle (angepasst auf Training Jan–Jul): V1 = Verspätung bleibt gleich (Ausfall: globale Rate 0,168 %); V1b = Persistenz + globale Δ-Verteilung (nur für Log-Loss/CRPS); V2 = empirische Δ-Verteilung bzw. Ausfallrate je (Station, Richtung, Stunde), mit Glättung (5 Pseudobeobachtungen aus der globalen Verteilung). MAE/RMSE/Bias der Verspätung an k+1 in min; Punktprognose des Modells = d_k + Median (bzw. Mittel) der Klassenverteilung; CRPS in min über die 18 Klassenschwellen (beobachtetes Δ auf [−3, 15] begrenzt); Log-Loss in nats. n = Übergänge mit Fahrt an k+1.

**2a. Verspätung, Rohwahrscheinlichkeiten (diese Variante nutzt der Simulator):**

| gruppe | modell | n | mae | rmse | bias | crps | logloss |
|---|---|---|---|---|---|---|---|
| gesamt | LightGBM (Median) | 675.888 | 0,4086 | 1,0219 | -0,0618 | 0,3062 | 0,8722 |
| gesamt | LightGBM (Mittel) | 675.888 | 0,4845 | 0,9810 | -0,0016 | 0,3062 | 0,8722 |
| gesamt | V1 Verspaetung bleibt (Delta=0) | 675.888 | 0,6019 | 1,1967 | -0,2653 | 0,5949 | – |
| gesamt | V1b Persistenz + globale Delta-Verteilung | 675.888 | 0,6019 | 1,1967 | -0,2653 | 0,4687 | 1,2653 |
| gesamt | V2 Gruppenverteilung (Median) | 675.888 | 0,5791 | 1,1626 | -0,1581 | 0,4317 | 1,1764 |
| gesamt | V2 Gruppenmittel (Punkt) | 675.888 | 0,6265 | 1,1273 | -0,0349 | 0,4317 | 1,1764 |
| Ost | LightGBM (Median) | 337.715 | 0,4202 | 1,0655 | -0,0754 | 0,3134 | 0,8915 |
| Ost | LightGBM (Mittel) | 337.715 | 0,4933 | 1,0212 | -0,0047 | 0,3134 | 0,8915 |
| Ost | V1 Verspaetung bleibt (Delta=0) | 337.715 | 0,5815 | 1,1976 | -0,2538 | 0,5728 | – |
| Ost | V1b Persistenz + globale Delta-Verteilung | 337.715 | 0,5815 | 1,1976 | -0,2538 | 0,4536 | 1,2401 |
| Ost | V2 Gruppenverteilung (Median) | 337.715 | 0,5792 | 1,1967 | -0,2152 | 0,4356 | 1,1903 |
| Ost | V2 Gruppenmittel (Punkt) | 337.715 | 0,6403 | 1,1581 | -0,0345 | 0,4356 | 1,1903 |
| West | LightGBM (Median) | 338.173 | 0,3970 | 0,9764 | -0,0482 | 0,2990 | 0,8529 |
| West | LightGBM (Mittel) | 338.173 | 0,4758 | 0,9391 | 0,0015 | 0,2990 | 0,8529 |
| West | V1 Verspaetung bleibt (Delta=0) | 338.173 | 0,6222 | 1,1959 | -0,2769 | 0,6170 | – |
| West | V1b Persistenz + globale Delta-Verteilung | 338.173 | 0,6222 | 1,1959 | -0,2769 | 0,4837 | 1,2905 |
| West | V2 Gruppenverteilung (Median) | 338.173 | 0,5791 | 1,1275 | -0,1011 | 0,4277 | 1,1625 |
| West | V2 Gruppenmittel (Punkt) | 338.173 | 0,6128 | 1,0957 | -0,0353 | 0,4277 | 1,1625 |

**2b. Ausfall, Rohwahrscheinlichkeiten** (n = alle Test-Übergänge; Brier-Skill = 1 − Brier / Brier(V2)):

| gruppe | modell | n | n_cancel | rate | brier | brier_skill_vs_V2 | logloss | auc |
|---|---|---|---|---|---|---|---|---|
| gesamt | LightGBM | 677.661 | 1.773 | 0,002616 | 0,002131 | 0,172171 | 0,010111 | 0,958226 |
| gesamt | V1 globale Rate | 677.661 | 1.773 | 0,002616 | 0,002610 | -0,014096 | 0,018395 | 0,500000 |
| gesamt | V2 Rate je Station/Richtung/Stunde | 677.661 | 1.773 | 0,002616 | 0,002574 | 0,000000 | 0,015472 | 0,861567 |
| Ost | LightGBM | 338.229 | 514 | 0,001520 | 0,001429 | 0,055536 | 0,008015 | 0,922303 |
| Ost | V1 globale Rate | 338.229 | 514 | 0,001520 | 0,001517 | -0,002551 | 0,011388 | 0,500000 |
| Ost | V2 Rate je Station/Richtung/Stunde | 338.229 | 514 | 0,001520 | 0,001514 | 0,000000 | 0,011211 | 0,739309 |
| West | LightGBM | 339.432 | 1.259 | 0,003709 | 0,002830 | 0,220618 | 0,012198 | 0,972352 |
| West | V1 globale Rate | 339.432 | 1.259 | 0,003709 | 0,003700 | -0,018891 | 0,025377 | 0,500000 |
| West | V2 Rate je Station/Richtung/Stunde | 339.432 | 1.259 | 0,003709 | 0,003631 | 0,000000 | 0,019718 | 0,900314 |

**2c. Mit Temperaturskalierung (T = 0,906, auf September gefittet):**

| gruppe | modell | n | mae | rmse | bias | crps | logloss |
|---|---|---|---|---|---|---|---|
| gesamt | LightGBM (Median) | 675.888 | 0,4087 | 1,0235 | -0,0684 | 0,3069 | 0,8786 |
| gesamt | LightGBM (Mittel) | 675.888 | 0,4766 | 0,9815 | -0,0266 | 0,3069 | 0,8786 |
| gesamt | V1 Verspaetung bleibt (Delta=0) | 675.888 | 0,6019 | 1,1967 | -0,2653 | 0,5949 | – |
| gesamt | V1b Persistenz + globale Delta-Verteilung | 675.888 | 0,6019 | 1,1967 | -0,2653 | 0,4687 | 1,2653 |
| gesamt | V2 Gruppenverteilung (Median) | 675.888 | 0,5791 | 1,1626 | -0,1581 | 0,4317 | 1,1764 |
| gesamt | V2 Gruppenmittel (Punkt) | 675.888 | 0,6265 | 1,1273 | -0,0349 | 0,4317 | 1,1764 |
| Ost | LightGBM (Median) | 337.715 | 0,4207 | 1,0672 | -0,0820 | 0,3143 | 0,8980 |
| Ost | LightGBM (Mittel) | 337.715 | 0,4852 | 1,0224 | -0,0325 | 0,3143 | 0,8980 |
| Ost | V1 Verspaetung bleibt (Delta=0) | 337.715 | 0,5815 | 1,1976 | -0,2538 | 0,5728 | – |
| Ost | V1b Persistenz + globale Delta-Verteilung | 337.715 | 0,5815 | 1,1976 | -0,2538 | 0,4536 | 1,2401 |
| Ost | V2 Gruppenverteilung (Median) | 337.715 | 0,5792 | 1,1967 | -0,2152 | 0,4356 | 1,1903 |
| Ost | V2 Gruppenmittel (Punkt) | 337.715 | 0,6403 | 1,1581 | -0,0345 | 0,4356 | 1,1903 |
| West | LightGBM (Median) | 338.173 | 0,3967 | 0,9779 | -0,0548 | 0,2995 | 0,8593 |
| West | LightGBM (Mittel) | 338.173 | 0,4681 | 0,9389 | -0,0206 | 0,2995 | 0,8593 |
| West | V1 Verspaetung bleibt (Delta=0) | 338.173 | 0,6222 | 1,1959 | -0,2769 | 0,6170 | – |
| West | V1b Persistenz + globale Delta-Verteilung | 338.173 | 0,6222 | 1,1959 | -0,2769 | 0,4837 | 1,2905 |
| West | V2 Gruppenverteilung (Median) | 338.173 | 0,5791 | 1,1275 | -0,1011 | 0,4277 | 1,1625 |
| West | V2 Gruppenmittel (Punkt) | 338.173 | 0,6128 | 1,0957 | -0,0353 | 0,4277 | 1,1625 |

(Ausfall unverändert, da die Platt-Kalibrierung abgelehnt wurde — Tabelle identisch mit 2b.) **Urteil:** Das Modell schlägt V1, V1b und V2 in MAE (0,409 gegen 0,579 für V2-Median, 0,627 für V2-Mittel), CRPS (0,306 gegen 0,432), Log-Loss (0,872 gegen 1,176) und Brier (0,002131 gegen 0,002574), in beiden Richtungen. Die Abbruchregel („Modell schlägt V2 nicht → nicht weiterbauen“) greift nicht. Die Median-Prognose hat einen kleinen negativen Bias (−0,06 min), die Mittelprognose praktisch keinen (−0,002 min).

## 3. Kalibrierung

Reliability-Tabellen (Test; Bins nach vorhergesagter Wahrscheinlichkeit; obs_lo/hi = 95-%-Wilson-Intervall).

**Ausfall (roh):**

| bin | n | mean_pred | obs_rate | obs_lo | obs_hi |
|---|---|---|---|---|---|
| [0, 0.0005) | 498.622 | 0,00016 | 0,00014 | 0,00011 | 0,00018 |
| [0.0005, 0.001) | 74.490 | 0,00070 | 0,00087 | 0,00068 | 0,00111 |
| [0.001, 0.002) | 43.740 | 0,00140 | 0,00165 | 0,00131 | 0,00207 |
| [0.002, 0.005) | 30.579 | 0,00310 | 0,00409 | 0,00343 | 0,00487 |
| [0.005, 0.01) | 12.182 | 0,00701 | 0,00846 | 0,00698 | 0,01024 |
| [0.01, 0.02) | 7.341 | 0,01404 | 0,01444 | 0,01195 | 0,01743 |
| [0.02, 0.05) | 5.437 | 0,03126 | 0,03587 | 0,03124 | 0,04115 |
| [0.05, 0.1) | 2.460 | 0,06994 | 0,10000 | 0,08876 | 0,11249 |
| [0.1, 0.2) | 1.490 | 0,14063 | 0,18389 | 0,16504 | 0,20437 |
| [0.2, 1) | 1.320 | 0,43533 | 0,39167 | 0,36568 | 0,41828 |

**Zuwachs, kumulative Ereignisse (roh)** (Δ ≤ 0 = Klassen 0–3; Δ ≥ +2 = Klassen 5–18; Δ ≥ +5 = Klassen 8–18):

| ereignis | bin | n | mean_pred | obs_rate | obs_lo | obs_hi |
|---|---|---|---|---|---|---|
| Delta<=0 | [0, 0.05) | 36.466 | 0,0286 | 0,0291 | 0,0274 | 0,0309 |
| Delta<=0 | [0.05, 0.1) | 28.729 | 0,0727 | 0,0813 | 0,0782 | 0,0845 |
| Delta<=0 | [0.1, 0.2) | 37.798 | 0,1481 | 0,1572 | 0,1535 | 0,1609 |
| Delta<=0 | [0.2, 0.3) | 28.876 | 0,2480 | 0,2646 | 0,2596 | 0,2697 |
| Delta<=0 | [0.3, 0.4) | 26.579 | 0,3498 | 0,3493 | 0,3436 | 0,3551 |
| Delta<=0 | [0.4, 0.5) | 28.325 | 0,4506 | 0,4565 | 0,4507 | 0,4623 |
| Delta<=0 | [0.5, 0.6) | 30.456 | 0,5520 | 0,5655 | 0,5600 | 0,5711 |
| Delta<=0 | [0.6, 0.7) | 41.688 | 0,6529 | 0,6563 | 0,6517 | 0,6608 |
| Delta<=0 | [0.7, 0.8) | 51.775 | 0,7519 | 0,7627 | 0,7590 | 0,7663 |
| Delta<=0 | [0.8, 0.9) | 93.088 | 0,8572 | 0,8657 | 0,8635 | 0,8679 |
| Delta<=0 | [0.9, 0.95) | 114.420 | 0,9280 | 0,9258 | 0,9243 | 0,9273 |
| Delta<=0 | [0.95, 1) | 157.688 | 0,9715 | 0,9710 | 0,9702 | 0,9718 |
| Delta>=+2 | [0, 0.05) | 481.268 | 0,0155 | 0,0142 | 0,0138 | 0,0145 |
| Delta>=+2 | [0.05, 0.1) | 80.724 | 0,0707 | 0,0703 | 0,0686 | 0,0721 |
| Delta>=+2 | [0.1, 0.2) | 54.087 | 0,1406 | 0,1356 | 0,1328 | 0,1385 |
| Delta>=+2 | [0.2, 0.3) | 24.352 | 0,2450 | 0,2366 | 0,2313 | 0,2420 |
| Delta>=+2 | [0.3, 0.4) | 13.898 | 0,3460 | 0,3254 | 0,3177 | 0,3333 |
| Delta>=+2 | [0.4, 0.5) | 8.508 | 0,4463 | 0,4275 | 0,4170 | 0,4380 |
| Delta>=+2 | [0.5, 0.6) | 5.526 | 0,5466 | 0,5331 | 0,5199 | 0,5462 |
| Delta>=+2 | [0.6, 0.7) | 3.251 | 0,6460 | 0,6438 | 0,6272 | 0,6601 |
| Delta>=+2 | [0.7, 0.8) | 2.262 | 0,7476 | 0,7352 | 0,7166 | 0,7530 |
| Delta>=+2 | [0.8, 0.9) | 1.647 | 0,8445 | 0,8470 | 0,8288 | 0,8636 |
| Delta>=+2 | [0.9, 0.95) | 347 | 0,9209 | 0,9481 | 0,9195 | 0,9669 |
| Delta>=+2 | [0.95, 1) | 18 | 0,9563 | 1,0000 | 0,8241 | 1,0000 |
| Delta>=+5 | [0, 0.05) | 664.159 | 0,0041 | 0,0042 | 0,0041 | 0,0044 |
| Delta>=+5 | [0.05, 0.1) | 7.914 | 0,0686 | 0,0719 | 0,0664 | 0,0778 |
| Delta>=+5 | [0.1, 0.2) | 3.117 | 0,1342 | 0,1373 | 0,1257 | 0,1498 |
| Delta>=+5 | [0.2, 0.3) | 537 | 0,2358 | 0,2253 | 0,1920 | 0,2625 |
| Delta>=+5 | [0.3, 0.4) | 131 | 0,3389 | 0,3053 | 0,2330 | 0,3888 |
| Delta>=+5 | [0.4, 0.5) | 24 | 0,4391 | 0,2500 | 0,1200 | 0,4490 |
| Delta>=+5 | [0.5, 0.6) | 4 | 0,5133 | 0,2500 | 0,0456 | 0,6994 |
| Delta>=+5 | [0.6, 0.7) | 1 | 0,6610 | 1,0000 | 0,2065 | 1,0000 |
| Delta>=+5 | [0.7, 0.8) | 1 | 0,7159 | 1,0000 | 0,2065 | 1,0000 |

**Mittlere vorhergesagte gegen beobachtete Häufigkeit je Klasse (roh):**

| klasse | pred_mittel | beobachtet | n_beobachtet |
|---|---|---|---|
| -3(<=) | 0,00220 | 0,00228 | 1.538 |
| -2 | 0,00963 | 0,01068 | 7.221 |
| -1 | 0,14588 | 0,13719 | 92.723 |
| +0 | 0,52299 | 0,53473 | 361.418 |
| +1 | 0,25437 | 0,25270 | 170.799 |
| +2 | 0,04184 | 0,04052 | 27.389 |
| +3 | 0,01212 | 0,01132 | 7.650 |
| +4 | 0,00520 | 0,00469 | 3.170 |
| +5 | 0,00243 | 0,00228 | 1.542 |
| +6 | 0,00114 | 0,00115 | 776 |
| +7 | 0,00066 | 0,00066 | 449 |
| +8 | 0,00045 | 0,00042 | 282 |
| +9 | 0,00022 | 0,00028 | 188 |
| +10 | 0,00017 | 0,00020 | 134 |
| +11 | 0,00012 | 0,00015 | 100 |
| +12 | 0,00007 | 0,00012 | 80 |
| +13 | 0,00006 | 0,00008 | 57 |
| +14 | 0,00005 | 0,00007 | 50 |
| +15(>=) | 0,00037 | 0,00048 | 322 |

ECE (Test, roh): Ausfall 0,00045; Zuwachs Δ ≤ 0: 0,0052; Δ ≥ +2: 0,0025; Δ ≥ +5: 0,00018. Nach Temperaturskalierung: 0,0144 / 0,0074 / 0,0021 (also schlechter). Nachkalibrierung nach Plan (Fit auf September; Entscheidung per Tageshälften: ungerade Tage fitten, gerade Tage prüfen): Temperatur 0,918 (ungerade) bzw. 0,906 (alle Tage); Log-Loss gerade Tage roh 0,23982 → kalibriert 0,23785 (Verbesserung, daher „angenommen“). Platt-Kalibrierung Ausfall: Brier gerade Tage roh 0,000302 → 0,000305, Log-Loss 0,001792 → 0,001793, also nicht angenommen (52 Ausfälle in dieser Hälfte; Kalibriermonat hat nur 101 Ausfälle, Rate 0,038 % gegenüber 0,262 % im Test, also untypisch). Beobachtung: Roh-Ausfallwahrscheinlichkeiten werden im Test bei 0,5–10 % leicht unterschätzt (z. B. vorhergesagt 7,0 %, beobachtet 10,0 %), weil die Ausfallrate im Test höher ist als im Training.

## 4. SHAP (TreeExplainer; 40.000 bzw. 20.000 zufällige Testzeilen)

Ausfallmodell, mittleres |SHAP| in log-Odds:

| merkmal | mean_abs_shap |
|---|---|
| cancel_net30 | 0,7883 |
| d_km1 | 0,4682 |
| rank | 0,3401 |
| d_pred_k1 | 0,2487 |
| hour | 0,2386 |
| act_gap_pred | 0,2260 |
| net_n15 | 0,1857 |
| d_k | 0,1772 |
| line | 0,1742 |
| cancel_sd30 | 0,1640 |
| net_mean15 | 0,1313 |
| plan_run | 0,1028 |
| plan_gap_pred | 0,0711 |
| richtung | 0,0681 |
| hvz | 0,0510 |

Zuwachsmodell, Summe über die 19 Klassen des mittleren |SHAP| (log-Odds), dazu die Werte für die Klassen Δ = 0 und Δ = +2:

| merkmal | mean_abs_shap_summe_klassen | mean_abs_shap_klasse_Delta0 | mean_abs_shap_klasse_Delta_plus2 |
|---|---|---|---|
| d_pred_k1 | 6,8263 | 0,0885 | 0,4536 |
| d_k | 6,3540 | 0,0792 | 0,2949 |
| rank | 6,1659 | 0,1950 | 0,5779 |
| plan_run | 5,7551 | 0,5973 | 0,5802 |
| net_mean15 | 5,1154 | 0,0625 | 0,2294 |
| d_km1 | 4,9024 | 0,0743 | 0,1782 |
| net_n15 | 3,9659 | 0,0143 | 0,0545 |
| hour | 3,5941 | 0,0179 | 0,0545 |
| act_gap_pred | 2,8926 | 0,0609 | 0,1169 |
| cancel_net30 | 2,3791 | 0,0051 | 0,0239 |
| plan_gap_pred | 2,2857 | 0,0211 | 0,1438 |
| line | 2,1747 | 0,0474 | 0,0778 |
| d_pred_k | 1,7437 | 0,0083 | 0,0206 |
| richtung | 1,4848 | 0,0557 | 0,1303 |
| wtype | 1,4119 | 0,0123 | 0,0405 |

Lesart: Das Ausfallmodell hängt am stärksten an der Zahl der Ausfälle der letzten 30 min auf der Strecke (cancel_net30). Das Zuwachsmodell nutzt am meisten die Verspätung des Vorgängers an k+1, die eigene Verspätung, die Station und die geplante Fahrzeit (plan_run, wichtig für Δ = 0).

## 5. Rollout

- Prüfsteintage: 30 aus den Testmonaten gezogen (Seed 4, nach Wochentagstyp geschichtet, ohne Sperrtage): 2025-10-05, 10-06, 10-15, 10-22, 10-25, 10-30, 11-04, 11-09, 11-13, 11-14, 11-15, 11-16, 11-17, 11-19, 11-20, 11-21, 11-27, 11-28, 11-29, 12-01, 12-04, 12-09, 12-10, 12-11, 12-12, 12-16, 12-25, 12-27, 12-28, 12-31. Je Tag 100 Läufe (gemeinsamer Seed 11) = 3.000 simulierte Tagesläufe; 20 Läufe je Tag für Prüfstein e.
- Mechanik: Minutentakt; Eintritt am ersten laut Daten gefahrenen Halt mit echter Verspätung; Ausfall absorbierend; Ziehung per Inversion mit je (Lauf, Zug, Pfadposition) festen Zufallszahlen (common random numbers).
- Ausfallquote (modellierte Positionen = Halte nach dem Eintritt; 230.800 Positionen/Lauf-Tag-Summe): Echt 2,55 %, Sim 2,03 %.
- Kausalitätsklammern: 152.314 über alle 3.000 Tagesläufe (ca. 0,6 % der Übergänge; Echt: 0,64 % der Übergänge haben Fahrzeit ≤ 0 min).

**Physikalisch unmögliche bzw. unplausible Zustände (nur gezählt):**

| Unnamed: 0 | ueberholungen | paare | ueberholquote | gleiche_minute | gleiche_minute_quote | abstand_hoechstens_1min_quote | n_abstaende |
|---|---|---|---|---|---|---|---|
| sim | 479.984 | 22.547.160 | 0,021 | 1.256.402 | 0,050 | 0,182 | 24.910.160 |
| real | 1.917 | 224.268 | 0,009 | 2.817 | 0,011 | 0,090 | 247.898 |

Spalten: `ueberholungen` = Zahl der Paare, die an k hintereinander ankommen und an k+1 in vertauschter Reihenfolge (strikt nach Ankunftsminute); `paare` = Zahl betrachteter benachbarter Paare; `gleiche_minute` = aufeinanderfolgende Ankünfte an einer Station in derselben Minute; `abstand_hoechstens_1min_quote` = Anteil Abstände ≤ 1 min. Sim-Summen sind über 100 Läufe, Echt über je einen Tag (Summen daher nicht vergleichbar, nur Quoten). Sim hat 2,5-mal so viele Überholungen, 4,4-mal so viele gleiche Minuten und doppelt so viele Abstände ≤ 1 min.

Mittlere Verspätung je Station/Richtung (Alle Ankünfte der 30 Tage; Sim = Mittel über Läufe):

| station_rang | richtung | echt_mittel_verspaetung | sim_mittel_verspaetung |
|---|---|---|---|
| 0 | Ost | 4,314 | 4,314 |
| 0 | West | 6,285 | 6,644 |
| 1 | Ost | 3,712 | 3,682 |
| 1 | West | 5,805 | 6,161 |
| 2 | Ost | 4,123 | 4,132 |
| 2 | West | 5,732 | 6,107 |
| 3 | Ost | 4,450 | 4,483 |
| 3 | West | 5,627 | 5,999 |
| 4 | Ost | 4,625 | 4,730 |
| 4 | West | 5,043 | 5,236 |
| 5 | Ost | 4,690 | 4,727 |
| 5 | West | 5,214 | 5,602 |
| 6 | Ost | 5,158 | 5,217 |
| 6 | West | 5,166 | 5,520 |
| 7 | Ost | 5,119 | 5,190 |
| 7 | West | 4,780 | 5,104 |
| 8 | Ost | 5,168 | 5,340 |
| 8 | West | 4,848 | 5,104 |
| 9 | Ost | 5,275 | 5,460 |
| 9 | West | 4,849 | 5,044 |
| 10 | Ost | 5,732 | 5,893 |
| 10 | West | 4,268 | 4,366 |
| 11 | Ost | 6,162 | 6,478 |
| 11 | West | 2,950 | 2,950 |

## 6. Prüfsteine (Simulation gegen Wirklichkeit, 30 Testtage)

Zusammenfassung mit Plan-Toleranzen:

| Prüfstein | Kriterium (vorab im Plan) | Ergebnis | Urteil |
|---|---|---|---|
| (a) Ankunft am Pfadende (22.848 Züge, 30 Tage) | 90-%-Band-Abdeckung in [85 %; 95 %]; \|Bias Median\| ≤ 0,5 min; MAE ≤ MAE Persistenz | Abdeckung 93,6 % (KI 92,7–94,5); Bias −0,11 min (−0,29…+0,04); MAE 1,71 min vs. Persistenz 2,87 min | **bestanden** |
| (b) Verspätungsaufbau über die Strecke (8.239 Züge Rang 0↔11) | \|Sim − Echt\| ≤ 0,5 min | Echt 3,31 min (2,88–3,74), Sim 3,69 min (3,36–4,00), Differenz +0,38 (+0,14…+0,58) | **bestanden** (knapp, systematisch zu hoch) |
| (c) Zuwachs Folgezug nach Vorgänger/Abstand (Tabelle 11.7) | ≥ 75 % der Zellen mit n ≥ 200 innerhalb ±0,3 min und Gradient wie Echt | 2 von 9 Zellen (22 %) innerhalb ±0,3; Sim in 9 von 9 Zellen zu hoch (+0,12…+0,78 min); Gradient in der Zeile „<5 min" vorhanden | **nicht bestanden** |
| (d) Ausfallquote nach Vorgänger/Abstand (Tabelle 11.8, nur modellierte Übergänge) | ≥ 75 % der Zellen mit n ≥ 500 innerhalb max(0,3 pp; 40 %) und Monotonie in Zeile „<5 min" | 8 von 14 Zellen (57 %); Monotonie in Echt und Sim vorhanden, aber Sim-Anstieg flacher (Zeile „<5 min": Echt 0,7 % → 4,7 %, Sim 1,0 % → 3,5 %) | **nicht bestanden** |
| (e) Erholungszeit je Schwereklasse (1.332 Episoden, 20 Läufe/Tag) | ≥ 4 von 5 Klassen mit n ≥ 50 innerhalb max(25 %; 5 min) bzw. Echt-Median in Lauf-Spanne und Monotonie | 3 von 5 Klassen (<3, 3–5, 5–10 ok; 10–20 und >20 nicht); Mediane monoton in Echt und Sim | **nicht bestanden** |


Konfidenzintervalle: Bootstrap über Betriebstage (1.000 Ziehungen); „Sim-Lauf q025/q975“ = Streuung über die 100 Läufe.

**(a) Ankunft am Pfadende** (Züge, die laut Daten bis zu ihrem letzten Halt fuhren und mindestens 2 Halte haben; Sim-Band aus den Läufen, in denen der Zug nicht ausfiel). Abdeckung = Anteil echter Werte in [q05 (untere Ordnungsstatistik), q95 (obere)]:

| gruppe | kennzahl | wert | ki_lo | ki_hi | n_zuege | n_tage |
|---|---|---|---|---|---|---|
| alle | abdeckung_90 | 0,936 | 0,927 | 0,945 | 22.848 | 30 |
| alle | abdeckung_50 | 0,654 | 0,638 | 0,671 | 22.848 | 30 |
| alle | bias_median_min | -0,110 | -0,286 | 0,040 | 22.848 | 30 |
| alle | mae_sim_median_min | 1,706 | 1,541 | 1,907 | 22.848 | 30 |
| alle | mae_persistenz_min | 2,871 | 2,604 | 3,167 | 22.848 | 30 |
| alle | bias_mittel_min | 0,316 | 0,144 | 0,461 | 22.848 | 30 |
| Ost | abdeckung_90 | 0,936 | 0,922 | 0,945 | 11.252 | 30 |
| Ost | abdeckung_50 | 0,652 | 0,633 | 0,670 | 11.252 | 30 |
| Ost | bias_median_min | -0,185 | -0,407 | -0,017 | 11.252 | 30 |
| Ost | mae_sim_median_min | 1,759 | 1,556 | 2,045 | 11.252 | 30 |
| Ost | mae_persistenz_min | 2,751 | 2,458 | 3,124 | 11.252 | 30 |
| Ost | bias_mittel_min | 0,246 | 0,039 | 0,415 | 11.252 | 30 |
| West | abdeckung_90 | 0,937 | 0,925 | 0,949 | 11.596 | 30 |
| West | abdeckung_50 | 0,655 | 0,634 | 0,678 | 11.596 | 30 |
| West | bias_median_min | -0,037 | -0,231 | 0,123 | 11.596 | 30 |
| West | mae_sim_median_min | 1,655 | 1,504 | 1,811 | 11.596 | 30 |
| West | mae_persistenz_min | 2,987 | 2,726 | 3,240 | 11.596 | 30 |
| West | bias_mittel_min | 0,383 | 0,194 | 0,539 | 11.596 | 30 |
| HVZ | abdeckung_90 | 0,933 | 0,920 | 0,943 | 10.457 | 30 |
| HVZ | abdeckung_50 | 0,642 | 0,621 | 0,663 | 10.457 | 30 |
| HVZ | bias_median_min | -0,104 | -0,389 | 0,128 | 10.457 | 30 |
| HVZ | mae_sim_median_min | 1,875 | 1,631 | 2,204 | 10.457 | 30 |
| HVZ | mae_persistenz_min | 3,250 | 2,863 | 3,706 | 10.457 | 30 |
| HVZ | bias_mittel_min | 0,337 | 0,059 | 0,566 | 10.457 | 30 |
| Nicht-HVZ | abdeckung_90 | 0,939 | 0,929 | 0,949 | 12.391 | 30 |
| Nicht-HVZ | abdeckung_50 | 0,664 | 0,645 | 0,683 | 12.391 | 30 |
| Nicht-HVZ | bias_median_min | -0,114 | -0,261 | 0,020 | 12.391 | 30 |
| Nicht-HVZ | mae_sim_median_min | 1,564 | 1,426 | 1,714 | 12.391 | 30 |
| Nicht-HVZ | mae_persistenz_min | 2,550 | 2,332 | 2,762 | 12.391 | 30 |
| Nicht-HVZ | bias_mittel_min | 0,298 | 0,153 | 0,432 | 12.391 | 30 |

**(b) Verspätungsaufbau Eintritt → Ende, Züge mit Rang 0 und 11 im Pfad, die am Streckenanfang eintraten und am Ende fuhren** (min):

| gruppe | n_zuege | echt_mittel_min | echt_lo | echt_hi | sim_mittel_min | sim_lo | sim_hi | diff_sim_minus_echt | diff_lo | diff_hi |
|---|---|---|---|---|---|---|---|---|---|---|
| alle | 8.239 | 3,305 | 2,877 | 3,740 | 3,690 | 3,361 | 4,003 | 0,385 | 0,141 | 0,583 |
| Ost | 4.146 | 2,296 | 1,850 | 2,768 | 2,685 | 2,334 | 3,005 | 0,389 | 0,120 | 0,612 |
| West | 4.093 | 4,328 | 3,884 | 4,776 | 4,708 | 4,376 | 5,017 | 0,380 | 0,115 | 0,612 |
| HVZ | 4.077 | 3,662 | 3,149 | 4,220 | 4,063 | 3,670 | 4,402 | 0,401 | 0,056 | 0,702 |
| Nicht-HVZ | 4.162 | 2,957 | 2,568 | 3,342 | 3,325 | 3,025 | 3,629 | 0,368 | 0,138 | 0,590 |
| Eintritt <=2 min | 4.745 | 3,407 | 3,029 | 3,818 | 3,764 | 3,430 | 4,114 | 0,358 | 0,155 | 0,542 |
| Eintritt >10 min | 598 | 4,425 | 3,336 | 5,613 | 4,393 | 3,675 | 4,938 | -0,031 | -0,715 | 0,571 |

**(c) Zuwachs des Folgezugs (Verspätung am Ende − am Eintritt), nach Vorgängerverspätung am Eintritt (≤ 2, 2–5, 5–10, > 10 min) und geplantem Abstand zum Vorgänger am Eintrittshalt (Tabelle 11.7-Definition, ganze Kernstrecke).** Vorgänger = vorheriger gefahrener Zug nach Planzeit am Eintrittshalt. Die Zelleneinteilung kommt aus den Echtdaten (Vorgängerverspätung am Eintritt ist Replay); n_sim_pro_lauf ist die mittlere Zahl je Lauf. Zelle gilt als geprüft bei n_Echt ≥ 200; `zelle_ok` = |Diff| ≤ 0,3 min (1 = ok, 0 = nicht ok, leer = nicht geprüft). Referenz aus BEFUNDE_VAR 11.7 (ganze Strecke, ganzes Jahr): z. B. Zeile „<5 min“: 2,79 → 3,87 → 4,01 → 4,54 min.

| abstand | vorgaenger | n_echt | echt_mittel | echt_lo | echt_hi | n_sim_pro_lauf | sim_mittel | sim_lauf_q025 | sim_lauf_q975 | diff | zelle_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <5min | <=2min | 2.709 | 3,287 | 2,922 | 3,660 | 2.669 | 3,587 | 3,463 | 3,699 | 0,300 | 0,000 |
| <5min | 2-5min | 1.117 | 3,848 | 3,433 | 4,329 | 1.094 | 4,193 | 3,991 | 4,373 | 0,346 | 0,000 |
| <5min | 5-10min | 624 | 4,087 | 3,574 | 4,649 | 608,980 | 4,616 | 4,323 | 4,895 | 0,529 | 0,000 |
| <5min | >10min | 322 | 4,665 | 3,341 | 6,193 | 309,470 | 4,829 | 4,497 | 5,231 | 0,165 | 1,000 |
| 5-10min | <=2min | 1.665 | 2,616 | 2,160 | 3,095 | 1.632 | 3,101 | 2,976 | 3,237 | 0,485 | 0,000 |
| 5-10min | 2-5min | 539 | 3,659 | 3,002 | 4,371 | 526,390 | 3,973 | 3,751 | 4,230 | 0,314 | 0,000 |
| 5-10min | 5-10min | 223 | 4,260 | 3,605 | 4,947 | 215,590 | 4,382 | 3,974 | 4,874 | 0,121 | 1,000 |
| 5-10min | >10min | 122 | 4,262 | 3,172 | 5,230 | 114,700 | 4,737 | 4,231 | 5,249 | 0,474 | – |
| 10-15min | <=2min | 236 | 3,110 | 2,578 | 3,733 | 232,260 | 3,513 | 3,102 | 3,851 | 0,403 | 0,000 |
| 10-15min | 2-5min | 83 | 4,120 | 3,494 | 4,772 | 81,200 | 4,457 | 3,765 | 5,129 | 0,336 | – |
| 10-15min | 5-10min | 30 | 4,900 | 3,500 | 6,808 | 29,320 | 4,719 | 3,723 | 6,018 | -0,181 | – |
| 10-15min | >10min | 27 | 5,630 | 2,635 | 7,001 | 26,220 | 6,106 | 4,844 | 7,921 | 0,477 | – |
| 15-20min | <=2min | 293 | 1,379 | 0,540 | 2,859 | 290,540 | 2,160 | 1,865 | 2,495 | 0,781 | 0,000 |
| 15-20min | 2-5min | 94 | 1,511 | 0,284 | 3,125 | 92,980 | 1,906 | 1,294 | 2,640 | 0,395 | – |
| 15-20min | 5-10min | 41 | 1,902 | 1,222 | 3,637 | 40,050 | 2,303 | 1,329 | 3,635 | 0,400 | – |
| 15-20min | >10min | 24 | 1,333 | -0,762 | 6,000 | 23,400 | 2,128 | 0,637 | 3,834 | 0,794 | – |
| 20-60min | <=2min | 39 | 1,821 | 0,823 | 2,943 | 38,270 | 2,041 | 1,247 | 2,961 | 0,220 | – |
| 20-60min | 2-5min | 17 | 1,118 | 0,000 | 2,000 | 16,870 | 1,973 | 0,734 | 4,247 | 0,855 | – |
| 20-60min | 5-10min | 3 | 2,000 | -1,000 | 6,000 | 2,970 | 0,603 | -2,000 | 3,508 | -1,397 | – |
| 20-60min | >10min | 1 | 10,000 | 0,000 | 10,000 | 0,990 | 3,710 | -0,525 | 10,525 | -6,290 | – |

**(d) Ausfallquote des Folgezugs nach Vorgängerverspätung und Abstand (Tabelle 11.8-Definition), nur an modellierten Übergängen** (Folgezug fuhr an der Vorstation; Vorgänger = unmittelbarer Vorgänger im Plan an derselben Station, nicht ausgefallen, Abstand 0 < Δ ≤ 60 min). `diff_pp` = Sim − Echt in Prozentpunkten; Toleranz max(0,3 pp; 40 % der Echtrate); geprüft bei n_Echt ≥ 500.

| abstand | vorgaenger | n_echt | ausfaelle_echt | echt_rate | echt_lo | echt_hi | n_sim_pro_lauf | sim_rate | sim_lauf_q025 | sim_lauf_q975 | diff_pp | zelle_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| <5min | <=2min | 61.741 | 420 | 0,007 | 0,004 | 0,011 | 57.012 | 0,010 | 0,008 | 0,012 | 0,310 | 0,000 |
| <5min | 2-5min | 68.207 | 1.540 | 0,023 | 0,011 | 0,037 | 68.529 | 0,017 | 0,015 | 0,020 | -0,530 | 1,000 |
| <5min | 5-10min | 43.318 | 1.503 | 0,035 | 0,017 | 0,056 | 46.498 | 0,025 | 0,021 | 0,029 | -0,981 | 1,000 |
| <5min | >10min | 18.624 | 881 | 0,047 | 0,031 | 0,068 | 20.593 | 0,035 | 0,029 | 0,042 | -1,182 | 1,000 |
| 5-10min | <=2min | 4.088 | 24 | 0,006 | 0,001 | 0,014 | 3.854 | 0,010 | 0,004 | 0,019 | 0,393 | 0,000 |
| 5-10min | 2-5min | 5.363 | 41 | 0,008 | 0,004 | 0,013 | 5.498 | 0,013 | 0,008 | 0,021 | 0,578 | 0,000 |
| 5-10min | 5-10min | 3.692 | 63 | 0,017 | 0,006 | 0,029 | 4.125 | 0,019 | 0,013 | 0,030 | 0,238 | 1,000 |
| 5-10min | >10min | 1.294 | 23 | 0,018 | 0,004 | 0,034 | 1.510 | 0,033 | 0,021 | 0,049 | 1,530 | 0,000 |
| 10-15min | <=2min | 674 | 3 | 0,004 | 0,000 | 0,013 | 626,670 | 0,006 | 0,000 | 0,020 | 0,139 | 1,000 |
| 10-15min | 2-5min | 532 | 6 | 0,011 | 0,000 | 0,031 | 535,400 | 0,007 | 0,000 | 0,024 | -0,425 | 1,000 |
| 10-15min | 5-10min | 238 | 0 | 0,000 | 0,000 | 0,000 | 272,290 | 0,007 | 0,000 | 0,027 | 0,701 | – |
| 10-15min | >10min | 97 | 0 | 0,000 | 0,000 | 0,000 | 104,630 | 0,011 | 0,000 | 0,046 | 1,051 | – |
| 15-20min | <=2min | 2.913 | 4 | 0,001 | 0,000 | 0,004 | 2.674 | 0,004 | 0,000 | 0,011 | 0,215 | 1,000 |
| 15-20min | 2-5min | 1.482 | 0 | 0,000 | 0,000 | 0,000 | 1.598 | 0,005 | 0,000 | 0,020 | 0,515 | 0,000 |
| 15-20min | 5-10min | 520 | 0 | 0,000 | 0,000 | 0,000 | 600,430 | 0,006 | 0,000 | 0,023 | 0,582 | 0,000 |
| 15-20min | >10min | 182 | 0 | 0,000 | 0,000 | 0,000 | 217,220 | 0,009 | 0,000 | 0,048 | 0,898 | – |
| 20-60min | <=2min | 513 | 1 | 0,002 | 0,000 | 0,007 | 454,810 | 0,003 | 0,000 | 0,020 | 0,109 | 1,000 |
| 20-60min | 2-5min | 262 | 0 | 0,000 | 0,000 | 0,000 | 312,350 | 0,007 | 0,000 | 0,032 | 0,707 | – |
| 20-60min | 5-10min | 111 | 1 | 0,009 | 0,000 | 0,031 | 117,560 | 0,007 | 0,000 | 0,072 | -0,182 | – |
| 20-60min | >10min | 35 | 0 | 0,000 | 0,000 | 0,000 | 37,700 | 0,010 | 0,000 | 0,151 | 1,042 | – |

**(e) Erholungszeit je Schwereklasse.** Messung exakt wie Stufe 3 (`recovery.measure`: Erholung = drei aufeinanderfolgende Züge an der Meldestation < 3 min Verspätung, ab t0+15 min, Kaplan-Meier-Median; Schwere = max. Verspätung in den ersten 15 min; Standard-Zensierung durch Folgeepisoden) auf den echten Episodenzeitpunkten der 30 Tage (1.332 Episoden; 965 auswertbar je Lauf), (i) auf echten Daten dieser Tage, (ii) auf 20 simulierten Läufen je Tag (Schwere und Erholung aus den simulierten Zügen). Mediane in min; „n.e.“ = nicht erreicht; Sim: Mittel der Lauf-Mediane und Spanne (2,5/97,5 %-Perzentil der 20 Lauf-Mediane, „–“ wenn Läufe n.e. liefern); `klasse_ok` nach Regel (|Sim − Echt| ≤ max(25 % Echt; 5 min) oder Echt-Median in Lauf-Spanne; bei Echt „n.e.“: > 50 % der Läufe n.e.).

| klasse | n_echt | zensiert_echt | median_echt | echt_lo | echt_hi | n_sim_pro_lauf | n_laeufe | zensiert_sim | median_sim_lauf_mittel | median_sim_q025 | median_sim_q975 | anteil_laeufe_ne | klasse_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| <3 | 80 | 0,387 | 31,983 | 18,550 | 53,367 | 69,750 | 20 | 0,394 | 24,489 | 18,987 | 37,230 | 0,000 | 1 |
| 3-5 | 125 | 0,512 | 32,567 | 19,900 | 48,033 | 123,900 | 20 | 0,545 | 39,556 | 27,834 | 56,419 | 0,000 | 1 |
| 5-10 | 391 | 0,675 | 88,183 | 56,333 | 106,817 | 356,750 | 20 | 0,686 | 86,742 | 68,465 | 130,967 | 0,000 | 1 |
| 10-20 | 293 | 0,795 | 132,400 | 100,350 | n.e. | 330,250 | 20 | 0,828 | 199,262 | 160,066 | – | 0,150 | 0 |
| >20 | 75 | 0,907 | n.e. | 118,917 | n.e. | 83,600 | 20 | 0,904 | 223,412 | 153,322 | – | 0,450 | 0 |

Vergleich mit Plot 6 der Survival-Stufe (Median-Erholungszeit ab t0+15 min, Störungen, ganzes Jahr, n = 1.358 / 1.844 / 4.228 / 2.451 / 608): 15,2 / 23,4 / 58,8 / 121,1 / 236,7 min (obere Klasse mit KI 124–n.e.). Die Testtage (kleinere Stichprobe, andere Monate) liefern 32,0 / 32,6 / 88,2 / 132,4 / n.e.; Sim 24,5 / 39,6 / 86,7 / 199,3 / 223,4. Alle drei zeigen monotones Ansteigen mit der Schwere. Starke Zensierung: Echt 39 % / 51 % / 68 % / 80 % / 91 %; Sim 39 % / 54 % / 69 % / 83 % / 90 %.

## 7. Szenarien

Aufbau: Tage 2025-11-12 und 2025-12-03 (Mittwoche). HVZ: Eintritt 7:00–9:00 bzw. 16:00–18:30 Uhr; Nicht-HVZ: 10:00–14:00 Uhr. Je Tag/Zeitfenster 2 Beispielzüge (bevorzugt Linie S3, Richtung Ost, Eintritt in Pasing, Verspätung am Eintritt ≤ 3 min, ≥ 8 Halte). Fenster: Simulation startet 60 min vor Eingriff mit echtem Zustand und endet 6 h danach. Je Szenario 100 Läufe; Basis und Szenario mit denselben Zufallszahlen. Definitionen: „+N“: Der Zug betritt die Strecke mit absolut N min Verspätung (die Änderung gegenüber der echten Verspätung steht in `eigener Zug Δ am Ende`). „betroffen“: Änderung der Verspätung am Pfadende ≥ 1 min im jeweiligen Lauf (andere Züge als X). `Δ je betr. Zug`: je Zug über die Läufe gemittelte Änderung am Pfadende, für Züge mit |Mittel| ≥ 0,25 min; Verteilung über diese Züge (Mittel, Median, q10, q90, max). Gesamt-Mehrverspätung = Summe der Änderungen am Pfadende aller anderen Züge je Lauf (Zug-Minuten). Zusätzliche Ausfälle = Zahl simulierter Ausfälle (Szenario − Basis) an modellierten Positionen anderer Züge. Abklingzeit (KI-Regel): ab dem Spitzenblock der Mittel-Differenz über 10-min-Blöcke der letzte Block, dessen 95-%-Intervall (über die Läufe) die 0 nicht einschließt, plus Blocklänge; Abklingzeit < 0,1: erster Block ab der Spitze, nach dem 3 Blöcke in Folge |mittlere Differenz| < 0,1 min. Alle Werte der Tabelle sind Mittel über die beiden Tage × je 2 Züge (n_sim = 4 Szenariosimulationen je Variante); Min/Max zeigt die Spanne zwischen den 4.

| variante | n_sim | betroffene Züge/Lauf (Mittel) | betr. q05 | betr. q95 | Δ je betr. Zug Mittel (min) | Median | q10 | q90 | max | Gesamt-Mehrverspätung Zug-min (Mittel) | Zug-min q05 | Zug-min q95 | Zug-min Mittel, min. Zug | Zug-min Mittel, max. Zug | zus. Ausfälle (Mittel) | Ausf. q05 | Ausf. q95 | Abklingzeit KI-Regel (min) | Abklingzeit <0,1 min (min) | eigener Zug Δ am Ende (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fahrt gestrichen, außerhalb HVZ | 4 | 12,54 | 2,49 | 28,07 | -0,07 | 0,02 | -0,63 | 0,38 | 0,74 | 2,35 | -10,09 | 18,29 | 0,51 | 5,94 | 1,82 | -5,50 | 15,29 | 95,00 | 37,50 | – |
| Fahrt gestrichen, HVZ | 4 | 21,44 | 6,22 | 44,86 | -0,14 | -0,10 | -0,53 | 0,17 | 0,20 | 0,94 | -18,40 | 23,22 | -1,20 | 3,17 | 0,63 | -9,54 | 12,34 | 227,50 | 40,00 | – |
| Verspätung +5 min, außerhalb HVZ | 4 | 7,57 | 0,75 | 23,55 | -0,41 | -0,42 | -0,65 | -0,16 | -0,09 | 0,14 | -9,09 | 10,55 | 0,00 | 0,27 | -0,86 | -6,50 | 2,00 | 115,00 | 40,00 | 4,18 |
| Verspätung +5 min, HVZ | 4 | 15,96 | 3,50 | 35,81 | 0,31 | 0,37 | -0,17 | 0,63 | 0,94 | 3,43 | -10,56 | 20,14 | 1,87 | 4,50 | 0,09 | -6,56 | 5,55 | 105,00 | 87,50 | 4,47 |
| Verspätung +10 min, außerhalb HVZ | 4 | 11,63 | 2,95 | 28,09 | 0,20 | 0,26 | -0,46 | 0,79 | 0,97 | 2,91 | -8,28 | 13,26 | 1,67 | 5,92 | -0,33 | -7,34 | 5,56 | 92,50 | 40,00 | 8,25 |
| Verspätung +10 min, HVZ | 4 | 20,70 | 6,00 | 42,83 | 0,34 | 0,36 | -0,19 | 0,81 | 1,38 | 6,28 | -11,81 | 29,10 | 4,05 | 8,34 | 0,05 | -7,53 | 6,32 | 105,00 | 50,00 | 10,10 |
| Verspätung +20 min, außerhalb HVZ | 4 | 15,51 | 5,71 | 29,06 | 0,37 | 0,35 | -0,18 | 0,95 | 1,42 | 6,52 | -6,54 | 20,06 | 5,02 | 9,05 | -0,12 | -9,30 | 8,86 | 72,50 | 57,50 | 19,53 |
| Verspätung +20 min, HVZ | 4 | 27,01 | 11,99 | 48,06 | 0,43 | 0,36 | -0,01 | 0,89 | 1,92 | 12,03 | -8,40 | 41,02 | 9,35 | 16,63 | -0,27 | -10,51 | 8,50 | 117,50 | 65,00 | 19,71 |


Einzelergebnisse je Simulation (alle Spalten):

| art | minuten | hvz | opday | zug | linie | eintritt_uhrzeit | eintritt_basis_verspaetung | n_zuege_fenster | betroffen_pro_lauf_mittel | betroffen_q05 | betroffen_q95 | zugaenderung_ende_n | zugaenderung_mittel | zugaenderung_median | zugaenderung_q10 | zugaenderung_q90 | zugaenderung_max | zugaenderung_min | gesamt_mehrverspaetung_zugmin_mittel | gesamt_zugmin_q05 | gesamt_zugmin_q95 | zusaetzliche_ausfaelle_mittel | zus_ausf_q05 | zus_ausf_q95 | spitze_diff_min | spitze_nach_min | abklingzeit_ki_min | abklingzeit_unter_0_1min | n_laeufe | eigener_zug_aenderung_ende_mittel | eigener_zug_eintritt_verspaetung_basis | eigener_zug_eintritt_verspaetung_szen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| verspaetung | 5 | 1 | 2025-11-12 | -3.282.464.980.179.847.168 | 2 | 16,48 | 0,00 | 668 | 17,70 | 3,00 | 40,00 | 12 | 0,21 | 0,30 | -0,40 | 0,42 | 0,98 | -0,42 | 4,50 | -15,10 | 28,00 | -0,19 | -8,05 | 11,00 | 0,10 | 30,00 | 70,00 | 40,00 | 100 | 4,15 | 0,00 | 5,00 |
| verspaetung | 10 | 1 | 2025-11-12 | -3.282.464.980.179.847.168 | 2 | 16,48 | 0,00 | 668 | 23,54 | 6,00 | 48,05 | 19 | 0,34 | 0,34 | -0,27 | 0,80 | 1,32 | -0,52 | 8,34 | -17,00 | 35,20 | 0,63 | -8,00 | 11,00 | 0,23 | 30,00 | 80,00 | 70,00 | 100 | 8,60 | 0,00 | 10,00 |
| verspaetung | 20 | 1 | 2025-11-12 | -3.282.464.980.179.847.168 | 2 | 16,48 | 0,00 | 668 | 31,88 | 14,00 | 51,10 | 30 | 0,49 | 0,39 | 0,20 | 0,95 | 1,80 | -0,56 | 16,63 | -11,20 | 49,05 | 0,25 | -11,00 | 11,00 | 0,30 | 30,00 | 90,00 | 80,00 | 100 | 18,69 | 0,00 | 20,00 |
| ausfall | 0 | 1 | 2025-11-12 | -3.282.464.980.179.847.168 | 2 | 16,48 | 0,00 | 668 | 21,46 | 4,00 | 45,00 | 12 | -0,08 | -0,29 | -0,45 | 0,36 | 0,37 | -0,65 | -1,20 | -27,25 | 21,50 | 0,26 | -9,10 | 11,00 | -0,04 | 40,00 | 230,00 | 40,00 | 100 | – | 0,00 | – |
| verspaetung | 5 | 1 | 2025-11-12 | -6.216.281.276.383.383.552 | 2 | 8,15 | 0,00 | 668 | 15,40 | 3,00 | 41,05 | 7 | 0,07 | 0,25 | -0,61 | 0,51 | 0,81 | -1,02 | 3,37 | -9,10 | 19,25 | 1,30 | -8,15 | 11,20 | -0,09 | 260,00 | 260,00 | 260,00 | 100 | 5,54 | 0,00 | 5,00 |
| verspaetung | 10 | 1 | 2025-11-12 | -6.216.281.276.383.383.552 | 2 | 8,15 | 0,00 | 668 | 16,08 | 4,00 | 37,10 | 10 | 0,19 | 0,33 | -0,49 | 0,64 | 1,39 | -1,02 | 4,05 | -12,20 | 31,00 | 0,58 | -11,00 | 11,00 | 0,15 | 30,00 | 60,00 | 40,00 | 100 | 11,47 | 0,00 | 10,00 |
| verspaetung | 20 | 1 | 2025-11-12 | -6.216.281.276.383.383.552 | 2 | 8,15 | 0,00 | 668 | 22,00 | 8,95 | 46,00 | 19 | 0,38 | 0,34 | -0,39 | 0,82 | 2,99 | -1,00 | 9,41 | -8,30 | 42,00 | -0,05 | -11,00 | 11,00 | 0,37 | 40,00 | 90,00 | 70,00 | 100 | 19,82 | 0,00 | 20,00 |
| ausfall | 0 | 1 | 2025-11-12 | -6.216.281.276.383.383.552 | 2 | 8,15 | 0,00 | 668 | 18,65 | 3,95 | 46,15 | 12 | 0,11 | 0,27 | -0,41 | 0,47 | 0,51 | -1,08 | 3,17 | -13,20 | 29,35 | 1,30 | -10,05 | 15,20 | 0,08 | 40,00 | 310,00 | 40,00 | 100 | – | 0,00 | – |
| verspaetung | 5 | 0 | 2025-11-12 | 432.986.483.584.246.336 | 2 | 11,20 | 3,00 | 486 | 6,42 | 0,00 | 26,10 | 1 | -0,43 | -0,43 | -0,43 | -0,43 | -0,43 | -0,43 | 0,06 | -10,10 | 11,10 | -2,16 | -15,00 | 0,00 | 0,01 | 70,00 | 70,00 | 70,00 | 100 | 1,40 | 3,00 | 5,00 |
| verspaetung | 10 | 0 | 2025-11-12 | 432.986.483.584.246.336 | 2 | 11,20 | 3,00 | 486 | 10,46 | 2,95 | 26,10 | 4 | 0,07 | -0,04 | -0,48 | 0,72 | 0,87 | -0,49 | 1,67 | -10,00 | 13,00 | -0,17 | -11,15 | 11,00 | 0,09 | 30,00 | 80,00 | 30,00 | 100 | 5,32 | 3,00 | 10,00 |
| verspaetung | 20 | 0 | 2025-11-12 | 432.986.483.584.246.336 | 2 | 11,20 | 3,00 | 486 | 14,43 | 3,95 | 29,05 | 7 | 0,32 | 0,32 | -0,37 | 1,02 | 1,04 | -0,51 | 5,76 | -8,00 | 18,05 | -0,66 | -15,00 | 11,00 | 0,19 | 30,00 | 80,00 | 60,00 | 100 | 17,48 | 3,00 | 20,00 |
| ausfall | 0 | 0 | 2025-11-12 | 432.986.483.584.246.336 | 2 | 11,20 | 3,00 | 486 | 10,70 | 2,00 | 27,20 | 2 | -0,14 | -0,14 | -0,46 | 0,18 | 0,26 | -0,54 | 0,51 | -11,20 | 14,05 | -0,14 | -11,00 | 10,05 | 0,03 | 30,00 | 80,00 | 30,00 | 100 | – | 3,00 | – |
| verspaetung | 5 | 0 | 2025-11-12 | 8.428.548.126.982.986.752 | 2 | 11,82 | 0,00 | 486 | 8,56 | 1,00 | 24,00 | 2 | -0,24 | -0,24 | -0,67 | 0,18 | 0,28 | -0,77 | 0,24 | -13,05 | 12,05 | -0,79 | -11,00 | 8,00 | 0,04 | 40,00 | 160,00 | 40,00 | 100 | 4,29 | 0,00 | 5,00 |
| verspaetung | 10 | 0 | 2025-11-12 | 8.428.548.126.982.986.752 | 2 | 11,82 | 0,00 | 486 | 11,76 | 2,90 | 30,05 | 8 | 0,05 | 0,29 | -0,44 | 0,41 | 0,45 | -0,80 | 1,95 | -10,10 | 15,00 | -0,30 | -11,00 | 11,25 | 0,12 | 30,00 | 160,00 | 40,00 | 100 | 8,69 | 0,00 | 10,00 |
| verspaetung | 20 | 0 | 2025-11-12 | 8.428.548.126.982.986.752 | 2 | 11,82 | 0,00 | 486 | 14,15 | 3,95 | 28,05 | 10 | 0,34 | 0,37 | -0,33 | 0,91 | 1,06 | -0,82 | 5,02 | -9,00 | 22,00 | 0,98 | -11,20 | 16,25 | 0,20 | 30,00 | 60,00 | 50,00 | 100 | 20,58 | 0,00 | 20,00 |
| ausfall | 0 | 0 | 2025-11-12 | 8.428.548.126.982.986.752 | 2 | 11,82 | 0,00 | 486 | 12,27 | 1,00 | 28,05 | 3 | 0,05 | 0,29 | -0,54 | 0,55 | 0,62 | -0,75 | 1,47 | -11,10 | 23,00 | 1,79 | -11,00 | 21,05 | 0,09 | 30,00 | 160,00 | 30,00 | 100 | – | 0,00 | – |
| verspaetung | 5 | 1 | 2025-12-03 | -3.976.403.648.475.326.976 | 2 | 7,00 | 1,00 | 856 | 15,55 | 3,00 | 33,05 | 2 | 0,56 | 0,56 | 0,41 | 0,72 | 0,76 | 0,37 | 1,87 | -10,05 | 13,25 | -0,75 | -10,00 | 0,00 | 0,08 | 20,00 | 40,00 | 20,00 | 100 | 2,66 | 1,00 | 5,00 |
| verspaetung | 10 | 1 | 2025-12-03 | -3.976.403.648.475.326.976 | 2 | 7,00 | 1,00 | 856 | 23,66 | 7,00 | 46,15 | 10 | 0,39 | 0,39 | -0,26 | 0,93 | 1,35 | -0,34 | 5,36 | -11,00 | 23,15 | -0,81 | -10,00 | 3,30 | 0,20 | 30,00 | 110,00 | 40,00 | 100 | 8,54 | 1,00 | 10,00 |
| verspaetung | 20 | 1 | 2025-12-03 | -3.976.403.648.475.326.976 | 2 | 7,00 | 1,00 | 856 | 27,91 | 12,00 | 49,05 | 14 | 0,38 | 0,39 | -0,13 | 0,70 | 1,14 | -0,42 | 9,35 | -9,00 | 31,00 | -0,45 | -10,05 | 11,00 | 0,20 | 40,00 | 210,00 | 50,00 | 100 | 20,18 | 1,00 | 20,00 |
| ausfall | 0 | 1 | 2025-12-03 | -3.976.403.648.475.326.976 | 2 | 7,00 | 1,00 | 856 | 26,61 | 9,00 | 50,15 | 7 | 0,08 | 0,26 | -0,45 | 0,35 | 0,36 | -0,60 | 2,18 | -17,10 | 24,00 | 1,00 | -10,00 | 20,00 | 0,08 | 40,00 | 190,00 | 40,00 | 100 | – | 1,00 | – |
| verspaetung | 5 | 1 | 2025-12-03 | -4.842.757.645.342.559.232 | 2 | 17,82 | 0,00 | 856 | 15,18 | 5,00 | 29,15 | 6 | 0,38 | 0,35 | -0,10 | 0,87 | 1,23 | -0,47 | 3,97 | -8,00 | 20,05 | 0,01 | -0,05 | 0,00 | 0,08 | 30,00 | 50,00 | 30,00 | 100 | 5,53 | 0,00 | 5,00 |
| verspaetung | 10 | 1 | 2025-12-03 | -4.842.757.645.342.559.232 | 2 | 17,82 | 0,00 | 856 | 19,51 | 7,00 | 40,00 | 11 | 0,45 | 0,39 | 0,28 | 0,88 | 1,45 | -0,71 | 7,38 | -7,05 | 27,05 | -0,20 | -1,10 | 0,00 | 0,21 | 30,00 | 170,00 | 50,00 | 100 | 11,78 | 0,00 | 10,00 |
| verspaetung | 20 | 1 | 2025-12-03 | -4.842.757.645.342.559.232 | 2 | 17,82 | 0,00 | 856 | 26,25 | 13,00 | 46,10 | 21 | 0,46 | 0,32 | 0,27 | 1,07 | 1,74 | -0,71 | 12,72 | -5,10 | 42,05 | -0,82 | -10,00 | 1,00 | 0,31 | 40,00 | 80,00 | 60,00 | 100 | 20,14 | 0,00 | 20,00 |
| ausfall | 0 | 1 | 2025-12-03 | -4.842.757.645.342.559.232 | 2 | 17,82 | 0,00 | 856 | 19,05 | 7,95 | 38,15 | 2 | -0,64 | -0,64 | -0,81 | -0,48 | -0,44 | -0,85 | -0,40 | -16,05 | 18,05 | -0,05 | -9,00 | 3,15 | 0,02 | 40,00 | 180,00 | 40,00 | 100 | – | 0,00 | – |
| verspaetung | 5 | 0 | 2025-12-03 | 3.662.156.649.843.842.560 | 2 | 11,82 | 0,00 | 591 | 7,12 | 1,00 | 21,05 | 3 | -0,23 | -0,30 | -0,77 | 0,33 | 0,49 | -0,89 | 0,00 | -6,00 | 9,00 | -0,31 | 0,00 | 0,00 | 0,04 | 30,00 | 180,00 | 30,00 | 100 | 5,21 | 0,00 | 5,00 |
| verspaetung | 10 | 0 | 2025-12-03 | 3.662.156.649.843.842.560 | 2 | 11,82 | 0,00 | 591 | 11,99 | 3,95 | 29,15 | 6 | 0,08 | 0,30 | -0,75 | 0,69 | 0,87 | -1,21 | 2,10 | -7,00 | 12,00 | -0,61 | -7,20 | 0,00 | 0,13 | 30,00 | 60,00 | 40,00 | 100 | 10,34 | 0,00 | 10,00 |
| verspaetung | 20 | 0 | 2025-12-03 | 3.662.156.649.843.842.560 | 2 | 11,82 | 0,00 | 591 | 16,56 | 6,95 | 30,10 | 7 | 0,36 | 0,38 | -0,31 | 1,00 | 1,79 | -1,22 | 6,23 | -6,10 | 19,10 | -0,21 | 0,00 | 5,00 | 0,18 | 30,00 | 70,00 | 60,00 | 100 | 20,31 | 0,00 | 20,00 |
| ausfall | 0 | 0 | 2025-12-03 | 3.662.156.649.843.842.560 | 2 | 11,82 | 0,00 | 591 | 12,75 | 3,00 | 27,05 | 3 | -0,42 | -0,33 | -1,06 | 0,19 | 0,32 | -1,24 | 1,47 | -11,05 | 18,00 | 3,84 | 0,00 | 20,00 | 0,04 | 30,00 | 60,00 | 30,00 | 100 | – | 0,00 | – |
| verspaetung | 5 | 0 | 2025-12-03 | -580.870.208.978.951.808 | 2 | 10,48 | 0,00 | 591 | 8,18 | 1,00 | 23,05 | 1 | -0,71 | -0,71 | -0,71 | -0,71 | -0,71 | -0,71 | 0,27 | -7,20 | 10,05 | -0,20 | 0,00 | 0,00 | 0,05 | 20,00 | 50,00 | 20,00 | 100 | 5,83 | 0,00 | 5,00 |
| verspaetung | 10 | 0 | 2025-12-03 | -580.870.208.978.951.808 | 2 | 10,48 | 0,00 | 591 | 12,30 | 2,00 | 27,05 | 7 | 0,60 | 0,49 | -0,15 | 1,35 | 1,69 | -0,84 | 5,92 | -6,00 | 13,05 | -0,22 | 0,00 | 0,00 | 0,17 | 30,00 | 70,00 | 50,00 | 100 | 8,64 | 0,00 | 10,00 |
| verspaetung | 20 | 0 | 2025-12-03 | -580.870.208.978.951.808 | 2 | 10,48 | 0,00 | 591 | 16,91 | 8,00 | 29,05 | 12 | 0,46 | 0,35 | 0,27 | 0,89 | 1,80 | -0,87 | 9,05 | -3,05 | 21,10 | -0,61 | -11,00 | 3,20 | 0,23 | 30,00 | 80,00 | 60,00 | 100 | 19,76 | 0,00 | 20,00 |
| ausfall | 0 | 0 | 2025-12-03 | -580.870.208.978.951.808 | 2 | 10,48 | 0,00 | 591 | 14,46 | 3,95 | 30,00 | 9 | 0,24 | 0,26 | -0,47 | 0,60 | 1,76 | -0,89 | 5,94 | -7,00 | 18,10 | 1,80 | 0,00 | 10,05 | 0,19 | 40,00 | 80,00 | 60,00 | 100 | – | 0,00 | – |

**Befund.** Verspätung eines Zuges wird im Modell kaum weitergegeben: bei +5/+10/+20 min Eintrittsverspätung ändert sich die Summe der Endverspätungen aller anderen Züge im Mittel um 0,1–6,5 Zug-Minuten (außerhalb HVZ) bzw. 3,4–12,0 (HVZ); die Zugwerte streuen stark (Minima der Zug-Minuten bis −10, Maxima bis +41 bei q95). Zusätzliche Ausfälle sind im Mittel nicht von 0 verschieden (Mittel zwischen −0,9 und +0,1 für Verspätung, +1,8/+0,6 für gestrichene Fahrt). Der Rauschpegel (zufälliges Umkippen der Ziehungen nach kleinen Änderungen der Wahrscheinlichkeiten) ist so groß wie der Effekt bei +5 min. Die Abklingzeiten (Mittel 72–228 min nach KI-Regel, 38–88 min nach der 0,1-min-Regel) sind unsicher, weil sie auf dem Rauschen aufsetzen.

**Sperrung (experimentell).** Regel: Station s von t1 bis t2 nicht passierbar; Züge, deren Ankunft an s in [t1, t2) läge, kommen frühestens bei t2 + Fahrzeit an, danach im Abstand ≥ 2 min; Rest Modell. Wartende Züge blockieren in der Simulation die Vorstation nicht. Datensuche: 18.698 Lücken > 15 min an Station/Richtung tagsüber (5–23 Uhr) im ganzen Jahr; die meisten an Randstationen/ Randzeiten; nur 1.693 mit ≥ 3 Zügen an der Vorstation in der Lücke; im Testzeitraum mit ≥ 8 Zügen und ≥ 20 min: 6 Fälle (nach Duplikatentfernung je Tag/Station):

| opday | rank | richtung | t1_m | t2_m | luecke_min | zuege_vorstation_in_luecke | monat |
|---|---|---|---|---|---|---|---|
| 2025-12-05 | 5 | West | 29.415.859 | 29.415.912 | 54 | 22,000 | 12 |
| 2025-12-01 | 7 | Ost | 29.410.189 | 29.410.237 | 49 | 18,000 | 12 |
| 2025-10-14 | 5 | West | 29.340.637 | 29.340.685 | 49 | 16,000 | 10 |
| 2025-10-05 | 5 | Ost | 29.327.979 | 29.328.011 | 33 | 9,000 | 10 |
| 2025-10-28 | 5 | Ost | 29.360.465 | 29.360.497 | 33 | 12,000 | 10 |
| 2025-10-15 | 9 | West | 29.341.776 | 29.341.804 | 29 | 11,000 | 10 |

Test (100 Läufe; Fenster t1−60 min … t2+150 min; Kennzahlen für Züge, die in den 60 min ab t2 an s bzw. der nächsten Station ankommen): Zahl der Züge, mittlere Verspätung, Minuten bis zur ersten Ankunft nach t2:

| opday | rank | richtung | t1_m | t2_m | luecke_min | zuege_vorstation_in_luecke | monat | echt_an_s_zuege_60min_nach_t2 | echt_an_s_mean_verspaetung_60min | echt_an_s_erste_ankunft_nach_t2_min | echt_naechste_zuege_60min_nach_t2 | echt_naechste_mean_verspaetung_60min | echt_naechste_erste_ankunft_nach_t2_min | basis_an_s_zuege_60min_nach_t2 | basis_an_s_mean_verspaetung_60min | basis_an_s_erste_ankunft_nach_t2_min | basis_naechste_zuege_60min_nach_t2 | basis_naechste_mean_verspaetung_60min | basis_naechste_erste_ankunft_nach_t2_min | sperrung_an_s_zuege_60min_nach_t2 | sperrung_an_s_mean_verspaetung_60min | sperrung_an_s_erste_ankunft_nach_t2_min | sperrung_naechste_zuege_60min_nach_t2 | sperrung_naechste_mean_verspaetung_60min | sperrung_naechste_erste_ankunft_nach_t2_min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-12-05 | 5 | West | 29.415.859 | 29.415.912 | 54 | 22,000 | 12 | 24,000 | 7,000 | 0,000 | 23,000 | 7,087 | 2,000 | 25,030 | 7,610 | 1,400 | 24,850 | 7,289 | 1,100 | 29,950 | 55,113 | 1,050 | 28,950 | 55,028 | 2,490 |
| 2025-12-01 | 7 | Ost | 29.410.189 | 29.410.237 | 49 | 18,000 | 12 | 22,000 | 2,818 | 0,000 | 22,000 | 2,727 | 4,000 | 22,710 | 2,906 | 0,920 | 23,180 | 3,101 | 0,820 | 29,650 | 43,874 | 1,350 | 28,550 | 44,618 | 3,220 |
| 2025-10-14 | 5 | West | 29.340.637 | 29.340.685 | 49 | 16,000 | 10 | 20,000 | 8,000 | 0,000 | 20,000 | 8,100 | 6,000 | 20,880 | 6,311 | 1,630 | 20,830 | 5,909 | 1,790 | 29,850 | 43,180 | 1,150 | 28,850 | 43,029 | 2,660 |
| 2025-10-05 | 5 | Ost | 29.327.979 | 29.328.011 | 33 | 9,000 | 10 | 22,000 | 3,273 | 0,000 | 21,000 | 3,143 | 2,000 | 20,640 | 2,497 | 1,060 | 20,820 | 2,685 | 0,700 | 29,080 | 23,391 | 2,010 | 28,520 | 24,310 | 3,610 |
| 2025-10-28 | 5 | Ost | 29.360.465 | 29.360.497 | 33 | 12,000 | 10 | 23,000 | 5,435 | 0,000 | 22,000 | 6,227 | 2,000 | 22,020 | 5,141 | 0,730 | 21,840 | 5,623 | 1,140 | 29,000 | 27,120 | 2,050 | 28,240 | 28,058 | 3,590 |
| 2025-10-15 | 9 | West | 29.341.776 | 29.341.804 | 29 | 11,000 | 10 | 4,000 | 20,500 | 0,000 | 5,000 | 30,000 | 2,000 | 9,650 | 15,416 | 3,540 | 9,720 | 15,525 | 4,800 | 13,350 | 24,271 | 3,730 | 13,250 | 24,500 | 5,870 |

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

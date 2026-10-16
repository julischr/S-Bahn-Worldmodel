# Erweiterung: `eintritt="puenktlich"` (Tag ohne Eingriff, alle Züge pünktlich)

**Bedienung.** `simulate(day, models, R, ..., eintritt="echt"|"puenktlich")` in `src/simulator/sim.py`; `simulate_day(datum, ..., eintritt=...)` und CLI `--eintritt puenktlich` in `api.py`. Standard bleibt `"echt"` (Verhalten unverändert; neue Parameter nur zusätzlich). Bei `"puenktlich"` treten **alle** Züge des Tages (auch die in den Daten ausgefallenen) am ersten Halt ihres Pfades mit 0 min Verspätung und ohne Ausfall ein; ab dort zieht das Modell. Nur ganze Tage (kein `t_start`, kein `forced`). Auswertung: `uv run python -m src.simulator.puenktlich 2025-11-12 2025-11-16 --runs 50` (Ausgaben `outputs/simulator/puenktlich_*.csv`).

Geprüft an zwei Tagen (Mi 2025-11-12, So 2025-11-16), je 50 Läufe; das sind Beispiele, keine Stichprobe.

## So sieht der Tag ohne Eingriff aus

Mittlere Verspätung (min) je Station, Mittel über Läufe; „echt“ = Replay der echten Eintrittsverspätungen (Sim), „pünktl.“ = alle 0 min:

| Rang | Mi Ost echt | Mi Ost pünktl. | Mi West echt | Mi West pünktl. | So Ost echt | So Ost pünktl. | So West echt | So West pünktl. |
|---|---|---|---|---|---|---|---|---|
| 0 Pasing | 4,60 | 0,00 | 7,44 | 2,42 | 1,76 | 0,00 | 4,53 | 3,26 |
| 3 Donnersbergerbr. | 5,63 | 0,70 | 6,96 | 2,09 | 2,46 | 0,79 | 2,43 | 1,51 |
| 6 Karlsplatz | 6,63 | 1,17 | 6,38 | 1,69 | 3,06 | 1,71 | 1,68 | 0,73 |
| 9 Rosenheimer Pl. | 6,90 | 1,15 | 5,81 | 1,36 | 3,60 | 2,05 | 2,54 | 1,30 |
| 11 Leuchtenbergring | 8,29 | 1,66 | 3,92 | 0,00 | 1,71 | 1,40 | 2,44 | 0,00 |

(Alle 12 Stationen in `puenktlich_profil.csv`.) Ein pünktlich startender Tag baut im Modell also auf der Strecke etwa 1,4–2,4 min Verspätung auf (Ost: 0 → 1,7 bzw. 1,4; West: 0 am Leuchtenbergring → 2,4 bzw. 3,3 min an Pasing); Mittel über alle Ankünfte 1,21 (Mi) bzw. 1,24 min (So).

Ausfallquote (Anteil der Halte nach dem Eintritt, die im Lauf ausfallen): Mi 0,11 % (echt-Replay-Sim: 3,2 %; in den Daten dieses Tages 1,2 %), So 0,28 % (echt-Replay-Sim 0,35 %). Ohne die in den Daten vorhandenen Ausfälle am Eintritt gibt es fast keine Ausfälle mehr; die in der Sim entstehenden Ausfälle sind seltene Teilausfälle unterwegs. Kausalitätsklammern: Mi 11.180 (echt-Replay 2.055) bei 50 Läufen, So 592 (804): am Mittwoch greift die Klammer im pünktlichen Tag 5-mal so oft ein, ein Zeichen, dass die Ziehungen dort häufiger Fahrzeiten ≤ 0 min ergeben.

## Wie oft kommen die entstehenden Merkmalswerte im Training vor?

Vergleich der Entscheidungszeilen des Simulators (Mi 468.205, So 119.327 Zeilen) mit dem Training (Jan–Jul, 1,58 Mio. Zeilen) und mit den echten Zeilen desselben Tages. Zelle = (HVZ, Eigenverspätung d_k, Netzmittel der letzten 15 min net_mean15 gerundet, Abstand zum Vorgänger act_gap_pred).

| Kennzahl | Mi pünktlich | Mi echte Zeilen | So pünktlich | So echte Zeilen |
|---|---|---|---|---|
| Zeilen in Zellen ohne Trainingszeile | 0,15 % | 0 % | 0,69 % | 0 % |
| … mit < 30 Trainingszeilen | 2,3 % | 3,4 % | 5,3 % | 0,7 % |
| … mit < 300 Trainingszeilen | 14,7 % | 29,4 % | 29,1 % | 14,8 % |
| **Anteil der HVZ-Zeilen mit Netzmittel < 1 min** (Training: 1,3 %) | **22,4 %** | 0 % | **47,9 %** | 2,5 % |
| net_mean15 unter Training-P1 (0,44 min) | 2,2 % | 0,4 % | 9,0 % | 0 % |

**Deutlicher Hinweis: Die Lage ist selten.** Das Einzelmerkmal liegt meist noch im Trainingsbereich, aber die Kombination „Hauptverkehrszeit bei praktisch verspätungsfreiem Netz“ kam im Training nur in 1,3 % der HVZ-Zeilen vor und tritt im pünktlichen Tag in 22 % (Mi) bzw. 48 % (So) der HVZ-Zeilen auf; am Sonntag liegt außerdem 9 % der Zeilen unter dem 1.-Perzentil des Netzmittels. Ebenso fehlen im pünktlichen Tag durchgehend Ausfälle (Zähler cancel_net30 im Mittel 0,3 gegen 40,9 am echten Mittwoch, wo die Trainings-P99 bei 63 liegt). Die Vorhersagen in diesen Zeilen sind daher teilweise Extrapolation; insbesondere der Mittwoch-Wert „1,2 min Aufbau“ ist nicht durch Prüfsteine abgesichert. Die Prüfsteine a–e betrafen ausschließlich echte Eintrittsverspätungen. Die Zellendefinition ist eine einfache Wahl (4 Merkmale, Rundung); sie zeigt Seltenheit, beweist aber keine Fehlvorhersage.

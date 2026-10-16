# Korrektur Datenausfall (Echtzeitfeed fehlt) — Methode, Ergebnisse, Grenzen

## Was war los
In `data/02_events/final` gibt es Zeiträume, in denen der Echtzeitfeed nichts geliefert hat. Die Zeilen sind dann reine Fahrplanzeilen: `update_timestamp` leer, `time_real = time_schedule`, `delay = 0`, keine Ausfälle, keine `message_codes`. Das sieht wie perfekte Pünktlichkeit aus. Betroffen sind 21 volle Betriebstage (2025-04-09, 2025-09-02/03, 09-09 bis 09-17, 09-19 bis 09-27) und viele Teilblöcke (u. a. 2025-01-11/12/13, 01-30, 03-19/20, 09-01, 09-04, 09-08, 09-18, 09-28, 10-03).

## Systematische Suche (`src/data/find_ausfall_bloecke.py` → `data/ausfall_bloecke.csv`, `data/ausfall_tage.csv`)
Alle Ankünfte aller Stationen/Richtungen 2025, Block = Station × Richtung × lokale Stunde:
| Regel | Kriterium | Blockstunden |
|---|---|---|
| A kein_update | ≥ 50 % der Zeilen ohne `update_timestamp` (n ≥ 2; Normalbetrieb Median 0,04 %) | 117.184 |
| B netz_nur_null | im ganzen Netz in der Stunde ≥ 95 % der Ankünfte mit delay = 0, n ≥ 200 | 524 Stunden (alle Stationen) |
| C station_nur_null | ≥ 10 aufeinanderfolgende Ankünfte einer Station/Richtung, alle delay = 0, keine Ausfälle, ≥ 60 min, außerhalb A/B | 6.166 Läufe = 47.767 Blockstunden (davon nur 264 an Stammstreckenstationen) |
| D codes_fehlen | im ganzen Netz keine einzige message_code in der Stunde (n ≥ 300) | 517 Stunden, nur Scope „nur_codes“, nicht entfernt |
| E konstante Verspätung ≠ 0 | n ≥ 8, delay konstant | 3 Blöcke, nur gemeldet |
Tage: voller Ausfalltag, wenn ≥ 90 % der Stammstrecken-Stundenblöcke 5–23 Uhr betroffen sind (21 Tage); 13 weitere Tage mit > 2 % betroffenen Blöcken (Teilausfälle). Regel C ist grob (sie erwischt auch zufällig sehr ruhige Stationen); sie betrifft die Stammstrecke kaum.

## Zentral eingebaut (nichts Altes überschrieben)
- `src/utils/ausfall_bloecke.py`: Schalter `S_BAHN_AUSFALL_FILTER=1`. Mit Schalter lesen alle Stufen `data/02_events/final_bereinigt/` (gleiches Schema; Zeilen der Blöcke A, B, C entfernt: 1,5 Mio. von 19 Mio. Zeilen, davon 73 % des Septembers, 4,4 % Januar, 3,5 % April) und `run_all.EXCLUDED_DATES` enthält zusätzlich die 21 vollen Ausfalltage. Ohne Schalter bleibt alles wie vorher (alte Outputs unverändert).
- `src/data/build_bereinigt.py` erzeugt die bereinigte Kopie; `BlockIndex` testet Zeitfenster gegen Blöcke.
- Die Neuläufe liefen in einer Kopie des Projekts (`../S-Bahn_bereinigt`, Skript `run_korrektur.sh`), damit keine alten Ergebnisse überschrieben werden. Vergleichstabellen: `outputs/korrektur_ausfall/vergleich_*.csv`; Skript `src/analysis/korrektur_ausfall_vergleich.py`.
- Survival: Episoden und Placebo-Zeitpunkte, deren Fenster [−60, +240] min einen Ausfall-Block der Station/Richtung (oder einen Netzblock) berührt, werden zusätzlich verworfen (97 von 14.336 Episoden, 5.587 von 46.123 bzw. 1.263 von 41.194 Placebos). Ohne diesen Schritt (nur Zeilen entfernen) entsteht eine neue Verzerrung: Placebos in entfernten Blöcken „warten“ bis zum ersten Zug nach der Lücke.
- Ein Fehler in `km.ci_perc` (NaN aus ∞ − ∞ bei nicht erreichten Medianen, Absturz) wurde behoben; ändert Altläufe nicht.

## Ergebnis: was sich ändert und was nicht
| Befund | alt | neu (bereinigt) | Urteil |
|---|---|---|---|
| BEG-Pünktlichkeit, Ankunft, Gesamtnetz | 82,70 % (n = 9.282.673) | 81,17 % (n = 8.525.180) | **ändert sich spürbar**: Lücke zu BEG 87,9 % wächst von 5,2 auf 6,7 pp; Stammstrecke 76,2 → 74,3 % |
| Ausfallquote (eigene) | 2,83 % | 3,07 % | steigt leicht (BEG 8,4 % bleibt weit entfernt) |
| Stufe 1, Code 34: Sockel [−60,−15] min | 80,7 s | 63,6 s | **sinkt um 21 %**; Spitze bei t0 151,8 → 136,1 s; Wert bei +60 min 49,5 → 33,8 s; Form (Spitze bei t0, Abklingen) bleibt |
| Survival A1 Median Störung | 60,7 min (54,6–68,1) | 61,4 (55,6–69,1) | unverändert |
| Survival A1 Median Placebo P1 / P2 | 11,5 / 9,0 min | 43,6 / 37,0 min | **ändert sich stark** |
| Survival A1 Differenz Störung − Placebo P1 / P2 | 49,2 (42,4–57,0) / 51,7 (45,2–59,6) min | 17,8 (11,9–24,4) / 24,4 (19,2–30,7) min | **schrumpft auf etwa ein Drittel bis die Hälfte; Richtung bleibt** |
| Survival A2 „ohne Wirkung“ Störung | 13,9 % | 13,9 % | unverändert |
| Survival A2 „ohne Wirkung“ Placebo P1 / P2 | 47,3 / 48,8 % | 22,7 / 22,1 % | **halbiert**; Abstand Störung–Placebo von 33 auf 9 pp |
| Survival A6 Störungsmediane (<3 … >20) | 15,2 / 23,4 / 58,8 / 121,1 / 236,7 | 15,2 / 23,4 / 59,1 / 121,6 / 236,7 | unverändert |
| Survival A6 Placebo P1 (<3 … >20) | 2,0 / 19,0 / 48,0 / 90,7 / 244,6 (n klein: 21.267) | 10,8 / 20,2 / 50,5 / 93,4 / 236,2 (n < 3: 6.819) | **nur Klasse <3 ändert sich** (66 % der alten Fälle lagen in Ausfallzeiten); Klassen ≥ 3 min ± 3 min |
| Survival Cox (Schwere-HR je +5 min / Vorbelastung / West / Sa / So) | 0,732 / 0,449 / 0,594 / 1,194 / 1,274 | 0,733 / 0,441 / 0,592 / 1,202 / 1,285 | unverändert |
| Zugebene 11.4.2 (Zuwachs vs Vorgänger, <5 min) | r = 0,064, Steigung 0,0145 | r = 0,056, Steigung 0,0129 | praktisch unverändert (n 867.069 → 796.613) |
| 11.8 Ausfallquote Folgezug | <5 min: 1,59 / 3,16 / 4,76 / 8,31 % | 1,93 / 3,16 / 4,76 / 8,28 % | nur Zelle „Vorgänger ≤ 2 min“ steigt (alle Abstände); Faktor 5,2 → 4,3; Gesamtrate 2,95 → 3,19 % |
| VAR Ost, BIC-Lag | p = 5 | p = 4 (647 statt 686 Episoden) | **Lagwahl ändert sich** |
| VAR 25-min-Peak Pasing→Leuchtenbergring bei **festem** p | Ost 25 min, 53,7 s; West (p = 17) 25 min, 45,1 s | Ost 25 min, 54,5 s; West 25 min, 46,8 s | **unverändert** |
| VAR Peak bei BIC-Lag p = 4 (Ost) | – | 5 min (Amplitude 2,6 s) | Peakzeit hängt am Lag: bereits in BEFUNDE_VAR 10.1 untersucht; hier ist der Wechsel von p = 5 auf p = 4 die Ursache, nicht die Ausbreitung |
| VAR mittlere Fremdvarianz (FEVD) Ost / West | 68,8 / 71,8 % | 67,9 / 70,5 % | −1 pp, Muster unverändert |
| VAR Netto-Kopplung Ost | Pasing +2,54, Ostbahnhof −0,79 | Pasing +2,52, Ostbahnhof −0,78 | unverändert |

## Grenzen / nicht gerechnet
- Stufe 1: nur die Hauptkurve Code 34 neu gerechnet, nicht die Nachdiagnosen (Terzile, Dauerklassen, neue Codes, Sommerzeit-Paare).
- Survival: Placebos und Episoden mit Block im Fenster werden verworfen (Regel von mir festgelegt); die Einordnung A4/A5/B-Befunde, Kipppunkt-Tests und Bootstrap-KIs von Cox wurden nicht einzeln verglichen, nur die Haupt-HR.
- VAR: Bootstrap-Bänder nicht neu bewertet; West mit 200 Ziehungen; „Quellen/Senken“ nur Ost-Netto-Kopplung verglichen (West-Haupttabellen der alten Läufe verwenden eine andere Cholesky-Ordnung, siehe BEFUNDE_VAR 10.1, daher kein direkter Vergleich).
- Plausibilitätstests außer 11.4.2/11.8 nicht neu gerechnet. Stufe 4 (Simulator) nicht neu trainiert: sein Trainingsdatensatz (Jan–Jul) enthält ca. 2 % Ausfallzeilen (alle Stationen); der Kalibrierungsmonat September war überwiegend Ausfall. Mit dem Schalter lässt sich Stufe 4 neu rechnen.
- Regel C und der Datenausfall in Zeiten mit Teil-Updates (z. B. 2025-01-30 Nachmittag: Updates vorhanden, aber alle Verspätungen 0) sind nur über die Nullbedingung erfasst; es kann weitere unentdeckte Muster geben.

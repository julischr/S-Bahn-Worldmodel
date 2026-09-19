# Befunde: Stufe 2 -- VAR-Modell auf dem Zustandsvektor der Stammstrecke

Alle Zahlen in diesem Dokument stammen aus dem tatsaechlichen Lauf des Codes unter
`src/analysis/var/` gegen `data/03_state_vector/state_vector_5min.parquet` (Jahr 2025,
5-Minuten-Raster). Primaeres Ergebnis ist die Richtung **Ost** (West->Ost-fahrende Zuege,
12 Stationen). Die Richtung **West** wurde zusaetzlich vollstaendig gerechnet (siehe
Abschnitt "Richtungsvergleich") und stuetzt die Ost-Befunde. Nichts an Stufe 1 (`BEFUNDE.md`,
`src/` ausserhalb von `src/analysis/var/`, bestehende `outputs/*`) wurde veraendert -- diese
Datei und `outputs/var/` sind komplett neu.

**Wichtigste Einschraenkung vorab:** Dies ist ein reduced-form VAR auf einer aggregierten
Kennzahl (`delay_mean` je 5-Min-Fenster und Station), kein strukturelles Kausalmodell. Die
Cholesky-Orthogonalisierung unterstellt eine rekursive kausale Ordnung (hier: geografisch
West->Ost) -- das ist eine Annahme, keine bewiesene Tatsache, auch wenn sie fuer eine
Zuglaufrichtung sehr plausibel ist (siehe unten, Robustheitscheck). Ausserdem sind die
Residuen weder normalverteilt noch vollstaendig weiss (siehe Abschnitt Residualdiagnostik)
-- die Konfidenzbaender aus dem Bootstrap sind trotzdem informativ, aber nicht im strengen
Sinn "exakt".

## 1. Datenaufbereitung

### 1.1 Pivotierung
`data/03_state_vector/state_vector_5min.parquet` enthaelt 2.523.144 Zeilen. Gefiltert auf
`richtung == "Ost"` und pivotiert auf die 12 Stammstrecken-Stationen (Spalten in geografischer
Reihenfolge West->Ost, siehe `src/analysis/var/stations.py`) ergibt eine durchgehende
5-Minuten-Matrix mit 105.129 Zeitpunkten (2025-01-01 01:05 bis 2026-01-01 01:45, Europe/Berlin).
Der Anteil fehlender `delay_mean`-Werte (keine Zuege im Fenster, vor allem nachts) liegt je
Station zwischen 23,1% (Karlsplatz/Stachus) und 42,5% (Pasing) -- die Randstationen der
Strecke haben spuerbar mehr Nachtluecken als die zentralen.

### 1.2 Taktentfernung (Plot 1: `outputs/var/plots/01_periodogramm.png`)
Fourier-Regression (OLS je Station) mit Tages- und Wochen-Harmonischen. Die Anzahl der
Harmonischen wurde ueber ein Gitter `n_daily in 1..8`, `n_weekly in 0..3` per BIC gewaehlt
(`select_harmonics`, an drei Beispielstationen getestet: Pasing, Hauptbahnhof (tief),
Ostbahnhof) -- BIC waehlt an allen dreien konsistent **n_daily=4, n_weekly=3** (Perioden
24h/12h/8h/6h und 168h/84h/56h), AIC waere leicht groesser gewesen (n_daily=7-8), der
Unterschied in der Residualvarianz ist aber marginal (< 0,1%). Mit dieser Wahl wurde fuer
alle 12 Stationen gefittet (`outputs/var/fourier_info.csv`).

**Ueberraschender Befund:** Die Fourier-Terme erklaeren nur **2,0% bis 5,7%** der
Gesamtvarianz von `delay_mean` (Pasing 2,3%, Hauptbahnhof (tief) 3,2%, Leuchtenbergring
5,7%). Der ganz ueberwiegende Teil der Varianz in den Verspaetungen ist NICHT der
Fahrplantakt, sondern stochastische Stoerungen. Das Lomb-Scargle-Periodogramm (Plot 1,
berechnet direkt auf den tatsaechlich beobachteten, nicht interpolierten Punkten, um kein
kuenstliches 24h-Artefakt durch Luecken-Interpolation zu erzeugen) zeigt: die exakten
24h/12h/8h/6h-Frequenzen werden durch die Regression vollstaendig auf Leistung 0 gesetzt
(rechnerisch exakt, da Teil der Regressorbasis), aber in der unmittelbaren Nachbarschaft
dieser Perioden bleibt sichtbare spektrale Leistung (Leckage) bestehen -- ein Hinweis darauf,
dass das tatsaechliche Tagesprofil nicht exakt periodisch mit konstanter Amplitude ist
(Werktag/Wochenende-Unterschiede, saisonale Variation), was ein einzelner, das ganze Jahr
ueber konstanter Fourier-Fit nicht vollstaendig einfangen kann. Mit den Residuen wird ab
hier weitergearbeitet.

### 1.3 Nachtluecken und Episoden
Kurze Luecken (<= 12 Schritte = 60 Minuten) wurden linear interpoliert
(`interpolate_short_gaps`, nur innerhalb bestehender Daten, keine Extrapolation an den
Raendern). Anschliessend wurde je Kalendertag in maximal zusammenhaengende Bloecke zerlegt,
in denen ALLE 12 Stationen gleichzeitig nicht-NaN sind (`build_daily_episodes`) --
Episodengrenzen liegen also nie ueber Mitternacht hinweg. Ergebnis: **686 Episoden**
(`outputs/var/episoden_meta.csv`), mittlere Laenge 10,2 Betriebsstunden (122 Beobachtungen),
Median 7,4 Stunden (89 Beobachtungen), laengste Episoden ca. 19,25 Stunden (231
Beobachtungen = ein voller Betriebstag 04:30-23:45 etwa). Dass viele Tage in zwei Episoden
zerfallen (Median deutlich unter dem Maximum), liegt vor allem an Pasing und
Leuchtenbergring: als Randstationen der Strecke haben sie laengere/mehr Nachtluecken, die
den gemeinsamen Beobachtungsblock aller 12 Stationen zerschneiden.

### 1.4 Auffaellige Betriebstage (`outputs/var/ausgeschlossene_betriebstage.csv`)
Datengetrieben identifiziert ueber `n_zuege` je Kalendertag im Verhaeltnis zum Median des
jeweiligen Wochentagtyps (z.B. alle Samstage). Tage mit weniger als 30% des
Wochentagtyp-Medians wurden ausgeschlossen -- **12 Tage**:

| Datum | Wochentag | n_zuege | Anteil am Wochentag-Median | Vermutlicher Grund |
|---|---|---|---|---|
| 2025-05-10/11 | Sa/So | 92 / 85 | 2,1% / 2,0% | Wochenend-Komplettsperrung |
| 2025-10-18/19 | Sa/So | 125 / 123 | 2,8% / 2,8% | Wochenend-Komplettsperrung (vom Nutzer benannt, bestaetigt) |
| 2026-01-01 | Do | 270 | 5,2% | Jahresgrenze -- Datensatz endet 2026-01-01 01:55, nur 1. Betriebsstunde vorhanden (Rand-Artefakt, kein echter Ausfall) |
| 2025-01-11/12 | Sa/So | 548 / 855 | 12,2% / 19,8% | Stark reduzierter Wochenendbetrieb |
| 2025-06-07 bis 06-12 | Sa-Do | 1193-1445 | 25,7-27,7% | Ganze Woche reduzierter Betrieb (vermutlich mehrtaegige Bauarbeiten) |

Zum Vergleich: der Median ueber alle 366 Tage liegt bei 5.139 Zuegen, das Minimum unter den
NICHT ausgeschlossenen Tagen liegt bei 1.507 (2025-11-16, ein Sonntag mit spuerbar, aber
nicht extrem reduziertem Betrieb). Eine zweite Gruppe auffaelliger Tage wurde bewusst NICHT
ausgeschlossen: im gesamten November 2025 (v.a. 10.-23.11.) und am 30./31.10.2025 liegt die
Ausfallquote (`n_ausfall / (n_zuege+n_ausfall)`) bei 10-26% statt sonst ueblichen 1-5% --
das ist regulaerer Betrieb mit hoher Stoerungslast (n_zuege bleibt nahe am Normalniveau),
also genau die Art von Ereignis, die ein Ausbreitungsmodell erfassen soll, kein
Datenqualitaetsproblem.

## 2. Stationaritaet (`outputs/var/adf_ergebnisse.csv`, `..._je_episode.csv`)
ADF-Test (`statsmodels.tsa.stattools.adfuller`, `autolag="AIC"`) je Station auf jeder
Episode mit mindestens 150 Beobachtungen (264 der 686 Episoden), Ergebnisse aggregiert:

| Station | Median ADF-Stat | Median p-Wert | Anteil Episoden stationaer (5%) |
|---|---|---|---|
| Pasing | -5,45 | 2,6e-06 | 82,9% |
| Laim | -5,38 | 3,7e-06 | 86,1% |
| Hirschgarten | -5,39 | 3,6e-06 | 87,5% |
| Donnersbergerbruecke | -5,09 | 1,4e-05 | 85,0% |
| Hackerbruecke | -5,07 | 1,6e-05 | 83,6% |
| Hauptbahnhof (tief) | -5,16 | 1,0e-05 | 88,2% |
| Karlsplatz/Stachus | -4,87 | 4,0e-05 | 86,4% |
| Marienplatz | -4,80 | 5,5e-05 | 83,6% |
| Isartor | -4,84 | 4,6e-05 | 84,0% |
| Rosenheimer Platz | -4,66 | 1,0e-04 | 85,7% |
| Ostbahnhof | -4,71 | 8,1e-05 | 82,9% |
| Leuchtenbergring | -4,38 | 3,2e-04 | 80,5% |

Alle Median-Teststatistiken liegen weit unter dem 5%-kritischen Wert (~-2,86) -- die
Residuenreihen sind ueberwiegend (80-88% der Episoden) stationaer. Das verbleibende
Sechstel/Fuenftel nicht-stationaerer Episoden ist erwartbar bei kurzen Episoden mit wenig
Beobachtungen (Testpower sinkt) und gelegentlichen laengeren Stoerepisoden mit trendartigem
Verlauf innerhalb eines Tages. Fuer die gepoolte VAR-Schaetzung wird das in Kauf genommen --
eine episodenweise Differenzierung wuerde die ohnehin kurzen Episoden weiter verkuerzen und
die fuer IRF-Interpretation zentrale Level-Information zerstoeren.

## 3. VAR-Modell

### 3.1 Schaetzverfahren
Episodenweise gepoolte Lag-Design-Matrix (Lags nie ueber Episodengrenzen hinweg, siehe
`build_lagged_design` in `src/analysis/var/var_model.py`), anschliessend Equation-by-Equation
OLS ueber alle 12 Gleichungen gemeinsam (aequivalent zur VAR-ML-Schaetzung).

### 3.2 Lag-Wahl (Plot 2: `outputs/var/plots/02_aic_bic.png`, Tabelle
`outputs/var/aic_bic_lag_wahl.csv`)
Fuer p=1..20 (5 bis 100 Minuten) berechnet nach Luetkepohl-Formel. **BIC hat ein klares,
inneres Minimum bei p=5** (25 Minuten, BIC=114,419), danach steigt BIC monoton bis p=20
(114,714). **AIC** faellt dagegen ueber den ganzen getesteten Bereich nur langsam und flach
weiter, mit einem lokalen Minimum bei p=17 (AIC=114,327) in einem insgesamt flachen Plateau
ab p~15. AIC und BIC divergieren hier deutlich (p=5 vs. p=17) -- klassisches Symptom von
BICs staerkerer Bestrafung der K²p-vielen Parameter (bei K=12 sind das 144 zusaetzliche
Parameter pro Lag-Schritt). **Gewaehlt wurde p=5** (BIC-Minimum): mit 80.530 nutzbaren
Beobachtungen und ohnehin schon 12*(12*5+1)=732 AR-Parametern ist das Risiko der
Ueberparametrisierung bei p=17 (12*(12*17+1)=2.460 Parameter) real, und BIC ist fuer diese
Groessenordnung an Beobachtungen die konservativere, robustere Wahl.

### 3.3 Stabilitaet (Plot 3: `outputs/var/plots/03_eigenwerte.png`, Tabelle
`outputs/var/eigenwerte.csv`)
Alle 60 Eigenwerte der Begleitmatrix liegen innerhalb des Einheitskreises: **maximaler
Betrag 0,9437**. Das VAR(5)-System ist stabil. Die Halbwertszeit des dominanten (reellen)
Eigenwerts betraegt ln(0,5)/ln(0,9437) * 5min = **59,8 Minuten** -- das ist die
systemweite "traegste" Zeitskala, mit der ein generischer Schock im System insgesamt
abklingt.

48 der 60 Eigenwerte sind komplex (24 konjugiert-komplexe Paare) -- das System hat also
oszillatorische Komponenten. Die groessten (nach Betrag) komplexen Paare haben Perioden von
ca. **19,8 Minuten**, **20,7 Minuten** und **47,7 Minuten** (Betraege 0,65/0,65/0,63,
Halbwertszeiten ca. 8-9 Minuten). Da der Fahrplantakt (24h/12h/8h/6h/168h/84h/56h) bereits
per Fourier-Regression entfernt wurde, sind das **Oszillationen, die NICHT vom Fahrplantakt
stammen** -- plausibel als Wechselwirkung aus Zugfolgeabstand/Kapazitaetsdynamik im
Tunnelkern (Perioden von 20-50 Minuten liegen in der Groessenordnung weniger
Zugumlaeufe/Taktzyklen der dichten Kern-Strecke). Plot 13
(`outputs/var/plots/13_komplexe_schwingung.png`) zeigt die Eigenantwort von Pasing auf sich
selbst als Beispiel einer gedaempften Schwingungskomponente.

### 3.4 Residualdiagnostik (`outputs/var/residual_diagnostik.csv`)
**Alle drei Modellannahmen sind klar verletzt -- ohne Beschoenigung:**
- **Ljung-Box (20 Lags):** in allen 12 Gleichungen p < 3*10^-12 (kleinstes p=1,1*10^-124 bei
  Leuchtenbergring) -- die Residuen sind NICHT weisses Rauschen, es bleibt Autokorrelation
  uebrig, auch nach 5 Lags. Das ist bei so granularen 5-Minuten-Verspaetungsdaten mit
  Stossartigen Ereignissen erwartbar; ein hoeheres p wuerde das nur graduell verbessern
  (siehe AIC-Kurve, die noch bei p=17 leicht sinkt), ohne es zu beheben.
- **Jarque-Bera:** in allen 12 Gleichungen p=0,0 (numerisch, Statistik in Millionenhoehe) --
  die Residuen sind deutlich nicht normalverteilt (schwere Flanken, typisch fuer
  Verspaetungsdaten mit seltenen grossen Ausreissern).
- **ARCH-Test (20 Lags):** in allen 12 Gleichungen p=0,0 -- klare Heteroskedastizitaet
  (Volatility Clustering: ruhige und turbulente Phasen wechseln sich ab, konsistent mit den
  oben genannten Stoerepisoden im November 2025).

Konsequenz: Die Punktschaetzungen der Koeffizienten/IRF bleiben konsistent (OLS ist unter
schwaecheren Annahmen konsistent), aber klassische t-Test-Standardfehler waeren irrefuehrend
optimistisch. Deshalb werden fuer die IRF ausschliesslich Bootstrap-Konfidenzbaender
verwendet (Abschnitt 4.3), keine asymptotischen Formeln.

## 4. Impulsantworten

### 4.1 Aufbau
MA-Darstellung (`phi_matrices`) rekursiv aus A_1..A_5 berechnet, orthogonalisiert per
Cholesky-Zerlegung von Sigma_u in der geografischen Ordnung West->Ost als Primaerordnung
(`structural_irf`). Horizont: 240 Minuten (48 Schritte).

### 4.2 IRF-Gitter (Plot 4: `outputs/var/plots/04_irf_gitter_12x12.png`)
Alle 144 Panels (12x12) wurden gerechnet und geplottet -- bei dieser Groesse noch
uebersichtlich genug fuer eine qualitative Musterpruefung, daher keine Reduktion noetig.
Deutlich sichtbar: Diagonal-Panels (Eigenantwort) haben scharfe Peaks bei t=0 und klingen am
schnellsten ab; Panels mit Schockquelle nahe der Diagonalen (benachbarte Stationen) zeigen
den charakteristischen "Anstieg-dann-Abklingen"-Verlauf mit Peak einige Schritte nach t=0 --
genau das erwartete Bild einer sich stromabwaerts ausbreitenden Stoerung.

### 4.3 Ausgewaehlte Schockstationen (Plot 5:
`outputs/var/plots/05_irf_ausgewaehlte_schocks.png`, mit 95%-Bootstrap-Konfidenzbaendern,
500 Wiederholungen episodenweiser Block-Bootstrap, Laufzeit 26,0 Sekunden)
Fuer Schocks an Pasing, Hauptbahnhof (tief), Ostbahnhof und Leuchtenbergring sind alle 12
Stationsantworten farbcodiert nach Position entlang der Strecke dargestellt. Sichtbares
Muster: je weiter eine Station (farblich) vom Schockort entfernt ist, desto spaeter und
gedaempfter reagiert sie -- eine klare West-Ost-Ausbreitungsstruktur.

### 4.4 Zeit bis zum Antwortmaximum (Plot 6:
`outputs/var/plots/06_heatmap_zeit_bis_peak.png`, Tabelle
`outputs/var/irf_zeit_bis_peak_min.csv`)
**Ein Stoss von Pasing braucht 25 Minuten bis zum Antwortmaximum an Leuchtenbergring**
(Amplitude an diesem Punkt: 53,7 Sekunden, siehe unten). Generell steigt die
Peak-Zeit systematisch mit dem Stationsabstand (z.B. Pasing->Marienplatz 55min,
Pasing->Rosenheimer Platz 50min, Pasing->Ostbahnhof 30min, Pasing->Leuchtenbergring 25min --
nicht perfekt monoton, aber der Trend ist eindeutig), konsistent mit einer sich
physikalisch stromabwaerts bewegenden Stoerung und nicht mit einem gleichzeitigen,
gemeinsamen Schock.

### 4.5 Amplitude (Plot 7: `outputs/var/plots/07_heatmap_amplitude.png`, Tabelle
`outputs/var/irf_amplitude.csv`)
Amplituden auf der Diagonalen (Eigenantwort bei t=0) sind mit Abstand am groessten (103-164
Sekunden je nach Station -- das ist ueber die Cholesky-Kette bedingt: die erst-geordnete
Station eines Schocks traegt die volle Innovationsgroesse). Cross-Amplituden liegen meist im
Bereich von wenigen bis knapp 60 Sekunden und nehmen mit Streckenabstand ab.

### 4.6 Abklingzeit (Plot 8: `outputs/var/plots/08_heatmap_halbwertszeit.png`, Tabelle
`outputs/var/irf_halbwertszeit_min.csv`)
Die Halbwertszeiten variieren stark (5 bis 120 Minuten je Stationspaar), am laengsten fuer
Antworten an Leuchtenbergring auf Schocks von Pasing/Laim (105-120 Minuten) -- die am
weitesten gereisten Stoerungen klingen am langsamsten ab, ein Hinweis darauf, dass sich
Verspaetungen ueber die Strecke hinweg nicht nur ausbreiten, sondern auch "verschmieren".

### 4.7 Varianzzerlegung (Plot 9: `outputs/var/plots/09_fevd.png`, Tabelle
`outputs/var/fevd.csv`), Horizont 240 Minuten
Eigenvarianzanteil je Station: Pasing 92,0%, Laim 65,1%, Hirschgarten 33,3%,
Donnersbergerbruecke 24,2%, Hackerbruecke 25,4%, Hauptbahnhof (tief) 17,9%,
Karlsplatz/Stachus 14,2%, Marienplatz 15,8%, Isartor 16,0%, Rosenheimer Platz 18,5%,
Ostbahnhof 19,3%, Leuchtenbergring 32,7%. **Im Mittel ueber alle 12 Stationen werden 68,8%
der Prognosefehlervarianz durch Schocks an ANDEREN Stationen erklaert** -- das System ist
stark gekoppelt. Wichtige Einschraenkung: die FEVD ist durch die Cholesky-Ordnung
mitbestimmt -- die erst-geordnete Station (Pasing) erhaelt mechanisch den groessten
Eigenvarianzanteil, weil ihr per Konstruktion kein "Fremdschock" bei t=0 zugerechnet werden
kann. Das ist Teil der ueblichen Cholesky-Identifikationslogik (rekursive Kausalordnung),
nicht per se ein Fehler, sollte aber bei der Interpretation von Pasings hohem
Eigenvarianzanteil mitgedacht werden.

### 4.8 Netto-Kopplung (Plot 10: `outputs/var/plots/10a_netto_kopplung_balken.png`,
`10b_netto_kopplung_strecke.png`, Tabelle `outputs/var/netto_kopplung.csv`)
Netto = Summe der FEVD-Anteile, die eine Station bei anderen erklaert (Export), minus Summe
der FEVD-Anteile, die andere bei ihr erklaeren (Import):

| Station | Netto-Kopplung | Rolle |
|---|---|---|
| Pasing | +2,54 | staerkste Netto-Quelle |
| Laim | +2,08 | Netto-Quelle |
| Hirschgarten | +0,16 | leicht Netto-Quelle |
| Hackerbruecke | -0,07 | annaehernd neutral |
| Donnersbergerbruecke | -0,12 | annaehernd neutral |
| Isartor | -0,55 | Netto-Senke |
| Leuchtenbergring | -0,56 | Netto-Senke |
| Hauptbahnhof (tief) | -0,59 | Netto-Senke |
| Rosenheimer Platz | -0,66 | Netto-Senke |
| Marienplatz | -0,69 | Netto-Senke |
| Karlsplatz/Stachus | -0,74 | Netto-Senke |
| Ostbahnhof | -0,79 | staerkste Netto-Senke |

Die westlichen Endstationen (Pasing, Laim) sind klare Netto-Quellen, die zentralen/oestlichen
Stationen (v.a. Ostbahnhof, Karlsplatz/Stachus, Marienplatz) Netto-Senken. **Auch hier gilt
die Cholesky-Einschraenkung aus 4.7**: ein Teil dieser West-Praeferenz ist durch die
geografische Ordnung selbst mitbedingt (frueh geordnete Stationen koennen per Konstruktion
nur "exportieren", nicht "importieren" bei t=0). Der Robustheitscheck mit umgekehrter
Ordnung (naechster Abschnitt) zeigt aber, dass die grundlegende Ausbreitungsrichtung
(West->Ost) real ist und nicht nur ein Artefakt der Ordnungswahl.

### 4.9 Cholesky-Robustheit (Plot 12: `outputs/var/plots/12_cholesky_robustheit.png`)
Vergleich der Eigenantworten (Schock=Antwort=gleiche Station) zwischen primaerer Ordnung
(West->Ost) und umgekehrter Ordnung (Ost->West) fuer die vier ausgewaehlten Schockstationen:
die Kurven liegen fast deckungsgleich uebereinander (sichtbare, aber kleine Abweichungen nur
bei Hauptbahnhof (tief) und Leuchtenbergring in der mittleren/spaeten Phase). Die
Eigenantwort-Form ist also robust gegenueber der Wahl der Cholesky-Ordnung -- erwartbar,
weil die Eigenantwort primaer durch die Diagonalelemente von Sigma_u bestimmt wird, die
ordnungsunabhaengig sind, waehrend Cross-Antworten (und damit FEVD/Netto-Kopplung, siehe
4.7/4.8) staerker ordnungsabhaengig sind.

## 5. Richtungsvergleich Ost vs. West (Plot 11:
`outputs/var/plots/11_richtungsvergleich.png`)
Die Richtung West (Ost->West fahrende Zuege) wurde vollstaendig parallel gerechnet
(gleiche Pipeline, `outputs/var/west/`). Wichtige Unterschiede zur Ost-Analyse:
- **Lag-Wahl:** BIC waehlt fuer West **p=17** (85 Minuten) statt p=5 -- deutlich laenger.
  Kontrolliert bis p=40 (siehe `outputs/var/west/aic_bic_lag_wahl.csv`): p=17 bleibt das
  globale BIC-Minimum, keine spaetere, tiefere Senke. Warum West laengere Lags braucht als
  Ost, ist aus den Daten allein nicht abschliessend zu klaeren -- eine Vermutung waere
  unterschiedliche Verkehrsdichte/Storungsmuster in Gegenrichtung, aber das ist Spekulation.
- **Stabilitaet:** maximaler Eigenwertbetrag 0,971 (vs. 0,944 bei Ost), Halbwertszeit ca.
  **119,6 Minuten** (doppelt so lang wie bei Ost) -- das West-System klingt insgesamt
  traeger ab.

**Wichtiger methodischer Punkt:** Die feste geografische Primaerordnung (West->Ost) ist fuer
West-Zuege physikalisch NICHT die richtige Kausalrichtung -- ein West-Zug durchfaehrt die
Strecke ja in umgekehrter Reihenfolge (erst Leuchtenbergring, zuletzt Pasing). Mit der
"falschen" Ordnung ergibt der Vergleich "Schock Pasing -> Antwort Leuchtenbergring" fuer
West nur eine schwache, spaete Reaktion (Peak 6,8s nach 40 Minuten) -- das ist ein
Ordnungsartefakt, keine echte Aussage ueber die Streckendynamik. Fuer einen fairen Vergleich
wurde daher die **Ost->West-Cholesky-Ordnung** (die fuer West-Zuege kausal richtige
Reihenfolge) herangezogen und "Schock Leuchtenbergring -> Antwort Pasing" betrachtet -- das
ist fuer West-Zuege das Aequivalent zu "Schock Pasing -> Antwort Leuchtenbergring" bei Ost.

**Ergebnis (Plot 11):** Peak bei West nach **25 Minuten** (identisch zu Ost!), Amplitude
45,1 Sekunden (vs. 53,7s bei Ost, gleiche Groessenordnung). Der Kurvenverlauf ist bis ca.
120 Minuten fast deckungsgleich, danach klingt West langsamer ab (konsistent mit dem
groesseren dominanten Eigenwert) und zeigt eine leichte, gedaempfte Welligkeit (Artefakt des
hoeheren p=17, mehr Freiheitsgrade fuer oszillierende Komponenten). **Das ist eine starke
Bestaetigung der Kernaussage**: eine Ende-zu-Ende-Ausbreitungszeit von rund 25 Minuten ueber
die volle Stammstrecke ist robust ueber beide Fahrtrichtungen hinweg reproduzierbar, wenn
man jeweils die kausal richtige Cholesky-Ordnung verwendet. Ein voller 13-Plot-Satz (alle
Heatmaps, FEVD, Netto-Kopplung) wurde fuer West NICHT ebenfalls komplett neu erzeugt --
Zeitgruenden folgend wurden fuer West nur die fuer den Richtungsvergleich noetigen
Kennzahlen (Lag-Wahl, Stabilitaet, IRF-Eigenantwort, ADF, Residualdiagnostik, FEVD,
Netto-Kopplung) in `outputs/var/west/` abgelegt; die vollen 12x12-Heatmaps/-Gitter
(Plots 4, 6-10) beziehen sich nur auf Ost.

## 6. Auswertung -- Antworten auf die Leitfragen

**Gibt es Oszillationen, die nicht der Fahrplantakt sind?** Ja. Nach vollstaendiger
Entfernung von Tages-/Wochentakt (Abschnitt 1.2) verbleiben komplexe Eigenwerte mit Perioden
von ca. 20 und 48 Minuten (Abschnitt 3.3, Plot 13) -- zu kurz fuer irgendeine
Fahrplanharmonische, am ehesten Zugfolge-/Kapazitaetsdynamik im Tunnelkern.

**Wie lange braucht ein Stoss von Pasing bis Leuchtenbergring?** 25 Minuten bis zum
Antwortmaximum (Plot 6), Amplitude an diesem Punkt 53,7 Sekunden (auf eine 1-Sigma-Schock-
Einheit in Pasing bezogen). In der Gegenrichtung (West, kausal richtige Ordnung) bestaetigt
sich exakt dieselbe Zeitspanne von 25 Minuten (Abschnitt 5).

**Welche Stationen sind Netto-Quellen, welche Netto-Senken?** Pasing und Laim (Westende) klar
Netto-Quellen, alle zentralen und oestlichen Stationen Netto-Senken, am staerksten
Ostbahnhof, Karlsplatz/Stachus und Marienplatz (Abschnitt 4.8) -- mit dem Vorbehalt, dass
dieses Muster teilweise durch die Cholesky-Ordnung mitbedingt ist.

**Wie stark ist das System insgesamt gekoppelt?** Im Mittel werden 68,8% der
Prognosefehlervarianz jeder Station durch Schocks an anderen Stationen erklaert
(Abschnitt 4.7) -- die Stammstrecke verhaelt sich klar als gekoppeltes System, nicht als
12 unabhaengige Zeitreihen.

## 7. Vergleich mit Stufe 1

Stufe 1 (`BEFUNDE.md`, Abschnitt "Impulsantwort auf message_code 34", Datei
`outputs/impulsantwort_code34.csv`, Plot `outputs/plots/impulsantwort_code34.png`) misst die
ereignisbasierte Impulsantwort auf einen konkreten Stoerungscode (34 = Signalstoerung,
gepoolt ueber Ereignisse derselben Station/desselben Zeitfensters relativ zu t0): Spitze von
**+152 Sekunden** (95%-CI 122-186s) genau bei t0, danach Abklingen ueber ca. 60-90 Minuten
auf ein **Plateau von ca. 30-40 Sekunden**, das im beobachteten Fenster bis +240 Minuten
NICHT auf null zurueckgeht (BEFUNDE.md Zeile 244-246, 339-346).

Die VAR-Eigenantwort (eine Station auf einen Schock in sich selbst, z.B. Pasing: 164s bei
t=0, Leuchtenbergring: 151s bei t=0 -- siehe `outputs/var/irf_amplitude.csv`, Diagonale)
liegt **in derselben Groessenordnung** wie die Code-34-Spitze (151-164s VAR vs. 152s Stufe 1)
-- eine bemerkenswerte Uebereinstimmung, obwohl die beiden Groessen methodisch komplett
verschieden gemessen sind (VAR: Innovationsgroesse aus Sigma_u, unbedingt auf irgendeine
Stoerursache; Stufe 1: bedingt auf den konkreten, beobachteten Stoerungscode 34).

**Bei der Abklingform gibt es einen erkennbaren Unterschied:** die VAR-Eigenantwort klingt
INNERHALB von 240 Minuten praktisch vollstaendig ab (Pasing: 164s -> 36,5s nach 25min ->
17,9s nach 60min -> 1,9s nach 240min; Hauptbahnhof (tief) noch schneller: 103s -> 2,1s
bereits nach 25 Minuten). Die Code-34-Kurve aus Stufe 1 dagegen bleibt bei einem Plateau von
30-40 Sekunden haengen und geht im beobachteten Fenster nicht auf null zurueck. Diese
Diskrepanz ist inhaltlich plausibel erklaerbar: Die VAR-Eigenantwort misst die Reaktion auf
eine EINZELNE statistische Innovation im stationaeren, detrendeten Residualprozess -- per
Modellannahme (Stationaritaet, Abschnitt 2) muss diese im Erwartungswert auf null
zurueckkehren. Die Stufe-1-Kurve dagegen ist bedingt auf einen tatsaechlich beobachteten,
realen Stoerungscode, dessen Nachwirkungen (z.B. Nachzuegler-Effekte, Ersatzfahrplan-
Anpassungen, in Stufe 1 selbst als "Mischung aus Ereignissen unterschiedlicher Dauer"
diskutiert, BEFUNDE.md Zeile 339-368) laenger andauern koennen als eine einzelne
VAR-Innovation. Beide Befunde ergaenzen sich: die VAR-Analyse zeigt die "typische",
mittlere Ausbreitungsdynamik des Systems bei generischen Schocks (inkl. Ausbreitung ueber
alle 12 Stationen), Stufe 1 zeigt die realistischere, laengere Nachwirkung eines konkreten,
schweren Stoerungstyps an einer einzelnen Station. Ein direkter quantitativer
Ausbreitungsvergleich (wie schnell sich ein Code-34-Ereignis raeumlich ueber die 12
Stationen ausbreitet) wurde in Stufe 1 nicht gerechnet (dort ist die Impulsantwort
stationsbezogen, nicht raeumlich) -- das waere ein natuerlicher naechster Schritt, um beide
Ansaetze direkt zu verzahnen.

## 8. Dateien

**Code:** `src/analysis/var/{stations,prepare,detrend,episodes,var_model,diagnostics,irf,
irf_metrics,plots,run_all}.py`. Einstiegspunkt fuer die Ost-Analyse:
```
uv run python -c "from src.analysis.var.run_all import run_direction, save_outputs_ost, make_plots_ost, STATION_NAMES_WEST_OST; r=run_direction('Ost', STATION_NAMES_WEST_OST); save_outputs_ost(r); make_plots_ost(r)"
```

**Tabellen (`outputs/var/`):** `fourier_info.csv`, `episoden_meta.csv`,
`aic_bic_lag_wahl.csv`, `adf_ergebnisse.csv`, `adf_ergebnisse_je_episode.csv`,
`residual_diagnostik.csv`, `eigenwerte.csv`, `irf_zeit_bis_peak_min.csv`,
`irf_amplitude.csv`, `irf_halbwertszeit_min.csv`, `fevd.csv`, `netto_kopplung.csv`,
`ausgeschlossene_betriebstage.csv`, `run_meta.json`. Analoge (reduzierte) Tabellen fuer
West in `outputs/var/west/`.

**Plots (`outputs/var/plots/`):** `01_periodogramm.png`, `02_aic_bic.png`,
`03_eigenwerte.png`, `04_irf_gitter_12x12.png`, `05_irf_ausgewaehlte_schocks.png`,
`06_heatmap_zeit_bis_peak.png`, `07_heatmap_amplitude.png`,
`08_heatmap_halbwertszeit.png`, `09_fevd.png`, `10a_netto_kopplung_balken.png`,
`10b_netto_kopplung_strecke.png`, `11_richtungsvergleich.png`,
`12_cholesky_robustheit.png`, `13_komplexe_schwingung.png` (dazu ergaenzend fuer West:
`02b_aic_bic_west.png`, `03b_eigenwerte_west.png`).

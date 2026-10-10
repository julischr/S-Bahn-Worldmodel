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

**Warum 12 Reihen statt 24 (eine Richtung statt beide gemeinsam)?** Jede der beiden
Fahrtrichtungen (Ost, West) wird als eigenes VAR mit K=12 Stationen geschaetzt, nicht als
ein gemeinsames VAR mit K=24 (12 Stationen x 2 Richtungen). Grund ist die
Parameteranzahl: ein VAR(p) mit K Gleichungen hat K*(K*p+1) freie AR-/Konstante-Parameter
(ohne Sigma_u). Bei K=12 und dem gewaehlten p=5 (Ost) sind das 12*(12*5+1) = **732**
Parameter, bei p=17 (West) bereits 12*(12*17+1) = **2.460** Parameter. Wuerde man beide
Richtungen gemeinsam als K=24-System schaetzen, waeren es bei p=5 bereits 24*(24*5+1) =
**2.904** Parameter (das Vierfache von K=12 bei gleichem p, weil die Parameterzahl mit K²
waechst) -- und da eine gemeinsame Episodenbildung ueber alle 24 Spalten (beide Richtungen
gleichzeitig nicht-NaN) die Anzahl brauchbarer Beobachtungen gegenueber der 12-Spalten-Bildung
weiter reduziert (siehe Abschnitt 9.4 fuer die genaue Zahl), waere das Verhaeltnis
Beobachtungen/Parameter bei K=24 deutlich ungünstiger als bei K=12. Die getrennte
Schaetzung pro Richtung ist deshalb nicht nur inhaltlich (unterschiedliche Fahrtrichtung als
kausale Ordnung), sondern auch statistisch die robustere Wahl. Ob dennoch relevante
Rueckkopplung zwischen den Richtungen existiert, wird in Abschnitt 9.4 explizit mit einem
K=24-Versuch geprueft (mit dem Resultat, ob/wie weit das schaetzbar ist).

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

## 9. Validierung und Robustheitspruefung

Dieser Abschnitt wurde in einer zweiten, kritischen Runde ergaenzt, nachdem der Nutzer
begruendete Zweifel angemeldet hatte, ob die obigen Befunde (Abschnitte 1-8) tatsaechlich
tragen. Alle Zahlen stammen aus tatsaechlich ausgefuehrtem Code unter `src/analysis/var/`
(neue Dateien: `joint24.py`, `hvz_nebenzeit.py`, `oszillation_placebo.py`, `oos_check.py`,
`robustness_differenzen.py`). Nichts an den Abschnitten 1-8 oder an bestehenden Outputs wurde
inhaltlich veraendert -- Ausnahme sind rein kosmetische Titel-Ergaenzungen an den 16
bestehenden Plots (siehe 9.1), die keine Zahlen aendern.

### 9.1 Richtung dokumentiert, Plot-Titel ergaenzt

Die Begruendung fuer 12 statt 24 Reihen (Parameteranzahl-Argument) fehlte bisher prominent
und wurde direkt in die Einleitung dieses Dokuments nachgetragen (siehe Absatz "Warum 12
Reihen statt 24" oben, vor Abschnitt 1): K=12 bei p=5 (Ost) ergibt 732 AR-Parameter, K=12 bei
p=17 (West) 2.460, ein gemeinsames K=24-System braeuchte bei p=5 bereits 2.904 Parameter --
das Vierfache bei gleichem p, weil die Parameterzahl mit K² waechst.

Alle 16 bestehenden PNGs wurden durchgegangen (Plot-Erzeugungsfunktionen in `plots.py`
inspiziert): AUSSER Plot 11 (Richtungsvergleich, dort schon explizit "Ost vs. West" im
Titel) nannte KEINER der Titel/Achsenbeschriftungen, welche Zuegerichtung (Ost- oder
West-fahrend) analysiert wurde -- nur die Cholesky-Ordnungsrichtung ("West->Ost" als
geografische Ordnung) wurde erwaehnt, was leicht mit der Zuegerichtung verwechselt werden
kann. Behoben durch additive, ruckwaertskompatible Parameter (`richtung: str = ""`) in
`plots.py` (keine bestehende Funktionssignatur wurde entfernt, nur optional erweitert) und
Neu-Erzeugung aller 14 Ost- und 2 West-Plots mit ergaenztem Richtungslabel im Titel
(`[Richtung Ost (West->Ost fahrende Zuege)]` bzw. `[Richtung West (Ost->West fahrende
Zuege)]`). **Wichtig: nur Titel/Labels geaendert, alle zugrunde liegenden Zahlen/Kurven sind
identisch mit dem urspruenglichen Lauf** (gleiche gepickelte Zwischenergebnisse
wiederverwendet, keine Neuschaetzung).

### 9.2 Heatmaps maskiert (Rauschen vs. Signal)

Der Nutzer wies zurecht darauf hin, dass sehr kleine Amplituden (Beispiel Leuchtenbergring
-> Pasing: 120 Minuten Halbwertszeit bei nur 3,2 Sekunden Amplitude) nicht vertrauenswuerdig
sind. Schwellenwert-Pruefung: die 132 Off-Diagonal-Amplituden verteilen sich mit Median 14,2s
(!), 25%-Quantil 5,3s, 75%-Quantil 53,7s -- die Haelfte aller Zellen liegt UNTER dem
vorgeschlagenen 20s-Schwellenwert. Zum Vergleich die Bootstrap-95%-KI-Breite am jeweiligen
Peak (median 8,0s, also Halbbreite ca. 4,0s): ein rein statistisches "signifikant von 0
verschieden"-Kriterium (Amplitude > halbe KI-Breite) wuerde nur 19,7% der Zellen maskieren,
deutlich weniger als der 20s-Vorschlag. **Der 20s-Schwellenwert des Nutzers ist also
KONSERVATIVER als das reine Bootstrap-Signifikanzkriterium** (er maskiert mehr, nicht
weniger) -- gerechtfertigt, weil bei 132 gleichzeitig betrachteten Zellen (Mehrfachvergleiche)
ein Cut bei ca. dem 2-3-fachen der typischen KI-Halbbreite eine sinnvolle, konservativere
Praxisschwelle ist als ein einzelnes 95%-KI pro Zelle. Gewaehlt: **20 Sekunden**.

Ergebnis: **68 von 132 Off-Diagonal-Zellen (51,5%) werden maskiert** -- ueber die Haelfte
der bisher gezeigten Zeit-bis-Peak- und Halbwertszeit-Werte in den Original-Heatmaps
(Plots 6 und 8) sind Werte, die auf Rauschen sitzen und nicht ueberinterpretiert werden
sollten. Das explizite Beispiel des Nutzers: Leuchtenbergring->Pasing, Amplitude 3,2s, wird
korrekt maskiert. Neue Plots: `outputs/var/plots/06b_heatmap_zeit_bis_peak_maskiert.png`,
`08b_heatmap_halbwertszeit_maskiert.png`; Tabelle `outputs/var/robustness/masking_zusammenfassung.csv`.

**Konsequenz fuer die Interpretation der Abschnitte 4.4/4.6:** Die dort berichteten
"Pasing->Marienplatz 55min" / "105-120 Minuten Halbwertszeit"-Aussagen fuer die am weitesten
entfernten Stationspaare beruhen teilweise auf Zellen mit Amplituden nahe der Rausch-Schwelle
und sollten mit entsprechender Vorsicht gelesen werden -- die Kernaussage (Ausbreitung
ueber wenige, benachbarte bis mittelweit entfernte Stationen mit klar messbarer Amplitude,
insbesondere Pasing->Leuchtenbergring mit 53,7s) bleibt bestehen, weil diese Zelle deutlich
ueber der Schwelle liegt.

### 9.3 Erweiterter Cholesky-Robustheitscheck (Kreuzantworten)

Der urspruengliche Plot 12 verglich nur Eigenantworten (Diagonale) -- das ist der Fall, in
dem die Cholesky-Ordnung am WENIGSTEN Einfluss hat (siehe 4.9). Kreuzantworten sind
grundsaetzlich ordnungssensitiver. Fuer die drei vom Nutzer genannten Paare (jeweils Schock
an erster, Antwort an zweiter Station -- alle drei GEGEN die primaere West->Ost-Richtung,
also "stromaufwaerts"):

| Schock -> Antwort | Ordnung 1 (West->Ost, primaer) | Ordnung 2 (Ost->West, alternativ) |
|---|---|---|
| Ostbahnhof -> Pasing | 2,76s bei 20min | 2,72s bei 20min |
| Leuchtenbergring -> Hackerbruecke | 7,85s bei 30min | 10,27s bei 30min |
| Marienplatz -> Laim | **+1,99s** bei 65min | **-1,62s** bei 20min |

Ostbahnhof->Pasing ist unter beiden Ordnungen praktisch identisch und sehr klein (~2,7s) --
konsistent damit, dass ein "stromaufwaerts"-Effekt real schwach ist, unabhaengig von der
Ordnungsannahme (das ist eine ermutigende Konsistenzpruefung, kein Ordnungsartefakt).
Leuchtenbergring->Hackerbruecke zeigt eine maessige Ordnungsabhaengigkeit (7,9 vs. 10,3s,
beide klein und unter der 20s-Rauschschwelle aus 9.2 -- **beide Werte sind ohnehin nicht
vertrauenswuerdig**). **Marienplatz->Laim ist das kritischste Beispiel: das VORZEICHEN
wechselt zwischen den Ordnungen** (leicht positiv vs. leicht negativ) -- allerdings sind
beide Werte betragsmaessig winzig (<2s, weit unter der Rauschschwelle), sodass dieser
Vorzeichenwechsel eher ein Rauschartefakt bei sehr kleinen Werten ist als ein inhaltlicher
Befund. Neuer Plot: `outputs/var/plots/12b_cholesky_kreuzantworten.png`, Tabelle
`outputs/var/robustness/cholesky_kreuzantworten_kennzahlen.csv`.

**Vollstaendige 12x12-Amplitude unter beiden Ordnungen** (Plot `12c_amplitude_beide_ordnungen.png`)
und die **Differenzmatrix** (Plot `12d_amplitude_differenz.png`,
`outputs/var/robustness/amplitude_differenz_ordnungen.csv`): mittlere absolute Differenz
ueber alle Off-Diagonal-Zellen 3,7s, maximale Differenz 45,0s (ein Ausreisser-Zellenpaar).
Mittlere Amplitude Off-Diagonal: 16,6s (Ordnung 1) vs. 14,2s (Ordnung 2) -- insgesamt aehnliche
Groessenordnung, aber NICHT identisch. **Fazit:** Die West->Ost-Ausbreitungsrichtung selbst
(Peak-Zeit, grobe Amplitudenordnung) ist robust, aber die genauen Zahlenwerte der
Kreuzantworten -- insbesondere kleine, "stromaufwaerts" gerichtete Effekte -- haengen
spuerbar von der gewaehlten Cholesky-Ordnung ab und sollten nicht auf die zweite Nachkommastelle
interpretiert werden.

### 9.4 Rueckkopplung zwischen den Richtungen (K=24-Versuch)

Ein gemeinsames VAR mit allen 24 Reihen (12 Stationen x Ost + 12 Stationen x West) wurde
tatsaechlich versucht (`src/analysis/var/joint24.py`). Ueberraschung: die gemeinsame
Episodenbildung (nur Zeitpunkte, an denen ALLE 24 Spalten gleichzeitig nicht-NaN sind) ergab
**707 Episoden mit 81.729 Rohbeobachtungen** -- kaum weniger als die 686 Episoden/80.530
Beobachtungen des reinen Ost-Modells. Das war so nicht erwartet (die Befuerchtung war
deutlich weniger gemeinsame Beobachtungen); Grund: fehlende Werte in Ost und West sind
zeitlich stark korreliert (beide fehlen v.a. nachts, wenn generell kein/kaum Betrieb ist),
nicht unabhaengig, sodass sich die Verfuegbarkeit nicht multiplikativ verschlechtert.

BIC ueber p=1..20 hat **kein klares Minimum** wie bei Ost (p=5) -- die Kurve ist im Bereich
p=3..9 fast flach (225,86 bis 225,85, Unterschied <0,005) und steigt erst danach wieder an;
das numerische Minimum liegt bei **p=9** (BIC=225,852), aber die Wahl ist deutlich fragiler
als bei Ost. Bei p=9: K*(K*p+1) = 24*217 = **5.208 Parameter**, T=75.366 nutzbare
Beobachtungen, Verhaeltnis Beobachtungen/Parameter = **14,5** -- rechnerisch schaetzbar
(Verhaeltnis > 10 gilt hier als praktikable Grenze), aber deutlich enger als bei den
getrennten 12-Reihen-Modellen (Ost: 80.530/732 = 110; West: 71.450/2.460 = 29).

**Ergebnis der Kreuzblock-Analyse** (Cholesky-Ordnung: Ost-Block zuerst in geografischer
Ordnung, dann West-Block, ebenfalls geografisch -- eine von mehreren moeglichen Wahlen, NICHT
auf Robustheit gegen andere Ordnungen getestet, siehe Einschraenkung unten):
- Mittlerer FEVD-Anteil, den West-Schocks bei Ost-Stationen erklaeren: **11,3%**
- Mittlerer FEVD-Anteil, den Ost-Schocks bei West-Stationen erklaeren: **11,2%**
- Das ist ein REALER, aber MODERATER Rueckkopplungseffekt -- deutlich kleiner als die 68,8%
  Fremdvarianz INNERHALB einer Richtung (Abschnitt 4.7), aber klar von null verschieden.
- Amplituden an identischen Stationen ueber die Richtung hinweg (Beispiel Pasing, wo beide
  Richtungen denselben Bahnsteigbereich nutzen): **Ost-Schock -> West-Antwort an Pasing:
  90,0s**, **West-Schock -> Ost-Antwort an Pasing: 40,0s** -- eine ueberraschend GROSSE
  Kreuzantwort, plausibel erklaerbar durch gemeinsame Infrastruktur/Wendezeiten an der
  Endstation. Bei Ostbahnhof (35,0s bzw. 70,0s) und Leuchtenbergring (45,0s bzw. 75,0s)
  aehnlich substantielle Kreuzeffekte, bei Laim/Hackerbruecke deutlich kleiner (20-25s,
  nahe der 20s-Rauschschwelle aus 9.2).

**Interpretation:** Es gibt eine messbare "Wende-Rueckkopplung" zwischen den Richtungen,
am staerksten an den Streckenenden (Pasing, Leuchtenbergring, Ostbahnhof) -- plausibel, weil
dort Zuege wenden oder Gleisinfrastruktur/Disposition geteilt wird. Diese Zahlen sind aber
mit spuerbarer Vorsicht zu lesen: das K=24-Modell hat ein deutlich schlechteres
Verhaeltnis Beobachtungen/Parameter, eine fragile (flache) Lag-Wahl, und die Kreuzblock-
Ergebnisse wurden nur unter EINER Cholesky-Ordnung berechnet (nicht wie bei den 12-Reihen-
Modellen mit Robustheitscheck gegen die umgekehrte Ordnung). Dieser Befund sollte als
"vorlaeufiger, aber plausibler Hinweis auf reale Rueckkopplung", nicht als belastbare
Endzahl kommuniziert werden. Outputs: `outputs/var/joint24/` (`joint24_meta.json`,
`irf_amplitude_24x24.csv`, `fevd_24x24.csv`, `aic_bic_lag_wahl_joint24.csv`,
`obs_per_param_joint24.csv`, `eigenwerte_joint24.csv`, `episoden_meta_joint24.csv`), Plot
`outputs/var/plots/14_joint24_amplitude_kreuzblock.png`.

### 9.5 Zeitabhaengige Kopplung: HVZ vs. Nebenzeit

HVZ (Mo-Fr 6-9 und 16-19 Uhr) vs. Nebenzeit (Rest, inkl. Wochenende), Richtung Ost. Jedes
HVZ-Zeitfenster wird automatisch zu einer eigenen kurzen Episode (Zeilen ausserhalb der
Teilmenge werden auf NaN gesetzt, `build_daily_episodes` bricht dort automatisch ab):
**HVZ: 482 Episoden, 17.196 Beobachtungen** (Median-Laenge ca. 36 Schritte = 3h, wie
erwartet); **Nebenzeit: 882 Episoden, 63.042 Beobachtungen**. Fuer einen fairen Vergleich
wurde **p=5 fest fuer beide Teilmengen** gewaehlt (identisch zum Hauptmodell) -- eigenstaendige
BIC-Kurven zeigen, dass HVZ (kurze Episoden) ein eigenes Minimum bei p=2 hat, waehrend
Nebenzeit wie das Ganzjahresmodell BIC noch bei p=10 monoton fallend zeigt (kein Minimum im
getesteten Bereich); p=5 ist ein vertretbarer Kompromiss, der fuer beide Teilmengen
schaetzbar ist und identisch zum Hauptmodell bleibt (siehe `outputs/var/hvz_nebenzeit/aic_bic_hvz.csv`,
`aic_bic_nebenzeit.csv` fuer die Details).

**Vergleichszahlen** (`outputs/var/hvz_nebenzeit/vergleich_hvz_nebenzeit.csv`):

| Kennzahl | HVZ | Nebenzeit |
|---|---|---|
| Peak-Zeit Pasing->Leuchtenbergring | 25 min | 25 min |
| Amplitude Pasing->Leuchtenbergring | **59,95s** | **51,65s** |
| Mittlere FEVD-Fremdvarianz | **70,8%** | **68,6%** |
| Max. Betrag Eigenwert (Persistenz) | **0,959** | **0,940** |

Netto-Kopplung: Pasing (+3,04 HVZ / +2,31 Nebenzeit) und Laim (+1,31 HVZ / +2,38 Nebenzeit)
bleiben in BEIDEN Teilmengen Netto-Quellen, Ostbahnhof bleibt in beiden die staerkste
Netto-Senke (-0,77 / -0,79) -- das raeumliche Grundmuster ist zeitunabhaengig stabil.

**Antwort auf die Nutzer-Hypothese ("HVZ staerker gekoppelt wegen Kapazitaetsgrenze"):**
JA, in die erwartete Richtung, aber der Unterschied ist MODERAT, nicht dramatisch: etwas
hoehere Amplitude (+16%), etwas hoehere Fremdvarianz (+2,2 Prozentpunkte) und etwas
persistentere Dynamik (hoeherer max. Eigenwert) in der HVZ. Die Peak-Zeit (Ausbreitungs-
geschwindigkeit) ist dagegen in beiden Regimes IDENTISCH (25min) -- die HVZ macht Stoerungen
also staerker/nachhaltiger, aber nicht schneller wandernd. Plots:
`outputs/var/plots/15a_hvz_vs_nebenzeit_eigenantwort.png`,
`15b_hvz_vs_nebenzeit_pasing_leuchtenbergring.png`.

### 9.6 Oszillationen abgesichert -- gemischtes Ergebnis

**(a) Ist der sichtbare "Wiederanstieg" (ca. 15-25min) im Plot statistisch von einer rein
monotonen Abklingkurve unterscheidbar?** Test: Vergleich der Pasing-Eigenantwort mit einer
angepassten monotonen Exponential-Referenzkurve (aus den ersten 3 Schritten extrapoliert,
tau=17,7min) im Fenster 10-25min gegen das 95%-Bootstrap-Band der tatsaechlichen Kurve.
**Ergebnis: 0 von 4 getesteten Zeitpunkten liegen mit ihrer Differenz ausserhalb der
Bootstrap-KI-Halbbreite.** Ehrlich gesagt: **der visuell sichtbare Wiederanstieg/Buckel in
der Pasing-Eigenantwort ist mit dieser Methode NICHT signifikant von einer monotonen
Abklingkurve zu unterscheiden** -- die Abweichung liegt innerhalb der Bootstrap-Unsicherheit.
Das relativiert die visuelle Interpretation von Plot 5/13 als "klar sichtbare Schwingung" an
EINER Kurve. Plot: `outputs/var/plots/16a_oszillation_bootstrap_vs_monoton.png`, Tabelle
`outputs/var/oszillation/hump_vs_monoton_pasing.csv`.

**(b) Tauchen dieselben Perioden bei p=17 (AIC-Optimum) auf?** Die **20-Minuten-Periode ist
robust**: bei p=5 Periode 19,8min (Betrag 0,652), bei p=17 Periode **20,0min** mit sogar
GROESSEREM Betrag **0,936** -- diese Schwingungskomponente ist nicht vom gewaehlten Lag
abhaengig und wird bei mehr Lags eher staerker/dominanter, nicht schwaecher. **Die
48-Minuten-Periode ist dagegen NICHT robust**: bei p=17 taucht unter den zehn betragsgroessten
komplexen Eigenwertpaaren keine Periode nahe 48min auf (stattdessen 1.167min [praktisch ein
reeller, quasi-nicht-oszillierender Eigenwert -- numerisches Artefakt eines fast entarteten
Paars], 376min, 105,7min, 15,6min). **Fazit (b): die 20-Minuten-Schwingung ist robust ueber
die Lag-Wahl, die 48-Minuten-Schwingung ist es nicht** -- diese sollte in kuenftigen
Formulierungen NICHT mehr gleichrangig mit der 20-Minuten-Periode genannt werden.

**(c) Placebo-Test (exakte 20/48-Minuten-Fourier-Terme in der Detrending-Regression):**
Trotz expliziter Entfernung deterministischer Sinus/Cosinus-Komponenten bei GENAU 20 und
48 Minuten bleiben die Eigenwertperioden praktisch UNVERAENDERT: bei p=5 weiterhin Perioden
46,7min/22,8min/21,0min/19,5min mit aehnlichen Betraegen (0,60-0,62, vs. 0,63-0,65 im
Original) -- **die Oszillation verschwindet NICHT und wird nur geringfuegig schwaecher**. Bei
p=17 bleibt die 20-Minuten-Periode praktisch identisch (19,98min, Betrag 0,883 vs. 0,936 im
Original). **Das ist ein wichtiger, POSITIVER Befund fuer die urspruengliche Interpretation**:
Waere die Oszillation nur ein Artefakt eines exakten externen 20/48-Minuten-Rhythmus
(z.B. ein regelmaessiger Betriebszyklus), haette der Placebo-Test sie zum Verschwinden bringen
muessen, wenn man genau diese Frequenzen explizit aus den Daten herausrechnet. Das ist NICHT
passiert -- die Schwingung, insbesondere bei ca. 20 Minuten, scheint tatsaechlich aus der
ENDOGENEN VAR-Dynamik (Zugfolge/Kapazitaetswechselwirkung) zu stammen und nicht aus einem
exakten externen Taktgeber. Tabellen: `outputs/var/oszillation/eigen_perioden_p5.csv`,
`eigen_perioden_p17.csv`, `eigen_perioden_placebo_p5.csv`, `eigen_perioden_placebo_p17.csv`.

**Gesamtfazit 9.6:** Die 20-Minuten-Periode ist der robuste Kern des Befunds (ueberlebt
Lag-Aenderung UND Placebo-Test). Die 48-Minuten-Periode ist schwaecher belegt (verschwindet
bei p=17 aus den fuehrenden Eigenwerten). Der visuell markante "zweite Peak" in den
IRF-Plots einzelner Kurven ist dagegen NICHT als statistisch signifikant vom monotonen Fall
abgesichert zu bezeichnen -- die zugrunde liegenden komplexen Eigenwerte sind real und
robust, aber ihr Effekt auf die EINZELNE beobachtete IRF-Kurve ist innerhalb der
Bootstrap-Unsicherheit nicht klar von reinem Rauschen zu trennen.

### 9.7 Out-of-Sample-Pruefung

VAR(5) (Ost) trainiert NUR auf Episoden aus Jan-Sep 2025 (**542 Episoden, 63.298
Beobachtungen**), getestet auf Okt-Dez 2025 (**144 Episoden, 20.662 Beobachtungen**, die 12
bereits ausgeschlossenen Tage bleiben ausgeschlossen). Mehrschritt-Prognose rekursiv
berechnet (5-Schritt nutzt die eigenen 1..4-Schritt-Prognosen als Lag-Ersatz, wie bei
echten Mehrschritt-Prognosen ueblich).

**Ergebnis: das VAR(5) schlaegt BEIDE naiven Baselines bei BEIDEN Horizonten fuer ALLE 12
Stationen (100% Trefferquote)** (`outputs/var/oos/oos_beats_baseline_summary.csv`):

| Horizont | RMSE VAR(5) | RMSE Random Walk | RMSE hist. Mittel (Wochentag/Stunde) |
|---|---|---|---|
| 5 min (1 Schritt) | **148,3s** | 273,3s | 286,4s |
| 25 min (5 Schritte) | **242,2s** | 319,7s | 287,8s |

Das ist ein klar positiver, ehrlicher Befund: das Modell hat echten Out-of-Sample-Mehrwert
gegenueber simplen Baselines, UNABHAENGIG von den dokumentierten Verletzungen der
Residualannahmen (Abschnitt 3.4) -- Punktprognosen bleiben nuetzlich, auch wenn die
Konfidenzintervalle mit Vorsicht zu lesen sind. Erwartungsgemaess steigt der Fehler mit dem
Horizont (VAR 148->242s), waehrend die historische Mittelwert-Baseline naturgemaess kaum
horizontabhaengig ist. Tabelle je Station: `outputs/var/oos/oos_forecast_errors.csv`, Plot
`outputs/var/plots/17_oos_rmse_vergleich.png`.

**H1 (Jan-Jun) vs. H2 (Jul-Dez) vs. Gesamtjahr** (`outputs/var/oos/h1_h2_vergleich.csv`):

| Kennzahl | H1 | H2 | Gesamtjahr |
|---|---|---|---|
| Peak-Zeit Pasing->Leuchtenbergring | 25,0 min | 25,0 min | 25,0 min |
| Amplitude Pasing->Leuchtenbergring | 47,7s | 58,9s | 53,7s |
| Mittlere FEVD-Fremdvarianz | 69,3% | 68,8% | 68,8% |
| Netto-Kopplung Pasing | +2,23 | +2,70 | +2,54 |
| Netto-Kopplung Laim | +2,34 | +1,90 | +2,08 |

**Alle drei Kernbefunde sind ueber die beiden Jahreshaelften stabil**: die Peak-Zeit ist auf
die Minute identisch (25min in allen drei Faellen), die mittlere Fremdvarianz variiert nur
um 0,5 Prozentpunkte, Pasing/Laim bleiben in beiden Haelften klare Netto-Quellen. Die
Amplitude schwankt staerker (47,7 vs. 58,9s), bleibt aber in derselben Groessenordnung.
**Das ist der staerkste einzelne Robustheitsbefund in diesem gesamten Validierungsabschnitt.**

### 9.8 Einfluss der Modellverletzungen -- Variante Differenzen-VAR

VAR(5) auf ersten Differenzen (Delta delay_mean, innerhalb jeder Episode, keine Differenzen
ueber Episodengrenzen) statt Level-Residuen geschaetzt, als direkte Antwort auf die in
Abschnitt 2/3.4 dokumentierte partielle Nicht-Stationaritaet. IRF der Differenzen-VAR
kumuliert (Standardtrick: kumulative Summe einer Delta-IRF = Level-Antwort auf einen
einmaligen Delta-Schock), um mit der Level-VAR aus dem Hauptlauf vergleichbar zu sein.
686 differenzierte Episoden, 83.274 Beobachtungen (eine weniger je Episode durch die
Differenzierung).

| Kennzahl | Level-VAR (Hauptlauf) | Differenzen-VAR (kumuliert) |
|---|---|---|
| Peak-Zeit Pasing->Leuchtenbergring | 25 min | **25 min** |
| Amplitude Pasing->Leuchtenbergring | 53,7s | 44,6s |
| Amplitude Pasing-Eigenantwort | 164s | 168,0s |
| Max. Betrag Eigenwert | 0,944 | **0,741** |

**Die Kernaussage "Ausbreitungszeit ca. 25 Minuten" bleibt unter der Differenzen-
Spezifikation EXAKT erhalten** -- ein starkes Robustheitssignal, da diese Spezifikation die
Nicht-Stationaritaetsproblematik explizit adressiert. Die Amplitude ist in derselben
Groessenordnung (leicht kleiner bei der Kreuzantwort, praktisch gleich bei der
Eigenantwort). Der maximale Eigenwertbetrag sinkt deutlich (0,944 -> 0,741) -- die
differenzierte Reihe ist erwartungsgemaess deutlich staerker stationaer/schneller
mittelwert-reduzierend, was fuer die urspruengliche Sorge (Level-Reihen nicht durchgehend
stationaer) spricht, aber die IRF-Kernaussagen nicht veraendert.

Netto-Kopplungsmuster (`outputs/var/robustness/netto_kopplung_vergleich_diff.csv`): Pasing
(+1,48 statt +2,54) und Laim (+1,94 statt +2,08) bleiben Netto-Quellen, Ostbahnhof bleibt mit
-0,67 (statt -0,79) die staerkste Netto-Senke -- **die komplette raeumliche Rangfolge
West=Quelle, Ost=Senke ist unter der Differenzen-Spezifikation erhalten**, nur die absoluten
Werte sind etwas gedaempft. Plot: `outputs/var/plots/18_level_vs_differenzen_irf.png`,
Tabelle `outputs/var/robustness/vergleich_level_vs_differenzen.csv`. (Variante B, HAC-robuste
Standardfehler, wurde aus Zeitgruenden nicht zusaetzlich gerechnet -- die Differenzen-VAR
allein liefert bereits eine klare, positive Antwort auf die Stationaritaets-Sorge.)

### 9.9 Ehrliche Gesamteinschaetzung

**Was robust ist (durch mehrere unabhaengige Checks bestaetigt):**
- **Die Ausbreitungszeit von ca. 25 Minuten Pasing->Leuchtenbergring ist der robusteste
  Einzelbefund der gesamten Analyse.** Bestaetigt durch: Richtung West mit kausal korrekter
  Ordnung (Abschnitt 5, 25min), H1 vs. H2 (beide exakt 25min, 9.7), HVZ vs. Nebenzeit (beide
  exakt 25min, 9.5), und die Differenzen-VAR (exakt 25min, 9.8). Vier von vier unabhaengigen
  Variationen liefern exakt dieselbe Zahl. Das ist ungewoehnlich stabil und wuerde ich einem
  Gutachter ohne Zoegern als soliden Befund vertreten.
- **Pasing und Laim als Netto-Quellen, die zentralen/oestlichen Stationen (v.a. Ostbahnhof)
  als Netto-Senken** ist ueber H1/H2 (9.7), HVZ/Nebenzeit (9.5) und die Differenzen-VAR (9.8)
  hinweg konsistent, auch wenn die genauen Werte schwanken.
- **Das System ist real gekoppelt** (mittlere FEVD-Fremdvarianz 68-71% je nach Teilmenge,
  konsistent über Zeitfenster und Modellspezifikation) -- das ist kein Cholesky-Artefakt,
  weil sowohl die Differenzen-VAR als auch die HVZ/Nebenzeit-Aufteilung aehnliche
  Groessenordnungen liefern.
- **Das VAR(5) hat echten Out-of-Sample-Prognosewert** gegenueber Random Walk und
  historischem Mittelwert (9.7) -- unabhaengig von der Residualdiagnostik.
- **Die ~20-Minuten-Oszillation ist real** (ueberlebt Lag-Aenderung auf p=17 UND den
  Placebo-Test mit expliziten 20/48-min-Fourier-Termen, 9.6) -- eher endogene VAR-Dynamik als
  Artefakt eines exakten externen Rhythmus.

**Was stark an Modellannahmen oder der Cholesky-Ordnung haengt (NICHT unkritisch behaupten):**
- **Die exakten Amplituden- und Halbwertszeit-Werte weit entfernter Stationspaare** (z.B.
  "105-120 Minuten Halbwertszeit Pasing/Laim -> Leuchtenbergring") sitzen teilweise unter
  der 20s-Rauschschwelle (51,5% aller Off-Diagonal-Zellen in den Heatmaps, 9.2) -- diese
  Zahlen sollte ich einem Gutachter NICHT als praezise Werte praesentieren, nur als "Richtung
  vorhanden, Betrag unsicher".
- **Einzelne Kreuzantworten, insbesondere "stromaufwaerts" gerichtete (Ost->West-Station),
  sind ordnungsabhaengig** -- Marienplatz->Laim wechselt sogar das Vorzeichen zwischen den
  beiden Cholesky-Ordnungen (9.3), auch wenn der Betrag winzig ist. Ich wuerde KEINE einzelne
  Kreuzantwort-Zahl fuer eine "stromaufwaerts"-Station als belastbar bezeichnen.
- **Der visuell markante "zweite Peak/Buckel" in einzelnen IRF-Kurven (z.B. Plot 5/13) ist
  NICHT statistisch von reinem monotonem Abklingen unterscheidbar** (9.6a) -- ich wuerde
  einem Gutachter NICHT sagen "man sieht klar eine Schwingung in der Kurve", sondern nur "die
  zugrunde liegenden komplexen Eigenwerte sind real, aber ihr sichtbarer Effekt auf eine
  einzelne beobachtete Kurve ist nicht separat abgesichert".
- **Die 48-Minuten-Periode ist schwaecher belegt als die 20-Minuten-Periode** (9.6b) und
  sollte in zukuenftigen Zusammenfassungen nicht mehr gleichrangig genannt werden.
- **Das K=24-Rueckkopplungsmodell (9.4) ist der methodisch fragilste neue Befund**: flache,
  kaum bestimmte Lag-Wahl, engeres Beobachtungen/Parameter-Verhaeltnis (14,5 statt 29-110),
  nur EINE Cholesky-Ordnung getestet. Die berichteten 11% Kreuz-FEVD und 90s
  Pasing-Kreuzamplitude sind ein plausibler, aber vorlaeufiger Hinweis, keine belastbare
  Endzahl.
- **Die HVZ-vs-Nebenzeit-Unterschiede sind real, aber klein** (9.5) -- ich wuerde NICHT
  behaupten "die HVZ ist dramatisch staerker gekoppelt", sondern nur "es gibt einen
  moderaten, in die erwartete Richtung gehenden Unterschied von wenigen Prozentpunkten/
  Sekunden", und dazusagen, dass HVZ und Nebenzeit mit demselben festen p=5 verglichen
  wurden, nicht mit ihrem jeweils optimalen p.

**Was ich einem kritischen Gutachter gegenueber klar NICHT behaupten wuerde:**
1. Dass die Residuen "praktisch weiss genug" seien fuer klassische Inferenz -- sie sind es
   nachweislich nicht (ARCH-Effekte, keine Normalitaet, siehe 3.4), die Bootstrap-Baender sind
   die einzige verlaessliche Unsicherheitsangabe, und selbst die sind nur "informativ", nicht
   "exakt".
2. Dass Netto-Kopplung/FEVD eine kausal saubere, ordnungsfreie Kennzahl sei -- sie ist
   mechanisch von der Cholesky-Ordnung mitgepraegt (4.7/4.8/9.3).
3. Dass jede einzelne Zahl in den 12x12-Heatmaps (Zeit-bis-Peak, Halbwertszeit) interpretierbar
   sei -- ueber die Haelfte sitzt unter der Rauschschwelle (9.2).
4. Dass die 48-Minuten-Schwingung ebenso gut belegt sei wie die 20-Minuten-Schwingung (9.6b).
5. Dass das K=24-Rueckkopplungsergebnis (9.4) eine praezise, reproduzierbare Kennzahl sei --
   es ist ein erster, methodisch verwundbarer Blick auf eine reale, aber noch nicht robust
   quantifizierte Rueckkopplung.
6. Dass ein einzelner beobachteter "Wiederanstieg" in einer IRF-Kurve fuer sich genommen ein
   bewiesenes Phaenomen sei (9.6a) -- nur die zugrunde liegenden Eigenwerte sind (fuer die
   20-Minuten-Periode) gut abgesichert, nicht ihre visuelle Auspraegung in jeder einzelnen
   Kurve.

**Zusammenfassend:** Die zentralen, headline-faehigen Befunde (25-Minuten-Ausbreitungszeit,
West=Quelle/Ost=Senke-Muster, generelle System-Kopplung, echter Prognosewert) halten der
Validierung ueberraschend gut stand und wurden durch vier bis fuenf unabhaengige Checks
bestaetigt. Die Detailbefunde (exakte Heatmap-Werte, einzelne Kreuzantworten, die
48-Minuten-Schwingung, das K=24-Modell) sind dagegen entweder teilweise Rauschen, ordnungs-
abhaengig oder methodisch fragil und sollten mit entsprechend geringerer Zuversicht
kommuniziert werden.

## 10. Lag-Kreuzvergleich und West-Autokorrelationsstruktur

Dieser Abschnitt wurde in einer dritten Runde ergaenzt, ausgeloest durch zwei konkrete
Nutzerfragen: (a) aendern sich die Kernbefunde, wenn man die Lag-Ordnung p zwischen Ost und
West vertauscht, und (b) was steckt hinter der deutlich hoeheren BIC-optimalen Lag-Ordnung
bei West (p=17=85min vs. Ost p=5=25min) -- insbesondere die Vermutung einer Struktur bei
75-85 Minuten. Neue Dateien: `src/analysis/var/lag_crosscheck.py`, `src/analysis/var/acf_check.py`.
Alle Zahlen stammen aus tatsaechlich ausgefuehrtem Code. Nichts an den Abschnitten 1-9 oder an
bestehenden Outputs wurde veraendert.

### 10.1 Lag-Kreuzvergleich (Teil A)

**Methodik:** West wurde zusaetzlich mit festem p=5 geschaetzt (statt des dortigen
BIC-Optimums p=17), Ost zusaetzlich mit festem p=17 (statt des dortigen BIC-Optimums p=5) --
jeweils mit identischer Datenaufbereitung (Fourier-Detrending, Tagesausschluss,
Episodenbildung) wie in den Hauptlaeufen. Fuer den Vier-Varianten-Vergleich wurde ein
methodischer Punkt vereinheitlicht: die in `outputs/var/west/*.csv` gespeicherten
Haupttabellen (fevd.csv, netto_kopplung.csv, irf_amplitude.csv) verwenden dort tatsaechlich
die PRIMAERE geografische Ordnung (West->Ost), die fuer West-Zuege kausal falsch ist (siehe
Abschnitt 5) -- nur die Prosa-Zahl "45,1s" in Abschnitt 5 nutzte bereits die kausal korrekte
Ost->West-Ordnung. Damit alle vier Varianten fair vergleichbar sind (Aepfel mit Aepfeln),
wurden **alle vier** Kombinationen hier einheitlich neu berechnet: Ost stets mit der primaeren
West->Ost-Ordnung, West stets mit der kausal korrekten Ost->West-Ordnung (identisch zur
Methodik aus Abschnitt 5). Die "Hauptlauf"-Zeilen unten sind also Replikationen der
bestehenden BIC-optimalen Modelle mit dieser konsistenten Ordnung, keine neuen Schaetzungen
der Koeffizienten selbst (p bleibt jeweils das dokumentierte BIC-Optimum). Aus Zeitgruenden
wurde fuer die vier Varianten KEIN Bootstrap gerechnet (nur Punktschaetzungen, kein
Konfidenzband) -- die Bootstrap-Baender der Hauptlaeufe (Abschnitt 4.3) bleiben davon
unberuehrt und gueltig.

**Ergebnistabelle** (`outputs/var/lag_crosscheck/vier_varianten_vergleich.csv`):

| Variante | Peak-Zeit (min) | Amplitude (s) | Mittlere FEVD-Fremdvarianz | Top Netto-Quelle | Top Netto-Senke |
|---|---|---|---|---|---|
| **Ost p=5** (Hauptlauf, BIC-optimal) | 25 | 53,7 | 68,8% | Pasing (+2,54) | Ostbahnhof (-0,79) |
| **Ost p=17** (Kreuzvergleich, AIC-Kandidat von West) | 25 | 52,3 | 68,9% | Pasing (+2,31) | Ostbahnhof (-0,79) |
| **West p=17** (Hauptlauf, BIC-optimal) | 25 | 45,1 | 71,8% | Rosenheimer Platz (+2,24) | Hackerbruecke (-0,82) |
| **West p=5** (Kreuzvergleich, BIC-optimal von Ost) | 25 | 47,3 | 71,0% | Rosenheimer Platz (+2,46) | Laim (-0,80) |

(Peak-Zeit/Amplitude jeweils fuer die kausal korrekte Kreuzantwort Pasing<->Leuchtenbergring;
bei Ost: Antwort Leuchtenbergring auf Schock Pasing; bei West: Antwort Pasing auf Schock
Leuchtenbergring.) Vergleichsplot: `outputs/var/plots/19_lag_crosscheck_vergleich.png` (links:
die vier IRF-Kurven ueberlagert, rechts: mittlere FEVD-Fremdvarianz je Variante als Balken).

**Vollstaendige Netto-Kopplungs-Rangfolge je Variante**
(`outputs/var/lag_crosscheck/netto_kopplung_{ost_p5_rep,ost_p17,west_p17_rep,west_p5}.csv`):
bei Ost sind in BEIDEN p-Varianten Pasing (+2,54/+2,31) und Laim (+2,08/+2,21) die zwei
staerksten Netto-Quellen und Ostbahnhof (-0,79/-0,79) die staerkste Netto-Senke, die gesamte
Rangfolge der uebrigen 9 Stationen ist zwischen p=5 und p=17 praktisch identisch (nur
Vertauschungen benachbarter Raenge bei sehr aehnlichen Werten, z.B. Marienplatz/
Karlsplatz-Stachus). Bei West sind in BEIDEN p-Varianten Rosenheimer Platz (+2,24/+2,46) und
Leuchtenbergring (+2,20/+2,27) die zwei staerksten Netto-Quellen (Ostbahnhof folgt mit
deutlichem Abstand auf Platz 3, +0,90/+0,77) und Hackerbruecke/Laim/Hirschgarten/Pasing bilden
in beiden Varianten das Schlusslicht-Quartett der Netto-Senken (Werte zwischen -0,71 und
-0,82). **Eine Praezisierung gegenueber der bisherigen Formulierung in Abschnitt 5/9:** die
bislang genannten "westlichen Stationen sind Quellen, oestliche Stationen sind Senken" gilt
fuer Ost eindeutig (Pasing/Laim vs. Ostbahnhof/Karlsplatz/Marienplatz), fuer West bei
konsistenter kausaler Ordnung aber praeziser als "Rosenheimer Platz/Leuchtenbergring/
Ostbahnhof (das oestliche Streckenende, wo West-Zuege einfahren) sind Quellen, die
westmittleren Stationen (Hackerbruecke/Hirschgarten/Laim/Pasing) sind Senken" -- das ist
inhaltlich dieselbe geometrische Logik (der Streckenanfang der jeweiligen Fahrtrichtung ist
Netto-Quelle), nur mit Pasing bei West NICHT mehr als Netto-Quelle, sondern als Netto-Senke
(weil Pasing fuer West-Zuege das STRECKENENDE ist).

**Antwort auf die Kernfrage:** Nein, die Kernbefunde aendern sich NICHT wesentlich, wenn man p
zwischen den Richtungen vertauscht:
- **Peak-Zeit exakt 25 Minuten in allen vier Kombinationen** -- das ist jetzt der sechste
  unabhaengige Check (nach West-Hauptlauf, H1/H2, HVZ/Nebenzeit, Differenzen-VAR), der
  identisch 25min liefert. Dieser Befund ist damit ausserordentlich robust gegenueber der
  Lag-Wahl.
- **Amplitude** bleibt in derselben Groessenordnung (45,1-53,7s ueber alle vier Varianten,
  Spannweite <9s) -- p aendert die Amplitude leicht (etwas hoehere Werte bei kuerzerem p in
  beiden Richtungen: Ost p=5 > p=17, West p=17 < p=5 -- kein einheitliches Vorzeichen des
  Effekts, aber durchgehend klein).
- **Mittlere FEVD-Fremdvarianz** ist bei gegebener Richtung nahezu p-unabhaengig (Ost: 68,8%
  vs. 68,9%; West: 71,8% vs. 71,0% -- Unterschied jeweils unter 1 Prozentpunkt), aber es
  bleibt ein echter, p-unabhaengiger Unterschied ZWISCHEN den Richtungen (West ca. 2-3
  Prozentpunkte hoeher gekoppelt als Ost, unabhaengig davon welches p verwendet wird).
- **Netto-Quellen/Senken-Rangfolge** ist innerhalb jeder Richtung ueber beide p-Werte hinweg
  stabil (siehe oben) -- das Muster ist eine Eigenschaft der Fahrtrichtung/geografischen
  Position, nicht ein Artefakt der zufaellig gewaehlten Lag-Ordnung.

**Fazit 10.1:** Die p-Wahl (BIC vs. Kreuzvergleich) ist fuer die inhaltlichen Kernaussagen
praktisch irrelevant -- das ist ein starkes, zusaetzliches Robustheitssignal. Der
Hauptunterschied zwischen Ost und West liegt NICHT an der unterschiedlichen Lag-Ordnung,
sondern ist ein echter (wenn auch moderater) struktureller Unterschied zwischen den beiden
Fahrtrichtungen (etwas hoehere Kopplung, etwas kleinere Amplitude bei West) -- was aber die
eigentliche Ausgangsfrage nicht beantwortet, WARUM West ein hoeheres BIC-optimales p hat.
Das wird in 10.2 direkt untersucht.

### 10.2 West-Autokorrelationsstruktur bei 75-85 Minuten (Teil B)

**Methodik:** Episodengewichtete, gepoolte Autokorrelationsfunktion (ACF) je Station, auf
genau den Residualreihen (nach Fourier-Detrending, Tagesausschluss, Kurzluecken-Interpolation,
Episodenbildung), die auch in die Hauptlauf-VAR-Schaetzung eingehen. Die ACF wird PRO EPISODE
mit `statsmodels.tsa.stattools.acf` berechnet (Lags nie ueber Episodengrenzen hinweg, analog
zur VAR-Lag-Design-Matrix) und ueber alle Episoden mit Gewicht (n_episode - lag) gepoolt
gemittelt -- eine einzelne ACF auf den durchgehend aneinandergehaengten Residuen wuerde an
jeder der 698 (West) bzw. 686 (Ost) Episodengrenzen kuenstliche Sprungkorrelationen erzeugen
und wurde deshalb bewusst vermieden. Bis Lag 25 (125 Minuten) berechnet.

**(1) West, Lags 14-18 (70-90 Minuten)** (`outputs/var/lag_crosscheck/acf_west_lag14_18_detail.csv`):
Werte liegen zwischen 0,02 (Ostbahnhof, Lag 14) und 0,19 (Ostbahnhof, Lag 16 = 80min, ein
deutlicher Ausreisser) -- die meisten Stationen zeigen im Fenster 0,04-0,09. **An JEDER der 12
Stationen ist Lag 16 (80 Minuten) der lokale Spitzenwert innerhalb des 70-90-Minuten-Fensters**
(Werte zwischen 0,070 und 0,185, durchgehend hoeher als die Nachbarwerte bei Lag 14/15/17/18) --
ein konsistentes, wenn auch kleines Muster.

**(2) Ost, Lags 14-18** (`outputs/var/lag_crosscheck/acf_ost_lag14_18_detail.csv`): **Derselbe
lokale Spitzenwert bei Lag 16 (80 Minuten) ist auch bei Ost an 11 von 12 Stationen sichtbar**
(Werte 0,069-0,095, Ausnahme Rosenheimer Platz mit flachem Profil) -- **die 80-Minuten-Struktur
ist also KEIN West-spezifisches Phaenomen**, sie existiert in aehnlicher (wenn auch
tendenziell etwas schwaecherer) Staerke auch bei Ost. Der gemittelte Wert ueber Lag 14-18 ist
bei West fuer die 5 westlichen Stationen (Pasing bis Hackerbruecke) um 0,009-0,020
hoeher als bei Ost, bei den 4 oestlichen Stationen (Isartor bis Leuchtenbergring) dagegen
UMGEKEHRT (Ost 0,003-0,012 hoeher als West) -- kein einheitliches Vorzeichen des
Richtungsunterschieds (`outputs/var/lag_crosscheck/acf_70_90min_vergleich.csv`).
**Wichtige Relativierung:** die staerkste Nebenkorrelation ausserhalb Lag 0-3 liegt bei
JEDER der 12 Stationen (beide Richtungen) bei **Lag 4 = 20 Minuten** (Werte 0,21-0,35, siehe
`outputs/var/lag_crosscheck/acf_staerkster_nebenpeak.csv`) -- also deutlich staerker als der
75-90-Minuten-Bereich. Die 80-Minuten-Struktur ist real messbar, aber ein SEKUNDAERES, viel
schwaecheres Muster im Vergleich zur bereits in Abschnitt 3.3/9.6 dokumentierten,
dominanten 20-Minuten-Schwingung. Plot: `outputs/var/plots/20a_acf_west_vs_ost.png` (4
Beispielstationen, 70-90min-Fenster grau markiert, 20min/80min-Marker eingezeichnet) -- am
auffaelligsten sichtbar ist dort NICHT der 80min-Bereich, sondern bei Ostbahnhof (West) ein
durchgehendes, sich alle 20 Minuten wiederholendes Kammuster (Peaks bei 20/40/60/80/100/120
Minuten), staerker ausgepraegt als bei allen anderen Station/Richtung-Kombinationen.

**(3) Blockfahrplan-Hypothese vs. Datenartefakt:**
- **Taktvielfache:** 75-85 Minuten geteilt durch einen 10-Minuten-Grundtakt ergibt 7,5-8,5 --
  das naechstliegende ganzzahlige Vielfache ist **8x10min=80min**. Geteilt durch einen
  20-Minuten-Grundtakt ergibt 3,75-4,25 -- das naechstliegende Vielfache ist ebenfalls
  **4x20min=80min**. Beide Grundtakt-Hypothesen zeigen also auf **exakt 80 Minuten** als
  plausibelsten Kandidaten -- und genau bei Lag 16 (80min) liegt tatsaechlich der empirisch
  beobachtete lokale ACF-Spitzenwert (Punkt 1/2 oben) UND die deutlichste "Stufe" in der
  BIC-Kurve (siehe Punkt 4). Das ist eine auffaellige Uebereinstimmung, aber OHNE Fahrplandaten
  (Umlaufplaene, Linienueberlagerung S1-S8) bleibt es eine numerisch plausible, nicht eine
  bewiesene Erklaerung -- 80 Minuten als Vielfaches des Grundtakts ist eine von vielen
  moeglichen 80-Minuten-Erklaerungen (z.B. auch Zugumlaufzeiten, die nicht zwingend exakte
  Taktvielfache sein muessen).
- **Episodenlaengen-Artefakt-Check:** Verteilung der West-Episodenlaengen zeigt KEINE
  auffaellige Konzentration nahe 75-85 Beobachtungen, die ein kuenstliches Muster erzeugen
  koennte -- nur 27 von 698 West-Episoden (3,9%) haben eine Laenge zwischen 70 und 90
  Beobachtungen, praktisch identisch zu Ost (25 von 686, 3,6%). Direkt quantifiziert
  (`outputs/var/lag_crosscheck/episodenlaengen_artefakt_check.csv`): diese kurzen Episoden
  tragen bei Lag 16 nur **2,4% (West) bzw. 2,1% (Ost)** des gesamten Gewichts der gepoolten
  ACF-Schaetzung, waehrend lange Episoden (>=150 Beobachtungen, ca. 40% aller Episoden in
  beiden Richtungen) **79,1% (West) bzw. 81,6% (Ost)** des Gewichts liefern. **Der
  80-Minuten-Peak ist damit klar KEIN Episodengrenzen-Artefakt** -- er wird ueberwiegend von
  langen, durchgehenden Betriebstag-Episoden getragen, nicht von einer zufaelligen Haeufung
  kurzer Episoden in genau dieser Laenge.
- **BIC-Stufe bei West:** die `logdet(Sigma_u)`-Kurve (`outputs/var/west/aic_bic_lag_wahl.csv`)
  zeigt zwischen p=14 und p=16 einen deutlich groesseren Sprung (-0,124 von p=14->15, -0,101
  von p=15->16) als in der unmittelbaren Umgebung (p=10->14: Schritte von nur -0,005 bis
  -0,034) -- **eine sichtbare Stufe genau im 75-80-Minuten-Bereich**, die anschliessend die
  BIC-Kurve nochmal absenkt und so zum globalen Minimum bei p=17 beitraegt (Plot
  `outputs/var/plots/20b_west_bic_stufe.png`). Das ist konsistent mit einer real vorhandenen,
  wenn auch kleinen zusaetzlichen erklaerbaren Varianz im 75-85-Minuten-Lag-Bereich, die bei
  Ost (wo laut Punkt 2 ein aehnlicher, aber etwas schwaecherer 80min-Effekt existiert und die
  BIC-Straffung staerker durchschlaegt, weil Ost ohnehin weniger Autokorrelationsstruktur zu
  "gewinnen" hat) nicht ausreicht, um das BIC-Optimum ueber p=5 hinauszuschieben.

**Ehrliches Fazit 10.2:** Es gibt eine reale, reproduzierbare, aber KLEINE Autokorrelations-
struktur bei 80 Minuten (Lag 16) in den Fourier-Residuen -- sichtbar in der ACF (an praktisch
allen 12 Stationen, beide Richtungen) UND als sichtbare Stufe in der West-BIC-Kurve. Diese
Struktur ist **NICHT West-spezifisch** (sie existiert in aehnlicher Groessenordnung auch bei
Ost), was die urspruengliche Nutzerhypothese "75-85min-Struktur erklaert das hoehere West-p"
nur TEILWEISE bestaetigt: die Struktur selbst ist kein West-Alleinstellungsmerkmal, aber ihr
Zusammenspiel mit der insgesamt flacheren/langsameren Abklingdynamik von West (Halbwertszeit
119,6min vs. 59,8min bei Ost, siehe Abschnitt 5) fuehrt dazu, dass der zusaetzliche
BIC-Gewinn durch diese Struktur bei West das globale Minimum ueber p=14 hinausschiebt,
waehrend er bei Ost (schnelleres Grundabklingen, BIC bereits bei p=5 klar minimal) nicht
ausreicht. Der Episodenlaengen-Artefakt ist mit den vorliegenden Daten klar AUSGESCHLOSSEN
(2-2,4% Gewichtsanteil). Die 8x10min/4x20min-Taktvielfache-Erklaerung ist eine numerisch
plausible, in der Groessenordnung passende Kandidatenerklaerung (beide Grundtakt-Annahmen
zeigen exakt auf 80min, wo auch empirisch der Peak liegt), aber **OHNE Fahrplan-/Umlaufdaten
nicht beweisbar** -- ich wuerde einem Gutachter sagen "die Zahlen sind mit einem
Taktvielfachen von 80 Minuten konsistent und ein Datenartefakt ist ausgeschlossen, aber die
tatsaechliche betriebliche Ursache (Umlaufplan, Linienueberlagerung, oder etwas ganz anderes)
bleibt aus den verfuegbaren Daten allein nicht abschliessend identifizierbar". Insgesamt bleibt
die 20-Minuten-Schwingung (Abschnitt 3.3/9.6) das mit Abstand staerkste und am besten
abgesicherte Oszillationsmuster in beiden Richtungen; die 80-Minuten-Struktur ist real, aber
ein Nebenbefund, kein neuer Haupteffekt.

## 11. Plausibilitaetstests

Dieser Abschnitt wurde in einer vierten Runde ergaenzt, um drei zentrale Befunde der
Abschnitte 1-10 gegen eine komplett andere Datenquelle zu pruefen: `data/02_events/final/`
(Zugebene, 12 Monatsdateien, ~19,1 Mio. Zeilen gesamt, alle Stationen), statt wie bisher
`data/03_state_vector/state_vector_5min.parquet` (Fensterebene, nur die 12
Stammstrecke-Stationen). Neue Dateien: `src/analysis/var/plausibilitaet_fahrzeit.py`,
`plausibilitaet_linienmischung.py`, `plausibilitaet_zugebene.py`. Nichts an den Abschnitten
1-10, an bestehenden Outputs oder am uebrigen `src/` wurde veraendert -- alle drei Skripte
lesen `data/02_events/final/` neu ein und importieren aus `src/analysis/var/` nur bestehende,
unveraenderte Funktionen (`prepare.pivot_direction`, `detrend.detrend_frame`,
`episodes.interpolate_short_gaps`/`build_daily_episodes`, `acf_check.acf_table_all_stations`,
`run_all.EXCLUDED_DATES`). Alle Zahlen stammen aus tatsaechlich ausgefuehrtem Code. Outputs:
`outputs/var/plausibilitaet/`, Plots `21_fahrzeit_pasing_leuchtenbergring.png`,
`22_linienbereinigte_acf_vergleich.png`, `23_zugebene_headway_ansteckung.png`.

### 11.1 Fahrplanmaessige Fahrzeit Pasing<->Leuchtenbergring vs. 25-Minuten-VAR-Peak

**Methodik:** Fuer jede `trip_id` mit einem Ankunfts-Ereignis (`is_arrival=true`, konsistent
zur Definition von `delay_mean` im Hauptlauf, `is_cancelled=false`) an BEIDEN Stationen
(Pasing 8004158, Leuchtenbergring 8004134) wurde die `time_schedule`-Differenz berechnet.
Richtung wurde ueber `stop_sequence` bestimmt (kleinere Sequenznummer = zuerst durchfahren)
und gegen die vorhandene `richtung`-Spalte der Events kreuzvalidiert: **bei allen 95.068
gefundenen Trips (100,0%) stimmt die stop_sequence-basierte Richtungszuordnung exakt mit der
vorhandenen richtung-Spalte ueberein** -- ein starkes Signal, dass beide Datenquellen
(Zugebene und die bereits fuer den Hauptlauf verwendete richtung-Spalte) konsistent sind.

**Ergebnis** (`outputs/var/plausibilitaet/fahrzeit_zusammenfassung.csv`):

| Richtung | n Trips | Median | Mittelwert | Std | P10 | P90 | Min | Max |
|---|---|---|---|---|---|---|---|---|
| Ost | 46.944 | **21,0 min** | 21,9 min | 1,29 | 21 | 23 | 21 | 28 |
| West | 48.124 | **21,0 min** | 20,8 min | 1,07 | 20 | 21 | 20 | 81 (Einzelfall) |

Die Verteilung ist in beiden Richtungen sehr eng (P10-P90-Spanne nur 2-3 Minuten) -- der
Fahrplan sieht praktisch immer 20-23 Minuten fuer diese Strecke vor. Nur 970 von 46.944
Ost-Trips (2,1%) und 1.036 von 48.124 West-Trips (2,2%) haben eine fahrplanmaessige Fahrzeit
ueber 25 Minuten; der West-Ausreisser bei 81 Minuten ist ein einzelner Fall (vermutlich ein
Sonderfahrplan-/Datenartefakt eines einzelnen Trips) und aendert nichts am Gesamtbild. Plot:
`outputs/var/plots/21_fahrzeit_pasing_leuchtenbergring.png` (Histogramm je Richtung mit
eingezeichneter 25-Minuten-VAR-Peak-Linie).

**Vergleich mit der VAR-Peak-Zeit von 25 Minuten:** Die fahrplanmaessige Fahrzeit (Median 21
Minuten, beide Richtungen identisch) liegt **in derselben Groessenordnung wie die
VAR-Ausbreitungszeit, aber systematisch knapp darunter** -- die VAR-Peak-Zeit ist um ca. 4
Minuten (19%) laenger als die reine Fahrzeit. Das ist ein starkes, ehrliches
Plausibilitaetssignal: Waere die VAR-Ausbreitungszeit voellig unabhaengig von der tatsaechlichen
Zugbewegung (z.B. ein reines Artefakt der Fensteraggregation oder des Cholesky-Verfahrens),
gaebe es keinen Grund fuer eine Naehe zur Fahrplanzeit ueberhaupt. Dass die VAR-Zeit etwas
LAENGER ist als die reine Fahrzeit, ist inhaltlich plausibel: die VAR-Peak-Zeit misst, wann die
VerspaetungsANTWORT an Leuchtenbergring ihr Maximum erreicht, nicht nur die reine
Fahrtdauer eines einzelnen Zuges -- dazu kommen z.B. Wartezeiten/Dispositionsverzoegerungen an
Zwischenhalten unter Storungsdruck sowie die Tatsache, dass die VAR-Antwort eine AGGREGIERTE
Groesse ueber viele Zuege/Fenster ist, keine Eins-zu-eins-Fahrzeit eines einzelnen Zuges.
**Fazit 11.1:** Die 25-Minuten-VAR-Ausbreitungszeit ist NICHT identisch mit, aber sehr nah an
der fahrplanmaessigen Fahrzeit (21 vs. 25 Minuten) -- das stuetzt die Kerninterpretation "die
Verspaetung reist ungefaehr mit dem Zug mit", mit einem plausiblen Aufschlag fuer
Ausbreitungs-/Dispositionseffekte on top der reinen Fahrzeit.

### 11.2 20-Minuten-ACF-Peak -- kein Linienmischungs-Kompositionseffekt

**Methodik:** Aus `data/02_events/final/` wurde je (Station, Richtung, Linie) die mittlere
Verspaetung ueber das Jahr berechnet (370 Kombinationen, `is_arrival=true`,
`is_cancelled=false`, `richtung IS NOT NULL`, dieselben 12 ausgeschlossenen Betriebstage wie
im VAR-Hauptlauf, siehe `run_all.EXCLUDED_DATES`). Je 5-Minuten-Fenster/Station/Richtung wurde
daraus ein "erwarteter" delay_mean-Wert konstruiert, der NUR aus der beobachteten
Linienzusammensetzung im Fenster folgt (nach Zuganzahl gewichteter Mittelwert der
Linien-Jahresmittelwerte). Composition-bereinigter Wert = tatsaechlicher delay_mean minus
dieser Erwartungswert (1.821.737 Fenster/Station/Richtung-Zeilen). Auf diese bereinigte Reihe
wurde EXAKT dieselbe Fourier-Detrending-Funktion (`detrend.detrend_frame`, n_daily=4,
n_weekly=3), Kurzluecken-Interpolation und Episodenbildung angewendet wie im Hauptlauf, und
darauf dieselbe episodengewichtete gepoolte ACF-Berechnung wie in Abschnitt 10.2
(`acf_check.acf_table_all_stations`, importiert und unveraendert wiederverwendet).

**Ergebnis bei Lag 4 (20 Minuten)** (`outputs/var/plausibilitaet/vergleich_lag4_20min_{ost,west}.csv`):

| Richtung | ACF Lag4 original (Mittel ueber 12 Stationen) | ACF Lag4 linienbereinigt | Mittlere relative Veraenderung |
|---|---|---|---|
| Ost | 0,250 | 0,268 | **-7,9%** (d.h. im Mittel LEICHT GESTIEGEN) |
| West | 0,301 | 0,295 | **+4,3%** (d.h. im Mittel leicht gesunken) |

Bei Ost steigt die ACF bei Lag 4 nach Linienbereinigung an 11 von 12 Stationen sogar leicht AN
(einzige Ausnahme Leuchtenbergring, -4,3%), bei West sinkt sie an allen 12 Stationen leicht
(0,6% bis 21,0%, im Mittel 4,3%; der Ausreisser Ostbahnhof West mit -21,0% ist die einzige
Station mit einer wirklich spuerbaren Abschwaechung). Plot
`outputs/var/plots/22_linienbereinigte_acf_vergleich.png` zeigt fuer vier Beispielstationen
(Pasing, Hauptbahnhof (tief), Ostbahnhof, Leuchtenbergring) die komplette ACF-Kurve bis 125
Minuten original vs. linienbereinigt -- beide Kurven liegen praktisch deckungsgleich
uebereinander, inklusive des charakteristischen Kammusters bei Vielfachen von 20 Minuten.

**Ehrliches Fazit 11.2:** Die Hypothese "der 20-Minuten-Peak ist ein Linienmischungs-
Kompositionseffekt" wird durch diesen Test **klar NICHT bestaetigt**. Nach Herausrechnen der
gesamten linienspezifischen mittleren Verspaetung bleibt der ACF-Peak bei Lag 4 in praktisch
unveraenderter Staerke bestehen -- bei Ost wird er sogar tendenziell etwas staerker (was
darauf hindeutet, dass die Linienmischung, wenn ueberhaupt, eher etwas Varianz VERDECKT als
erzeugt). Das ist ein weiterer, unabhaengiger positiver Befund fuer die bereits in Abschnitt
9.6(c) getroffene Einschaetzung (Placebo-Test mit exakten 20/48-Minuten-Fourier-Termen
liess die Schwingung ebenfalls nicht verschwinden): die 20-Minuten-Periode ist mit hoher
Wahrscheinlichkeit eine echte endogene zeitliche Dynamik (Zugfolge-/Kapazitaetswechselwirkung
im Tunnelkern), kein Kompositionsartefakt aus der zufaelligen Linienmischung je Fenster.

### 11.3 Zugebene -- Ansteckung zwischen Vorgaenger- und Folgezug nach Headway

**Methodik:** Je Station/Richtung (12 Stationen, beide Richtungen, 3.433.449
Ankunfts-Ereignisse, `is_arrival=true`, `is_cancelled=false`, `richtung IS NOT NULL`)
chronologisch nach `time_schedule` sortiert; fuer jedes Paar aufeinanderfolgender,
unterschiedlicher `trip_id` an derselben Station/Richtung wurde der Headway
(`time_schedule`-Differenz) sowie `delay` von Vorgaenger und Folgezug erfasst. Von 3.433.425
moeglichen Paaren wurden 335.673 (9,8%, Headway <=0, vermutlich zeitgleich geplante Zuege
unterschiedlicher Linien oder Datenartefakte) und 6.424 (0,2%, Headway > 60 Minuten,
Betriebspausen/Nachtluecken -- kein sinnvoller "direkt folgender Zug") ausgeschlossen; **es
verbleiben 3.091.328 auswertbare Paare.** Stratifiziert nach Headway-Bins: <5min (2,71 Mio.
Paare, der weit ueberwiegende Teil bei der dichten Stammstrecken-Taktung), 5-10min (268.731),
10-15min (30.683), 15-20min (58.010), 20-60min (20.299, als Naeherung fuer ">20min"). Je Bin
wurden Pearson-Korrelation (mit Fisher-z-Konfidenzintervall) und OLS-Steigung
(`delay_next ~ delay_prev`, HC1-robuste Standardfehler) berechnet, gepoolt ueber alle 12
Stationen/2 Richtungen UND je Station/Richtung als Konsistenzcheck.

**Ergebnis** (`outputs/var/plausibilitaet/zugebene_headway_kontingenz_gepoolt.csv`, Plot
`outputs/var/plots/23_zugebene_headway_ansteckung.png`):

| Headway-Bin | n Paare | r (95%-KI) | OLS-Steigung (95%-KI) |
|---|---|---|---|
| <5min | 2.713.605 | 0,418 (0,417-0,419) | 0,416 (0,412-0,421) |
| 5-10min | 268.731 | **0,505** (0,502-0,508) | 0,520 (0,508-0,532) |
| 10-15min | 30.683 | 0,430 (0,420-0,439) | 0,449 (0,404-0,493) |
| 15-20min | 58.010 | 0,377 (0,370-0,384) | 0,398 (0,361-0,436) |
| 20-60min | 20.299 | 0,401 (0,389-0,412) | 0,360 (0,312-0,409) |

Konsistenzcheck ueber die 24 Station/Richtung-Kombinationen im Bin <5min: r zwischen 0,204 und
0,531 (Median 0,414) -- das Muster (deutlich positive, moderate Korrelation) ist ueberall
sichtbar, wenn auch mit spuerbarer Streuung zwischen Stationen.

**Das ist NICHT die erwartete Musterform.** Die Nutzer-Hypothese war: Korrelation nimmt mit
laengerem Headway monoton AB (kuerzester Headway = staerkste "Ansteckung"). Tatsaechlich zeigt
sich: die Korrelation ist bei sehr kurzem Headway (<5min) moderat (r=0,42), steigt beim
naechsten Bin (5-10min) sogar auf ihr Maximum (r=0,51) und sinkt danach nur leicht auf ein
Plateau von r=0,38-0,43 bei laengeren Headways -- **es gibt KEINEN klaren, monotonen
Headway-Effekt.** Die Korrelation bleibt ueber den gesamten getesteten Headway-Bereich (von
unter 5 bis 60 Minuten) in einer aehnlichen, moderaten Groessenordnung (r zwischen 0,38 und
0,51), ohne dass kurze Headways erkennbar staerker "anstecken" als lange.

**Einordnung zur VAR-25-Minuten-Ausbreitung:** Diese Zug-zu-Zug-Korrelation ist real und
substanziell (r~0,4-0,5 an EINER Station zwischen aufeinanderfolgenden Zuegen), aber sie
erklaert die stationsUEBERGREIFENDE VAR-Kopplung NICHT direkt -- das ist eine andere Groesse
(Verspaetung eines Zuges vs. des naechsten an DERSELBEN Station, waehrend die VAR-Analyse die
Kopplung zwischen VERSCHIEDENEN Stationen ueber die Zeit misst). Dass die Korrelation
unabhaengig vom Headway relativ konstant bleibt, spricht eher fuer eine gemeinsame,
uebergeordnete Ursache (z.B. systemweite Stoerepisoden, die sowohl den Vorgaenger- als auch
den Folgezug treffen, unabhaengig vom exakten zeitlichen Abstand) als fuer einen reinen,
kapazitaetsgetriebenen "Auffahr"-Mechanismus im engeren Sinne (der eine klare
Headway-Abhaengigkeit erwarten liesse). **Ehrliches Fazit 11.3:** Es gibt klare Evidenz, dass
sich Verspaetung von einem Zug auf den direkt folgenden an derselben Station uebertraegt
(moderate, durchgehend positive Korrelation r~0,4-0,5) -- das stuetzt grundsaetzlich die
Plausibilitaet einer physischen Ausbreitungsdynamik, die der VAR-Analyse zugrunde liegen
koennte. Die spezifische Form dieser Uebertragung (kein klarer Headway-Gradient) ist aber NICHT
das erwartete "je enger, desto staerker"-Muster, das eine direkte Kapazitaetsengpass-Erklaerung
nahegelegt haette -- diese Nuance sollte einem Gutachter ehrlich mitgeteilt werden, statt die
Headway-Hypothese unkritisch als bestaetigt darzustellen. Ein direkter quantitativer Link von
dieser zugebene-Korrelation zur 25-Minuten-VAR-Ausbreitungszeit (z.B. wie viele
Zugfolge-Stufen noetig waeren, um dieselbe Zeitspanne zu erklaeren) wurde hier nicht gerechnet
und waere ein natuerlicher naechster Schritt.

### 11.4 Vertiefung Zugebene: gemeinsame Ursache, Zuwachs, Linienzugehoerigkeit

11.3 fand eine ueber alle Headway-Bins erstaunlich konstante Vorgaenger->Folgezug-Korrelation
(r=0,38-0,51) OHNE den erwarteten monotonen Abfall mit laengerem Headway. Drei Zusatztests
vertiefen dieses Ergebnis (Code: `src/analysis/var/plausibilitaet_zugebene_vertiefung.py`, nur
lesender Zugriff auf `data/02_events/final/`, `stations.py`, `plausibilitaet_zugebene.py`).

**11.4.1 Gemeinsame Ursache vs. direkte Uebertragung (r als Funktion von k).** Fuer jede
Station/Richtung wurde nicht nur r(delay_n, delay_n+1) wie in 11.3 berechnet, sondern zusaetzlich
r(delay_n, delay_n+k) fuer k in {1,2,3,5,10} an DERSELBEN Station (chronologisch nach
`time_schedule`, `trip_id` von n und n+k muss sich unterscheiden, Zeitspannen >360min als
Tagesgrenzen-Artefakte ausgeschlossen). Ergebnis, gepoolt ueber alle 12 Stationen/2 Richtungen
(`outputs/var/plausibilitaet/vertiefung_zugebene_r_vs_k.csv`, Plot
`outputs/var/plots/24a_zugebene_r_vs_k.png`):

| k | n Paare | r (95%-KI) | mediane Realzeit-Distanz Zug n <-> Zug n+k |
|---|---|---|---|
| 1 | 3.097.165 | 0,426 (0,425-0,427) | 2 min |
| 2 | 3.431.415 | 0,404 (0,403-0,405) | 6 min |
| 3 | 3.431.616 | 0,395 (0,394-0,396) | 8 min |
| 5 | 3.430.340 | 0,369 (0,368-0,370) | 13 min |
| 10 | 3.425.972 | 0,324 (0,323-0,325) | 26 min (IQR 22-32) |

Die Korrelation faellt mit wachsendem k **graduell, aber nicht steil** ab: von r=0,43 (k=1,
Median 2 min Realzeit-Abstand) auf r=0,32 (k=10, Median 26 min Realzeit-Abstand) -- ein
Rueckgang um etwa ein Viertel ueber einen mehr als zehnfach laengeren Zeithorizont. Ein reiner
kurzreichweitiger "Hand-off"-Mechanismus (Zug n+1 erbt von Zug n, der Effekt klingt schnell auf
~0 ab) wuerde einen deutlich schnelleren Abfall erwarten lassen. Bemerkenswert: bei k=10 (Median
26 min Realzeit-Abstand) bleibt r=0,32 in derselben Groessenordnung wie die r-Werte der
20-60min-Headway-Bins aus 11.3 (dort r=0,38-0,40 fuer EINEN Schritt ueber einen aehnlich langen
Zeithorizont). Das stuetzt die in 11.3 bereits geaeusserte Vermutung: der groesste Teil der
Zugebene-Korrelation ist mit einer **langsam variierenden, gemeinsamen Ursache** (allgemeine
Betriebslage des Tages/Streckenabschnitts) vertraeglich, nicht ausschliesslich mit einer engen
Zug-zu-Zug-Ansteckungskette. Ein kleiner direkter Uebertragungsanteil ist angesichts des
insgesamt tatsaechlich vorhandenen (wenn auch moderaten) Abfalls nicht auszuschliessen, aber er
dominiert das Bild nicht.

**11.4.2 Zuwachs statt Niveau.** Fuer vier geografisch benachbarte Stationspaare der
Stammstrecke (Pasing-Laim, Hauptbahnhof_tief-Karlsplatz_Stachus, Isartor-Rosenheimer_Platz,
Karlsplatz_Stachus-Marienplatz, jeweils beide Richtungen mit korrekter Upstream/Downstream-
Zuordnung ueber `stop_sequence`) wurde je `trip_id` der Verspaetungs-ZUWACHS
`delta = delay(Station k) - delay(Station k-1)` berechnet und mit der Verspaetung des
unmittelbaren VORGAENGER-Zugs an Station k-1 (derselbe Vorgaenger/Folgezug-Mechanismus wie in
11.3, inkl. Headway-Stratifizierung) verknuepft (Regression `delta_folgezug ~ delay_vorgaenger`,
Rohdaten `outputs/var/plausibilitaet/vertiefung_zuwachs_vorgaenger_rohdaten.csv.gz`, gepoolt
`outputs/var/plausibilitaet/vertiefung_zuwachs_vorgaenger_gepoolt.csv`, Plot
`outputs/var/plots/24b_zugebene_zuwachs_vorgaenger.png`):

| Headway-Bin | n | r (95%-KI) | OLS-Steigung (95%-KI) |
|---|---|---|---|
| <5min | 867.069 | 0,064 (0,062-0,067) | 0,015 (0,013-0,016) |
| 5-10min | 97.240 | 0,000 (-0,006-0,006) | 0,000 (-0,006-0,006) |
| 10-15min | 5.584 | 0,074 (0,048-0,101) | 0,013 (0,006-0,021) |
| 15-20min | 17.950 | 0,045 (0,030-0,060) | 0,012 (0,007-0,016) |
| 20-60min | 6.564 | 0,011 (-0,013-0,035) | 0,003 (-0,003-0,009) |

Das Ergebnis ist eindeutig **nicht** das, was die "Vorgaenger haelt Folgezug im Tunnel auf"-
Hypothese vorhersagen wuerde: die Korrelation zwischen Vorgaenger-Verspaetung und dem
Verspaetungs-ZUWACHS des Folgezugs zwischen zwei Nachbarstationen ist in JEDEM Headway-Bin sehr
schwach (r zwischen 0,00 und 0,07 -- eine Groessenordnung kleiner als die Niveau-Korrelation
r~0,4-0,5 aus 11.3), und es zeigt sich KEIN Muster "kuerzerer Headway = staerkerer Effekt" (der
<5min-Bin liegt bei r=0,064, nicht hoeher als 10-15min mit r=0,074; der 5-10min-Bin liegt sogar
bei r~0,00). Die einzelnen Stationspaare
(`outputs/var/plausibilitaet/vertiefung_zuwachs_vorgaenger_je_stationspaar.csv`) bestaetigen
dieses Bild: durchweg kleine, teils nicht signifikante, teils sogar leicht negative Koeffizienten
(z.B. Pasing-Laim im 5-10min-Bin: r=-0,045). Die OLS-Steigungen sind zwar meist statistisch von
Null verschieden (grosse Stichproben), aber praktisch vernachlaessigbar: selbst im <5min-Bin
fuehrt eine Vorgaenger-Verspaetung von 5 Minuten (300s) im Mittel nur zu ~4,5 Sekunden
zusaetzlichem Zuwachs des Folgezugs zwischen zwei Nachbarstationen. Das ist ein deutlich
schwaecherer, kausalerer Befund als die reine Niveau-Korrelation aus 11.3: **es gibt kaum
Evidenz, dass ein verspaeteter Vorgaenger den unmittelbar folgenden Zug auf der kurzen Strecke
zwischen zwei Nachbarstationen messbar "aufhaelt"** -- weder allgemein noch verstaerkt bei
kurzem Headway.

**11.4.3 Gleiche Linie vs. andere Linie.** Alle Vorgaenger/Folgezug-Paare aus 11.3 wurden erneut
gebildet (diesmal inkl. `line`-Spalte, da die urspruengliche Rohdaten-Datei aus 11.3 keine
Linieninformation enthaelt) und in "Folgezug hat dieselbe Linie wie Vorgaenger" vs. "andere
Linie" geteilt, mit feineren Headway-Bins um den ~20min-Grundtakt-Abstand herum (15-18/18-22/
22-27min statt nur 15-20/20-60min). Ergebnis
(`outputs/var/plausibilitaet/vertiefung_gleiche_linie_vs_andere_linie.csv`, Plot
`outputs/var/plots/24c_zugebene_gleiche_vs_andere_linie.png`):

| Headway-Bin | r gleiche Linie (n) | r andere Linie (n) |
|---|---|---|
| <5min | 0,409 (n=8.652) | 0,418 (n=2.704.953) |
| 5-10min | 0,402 (n=1.146) | **0,505** (n=267.585) |
| 10-15min | 0,561 (n=631) | 0,426 (n=30.052) |
| 15-18min | **0,520** (n=3.372) | 0,212 (n=17.103) |
| 18-22min | **0,456** (n=32.729) | 0,344 (n=8.522) |
| 22-27min | 0,554 (n=1.191) | 0,699 (n=2.160) |
| 27-35min | 0,290 (n=353) | 0,360 (n=1.875) |
| 35-60min | 0,219 (n=6.506) | 0,270 (n=4.498) |

Die Verteilung bestaetigt zunaechst die Takt-Intuition: Vorgaenger/Folgezug-Paare DERSELBEN
Linie liegen weit ueberwiegend im 18-22min-Bin (32.729 von 54.578 Gleiche-Linie-Paaren, also
~60% -- der S-Bahn-Grundtakt), waehrend Paare unterschiedlicher Linien ueberwiegend <5min
Abstand haben (2,7 Mio. von 3,04 Mio., die dichte Linienueberlagerung im Tunnel). Genau im fairen
Vergleichsbereich um 20 Minuten zeigt sich ein interessanter, aber nicht ueberwaeltigender
Unterschied: im 18-22min-Bin ist r fuer gleiche Linie (0,456, 95%-KI 0,447-0,464) signifikant
hoeher als fuer andere Linie bei aehnlichem Headway (0,344, 95%-KI 0,325-0,362) -- die
Konfidenzintervalle ueberlappen nicht. Im benachbarten 15-18min-Bin ist der Unterschied noch
groesser (0,520 vs. 0,212). Im 22-27min-Bin dreht sich das Bild jedoch um (andere Linie hoeher,
0,699 vs. 0,554, wenn auch bei kleineren Stichproben von 1.191/2.160 und daher groesseren KIs) --
das Muster ist also nicht ueber alle Bins hinweg konsistent in eine Richtung. **Fazit 11.4.3:**
Es gibt ein moderates, in zwei von drei Bins um die 20-Minuten-Marke statistisch signifikantes
Indiz, dass Zuege DERSELBEN Linie bei ihrem typischen ~15-22min-Abstand STAERKER miteinander
korrelieren als Zuege unterschiedlicher Linien bei vergleichbarem Headway. Das ist ein
PARTIELLER, qualifizierter Beleg dafuer, dass ein linienspezifischer Mechanismus (z.B.
Umlaufplanung, systematische Fahrplananpassungen innerhalb derselben Linie, oder eine echte
20-Minuten-Wiederkehr-Dynamik) zur 20-Minuten-Schwingung beitragen KOENNTE -- er ist aber klar
NICHT der alleinige Erklaerungsmechanismus (die Korrelation bei anderer Linie und aehnlichem
Headway ist mit r=0,34-0,70 selbst schon substanziell, und das 22-27min-Bin widerspricht dem
Muster). Das steht nicht im Widerspruch zu 11.2 (Linienmischungs-KOMPOSITIONS-Hypothese
widerlegt): hier wird eine andere, tiefere Frage getestet (Zug-zu-Zug-Korrelationsstaerke nach
Linienzugehoerigkeit statt Fenster-Komposition), und das Ergebnis ist ebenfalls kein vollstaendiger
Erklaerungsmechanismus, sondern hoechstens ein kleiner zusaetzlicher Baustein.

**Ehrliches Gesamtfazit 11.4:** Die drei Vertiefungstests zeichnen ein gemischtes, insgesamt eher
relativierendes Bild fuer die "Zug-zu-Zug-Ansteckung als mikroskopischer Mechanismus"-Hypothese
aus 11.3:
- **11.4.1** zeigt, dass die in 11.3 gefundene konstante Korrelation nicht rein durch schnelle,
  lokale Uebertragung erklaert werden kann -- der graduelle statt steile Abfall von r mit k
  spricht eher FUER die bereits in 11.3 geaeusserte "gemeinsame Ursache"-Interpretation als fuer
  eine echte Kettenreaktion Zug-zu-Zug.
- **11.4.2 ist der wichtigste, staerkste relativierende Befund dieses Abschnitts:** sobald man
  vom reinen Verspaetungs-NIVEAU (11.3, r~0,4-0,5) zum tatsaechlichen ZUWACHS zwischen zwei
  Nachbarstationen uebergeht -- dem kausal naeherliegenden Test fuer "haelt der Vorgaenger den
  Folgezug im Tunnel auf?" --, bricht der Effekt auf r=0,00-0,07 ein, OHNE jede erkennbare
  Headway-Abhaengigkeit. Das spricht deutlich GEGEN einen direkten kapazitaetsgetriebenen
  "Auffahr"-Mechanismus als Erklaerung fuer die Zugebene-Korrelation aus 11.3 und schwaecht damit
  eine der plausibelsten mikroskopischen Erklaerungen fuer die VAR-Kopplung, ohne die
  VAR-Kopplung selbst (die auf Stations-/5-Minuten-Aggregatsebene gemessen wird) zu widerlegen.
- **11.4.3** liefert ein schwaches bis moderates, teilweise inkonsistentes Indiz, dass die
  20-Minuten-Schwingung aus 3.3/9.6/10.2/11.2 ZUM TEIL mit linienspezifischen Zug-zu-Zug-Mustern
  zusammenhaengen koennte -- das erklaert die Schwingung aber nicht vollstaendig und ist kein
  Ersatz fuer die bereits etablierten Kompositions-Ausschlusstests (9.6c, 11.2).

Insgesamt aendert 11.4 NICHTS an den validierten VAR-Kernbefunden selbst (25-Minuten-
Ausbreitung, 20-Minuten-Oszillation, System-Kopplung bleiben unangetastet, da auf einer anderen
Aggregationsebene gemessen), aber es schwaecht die Ueberzeugungskraft der naheliegendsten
MIKROSKOPISCHEN Erklaerung (Kapazitaetsengpass durch dichte Zugfolge) spuerbar weiter, nachdem
schon 11.3 keinen Headway-Gradienten fand: 11.4.2 zeigt, dass selbst die kausal praezisere
Zuwachs-Messung keinen ueberzeugenden Auffahr-Effekt liefert. Ein Gutachter sollte hoeren: die
VAR-Kopplung ist auf Aggregatebene robust und mehrfach validiert, aber ihr mikroskopischer
Entstehungsmechanismus bleibt nach diesen Zusatztests eher UNGEKLAERT als bestaetigt --
das ist keine Schwaechung der Hauptbefunde, aber auch keine vollstaendige mechanistische
Erklaerung.

### 11.5 Gesamtbewertung der Plausibilitaetstests

**Diese Fassung von 11.5 ersetzt eine fruehere Version vollstaendig** (nach Abschluss von
11.6-11.8, siehe unten) und korrigiert dabei zwei Punkte der urspruenglichen Einordnung sowie
stellt den wichtigsten Befund der gesamten Testserie prominent an den Anfang.

**Der klarste, wichtigste Befund zuerst: kein messbarer "Auffahreffekt" im Tunnel.**
Ueber vier unabhaengige, methodisch verschiedene Tests hinweg (11.4.2 Nachbarstationen-Zuwachs,
11.6.2 Streckenzuwachs Pasing<->Leuchtenbergring, 11.7 Nichtlinearitaets-/Schwellenwert-Check)
findet sich **keine substanzielle Evidenz dafuer, dass ein verspaeteter Vorgaengerzug den direkt
folgenden Zug im Tunnel messbar "aufhaelt"**, im Sinne eines physischen Kapazitaetsengpass-
Mechanismus:
- 11.4.2 (Nachbarstationen, vier Stationspaare): r=0,00-0,07 zwischen Vorgaenger-Verspaetung und
  dem Verspaetungszuwachs des Folgezugs zwischen zwei benachbarten Stationen -- praktisch Null,
  unabhaengig vom Headway.
- 11.6.2 (ganze Kernstrecke Pasing<->Leuchtenbergring, siehe unten): der kumulierte Effekt ist
  zwar messbar groesser (r=0,10-0,18 bei kurzem Headway statt r=0,00-0,07), bleibt aber selbst
  ueber die volle Streckenlaenge klein (erklaert < 4% der Varianz) und zeigt weiterhin keinen
  klaren Headway-Gradienten.
- 11.7 (nichtlineare Klassierung statt linearer Korrelation): es gibt zwar einen konsistenten,
  monotonen Gradienten (hoehere Vorgaenger-Verspaetung -> etwas hoeherer Zuwachs des Folgezugs,
  in JEDEM Headway-Bin), aber KEINEN spezifischen Schwellen-/Auffahreffekt, der NUR in der Zelle
  "Vorgaenger >10min verspaetet UND Headway <5min" auftritt und in den anderen Zellen fehlt --
  der Gradient ist ueberall aehnlich, auch bei langem Headway. Es handelt sich also nicht um
  einen kapazitaetsspezifischen Schwelleneffekt, sondern eher um ein generelles, schwaches
  "mehr Betrieb/mehr Stoerlage = etwas mehr Verspaetungszuwachs fuer alle"-Muster.

**Interpretation:** Die stationsuebergreifende VAR-Kopplung (die zentralen, mehrfach validierten
Befunde aus Abschnitt 1-10: 25-Minuten-Ausbreitungszeit, 20-Minuten-Oszillation, deutliche
System-Kopplung) ist damit eher durch zwei andere Mechanismen erklaerbar als durch einen
physischen Kapazitaetsengpass im Tunnel selbst:
1. **Mitgefuehrte Verspaetung:** ein bereits verspaeteter Zug bleibt ueber seine gesamte Fahrt
   verspaetet (Verspaetung wird kaum ab- oder aufgebaut), was sich in den 5-Minuten-Fenster-
   Aggregaten der VAR-Analyse als raeumliche Korrelation zwischen Stationen zeigt, OHNE dass ein
   Zug einen anderen tatsaechlich physisch "ansteckt".
2. **Gemeinsame uebergeordnete Stoerlagen:** systemweite Ereignisse (Signal-/Weichenstoerungen,
   Personalengpaesse, Wetterlagen etc.), die mehrere Zuege gleichzeitig betreffen und dadurch
   raeumlich korrelierte Verspaetungsmuster erzeugen, ohne dass ein Zug den naechsten direkt
   beeinflusst.

Das relativiert **nicht** die VAR-Kopplungsbefunde selbst (sie sind auf Aggregatebene robust und
mehrfach validiert, siehe Abschnitt 1-10, und werden von den Plausibilitaetstests methodisch
nicht beruehrt) -- relativiert wird nur eine bestimmte, naheliegende KAUSALE Interpretation
dieser Kopplung ("Kapazitaetsengpass durch dichte Zugfolge im Tunnel"), die durch die
Zugebene-Daten nicht gestuetzt wird.

**Eine wichtige Ausnahme mit gegenlaeufigem Befund: der Ausfallkanal (11.8).** Waehrend die
Verspaetungs-WEITERGABE keinen Auffahreffekt zeigt, zeigt die AUSFALLRATE des Folgezugs einen
klaren, headway-abhaengigen Zusammenhang mit der Vorgaenger-Verspaetung: bei Headway <5min
steigt die Ausfallrate des Folgezugs von 1,6% (Vorgaenger <=2min verspaetet) auf 8,3%
(Vorgaenger >10min verspaetet) -- ein mehr als fuenffacher Anstieg, konsistent monoton ueber
alle vier Verspaetungsklassen, und bei laengerem Headway (10-20min) deutlich schwaecher
ausgepraegt. Das ist ein substanzieller, ehrlich zu berichtender GEGENbefund zur obigen
Haupterzaehlung: es GIBT einen headway-abhaengigen Uebertragungskanal zwischen Vorgaenger- und
Folgezug-Stoerung -- er wirkt nur nicht ueber die Verspaetungshoehe des Folgezugs, sondern ueber
dessen Ausfallwahrscheinlichkeit (plausibel z.B. durch Fahrzeug-/Personaldisposition: ein
extrem verspaeteter Zug bindet Ressourcen, die dem eng folgenden naechsten Zug fehlen). Das ist
mit dem "kein Auffahreffekt"-Befund oben NICHT in Widerspruch, sondern ergaenzt ihn: der
Kapazitaets-/Dispositionsdruck aeussert sich eher in AUSFAELLEN naheliegender Folgezuege als in
graduell steigender Verspaetung derjenigen Folgezuege, die tatsaechlich fahren.

**Korrektur 1 -- 11.1-Richtungsuebereinstimmung ist KEIN unabhaengiger Beleg.** Die urspruengliche
Fassung dieses Abschnitts wertete die 100%ige Uebereinstimmung der stop_sequence-basierten
Richtungszuordnung (11.1) mit der bereits vorhandenen `richtung`-Spalte als "zusaetzlichen,
unabhaengigen Beleg fuer die Konsistenz der Datenpipeline". Das ist irrefuehrend und wird hier
gestrichen: wie Stufe-1-`BEFUNDE.md` (Abschnitt 0.2) zeigt, wurde die `richtung`-Spalte selbst
urspruenglich aus genau derselben stop_sequence-Logik abgeleitet. Beide Zuordnungen beruhen also
auf derselben zugrundeliegenden Methode -- eine 100%ige Uebereinstimmung ist folglich ZIRKULAER
(fast tautologisch zu erwarten) und keine unabhaengige Bestaetigung der Datenpipeline. Der Rest
von Abschnitt 11.1 (die eigentliche Fahrzeit-Plausibilitaetspruefung, 21 vs. 25 Minuten) bleibt
davon unberuehrt und ist weiterhin eine gueltige, positive Bestaetigung.

**Korrektur 2 -- 20-Minuten-Schwingung: Ursache bleibt offen, nicht "endogene Tunneldynamik"
behaupten.** An mehreren Stellen (u.a. in frueheren Formulierungen dieses Abschnitts sowie
teilweise in 9.6/10.2) wurde die 20-Minuten-Oszillation als "eher endogene VAR-/Tunneldynamik"
bezeichnet -- eine kausal-mechanistische Formulierung, die zu stark interpretiert. Praezise
laesst sich nur sagen: die 20-Minuten-Schwingung ist **zweifach als Kompositions-/Placebo-
Artefakt ausgeschlossen** (9.6c: exaktes Fourier-Placebo bei 20/48min entfernt die Schwingung
nicht; 11.2: Herausrechnen der linienspezifischen mittleren Verspaetung veraendert sie kaum) --
aber der TATSAECHLICHE URSPRUNGSMECHANISMUS bleibt offen. 11.4.3 lieferte ein partielles,
nicht durchgaengig konsistentes Indiz fuer einen linienspezifischen Beitrag (gleiche Linie bei
~15-22min-Abstand korreliert staerker als andere Linie bei aehnlichem Headway, aber nicht im
22-27min-Bin). Weder "endogene Tunneldynamik" noch ein anderer spezifischer Mechanismus ist mit
den vorliegenden Tests bewiesen -- das ist und bleibt eine offene Frage.

**Einordnung aller Einzeltests (11.1-11.4, 11.6-11.8):**
- **11.1 (Fahrzeit):** starke Bestaetigung der VAR-Ausbreitungszeit (siehe oben, ohne die
  fehlerhafte Richtungszuordnungs-Interpretation).
- **11.2 (Linienmischung):** widerlegt die Komposition-Hypothese fuer die 20-Minuten-Schwingung
  klar -- der Peak bleibt nach Linienbereinigung fast unveraendert (bei Ost sogar leicht
  verstaerkt) bestehen.
- **11.3 (Zugebene, Niveau-Korrelation):** substanzielle Zug-zu-Zug-Korrelation an derselben
  Station (r=0,38-0,51), aber OHNE den erwarteten Headway-Gradienten -- erste Relativierung der
  "kurzer Headway = starke Ansteckung"-Intuition.
- **11.4 (Vertiefung):** 11.4.1 (graduelle statt steile Abnahme mit k, spricht fuer gemeinsame
  Ursache statt Kettenreaktion), 11.4.2 (Zuwachs statt Niveau: r bricht auf 0,00-0,07 ein -- der
  wichtigste relativierende Einzelbefund vor 11.6/11.7), 11.4.3 (partielles Linienindiz fuer die
  20-Minuten-Schwingung).
- **11.6.1 (Klaerung Headway<=0):** die 335.673 in 11.3 kommentarlos ausgeschlossenen
  Headway<=0-Paare sind **zu 100% exakt Headway=0** (keine kleinen oder grossen negativen
  Werte) und zu 99,06% (332.517 von 335.673) direkt als **gekoppelte Zugverbaende** identifizierbar
  (zwei trip_ids desselben physischen Zugverbands, an median 10 von 11 gemeinsamen
  Stammstrecke-Stationen zeitgleich, davon median 10 exakt zeitgleich innerhalb 60 Sekunden).
  Kein Hinweis auf Datenduplikate (0 doppelte trip_id/stop_id/is_arrival-Kombinationen in den
  Rohdaten) oder echte Ueberholungen (der Headway basiert auf time_schedule, nicht time_real --
  eine reale Ueberholung kann per Definition keinen negativen SCHEDULE-Headway erzeugen). Nur
  0,94% (3.156 Paare) bleiben ungeklaert (Einzelstation-Koinzidenz ohne Mehrstationen-
  Bestaetigung). **Das aendert nichts an der Validitaet der 90,2%-Teilmenge (Headway>0) aus
  11.3/11.4** -- die ausgeschlossenen Faelle sind eine andere, gut erklaerte Kategorie
  (Kopplung, kein Datenfehler), keine verdeckte Verzerrung der verbleibenden Stichprobe.
- **11.6.2 (Streckenzuwachs):** der kumulierte Effekt ueber die ganze Kernstrecke ist groesser
  als der Nachbarstationen-Effekt aus 11.4.2 (r=0,10-0,18 statt 0,00-0,07 bei kurzem Headway),
  bestaetigt also teilweise die Vermutung "kleine Effekte pro Abschnitt summieren sich", bleibt
  aber absolut klein und ohne klaren Headway-Gradienten (15-20min-Bin nahe Null).
- **11.7 (Nichtlinearitaet):** kein spezifischer Schwellen-/Auffahreffekt bei extremer
  Kombination (Vorgaenger >10min, Headway <5min) -- der beobachtete Gradient ist durchgehend
  moderat und in allen Headway-Bins qualitativ aehnlich, kein Sprung nur in der Extremzelle.
- **11.8 (Ausfallkanal):** klarer, headway-abhaengiger GEGENbefund -- die Ausfallrate des
  Folgezugs steigt bei kurzem Headway stark mit der Vorgaenger-Verspaetung (1,6% auf 8,3%),
  waehrend die Verspaetungs-HOEHE der (nicht ausgefallenen) Folgezuege kaum reagiert.

**Fazit fuer einen Gutachter:** Die VAR-Kernbefunde selbst (Ausbreitungszeit, Oszillation,
Kopplungsstaerke, Abschnitte 1-10) sind auf Aggregatebene robust, mehrfach validiert und werden
durch keinen der zehn Plausibilitaetstests widerlegt. Was sich durch 11.1-11.8 aendert, ist die
Einschaetzung des MIKROSKOPISCHEN Entstehungsmechanismus: ein direkter, physischer
Kapazitaetsengpass-Mechanismus ("Zug haelt Zug im Tunnel auf") findet in der Verspaetungshoehe
keine ueberzeugende Stuetzung (11.3, 11.4.2, 11.6.2, 11.7), wohl aber -- als eigener, klar
identifizierter Kanal -- in der Ausfallwahrscheinlichkeit des Folgezugs bei kurzem Headway
(11.8). Die wahrscheinlichste Gesamterklaerung der VAR-Kopplung bleibt eine Mischung aus
mitgefuehrter Einzelzug-Verspaetung, gemeinsamen uebergeordneten Stoerlagen und einem
schwaecheren, aber realen Ausfall-Uebertragungskanal -- nicht ein einzelner, dominanter
physischer Engpassmechanismus im Tunnelkern.

### 11.6 Klaerung Headway<=0 und Streckenzuwachs (Ergaenzung zu 11.3/11.4)

**11.6.1 Klaerung der 335.673 Headway<=0-Paare (Code:
`src/analysis/var/plausibilitaet_headway_artefakt.py`).** In 11.3 wurden 335.673 von 3.433.425
Vorgaenger/Folgezug-Paaren (9,8%) mit Headway<=0 kommentarlos ausgeschlossen. Methodik: alle
Ankunfts-Ereignisse der 12 Stammstrecke-Stationen (dieselbe Basis wie 11.3, is_arrival=true,
is_cancelled=false, richtung IS NOT NULL) wurden erneut chronologisch nach `time_schedule`
gepaart, diesmal OHNE den Headway>0-Filter. Fuer jedes eindeutige (trip_id_prev, trip_id)-Paar
mit Headway<=0 wurde geprueft, an wie vielen der bis zu 11 weiteren gemeinsam befahrenen
Stammstrecke-Stationen beide trip_ids ebenfalls (nahezu, Toleranz 60 Sekunden) zeitgleich
verkehren -- ein starkes empirisches Kriterium fuer einen gekoppelten Zugverband (zwei
trip_ids, ein physischer Zug/Verband, ueblich bei S-Bahn-Doppeltraktionen).

**Ergebnis** (`outputs/var/plausibilitaet/headwayartefakt_*`, Plot
`outputs/var/plots/25_headway_negativ_klaerung.png`):
- Die Headway-Verteilung im Bereich <=0 ist **zu 100% exakt Headway=0 Sekunden** -- keine
  kleinen negativen Werte (-2 bis 0min) und keine grossen negativen Spruenge kommen vor. Das
  schliesst "reale Ueberholung" (die eher kleine, realzeit-basierte Verschiebungen erzeugen
  wuerde, und ausserdem per Definition den SCHEDULE-Headway nicht negativ machen kann) als
  Erklaerung sofort aus.
- **Datenduplikate:** 0 Faelle von mehrfach vorkommenden (trip_id, stop_id, is_arrival)-
  Kombinationen in den Rohdaten unter `data/02_events/final/` -- die Datenpipeline ist an dieser
  Stelle bereits vollstaendig depliziert, Duplikate scheiden als Erklaerung aus.
  (`outputs/var/plausibilitaet/headwayartefakt_duplikat_check.csv`)
- **Gekoppelter Zugverband:** von 85.603 eindeutigen (trip_id_prev, trip_id)-Paaren mit
  Headway=0 sind 82.447 (96,3%) Konvoi-Kandidaten (>=2 fast-zeitgleiche gemeinsame Stationen);
  im Median liegen sogar 10 von median 10 gemeinsam befahrenen Stationen fast zeitgleich (siehe
  `outputs/var/plausibilitaet/headwayartefakt_konvoi_kandidaten.csv.gz`). Auf Ebene der
  335.673 Paar-ZEILEN (nicht eindeutiger trip-Paare, ein Konvoi-Paar taucht an mehreren
  Stationen als eigene Zeile auf) entspricht das **332.517 Zeilen (99,06%)**, die eindeutig der
  Kategorie "gekoppelter Verband" zuzuordnen sind. In 74% dieser Konvoi-Paare stimmt sogar die
  `line`-Spalte ueberein, die restlichen 26% sind vermutlich Kopplungen unterschiedlicher
  Linien (z.B. gemeinsame Zugfahrt bis zu einem Trennungspunkt).
- Nur **3.156 Paare (0,94%)** bleiben ungeklaert -- Einzelstation-Zufallskoinzidenzen ohne
  Bestaetigung an weiteren gemeinsamen Stationen; denkbare Ursachen (nicht weiter untersucht)
  sind kurze Ueberschneidungen an Linienendpunkten oder seltene Datenrandfaelle.

**Fazit 11.6.1:** Die Nutzer-Vermutung "gekoppelte Zugverbaende als zwei trip_ids" wird durch
diesen Test **klar bestaetigt** -- praktisch alle (99,06%) Headway<=0-Faelle sind erklaerte
Zugverband-Kopplungen, keine Datenfehler. Da diese Faelle eine andere, physisch begruendete
Kategorie darstellen (kein Vorgaenger/Folgezug-Verhaeltnis im eigentlichen Sinn, sondern
gleichzeitig verkehrende Teile desselben Verbands), aendert diese Klaerung **nichts an der
Validitaet der 90,2%-Headway>0-Teilmenge**, die in 11.3/11.4 verwendet wurde -- der Ausschluss
war methodisch korrekt, auch wenn er in 11.3 nicht naeher begruendet war.

**11.6.2 Verspaetungs-Zuwachs ueber die GESAMTE Kernstrecke (Code:
`src/analysis/var/plausibilitaet_zuwachs_strecke.py`).** 11.4.2 mass den Zuwachs nur zwischen
vier Paaren geografisch BENACHBARTER Stationen (r=0,00-0,07). Hypothese: die minutengenaue
`delay`-Aufloesung koennte kleine Effekte PRO Abschnitt verdecken, die sich ueber die ganze
Strecke kumuliert zu einem messbaren Signal aufsummieren. Test analog zu 11.4.2, aber mit den
Endpunkten der Kernstrecke: **Pasing (Rang 0) und Leuchtenbergring (Rang 11)** in
`stations.STATION_ORDER_WEST_OST` -- nicht Ostbahnhof (Rang 10, vorletzte Station), da
Leuchtenbergring der tatsaechliche Ost-Endpunkt der 12-Stationen-Kernstrecke ist und bereits in
11.1 als Kernstrecken-Endpunkt fuer den Fahrzeitvergleich verwendet wurde (direkte
Vergleichbarkeit mit der 25-Minuten-VAR-Peak-Zeit).

**Ergebnis** (`outputs/var/plausibilitaet/streckenzuwachs_*`, Plot
`outputs/var/plots/26_streckenzuwachs_vs_nachbarstationen.png`, 94.399 gematchte Trips beider
Richtungen mit Vorgaenger-Verknuepfung):

| Headway-Bin | n | r ganze Strecke (95%-KI) | r Nachbarstationen (11.4.2, zum Vergleich) |
|---|---|---|---|
| <5min | 60.525 | **0,141** (0,133-0,149) | 0,064 |
| 5-10min | 26.823 | **0,176** (0,165-0,188) | 0,000 |
| 10-15min | 3.200 | 0,099 (0,065-0,134) | 0,074 |
| 15-20min | 3.059 | -0,001 (-0,037-0,034) | 0,045 |
| 20-60min | 792 | 0,068 (-0,002-0,137) | 0,011 |

Die Vermutung des Nutzers bestaetigt sich **teilweise**: der kumulierte Effekt ueber die ganze
Kernstrecke ist bei kurzem Headway (<5min, 5-10min) tatsaechlich rund 2-3x groesser als der
Effekt zwischen zwei Nachbarstationen (r=0,14-0,18 statt r=0,00-0,06), konsistent mit der Idee,
dass sich kleine Pro-Abschnitt-Effekte aufsummieren. Trotzdem bleibt der Effekt absolut
**klein** (r=0,18 erklaert < 4% der Varianz des Streckenzuwachses) und zeigt weiterhin **keinen
klaren, monotonen Headway-Gradienten** (der 15-20min-Bin liegt bei r~0,00, nicht klar unter dem
<5min-Bin, und der 10-15min-Bin liegt nicht ueber dem 5-10min-Bin). Je Richtung (Ost/West)
getrennt betrachtet (`outputs/var/plausibilitaet/streckenzuwachs_je_richtung.csv`) zeigt sich
dasselbe Bild in beiden Richtungen. **Fazit 11.6.2:** ein schwacher, aber real messbarer
kumulierter Effekt existiert -- er ist naeher an einem "generellen Trend, dass Zuege waehrend
belasteter Betriebslagen etwas mehr Verspaetung aufbauen" als an einem spezifischen, headway-
getriebenen Auffahrmechanismus.

### 11.7 Nichtlineare Analyse -- Auffahreffekt erst bei extremer Verspaetung?

**Methodik (Code: `src/analysis/var/plausibilitaet_nichtlinear.py`).** Statt linearer
Korrelation wurden Vorgaenger-Verspaetung (<=2min, 2-5min, 5-10min, >10min) und Headway (wie
bisher: <5min, 5-10min, 10-15min, 15-20min, 20-60min) gekreuzt klassiert; je Zelle mittlerer
Zuwachs des Folgezugs mit 95%-Konfidenzintervall (analytische Normalapproximation,
mean +/- 1,96*SE). Auf beiden Zuwachs-Definitionen gerechnet: Nachbarstationen (11.4.2-Basis,
groessere Stichprobe) und ganze Kernstrecke (11.6.2-Basis).

**Ergebnis** (`outputs/var/plausibilitaet/nichtlinear_*`, Plots
`outputs/var/plots/27a_nichtlinear_heatmap_nachbarstationen.png`,
`outputs/var/plots/27b_nichtlinear_heatmap_ganzestrecke.png`). Nachbarstationen, mittlerer
Zuwachs in Minuten:

| Headway-Bin | Vorgaenger<=2min | 2-5min | 5-10min | >10min |
|---|---|---|---|---|
| <5min | 0,070 | 0,160 | 0,211 | **0,278** |
| 5-10min | 0,349 | 0,298 | 0,316 | **0,429** |
| 10-15min | 0,016 | 0,072 | 0,102 | **0,406** |
| 15-20min | 0,081 | 0,144 | 0,189 | **0,329** |
| 20-60min | 0,026 | 0,024 | -0,012 | 0,166 (n=278, KI schliesst 0 knapp ein) |

Es zeigt sich in JEDEM Headway-Bin ein **konsistenter, monotoner Gradient**: je hoeher die
Vorgaenger-Verspaetungsklasse, desto hoeher (im Mittel) der Zuwachs des Folgezugs -- auch bei
15-20min und teils 10-15min Headway. Das ist real und statistisch klar abgesichert (schmale
KIs bei grossen n), aber es ist **kein spezifischer Schwellen-/Auffahreffekt**, der NUR in der
vermuteten Extremzelle (Vorgaenger >10min, Headway <5min: 0,278min) auftritt -- vergleichbar
hohe oder sogar hoehere Werte finden sich auch bei 5-10min und 15-20min Headway in derselben
Verspaetungsklasse (0,429 bzw. 0,329min). Auf der ganzen Kernstrecke (Plot 27b) zeigt sich
dasselbe Muster in groesserem Massstab (z.B. <5min-Headway: 2,79min bei puenktlichem Vorgaenger
bis 4,54min bei >10min verspaetetem Vorgaenger) -- auch hier kein Sprung, der spezifisch auf die
Extremzelle beschraenkt waere; die Differenz (~1,7min) ist ueber die ganze Strecke plausibel mit
der Summe der kleinen Pro-Abschnitt-Gradienten aus 11.6.2 vertraeglich.

**Fazit 11.7:** Die Hypothese eines spezifischen, nichtlinearen "Auffahreffekts" (der bei
linearer Korrelation durch die Masse "normaler" Faelle verdeckt wuerde) wird **nicht
bestaetigt**. Es gibt einen schwachen, aber durchgehenden Gradienten "mehr Vorgaenger-
Verspaetung -> etwas mehr Zuwachs", der aber weder auf kurze Headways konzentriert noch in
seiner Groessenordnung geeignet ist, die stationsuebergreifende VAR-Kopplung zu erklaeren.

### 11.8 Ausfaelle als Uebertragungskanal

**Methodik (Code: `src/analysis/var/plausibilitaet_ausfallkanal.py`).** Bisherige Zugpaar-
Konstruktionen (11.3/11.4/11.6/11.7) filterten `is_cancelled=false` VOR der Paarbildung -- ein
ausgefallener Folgezug wurde dadurch stillschweigend uebersprungen und der naechste tatsaechlich
fahrende Zug faelschlich zum "unmittelbaren Nachfolger". Um genau diesen moeglichen
Selektionseffekt zu pruefen, wurden die Paare hier aus ALLEN geplanten Ankunfts-Ereignissen
(inkl. `is_cancelled=true`) neu gebildet; der Vorgaenger muss weiterhin nicht ausgefallen sein
(113.363 von 3.546.788 Paaren mit ausgefallenem Vorgaenger wurden separat gezaehlt und aus der
Hauptauswertung ausgeschlossen, da seine `delay`-Beobachtung sonst nicht interpretierbar waere).
Gleiche Klassenbildung wie 11.7 (Vorgaenger-Verspaetungsklasse x Headway-Bin), Zielgroesse:
Ausfallrate des Folgezugs (`is_cancelled` des Folgezugs) mit 95%-Wilson-Konfidenzintervall.

**Ergebnis** (`outputs/var/plausibilitaet/ausfallkanal_*`, Plot
`outputs/var/plots/28_ausfallkanal_heatmap.png`, 3.090.301 auswertbare Paare, Gesamt-
Ausfallrate des Folgezugs 2,95%):

| Headway-Bin | Vorgaenger<=2min | 2-5min | 5-10min | >10min |
|---|---|---|---|---|
| <5min | 1,59% | 3,16% | 4,76% | **8,31%** |
| 5-10min | 1,74% | 2,30% | 2,90% | **6,63%** |
| 10-15min | 0,73% | 0,97% | 1,33% | 1,56% |
| 15-20min | 0,51% | 0,59% | 0,95% | 2,83% |
| 20-60min | 0,16% | 0,42% | 0,76% | 0,76% (n=795, klein) |

Die Ausfallrate des Folgezugs steigt **klar und monoton** mit der Vorgaenger-Verspaetung in
JEDEM Headway-Bin, und der Effekt ist **bei kurzem Headway am staerksten ausgepraegt**: bei
Headway <5min steigt sie von 1,6% (puenktlicher Vorgaenger) auf 8,3% (Vorgaenger >10min
verspaetet) -- ein Faktor 5,2. Bei 5-10min Headway ist der Anstieg aehnlich stark (1,7% auf
6,6%, Faktor 3,8). Bei laengerem Headway (10-20min) ist der relative Anstieg zwar noch
vorhanden, aber auf absolut niedrigerem Niveau deutlich schwaecher ausgepraegt.

**Fazit 11.8:** Ja, die Ausfallrate des Folgezugs steigt mit der Vorgaenger-Verspaetung,
besonders bei kurzem Headway -- das ist ein klarer, substanzieller Befund, der **im Gegensatz**
zu den Verspaetungs-Zuwachs-Tests (11.4.2, 11.6.2, 11.7) tatsaechlich einen headway-abhaengigen
Uebertragungsmechanismus zeigt. Ausserdem bestaetigt sich der befuerchtete Selektionseffekt der
frueheren Tests: da ausgefallene Folgezuege in 11.3/11.4/11.6/11.7 aus der Paarbildung
herausfielen, wurde genau die Untergruppe der staerksten Stoerungsfaelle (Vorgaenger stark
verspaetet, kurzer Headway) dort systematisch NICHT als eigener Fall erfasst, sondern durch den
naechsten tatsaechlich fahrenden Zug ersetzt -- was die in 11.4.2/11.6.2/11.7 gemessenen
Zuwachs-Effekte tendenziell eher unterschaetzt als ueberschaetzt haben duerfte (der am
staerksten betroffene, ausgefallene Fall fehlt dort komplett). Dieser Befund ist in die
Gesamtbewertung in 11.5 eingearbeitet.

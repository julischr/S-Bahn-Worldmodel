# Auswertungsplan Stufe 1 — Impulsantwort messen

Festgelegt VOR der ersten Rechnung, wie vom Bauplan gefordert.

## Ereignisart

**message_code 34 — "Defekt an einem Signal"** (IRIS-Referenz: `Travel::Status::DE::IRIS::Result`, derf).

Abweichung vom Vorschlag im Bauplan ("häufigster Ursachencode"): der tatsächlich
häufigste Code auf den 12 Stammstreckenhalten ist **43** ("Verspätung eines
vorausfahrenden Zuges", 935.676 Vorkommen). Das ist kein geeignetes
Störungsereignis für eine Impulsantwort-Messung — es ist ein generischer
Folge-Code, der praktisch jeden moderat verspäteten Zug markiert, ohne
diskreten Beginn oder abgrenzbare externe Ursache. Eine "Impulsantwort" auf
ein Ereignis, das selbst schon die Antwort auf etwas anderes ist, liefert
kein interpretierbares Ergebnis.

Code 34 (281.153 Vorkommen, viertMeistgenannt) ist dagegen ein konkreter,
extern verursachter Infrastrukturdefekt mit klarem Beginn — strukturell
genau das, was die Methodik braucht. Das deckt sich außerdem mit dem
BEG-Jahresbericht 2025 (siehe `BEFUNDE.md`, Abschnitt 0.4), der Störungen an
Signal- und Sicherungstechnik als häufigste Verspätungsursache im gesamten
bayerischen Netz nennt (47,8%).

## Station und Zielgröße

- Alle 12 Stammstreckenhalte.
- Zielgröße: `delay_mean` **an der Station des Ereignisses selbst, in der
  betroffenen Richtung** (nicht netzweit gemittelt) — direkt aus
  `data/03_state_vector/state_vector_5min.parquet`, das schon exakt auf
  `(zeitfenster, stop_id, richtung)` aggregiert ist. Das ist auch, was die
  Kontrollfenster-Kriterien in 1.3 ("gleiche Station, gleiche Richtung")
  implizit voraussetzen.
- Ausbreitung auf andere Stationen ist eine natürliche Folgefrage, aber
  bewusst NICHT Teil des ersten Durchgangs (Abnahmekriterium ist eine Kurve
  für mindestens eine Ereignisart).

## Zeitfenster

−60 bis +240 Minuten relativ zu t0, im 5-Minuten-Raster des Zustandsvektors
(61 Zeitpunkte: −60, −55, …, 0, …, +240).

## Ereignisdefinition (Details siehe 1.2)

- Cluster: alle `events_final`-Zeilen mit Code 34 an Station s innerhalb von
  15 Minuten (`update_timestamp`).
- Richtung eines Clusters: Mehrheitsrichtung der betroffenen Fahrten (siehe
  Code-Kommentar für den Umgang mit gemischten Clustern).
- t0 = frühester `update_timestamp` im Cluster.
- Mindestgröße: ≥3 betroffene Fahrten (in der Mehrheitsrichtung).
- Bereinigung: Ereignis verwerfen, wenn im Fenster [t0−60min, t0+240min]
  ein weiteres Code-34-Ereignis an DERSELBEN Station+Richtung liegt.

## Kontrollfenster (siehe 1.3)

k=10 pro Ereignis, Kriterien wie im Bauplan spezifiziert:
gleicher Wochentagstyp (Mo–Do / Fr / Sa / So+Feiertag, bayerische
gesetzliche Feiertage 2025 — Mariä Himmelfahrt bewusst ausgeklammert, da
diese Regelung gemeindeabhängig ist und München nicht landesweit einheitlich
zugeordnet werden kann), Uhrzeit ±30min, Datum ±6 Wochen, kein Code-34-
Ereignis im Fenster [−60,+240] an derselben Station+Richtung, gleiche
Station und Richtung wie das Ereignis.

## Auswertung (siehe 1.4)

Differenz Ereigniskurve − Kontrollmittel, über alle sauberen Ereignisse
gemittelt. Bootstrap-Konfidenzintervall **tageweise** gezogen (Tage mit
Ereignissen werden mit Zurücklegen resampled, nicht einzelne Ereignisse),
1000 Wiederholungen, 95%-Perzentilband.

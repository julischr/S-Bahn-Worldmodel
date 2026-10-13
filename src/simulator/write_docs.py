"""Setzt BEFUNDE_SIMULATOR.md und outputs/simulator/UEBERGABE_KI.md aus Text + Tabellen (tabellen.md) zusammen."""
import re
import pandas as pd
from .data import OUT, PROJECT_ROOT
from .report import md, main as make_tables

make_tables()
T = {}
cur = None
for line in open(OUT / "tabellen.md"):
    m = re.match(r"<!-- (\S+) -->", line)
    if m:
        cur = m.group(1); T[cur] = []
    elif cur and not line.startswith("### "):
        T[cur].append(line)
T = {k: "".join(v).strip() for k, v in T.items()}

# kompakte Szenariotabelle
z = pd.read_csv(OUT / "szenario_zusammen_szenarien.csv")
z["variante"] = z.apply(lambda r: ("Verspätung +%d min" % r.minuten if r.art == "verspaetung" else "Fahrt gestrichen") + (", HVZ" if r.hvz else ", außerhalb HVZ"), axis=1)
SZ = md(z.rename(columns={
    "betroffen_pro_lauf_mittel_mittel_ueber_zuege": "betroffene Züge/Lauf (Mittel)", "betroffen_q05_mittel_ueber_zuege": "betr. q05", "betroffen_q95_mittel_ueber_zuege": "betr. q95",
    "zugaenderung_mittel_mittel_ueber_zuege": "Δ je betr. Zug Mittel (min)", "zugaenderung_median_mittel_ueber_zuege": "Median", "zugaenderung_q10_mittel_ueber_zuege": "q10", "zugaenderung_q90_mittel_ueber_zuege": "q90",
    "zugaenderung_max_mittel_ueber_zuege": "max", "gesamt_mehrverspaetung_zugmin_mittel_mittel_ueber_zuege": "Gesamt-Mehrverspätung Zug-min (Mittel)",
    "gesamt_zugmin_q05_mittel_ueber_zuege": "Zug-min q05", "gesamt_zugmin_q95_mittel_ueber_zuege": "Zug-min q95",
    "zusaetzliche_ausfaelle_mittel_mittel_ueber_zuege": "zus. Ausfälle (Mittel)", "zus_ausf_q05_mittel_ueber_zuege": "Ausf. q05", "zus_ausf_q95_mittel_ueber_zuege": "Ausf. q95",
    "abklingzeit_ki_min_mittel_ueber_zuege": "Abklingzeit KI-Regel (min)", "abklingzeit_unter_0_1min_mittel_ueber_zuege": "Abklingzeit <0,1 min (min)",
    "eigener_zug_aenderung_ende_mittel_mittel_ueber_zuege": "eigener Zug Δ am Ende (min)",
    "gesamt_mehrverspaetung_zugmin_mittel_min_zug": "Zug-min Mittel, min. Zug", "gesamt_mehrverspaetung_zugmin_mittel_max_zug": "Zug-min Mittel, max. Zug"}),
    2, ["variante", "n_sim", "betroffene Züge/Lauf (Mittel)", "betr. q05", "betr. q95", "Δ je betr. Zug Mittel (min)", "Median", "q10", "q90", "max",
        "Gesamt-Mehrverspätung Zug-min (Mittel)", "Zug-min q05", "Zug-min q95", "Zug-min Mittel, min. Zug", "Zug-min Mittel, max. Zug",
        "zus. Ausfälle (Mittel)", "Ausf. q05", "Ausf. q95", "Abklingzeit KI-Regel (min)", "Abklingzeit <0,1 min (min)", "eigener Zug Δ am Ende (min)"])
T["szenario_kompakt"] = SZ

STONES = """| Prüfstein | Kriterium (vorab im Plan) | Ergebnis | Urteil |
|---|---|---|---|
| (a) Ankunft am Pfadende (22.848 Züge, 30 Tage) | 90-%-Band-Abdeckung in [85 %; 95 %]; \\|Bias Median\\| ≤ 0,5 min; MAE ≤ MAE Persistenz | Abdeckung 93,6 % (KI 92,7–94,5); Bias −0,11 min (−0,29…+0,04); MAE 1,71 min vs. Persistenz 2,87 min | **bestanden** |
| (b) Verspätungsaufbau über die Strecke (8.239 Züge Rang 0↔11) | \\|Sim − Echt\\| ≤ 0,5 min | Echt 3,31 min (2,88–3,74), Sim 3,69 min (3,36–4,00), Differenz +0,38 (+0,14…+0,58) | **bestanden** (knapp, systematisch zu hoch) |
| (c) Zuwachs Folgezug nach Vorgänger/Abstand (Tabelle 11.7) | ≥ 75 % der Zellen mit n ≥ 200 innerhalb ±0,3 min und Gradient wie Echt | 2 von 9 Zellen (22 %) innerhalb ±0,3; Sim in 9 von 9 Zellen zu hoch (+0,12…+0,78 min); Gradient in der Zeile „<5 min" vorhanden | **nicht bestanden** |
| (d) Ausfallquote nach Vorgänger/Abstand (Tabelle 11.8, nur modellierte Übergänge) | ≥ 75 % der Zellen mit n ≥ 500 innerhalb max(0,3 pp; 40 %) und Monotonie in Zeile „<5 min" | 8 von 14 Zellen (57 %); Monotonie in Echt und Sim vorhanden, aber Sim-Anstieg flacher (Zeile „<5 min": Echt 0,7 % → 4,7 %, Sim 1,0 % → 3,5 %) | **nicht bestanden** |
| (e) Erholungszeit je Schwereklasse (1.332 Episoden, 20 Läufe/Tag) | ≥ 4 von 5 Klassen mit n ≥ 50 innerhalb max(25 %; 5 min) bzw. Echt-Median in Lauf-Spanne und Monotonie | 3 von 5 Klassen (<3, 3–5, 5–10 ok; 10–20 und >20 nicht); Mediane monoton in Echt und Sim | **nicht bestanden** |
"""
T["steine"] = STONES

def sub(text):
    return re.sub(r"\{\{(\w+)\}\}", lambda m: T.get(m.group(1), "(fehlt: %s)" % m.group(1)), text)

BEF = open(OUT / "vorlage_befunde.md").read()
UEB = open(OUT / "vorlage_uebergabe.md").read()
(PROJECT_ROOT / "BEFUNDE_SIMULATOR.md").write_text(sub(BEF))
(OUT / "UEBERGABE_KI.md").write_text(sub(UEB))
print("ok")

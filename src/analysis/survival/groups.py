"""Gruppierung der Codes (>= 50 Episoden einzeln, Rest 'selten') und Codeliste."""

import pandas as pd

from src.analysis.stoerungsinventur import IRIS_CODES
from .data import AUSLOESER, OUT

MIN_EP = 50


def code_table(ep: pd.DataFrame) -> pd.DataFrame:
    vc = ep["code"].value_counts()
    rows = []
    for c in AUSLOESER:
        n = int(vc.get(c, 0))
        grund = "Störungscode 2–69 der Inventur, kein Folgecode (43/44/45/46/48)"
        if c == 23:
            grund += "; Vorbehalt: Betriebsstabilisierung ist eher Maßnahme als Ursache"
        if c in (21, 25, 32, 47, 51, 57, 61):
            grund += "; eher betriebliche Meldung als externe Störung (nicht ausgeschlossen, da Inventur nicht)"
        rows.append(dict(code=c, bedeutung=IRIS_CODES.get(c, "?"), n_episoden=n,
                         gruppe=str(c) if n >= MIN_EP else "selten", begruendung=grund))
    t = pd.DataFrame(rows).sort_values("n_episoden", ascending=False)
    t.to_csv(OUT / "auslosercodes.csv", index=False)
    return t


def gruppe_map(t: pd.DataFrame) -> dict:
    return dict(zip(t["code"], t["gruppe"]))

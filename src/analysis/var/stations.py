"""Stationsliste und geografische Reihenfolge der Stammstrecke (West -> Ost).

Reihenfolge und stop_ids stammen aus BEFUNDE.md (Aufgabe 3, verifiziert über eine
echte Trip-Durchfahrt). Nur lesend uebernommen, nicht veraendert.
"""

STATION_ORDER_WEST_OST = [
    (8004158, "Pasing"),
    (8004151, "Laim"),
    (8004179, "Hirschgarten"),
    (8004128, "Donnersbergerbruecke"),
    (8004129, "Hackerbruecke"),
    (8098263, "Hauptbahnhof_tief"),
    (8004132, "Karlsplatz_Stachus"),
    (8004135, "Marienplatz"),
    (8004131, "Isartor"),
    (8004136, "Rosenheimer_Platz"),
    (8000262, "Ostbahnhof"),
    (8004134, "Leuchtenbergring"),
]

STOP_ID_TO_NAME = {sid: name for sid, name in STATION_ORDER_WEST_OST}
STATION_NAMES_WEST_OST = [name for _, name in STATION_ORDER_WEST_OST]
STATION_NAMES_OST_WEST = list(reversed(STATION_NAMES_WEST_OST))

# Distanz-Rang entlang der Strecke (0 = Pasing ... 11 = Leuchtenbergring), fuer
# Farbcodierung nach Entfernung in den Plots.
STATION_POSITION = {name: i for i, name in enumerate(STATION_NAMES_WEST_OST)}

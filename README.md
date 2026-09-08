S-Bahn Worldmodel
=================

Projektstruktur und Pfade:

- Rohdaten liegen unter `data/raw/`
- Verarbeitete 5-Minuten-Matrizen liegen unter `data/processed/`
- Ausgaben, Plots und Logs liegen unter `outputs/`
- Die Skripte `explore.py` und `analyse.py` lesen standardmäßig `data/raw/2025-01-01.parquet`

Ausführen:

- `uv run python explore.py`
- `uv run python analyse.py`

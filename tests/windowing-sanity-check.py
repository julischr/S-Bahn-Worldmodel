# %% [markdown]
# # Sanity Check: 5-Minuten-Zustandsmatrix & Count-Matrix

# %%
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("../data/processed")

# Matrizen laden
state_path = PROCESSED_DIR / "stammstrecke_5min.parquet"
count_path = PROCESSED_DIR / "stammstrecke_5min_counts.parquet"

if not state_path.exists():
    raise FileNotFoundError(f"Konnt {state_path} nicht finden. Erst windowing.py ausführen!")

state = pd.read_parquet(state_path)
count = pd.read_parquet(count_path)

print(f"✅ Matrizen erfolgreich geladen!")
print(f"Shape State-Matrix: {state.shape}")
print(f"Shape Count-Matrix: {count.shape}")
print(f"Zeitraum: {state.index.min()} bis {state.index.max()}")

# %% [markdown]
# ## 1. Spalten- und Stationsprüfung

# %%
print("Spalten (West -> Ost):")
for col in state.columns:
    print(f" - {col}")

# %% [markdown]
# ## 2. Imputations-Check am Hauptbahnhof
# Vergleicht echte Messpunkte (Count > 0) mit imputierten Werten (Count == 0)

# %%
station = "Hauptbahnhof (tief)"
if station in state.columns:
    check_df = pd.concat([
        state[station].rename('Delay_Min'), 
        count[station].rename('Zug_Count')
    ], axis=1)

    # Zeige Fenster mit Imputation (Count == 0, aber Verspätung vorhanden)
    imputed_mask = (check_df['Zug_Count'] == 0) & (check_df['Delay_Min'].notna())
    
    print(f"\nBeispiele für imputierte Fenster ({imputed_mask.sum()} gesamt):")
    print(check_df[imputed_mask].head(50).to_string())

    # Zeigt die echten Datenlücken (wo nach 30 Min keine Züge mehr fuhren)
    nan_mask = check_df['Delay_Min'].isna()

    print(f"Echte Nacht-Lücken (NaN): {nan_mask.sum()} Fenster")
    print(check_df[nan_mask].head(50).to_string())

    # Zeigt alle Abstände zwischen aufeinanderfolgenden Zeile
    step_sizes = state.index.to_series().diff().value_counts()

    print("Verteilung der Zeitschritte im Index:")
    print(step_sizes)
else:
    print(f"Station {station} nicht in den Spalten gefunden.")
"""Project-wide path constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

RAW_2025_01_01 = RAW_DATA_DIR / "2025-01-01.parquet"
RAW_2025_TAR = RAW_DATA_DIR / "2025.tar"
PROCESSED_STAMMSTRECKE_5MIN = PROCESSED_DATA_DIR / "stammstrecke_5min.parquet"
PROCESSED_MVV_FULL_5MIN = PROCESSED_DATA_DIR / "mvv_full_5min.parquet"

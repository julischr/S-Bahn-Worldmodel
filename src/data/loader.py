"""Helpers for locating and loading raw parquet files."""

from pathlib import Path

from src.utils.constants import RAW_DATA_DIR

DEFAULT_RAW_FILENAME = "2025-01-01.parquet"


def raw_data_path(filename: str = DEFAULT_RAW_FILENAME) -> Path:
    return RAW_DATA_DIR / filename


def raw_parquet_glob() -> str:
    return str(RAW_DATA_DIR / "*.parquet")

"""Helpers for 5-minute windowing and processed output paths."""

from pathlib import Path

from src.utils.constants import PROCESSED_DATA_DIR


def processed_data_path(filename: str) -> Path:
    return PROCESSED_DATA_DIR / filename


def processed_parquet_glob() -> str:
    return str(PROCESSED_DATA_DIR / "*.parquet")

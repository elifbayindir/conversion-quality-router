"""Loading helpers for the raw Online Shoppers dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "online_shoppers_intention.csv"


def load_raw_dataset(path: Path | str = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """Load the raw dataset CSV with pandas' default type inference.

    Boolean columns (Weekend, Revenue) and numeric columns are inferred
    correctly by pandas from this source file's literal TRUE/FALSE and
    numeric text without extra conversion.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Run 'python scripts/download_data.py' first."
        )
    return pd.read_csv(path)

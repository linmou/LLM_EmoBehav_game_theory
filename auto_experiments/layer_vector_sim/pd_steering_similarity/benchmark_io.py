"""
Helpers for loading existing PD benchmark raw results.
"""

import json
from pathlib import Path
from typing import Any


def load_raw_results(path: Path) -> Any:
    """
    Load raw_results.json content.
    """
    raw_path = Path(path)
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    return json.loads(raw_path.read_text())

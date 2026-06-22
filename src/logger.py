from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


class CSVLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []

    def add(self, row: dict) -> None:
        self.rows.append(row)

    def save(self) -> None:
        if not self.rows:
            return
        pd.DataFrame(self.rows).to_csv(self.path, index=False)

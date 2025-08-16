# src/utils/storage.py
from pathlib import Path
import json
from typing import Iterable, Any


def save_jsonl(path: Path, rows: Iterable[Any]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

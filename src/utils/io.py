from pathlib import Path
from typing import Union
import json

def save_json(path: Union[str, Path], obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def read_text(path: Union[str, Path]) -> str:
    return Path(path).read_text(encoding="utf-8")

def write_text(path: Union[str, Path], text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

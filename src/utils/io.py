from pathlib import Path
from typing import Union
import json

def save_json(path: Union[str, Path], obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        
def delete_file(path: Union[str, Path]) -> None:
    path = Path(path)
    if path.exists():
        path.unlink()

def read_text(path: Union[str, Path]) -> str:
    return Path(path).read_text(encoding="utf-8")

def write_text(path: Union[str, Path], text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def read_json(path: Union[str, Path]):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
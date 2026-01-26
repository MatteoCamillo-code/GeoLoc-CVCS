from __future__ import annotations
from pathlib import Path
from typing import Optional

def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Walk upwards until we find a marker of repo root.
    Markers: .git, pyproject.toml, requirements.txt
    """
    start = (start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if (p / ".git").exists() or (p / "pyproject.toml").exists() or (p / "requirements.txt").exists():
            return p
    # fallback
    return start

def abs_path(*parts: str | Path, start: Optional[Path] = None) -> Path:
    """
    Build an absolute path from project root + parts.
    """
    root = find_project_root(start=start)
    out = root
    for part in parts:
        out = out / part
    return out.resolve()


from __future__ import annotations
from pathlib import Path
from typing import Optional
import re

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

def get_next_version(history_dir, base_name):
    """
    Find the highest version number for a file and return the next version.
    
    Args:
        history_dir: Path to the history directory
        base_name: Base filename without extension (e.g., 'baseline_multihead_natural')
    
    Returns:
        Next version number (starting from 1 if no versioned files exist)
    """
    history_dir = Path(history_dir)
    pattern = rf"^{re.escape(base_name)}_v(\d+)_history\.json$"
    
    max_version = 0
    if history_dir.exists():
        for file in history_dir.iterdir():
            match = re.match(pattern, file.name)
            if match:
                version = int(match.group(1))
                max_version = max(max_version, version)
    
    return max_version + 1

def get_current_version(history_dir, base_name):
    """
    Find the highest version number for a file and return it.
    
    Args:
        history_dir: Path to the history directory
        base_name: Base filename without extension (e.g., 'baseline_multihead_natural')
    Returns:
        Current highest version number (0 if no versioned files exist)
    """
    history_dir = Path(history_dir)
    pattern = rf"^{re.escape(base_name)}_v(\d+)_history\.json$"
    
    max_version = 0
    if history_dir.exists():
        for file in history_dir.iterdir():
            match = re.match(pattern, file.name)
            if match:
                version = int(match.group(1))
                max_version = max(max_version, version)
    
    return max_version


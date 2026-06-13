#!/usr/bin/env python3
"""Project-wide path anchors.

All data and result paths are resolved relative to the project root (the
directory that contains ``pyproject.toml``), not relative to any single source
file. This keeps paths stable no matter where a script lives under ``src/``.
"""

from __future__ import annotations

from pathlib import Path


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    # Fallback: src/core/paths.py -> src/core -> src -> project root
    return start.resolve().parents[2]


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())

DATA_DIR = PROJECT_ROOT / "data"
OFFICIAL_DATA = DATA_DIR / "official_data"
RESULTS = PROJECT_ROOT / "results"
CONFIG_DIR = PROJECT_ROOT / "config"

"""Small IO helpers shared across the project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path.

    Args:
        path: Directory to create.

    Returns:
        The resolved :class:`Path`.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str | Path) -> None:
    """Serialize ``data`` to a JSON file.

    Args:
        data: JSON-serialisable object.
        path: Destination file path.
    """
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    """Load a JSON file.

    Args:
        path: Source file path.

    Returns:
        The deserialised object.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

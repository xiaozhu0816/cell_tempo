from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML or JSON configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix.lower() in {".yml", ".yaml"}:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported config extension: {path.suffix}")

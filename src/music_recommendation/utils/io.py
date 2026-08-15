from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_joblib(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    joblib.dump(payload, path)

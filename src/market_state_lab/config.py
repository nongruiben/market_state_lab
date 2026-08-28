from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "settings.yml"
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_meta"] = {
        "config_path": str(config_path.resolve()),
        "project_root": str(PROJECT_ROOT),
    }
    return config


def project_path(config: dict[str, Any], *parts: str) -> Path:
    root = Path(config.get("_meta", {}).get("project_root", PROJECT_ROOT))
    return root.joinpath(*parts)


def ensure_runtime_directories(config: dict[str, Any]) -> None:
    for relative in (
        ("data", "raw"),
        ("data", "processed"),
        ("reports",),
        ("models",),
    ):
        project_path(config, *relative).mkdir(parents=True, exist_ok=True)


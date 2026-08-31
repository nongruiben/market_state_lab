from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_SECRET_NAMES = {"DEEPSEEK_API_KEY", "FRED_API_KEY"}


def _load_local_secrets(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name not in _LOCAL_SECRET_NAMES or name in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if value:
            os.environ[name] = value


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    _load_local_secrets(PROJECT_ROOT / ".env")
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

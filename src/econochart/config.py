from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """Raised when a project configuration is missing or internally inconsistent."""


def load_config(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Config file does not exist: {config_path}")
    seen = set() if _seen is None else set(_seen)
    if config_path in seen:
        chain = " -> ".join(str(value) for value in [*seen, config_path])
        raise ConfigError(f"Circular config inheritance detected: {chain}")
    seen.add(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"Config root must be a mapping: {config_path}")
    parent_value = payload.pop("extends", None)
    if parent_value:
        parent_path = Path(parent_value).expanduser()
        if not parent_path.is_absolute():
            project_candidate = (ROOT / parent_path).resolve()
            parent_path = project_candidate if project_candidate.is_file() else (config_path.parent / parent_path).resolve()
        parent = load_config(parent_path, seen)
        parent.pop("_config_path", None)
        payload = deep_merge(parent, payload)
    payload["_config_path"] = str(config_path)
    return payload


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def resolve_model_path(config: dict[str, Any], cli_value: str | None = None) -> str:
    """Resolve model location without committing an AutoDL-specific absolute path."""
    value = cli_value or os.getenv("ECONOCHART_MODEL_PATH") or config.get("model", {}).get("name_or_path")
    if not value:
        raise ConfigError(
            "No base model configured. Set ECONOCHART_MODEL_PATH or model.name_or_path in the YAML file."
        )
    candidate = Path(value).expanduser()
    return str(candidate.resolve()) if candidate.exists() else str(value)


def resolve_adapter_path(config: dict[str, Any], cli_value: str | None = None) -> str | None:
    value = cli_value or os.getenv("ECONOCHART_ADAPTER_PATH") or config.get("model", {}).get("adapter_path")
    if not value:
        return None
    candidate = Path(value).expanduser()
    return str(candidate.resolve()) if candidate.exists() else str(value)


def require(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ConfigError(f"Missing required config key: {dotted_key}")
        current = current[part]
    return current

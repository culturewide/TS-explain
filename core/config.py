from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any, Dict


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML while remaining usable before PyYAML is installed.

    Project config files are written as JSON-compatible YAML. If PyYAML exists
    it will be used; otherwise json/ast parsing keeps bootstrap commands alive.
    """

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data or {}
    except Exception:
        pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def resolve_path(value: str | Path, base: str | Path | None = None) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    if path.is_absolute():
        return path
    return (Path(base) if base else project_root()).joinpath(path).resolve()


def load_project_config(path: str | Path | None = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else project_root() / "config" / "config.yaml"
    config = load_yaml(cfg_path)
    config["_config_path"] = str(cfg_path.resolve())
    config["_project_root"] = str(project_root())
    return config


def env_or_config(config: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cursor: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    if isinstance(cursor, str) and cursor.startswith("${") and cursor.endswith("}"):
        return os.getenv(cursor[2:-1], default)
    return cursor


def load_env_files(*paths: str | Path, override: bool = False) -> None:
    """Load simple KEY=VALUE pairs from .env files.

    This intentionally avoids an extra python-dotenv dependency. It is used so
    Codex/terminal child processes can pick up API keys and proxy variables from
    C:\\Users\\Jack\\.codex\\.env without restarting the whole app.
    """

    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.exists() or not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and (override or key not in os.environ):
                os.environ[key] = value


def load_standard_env_files(project: str | Path | None = None, override: bool = False) -> None:
    root = Path(project) if project else project_root()
    load_env_files(
        Path.home() / ".codex" / ".env",
        root / ".env",
        root.parent / ".env",
        override=override,
    )

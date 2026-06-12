"""Loader for strategy_config.json — single source of truth for strategy params."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "strategy_config.json"

_cached_config: StrategyConfigData | None = None


@dataclass(frozen=True)
class ParamMeta:
    default: float
    min: float
    max: float
    step: float
    integer: bool
    label: str
    description: str


@dataclass(frozen=True)
class StrategyConfigData:
    params: dict[str, ParamMeta]
    presets: dict[str, dict]


def _load_from_disk() -> StrategyConfigData:
    raw = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    params = {
        name: ParamMeta(
            default=p["default"],
            min=p["min"],
            max=p["max"],
            step=p["step"],
            integer=p.get("integer", False),
            label=p["label"],
            description=p["description"],
        )
        for name, p in raw["params"].items()
    }
    presets = dict(raw["presets"])
    return StrategyConfigData(params=params, presets=presets)


def get_strategy_config() -> StrategyConfigData:
    global _cached_config
    if _cached_config is None:
        _cached_config = _load_from_disk()
    return _cached_config


def reload_strategy_config() -> StrategyConfigData:
    global _cached_config
    _cached_config = _load_from_disk()
    return _cached_config


def get_default(name: str) -> float:
    return get_strategy_config().params[name].default


def get_range(name: str) -> tuple[float, float]:
    meta = get_strategy_config().params[name]
    return (meta.min, meta.max)


def clamp(name: str, value: float) -> float:
    meta = get_strategy_config().params[name]
    return max(meta.min, min(meta.max, value))

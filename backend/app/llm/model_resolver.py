"""Resolve a per-run LLM model from a frontend selection.

The frontend can either pick a preset (defined in ``.env`` via ``MODEL_N_*``)
or supply a fully custom model config. This module exposes:

- :func:`list_presets`  — safe metadata for the model picker (never returns keys)
- :func:`resolve_model_config` — turn a selection payload into a ``ModelConfig``

Security: preset api_keys live only in ``.env`` and are never returned by
:func:`list_presets`. Custom-model keys arrive in the request body and stay in
memory for the run; they must never be echoed back to the client.
"""

from __future__ import annotations

from typing import Any, Optional

from app.config import ModelConfig, settings


class ModelResolutionError(ValueError):
    """Raised when a model selection cannot be resolved to a usable config."""


def list_presets() -> list[dict[str, Any]]:
    """Return safe preset metadata for the frontend (no api_keys).

    ``configured`` is True when the preset has an api_key, i.e. it is usable.
    ``default`` marks the preset that an unspecified selection falls back to.
    """
    presets = settings.model_presets
    default = _default_preset(presets)
    default_id = default.id if default is not None else None
    return [
        {
            "id": p.id,
            "label": p.label,
            "model": p.model,
            "supports_image": p.supports_image,
            "configured": bool(p.api_key),
            "default": p.id == default_id,
        }
        for p in presets
    ]


def _default_preset(presets: list[ModelConfig]) -> Optional[ModelConfig]:
    """First configured preset, else the first preset, else None."""
    for p in presets:
        if p.api_key:
            return p
    return presets[0] if presets else None


def resolve_model_config(selection: Optional[dict[str, Any]]) -> ModelConfig:
    """Resolve a selection payload into a usable :class:`ModelConfig`.

    Selection shapes:
      - ``None`` / ``{}`` → default (first configured preset, else .env default).
      - ``{"preset_id": "..."}`` → look up the preset; error if unknown or
        unconfigured (empty api_key).
      - ``{"base_url", "model", "api_key", ...}`` → a fully custom model.

    Raises:
        ModelResolutionError: on unknown preset, unconfigured preset, or a
            custom model missing required fields.
    """
    if not selection:
        return _default_or_raise()

    preset_id = selection.get("preset_id")
    if preset_id:
        for p in settings.model_presets:
            if p.id == preset_id:
                if not p.api_key:
                    raise ModelResolutionError(
                        f"preset model '{preset_id}' is not configured (missing API key)"
                    )
                return p
        raise ModelResolutionError(f"unknown preset model id: {preset_id!r}")

    # Custom model — require base_url + model + api_key.
    base_url = (selection.get("base_url") or "").strip()
    model = (selection.get("model") or "").strip()
    api_key = (selection.get("api_key") or "").strip()
    missing = [
        name
        for name, value in (("base_url", base_url), ("model", model), ("api_key", api_key))
        if not value
    ]
    if missing:
        raise ModelResolutionError(
            "custom model is missing required field(s): " + ", ".join(missing)
        )

    label = (selection.get("label") or model).strip()
    return ModelConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        proxy=settings.proxy,
        supports_image=bool(selection.get("supports_image", False)),
        id=selection.get("id") or "custom",
        label=label,
    )


def _default_or_raise() -> ModelConfig:
    default = _default_preset(settings.model_presets)
    if default is not None and default.api_key:
        return default
    # Fall back to the legacy single-model .env config.
    return settings.llm

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

import ipaddress
import os
import socket
from typing import Any, Optional
from urllib.parse import urlsplit

from p2s_agent.config import ModelConfig, settings


class ModelResolutionError(ValueError):
    """Raised when a model selection cannot be resolved to a usable config."""


# Item 4 (SSRF) — schemes allowed for a custom-model base_url.
_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})


def _allowlisted_hosts() -> frozenset[str]:
    """Hosts explicitly allowed to bypass the SSRF block (env, comma-separated).

    Configured via ``LLM_BASE_URL_ALLOWLIST`` (e.g. for a self-hosted LLM on
    localhost). Matched case-insensitively against the URL host and against
    each resolved IP literal.
    """
    raw = os.getenv("LLM_BASE_URL_ALLOWLIST", "")
    return frozenset(h.strip().lower() for h in raw.split(",") if h.strip())


def _is_blocked_ip(ip: "ipaddress._BaseAddress") -> bool:
    """True for SSRF-relevant internal addresses.

    Blocks the cloud metadata IP (169.254.169.254, link-local), 127.0.0.0/8,
    ::1, the RFC1918 ranges (10/8, 172.16/12, 192.168/16) and 0.0.0.0.

    Note: ``ipaddress`` classifies RFC1918 + loopback + link-local + unspecified
    all under ``is_private``; we list the others explicitly for clarity. We do
    NOT block ``is_reserved``/``is_multicast`` here — those are not the SSRF
    targets we care about and over-blocking would reject legitimate public hosts
    in some environments.
    """
    return (
        ip.is_loopback
        or ip.is_link_local
        or ip.is_private
        or ip.is_unspecified
    )


def _validate_custom_base_url(base_url: str) -> None:
    """Reject a custom-model base_url that could enable SSRF.

    Requires an http/https scheme and a host that is not loopback / link-local
    (incl. the 169.254.169.254 metadata IP) / private / unspecified — unless the
    host (or a resolved IP) is on the ``LLM_BASE_URL_ALLOWLIST``.

    Raises:
        ModelResolutionError: when the URL is malformed or points at a blocked
            (internal / metadata) address.
    """
    parts = urlsplit(base_url)
    scheme = (parts.scheme or "").lower()
    if scheme not in _ALLOWED_URL_SCHEMES:
        raise ModelResolutionError(
            f"custom model base_url scheme must be http or https, got {scheme or 'none'!r}"
        )

    host = parts.hostname
    if not host:
        raise ModelResolutionError("custom model base_url has no host")

    allowlist = _allowlisted_hosts()
    host_lower = host.lower()
    if host_lower in allowlist:
        return

    # If the host is an IP literal, check it directly.
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        ip = None

    if ip is not None:
        if str(ip) in allowlist:
            return
        if _is_blocked_ip(ip):
            raise ModelResolutionError(
                f"custom model base_url host {host!r} is a blocked (internal/metadata) address"
            )
        return

    # Hostname — block obvious loopback names, then resolve and check every IP.
    if host_lower in {"localhost"} or host_lower.endswith(".localhost"):
        raise ModelResolutionError(
            f"custom model base_url host {host!r} resolves to a blocked (loopback) address"
        )
    try:
        infos = socket.getaddrinfo(host, parts.port or None, proto=socket.IPPROTO_TCP)
    except OSError:
        # Cannot resolve — do not let an unresolvable host through.
        raise ModelResolutionError(
            f"custom model base_url host {host!r} could not be resolved"
        )
    for info in infos:
        sockaddr = info[4]
        ip_str = sockaddr[0]
        if ip_str in allowlist:
            continue
        try:
            resolved = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if _is_blocked_ip(resolved):
            raise ModelResolutionError(
                f"custom model base_url host {host!r} resolves to a blocked "
                f"(internal/metadata) address {ip_str}"
            )


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

    # Item 4 (SSRF) — validate the client-supplied base_url before it ever
    # reaches the LLM client.
    _validate_custom_base_url(base_url)

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

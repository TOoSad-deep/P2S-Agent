"""SSRF guard tests for custom-model base_url resolution (Item 4).

A fully custom model's ``base_url`` arrives in the request body. Without
validation it can point at internal/metadata endpoints (cloud metadata IP,
loopback, RFC1918) → SSRF. ``resolve_model_config`` must reject such URLs.
"""

from __future__ import annotations

import socket

import pytest

from app.llm.model_resolver import ModelResolutionError, resolve_model_config


@pytest.fixture
def _public_dns(monkeypatch):
    """Make every hostname resolve to a fixed public IP (8.8.8.8).

    The offline test sandbox resolves all hostnames to the 198.18.0.0/15
    benchmark range (classified is_private), so we stub DNS to deterministically
    exercise the "public host is accepted" path of the SSRF validator.
    """
    def _fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.8.8", port or 0))]

    monkeypatch.setattr(
        "app.llm.model_resolver.socket.getaddrinfo", _fake_getaddrinfo
    )


def _custom(base_url: str) -> dict:
    return {
        "base_url": base_url,
        "model": "gpt-4o",
        "api_key": "sk-test",
        "supports_image": False,
    }


@pytest.mark.parametrize(
    "base_url",
    [
        "http://169.254.169.254/v1",          # cloud metadata IP (link-local)
        "http://169.254.170.2/v1",            # ECS task metadata (link-local)
        "http://127.0.0.1/v1",                # loopback
        "http://127.0.0.1:8001/v1",           # loopback w/ port
        "http://localhost/v1",                # loopback hostname
        "http://10.0.0.5/v1",                 # RFC1918 10/8
        "http://172.16.0.10/v1",              # RFC1918 172.16/12
        "http://192.168.1.1/v1",              # RFC1918 192.168/16
        "http://[::1]/v1",                    # IPv6 loopback
        "http://0.0.0.0/v1",                  # unspecified / wildcard
        "ftp://example.com/v1",               # disallowed scheme
        "file:///etc/passwd",                 # disallowed scheme
        "not-a-url",                          # no scheme/host
    ],
)
def test_custom_base_url_blocked(base_url):
    with pytest.raises(ModelResolutionError):
        resolve_model_config(_custom(base_url))


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.openai.com/v1",
        "http://api.example.com:8080/v1",
        "https://my-llm.internal-corp.example.com/v1",
    ],
)
def test_normal_base_url_accepted(_public_dns, base_url):
    cfg = resolve_model_config(_custom(base_url))
    assert cfg.base_url == base_url
    assert cfg.model == "gpt-4o"
    assert cfg.api_key == "sk-test"


def test_allowlist_env_permits_blocked_host(monkeypatch):
    """An explicit allowlist env entry permits an otherwise-blocked host."""
    monkeypatch.setenv("LLM_BASE_URL_ALLOWLIST", "127.0.0.1,169.254.169.254")
    cfg = resolve_model_config(_custom("http://127.0.0.1:1234/v1"))
    assert cfg.base_url == "http://127.0.0.1:1234/v1"


def test_preset_selection_unaffected():
    """A preset selection path does not go through base_url validation."""
    # No presets configured in test env → unknown preset raises, not SSRF error.
    with pytest.raises(ModelResolutionError):
        resolve_model_config({"preset_id": "does-not-exist"})

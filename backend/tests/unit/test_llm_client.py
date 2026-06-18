"""Tests for BaseAgent httpx client caching/reuse (no network).

Bug 1: each BaseAgent created an httpx.Client that was never closed, leaking
sockets/FDs across many VLM/generation calls. We cache one BaseAgent (and its
httpx.Client) per ModelConfig identity so repeated calls reuse one keep-alive
client; different configs get different clients; the cache is bounded.
"""
import app.llm.client as client_mod
from app.config import ModelConfig
from app.llm.client import BaseAgent, get_agent


def _cfg(model="gpt-4o", base_url="https://api.openai.com/v1", api_key="k", proxy=None):
    return ModelConfig(api_key=api_key, base_url=base_url, model=model, proxy=proxy)


def setup_function(_fn):
    client_mod.clear_agent_cache()


def test_same_config_reuses_same_agent_and_http_client():
    cfg = _cfg()
    a1 = get_agent(cfg)
    a2 = get_agent(_cfg())  # equal identity, distinct object
    assert a1 is a2
    assert a1._http_client is a2._http_client


def test_different_config_gets_different_client():
    a_openai = get_agent(_cfg(model="gpt-4o"))
    a_other = get_agent(_cfg(model="gpt-4o-mini"))
    assert a_openai is not a_other
    assert a_openai._http_client is not a_other._http_client

    a_url = get_agent(_cfg(base_url="https://other.example/v1"))
    assert a_url is not a_openai

    a_proxy = get_agent(_cfg(proxy="http://127.0.0.1:7890"))
    assert a_proxy is not a_openai


def test_cache_is_bounded():
    # Far exceed the cache cap; the cache must not grow without limit.
    for i in range(client_mod._AGENT_CACHE_MAX + 50):
        get_agent(_cfg(model=f"m{i}"))
    assert len(client_mod._AGENT_CACHE) <= client_mod._AGENT_CACHE_MAX


def test_close_path_releases_client():
    cfg = _cfg()
    agent = get_agent(cfg)
    http = agent._http_client
    agent.close()
    assert http.is_closed
    # after closing, the cache should no longer hand back the closed agent
    fresh = get_agent(cfg)
    assert fresh is not agent
    assert not fresh._http_client.is_closed


# --- Bug 3: browser_render screenshot temp-file cleanup helpers --------------
# The Playwright-driven render functions can't run in unit tests, but the
# temp-file lifecycle (the actual leak) is extracted into pure helpers TDD'd here.
import os as _os

from app.services.browser_render import (
    _cleanup_paths,
    _new_screenshot_path,
    _unlink_quietly,
)


def test_new_screenshot_path_creates_unique_png_files():
    p1 = _new_screenshot_path("vfx_test_")
    p2 = _new_screenshot_path("vfx_test_")
    try:
        assert p1 != p2
        assert p1.endswith(".png") and p2.endswith(".png")
        assert _os.path.exists(p1) and _os.path.exists(p2)
    finally:
        _unlink_quietly(p1)
        _unlink_quietly(p2)


def test_unlink_quietly_removes_and_is_idempotent():
    p = _new_screenshot_path("vfx_test_")
    assert _os.path.exists(p)
    _unlink_quietly(p)
    assert not _os.path.exists(p)
    # second call on a missing path must not raise
    _unlink_quietly(p)


def test_cleanup_paths_removes_all_screenshots():
    paths = [_new_screenshot_path("vfx_test_") for _ in range(3)]
    assert all(_os.path.exists(p) for p in paths)
    _cleanup_paths(paths)
    assert not any(_os.path.exists(p) for p in paths)

"""Global worker backpressure tests for the PNG-to-Shader router.

When the global top-level worker capacity is saturated, /run (and the other
top-level pipeline endpoints) must reject new submissions with HTTP 429 instead
of spawning unbounded daemon threads. When a slot frees, new runs are accepted
again.

These tests are deterministic: the heavy background worker is monkeypatched so
no real pipeline thread runs. We control occupancy by acquiring/releasing the
global worker semaphore directly.
"""

from __future__ import annotations

import threading

import pytest

fastapi = pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")

import app.routers.png_shader as ps
from app.routers.png_shader import _run_store, router

FastAPI = fastapi.FastAPI
TestClient = testclient.TestClient


@pytest.fixture(autouse=True)
def _isolate_run_index(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.routers.png_shader._RUN_INDEX_PATH",
        str(tmp_path / "run_index.jsonl"),
    )


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _png_bytes(tmp_path) -> bytes:
    from PIL import Image

    path = tmp_path / "input.png"
    Image.new("RGBA", (16, 16), (180, 80, 40, 255)).save(path)
    return path.read_bytes()


@pytest.fixture(autouse=True)
def _no_real_worker(monkeypatch):
    """Replace the background worker with a no-op so no pipeline thread runs.

    The global capacity slot is still acquired/released by _start_pipeline_worker,
    so backpressure semantics are exercised without the heavy pipeline. We DO NOT
    release the global slot here — the test controls occupancy explicitly via the
    semaphore so submissions stay "active" until the test frees them.
    """

    def _fake_worker(**kwargs):  # pragma: no cover - never actually invoked
        return None

    monkeypatch.setattr(ps, "_run_png_shader_background", _fake_worker)


def _small_cap_semaphore(monkeypatch, capacity: int):
    """Install a fresh bounded global semaphore with *capacity* slots."""
    sem = threading.BoundedSemaphore(capacity)
    monkeypatch.setattr(ps, "_global_worker_semaphore", sem)
    return sem


def test_run_returns_429_when_global_capacity_saturated(tmp_path, monkeypatch):
    _run_store.clear()
    sem = _small_cap_semaphore(monkeypatch, capacity=1)
    # Saturate: hold the single global slot so no top-level run can acquire one.
    assert sem.acquire(blocking=False) is True

    client = _client()
    response = client.post(
        "/png-shader/run",
        files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
    )

    assert response.status_code == 429
    body = response.json()["detail"]
    assert "capacity" in str(body).lower() or "retry" in str(body).lower()


def test_run_accepted_again_after_slot_frees(tmp_path, monkeypatch):
    _run_store.clear()
    sem = _small_cap_semaphore(monkeypatch, capacity=1)
    assert sem.acquire(blocking=False) is True

    client = _client()
    # First submission is rejected (saturated).
    blocked = client.post(
        "/png-shader/run",
        files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
    )
    assert blocked.status_code == 429

    # Free the held slot; the next submission must be accepted.
    sem.release()
    accepted = client.post(
        "/png-shader/run",
        files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
    )
    assert accepted.status_code == 200
    assert accepted.json()["status"] == "running"


def test_nplus1_run_rejected_until_a_worker_finishes(tmp_path, monkeypatch):
    """With N global slots, the first N /run submissions are accepted and the
    (N+1)th is rejected. After a worker releases a slot, a new run is accepted."""
    _run_store.clear()
    capacity = 2
    sem = _small_cap_semaphore(monkeypatch, capacity=capacity)
    client = _client()

    for _ in range(capacity):
        ok = client.post(
            "/png-shader/run",
            files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
        )
        assert ok.status_code == 200

    # (N+1)th submission: no slots left -> 429.
    over = client.post(
        "/png-shader/run",
        files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
    )
    assert over.status_code == 429

    # Simulate a worker finishing: it releases one global slot in its finally.
    sem.release()
    again = client.post(
        "/png-shader/run",
        files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
    )
    assert again.status_code == 200

"""Security tests for the /png-shader/run upload guard (Item 1).

Covers:
  - oversized Content-Length header → 413
  - non-image content-type → rejected (415)
  - corrupt / non-image bytes (valid content-type) → rejected (422)
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

fastapi = pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")

from app.routers import png_shader
from app.routers.png_shader import _MAX_UPLOAD_BYTES, _run_store, router

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


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (16, 16), (10, 20, 30, 255)).save(buf, format="PNG")
    return buf.getvalue()


def test_oversized_content_length_rejected_413(monkeypatch):
    """A request whose Content-Length exceeds the cap is rejected up front."""
    _run_store.clear()
    client = _client()

    # Send a Content-Length larger than the cap without buffering the whole body.
    too_big = _MAX_UPLOAD_BYTES + 1
    headers = {
        "content-type": "image/png",
        "content-length": str(too_big),
    }
    response = client.post(
        "/png-shader/run",
        content=b"x" * 16,  # actual tiny body; the header is what trips the guard
        headers=headers,
    )
    assert response.status_code == 413


def test_non_image_content_type_rejected(tmp_path):
    """An upload whose content-type is not an allowed image type is rejected."""
    _run_store.clear()
    client = _client()

    response = client.post(
        "/png-shader/run",
        files={"image": ("payload.txt", b"hello world not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_corrupt_image_bytes_rejected(tmp_path):
    """Bytes that claim image/png but are not a real image are rejected (422)."""
    _run_store.clear()
    client = _client()

    response = client.post(
        "/png-shader/run",
        files={"image": ("input.png", b"NOT-A-REAL-PNG-FILE", "image/png")},
    )
    assert response.status_code == 422


def test_valid_png_accepted(tmp_path):
    """A genuine PNG within limits still passes the guard (regression)."""
    _run_store.clear()
    client = _client()

    response = client.post(
        "/png-shader/run",
        files={"image": ("input.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["run_id"]

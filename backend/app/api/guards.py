"""Web-layer upload guards for the PNG-shader API.

These helpers are intentionally HTTP-coupled (they raise ``HTTPException``) and
belong in the ``app/`` web layer.  They are separated here so the router module
stays thin and so the guards can be unit-tested in isolation.
"""

from __future__ import annotations

import io
import os

from fastapi import HTTPException, Request, UploadFile


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    """Read a positive int from the environment, falling back to *default*."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Item 1 — cap the multipart upload body so an unauthenticated client cannot
# OOM the worker by streaming an enormous image. Default ~25 MB.
_MAX_UPLOAD_BYTES = _env_int("MAX_UPLOAD_BYTES", 25 * 1024 * 1024)

# Item 1 — content-type allowlist for the uploaded image.
_ALLOWED_IMAGE_CONTENT_TYPES: frozenset[str] = frozenset({
    "image/png", "image/jpeg", "image/jpg", "image/webp",
})


# ---------------------------------------------------------------------------
# Route guards
# ---------------------------------------------------------------------------

def _guard_upload(request: Request, image: UploadFile, contents: bytes) -> None:
    """Validate an uploaded image: content-type allowlist + real-image bytes.

    Content-Length is checked separately (up front) before the body is read.
    Here we verify the declared content-type is an allowed image type (415)
    and that the bytes actually decode as an image (422).
    """
    content_type = (image.content_type or "").split(";", 1)[0].strip().lower()
    if content_type not in _ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported image content-type: {image.content_type!r}",
        )
    # Enforce the size cap on the actual bytes too (Content-Length can lie or
    # be absent under chunked transfer-encoding).
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"upload exceeds maximum size of {_MAX_UPLOAD_BYTES} bytes",
        )
    # Verify the bytes are a real image (defends against content-type spoofing).
    try:
        from PIL import Image as _PILImage

        with _PILImage.open(io.BytesIO(contents)) as img:
            img.verify()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="uploaded file is not a valid image",
        ) from exc


def _check_content_length(request: Request) -> None:
    """Reject (413) up front when the declared Content-Length exceeds the cap.

    Used as a FastAPI route dependency so it runs BEFORE the multipart body is
    parsed/validated — i.e. before the worker buffers the body in memory.
    """
    raw = request.headers.get("content-length")
    if not raw:
        return
    try:
        declared = int(raw)
    except (TypeError, ValueError):
        return
    if declared > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"upload exceeds maximum size of {_MAX_UPLOAD_BYTES} bytes",
        )

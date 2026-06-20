"""Decomposition-based candidate: measured geometry via color quantization
+ connected components + primitive fitting (see p2s_agent.core.pipeline.decompose)."""

from __future__ import annotations

import logging
from pathlib import Path

from p2s_agent.core.pipeline.decompose import DECOMPOSE_AVAILABLE, decompose_to_dsl
from p2s_agent.core.pipeline.preprocess import feature_num

logger = logging.getLogger(__name__)

PHOTO_LIKE_SKIP_THRESHOLD = 0.7


def generate_decompose_candidate(
    preprocess: dict,
    image_path: "str | Path | None",
    canvas_width: int = 512,
    canvas_height: int = 512,
) -> dict | None:
    if not DECOMPOSE_AVAILABLE or image_path is None:
        return None
    # photo-like inputs decompose into fragments — leave them to other candidates
    if feature_num(preprocess, "photo_like_score", 0.0) > PHOTO_LIKE_SKIP_THRESHOLD:
        return None
    dsl = decompose_to_dsl(image_path, canvas_width, canvas_height)
    if dsl is None:
        return None
    dsl["_meta"] = {"source": "decompose", "priority": 1}
    return dsl

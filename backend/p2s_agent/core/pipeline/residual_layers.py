"""Residual-driven layer addition (geometrize-style greedy construction).

After base optimization, repeatedly: render the current DSL, locate the
region deviating most from the reference, fit a primitive to that region
of the *reference* (analytic initialization — no random search), and keep
the new layer only when the objective score improves.

Unlike the optimizer/revision stages this stage intentionally changes
layer_count: it is a construction stage, not a mutation stage, so the
protected_aspects contract does not apply here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from p2s_agent.core.pipeline.decompose import fit_primitive_layer

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:  # pragma: no cover - environment-dependent
    cv2 = None

RESIDUAL_SIZE = 128       # residual analysis resolution
MIN_REGION_FRAC = 0.004   # ignore hot regions smaller than this
ACCEPT_MIN_DELTA = 0.003  # required score gain to keep a new layer


@dataclass
class ResidualAddResult:
    final_dsl: dict
    initial_score: float
    final_score: float
    layers_added: int
    log: list[dict] = field(default_factory=list)


def _load_rgb01(path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def add_residual_layers(
    dsl: dict,
    reference_path: "str | Path",
    *,
    score_fn,
    render_fn,
    max_added: int = 4,
    max_layers_total: int = 12,
) -> ResidualAddResult:
    """Greedily add primitives where the render deviates most from the reference.

    Args:
        dsl: Starting DSL (not mutated).
        reference_path: Reference image path.
        score_fn: callable(dsl) -> float, higher is better.
        render_fn: callable(dsl) -> Path | None rendering the DSL to an image.
        max_added: Max number of layers to add.
        max_layers_total: Hard cap on total layer count (shader budget guard).
    """
    initial_score = score_fn(dsl)
    if cv2 is None:
        return ResidualAddResult(dsl, initial_score, initial_score, 0)

    ref = _load_rgb01(reference_path, RESIDUAL_SIZE)
    current = dsl
    current_score = initial_score
    log: list[dict] = []

    for step in range(max_added):
        if len(current.get("layers", [])) >= max_layers_total:
            break
        render_path = render_fn(current)
        if render_path is None:
            break
        rnd = _load_rgb01(render_path, RESIDUAL_SIZE)
        residual = np.abs(ref - rnd).mean(axis=-1)
        residual = cv2.blur(residual, (5, 5))

        threshold = max(float(residual.mean() + 2.0 * residual.std()), 0.10)
        hot = (residual > threshold).astype(np.uint8)
        n, comp_map, stats, _ = cv2.connectedComponentsWithStats(hot, connectivity=8)
        if n <= 1:
            break
        comp = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        area_frac = stats[comp, cv2.CC_STAT_AREA] / float(hot.size)
        if area_frac < MIN_REGION_FRAC:
            break
        region_mask = comp_map == comp

        mean_color = (ref[region_mask].mean(axis=0) * 255.0 + 0.5).astype(int)
        color_hex = "#{:02x}{:02x}{:02x}".format(*np.clip(mean_color, 0, 255))
        layer = fit_primitive_layer(region_mask, color_hex=color_hex)
        if layer is None:
            break
        layer["id"] = f"res_{step:02d}_{layer['type']}"

        candidate = {**current, "layers": [*current.get("layers", []), layer]}
        new_score = score_fn(candidate)
        accepted = new_score >= current_score + ACCEPT_MIN_DELTA
        log.append({
            "step": step + 1,
            "layer_id": layer["id"],
            "layer_type": layer["type"],
            "area_frac": round(float(area_frac), 4),
            "score_before": round(current_score, 4),
            "score_after": round(new_score, 4),
            "accepted": accepted,
        })
        logger.info(
            "residual layer step=%d layer=%s before=%.4f after=%.4f accepted=%s",
            step + 1, layer["id"], current_score, new_score, accepted,
        )
        if not accepted:
            break
        current = candidate
        current_score = new_score

    return ResidualAddResult(
        final_dsl=current,
        initial_score=initial_score,
        final_score=current_score,
        layers_added=sum(1 for e in log if e["accepted"]),
        log=log,
    )

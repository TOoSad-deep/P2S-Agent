"""VLM-as-judge: rubric scoring and pairwise comparison of renders vs reference.

Design contract:
- NEVER called from inner optimization loops — decision points only
  (near-tie candidate selection, refinement arbitration, final gate).
- Every failure path returns None so callers degrade to objective metrics.
- Pairwise comparisons ask twice with panel order swapped; disagreement = tie.
- Results are cached in-process by content hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Callable, Optional, Union

from PIL import Image, ImageDraw

from app.config import settings

logger = logging.getLogger(__name__)

JudgeClient = Callable[[str, str, "Optional[list[str]]"], "Union[str, dict, None]"]

_CACHE: dict[str, object] = {}

PANEL_HEIGHT = 384

RUBRIC_SYSTEM_PROMPT = """You are a strict visual QA judge for an image-to-shader system.
The image shows two labeled panels: REFERENCE (the target) and RENDER (the shader output).
First list concrete visual differences, then score the render against the reference.
Scoring anchors: 1.0 = visually identical, 0.8 = minor deviations, 0.5 = same concept but clearly off, 0.2 = barely related.
Respond ONLY with JSON:
{"differences": ["..."],
 "shape_fidelity": 0.0-1.0,
 "position_layout": 0.0-1.0,
 "color_fidelity": 0.0-1.0,
 "effects_fidelity": 0.0-1.0,
 "failure_type": "none" | "structure" | "parameter" | "color" | "layer_order",
 "revision_hints": ["actionable scene-level change", "..."]}"""

PAIRWISE_SYSTEM_PROMPT = """You compare two shader renders against a reference image.
The image shows three labeled panels: REFERENCE, A, B.
Decide which of A or B is visually closer to REFERENCE overall (shape, position, color, effects).
Respond ONLY with JSON: {"winner": "A" | "B" | "tie", "reason": "one sentence"}"""


def _file_digest(*paths) -> str:
    h = hashlib.sha1()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def _compose_panel(labeled_paths: "list[tuple[str, Path]]", out_path: Path) -> Path:
    """Concatenate labeled image panels horizontally into a single image."""
    panels = []
    for label, p in labeled_paths:
        img = Image.open(p).convert("RGB")
        scale = PANEL_HEIGHT / max(1, img.height)
        img = img.resize((max(1, int(img.width * scale)), PANEL_HEIGHT), Image.LANCZOS)
        labeled = Image.new("RGB", (img.width, PANEL_HEIGHT + 28), (24, 24, 24))
        labeled.paste(img, (0, 28))
        ImageDraw.Draw(labeled).text((8, 6), label, fill=(255, 255, 255))
        panels.append(labeled)
    total_w = sum(p.width for p in panels) + 12 * (len(panels) - 1)
    canvas = Image.new("RGB", (total_w, PANEL_HEIGHT + 28), (24, 24, 24))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + 12
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def _default_client(system_prompt: str, user_prompt: str, image_paths: "Optional[list[str]]"):
    if not settings.llm.api_key:
        return None
    from app.llm.client import BaseAgent

    agent = BaseAgent(settings.llm)
    paths = image_paths if image_paths and settings.llm.supports_image else None
    return agent.chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_paths=paths,
        temperature=0.0,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )


def _parse_json(response) -> "Optional[dict]":
    if response is None:
        return None
    if isinstance(response, dict):
        return response
    text = str(response).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def judge_rubric(
    reference_path,
    render_path,
    *,
    work_dir,
    judge_client: "Optional[JudgeClient]" = None,
) -> "Optional[dict]":
    """Score a render against the reference on semantic dimensions.

    Returns {"semantic_scores": {...4 dims...}, "failure_type": str,
    "revision_hints": [...], "differences": [...]} or None on any failure.
    """
    try:
        cache_key = "rubric:" + _file_digest(reference_path, render_path)
        if cache_key in _CACHE:
            return _CACHE[cache_key]  # type: ignore[return-value]
        panel = _compose_panel(
            [("REFERENCE", Path(reference_path)), ("RENDER", Path(render_path))],
            Path(work_dir) / "rubric_panel.png",
        )
        client = judge_client or _default_client
        raw = client(RUBRIC_SYSTEM_PROMPT, "Evaluate the RENDER against the REFERENCE.", [str(panel)])
        data = _parse_json(raw)
        if data is None:
            return None
        semantic: dict[str, float] = {}
        for dim in ("shape_fidelity", "position_layout", "color_fidelity", "effects_fidelity"):
            try:
                semantic[dim] = max(0.0, min(1.0, float(data[dim])))
            except (KeyError, TypeError, ValueError):
                return None  # malformed — degrade entirely rather than half-trust
        result = {
            "semantic_scores": semantic,
            "failure_type": str(data.get("failure_type", "none")),
            "revision_hints": [str(x) for x in data.get("revision_hints", [])][:5],
            "differences": [str(x) for x in data.get("differences", [])][:8],
        }
        _CACHE[cache_key] = result
        return result
    except Exception:
        logger.warning("VLM rubric judge failed", exc_info=True)
        return None


def judge_pairwise(
    reference_path,
    a_path,
    b_path,
    *,
    work_dir,
    judge_client: "Optional[JudgeClient]" = None,
) -> "Optional[str]":
    """Return "A", "B", or "tie" (order-debiased), or None on any failure."""
    try:
        cache_key = "pair:" + _file_digest(reference_path, a_path, b_path)
        if cache_key in _CACHE:
            return _CACHE[cache_key]  # type: ignore[return-value]
        client = judge_client or _default_client
        verdicts: list[str] = []
        for tag, first, second in (("fwd", a_path, b_path), ("rev", b_path, a_path)):
            panel = _compose_panel(
                [("REFERENCE", Path(reference_path)), ("A", Path(first)), ("B", Path(second))],
                Path(work_dir) / f"pair_panel_{tag}.png",
            )
            data = _parse_json(client(PAIRWISE_SYSTEM_PROMPT, "Which render is closer to REFERENCE?", [str(panel)]))
            if data is None:
                return None
            verdicts.append(str(data.get("winner", "tie")).strip().upper())
        fwd, rev = verdicts
        rev_mapped = {"A": "B", "B": "A"}.get(rev, "tie")  # rev call had panels swapped
        result = fwd if fwd == rev_mapped and fwd in ("A", "B") else "tie"
        _CACHE[cache_key] = result
        return result
    except Exception:
        logger.warning("VLM pairwise judge failed", exc_info=True)
        return None


def judge_directed_pairwise(
    reference_path,
    current_render_path,
    candidate_render_path,
    *,
    user_feedback: str,
    work_dir,
    judge_client: "Optional[JudgeClient]" = None,
) -> "Optional[str]":
    """Goal-aware pairwise judge for human-in-loop directed acceptance.

    A is the current best; B is a new candidate. Returns "B" only when B better
    satisfies ``user_feedback`` *without* clearly breaking fidelity to the
    reference; otherwise "A"/"tie". Order-debiased; None on any failure so the
    caller degrades to metric-only acceptance.
    """
    try:
        goal = (user_feedback or "").strip()
        cache_key = (
            "directed:"
            + hashlib.sha1(goal.encode("utf-8")).hexdigest()[:10]
            + ":"
            + _file_digest(reference_path, current_render_path, candidate_render_path)
        )
        if cache_key in _CACHE:
            return _CACHE[cache_key]  # type: ignore[return-value]
        system_prompt = (
            "You compare two shader renders (A and B) against a reference image, "
            "given a user's goal.\n"
            "The image shows three labeled panels: REFERENCE, A, B.\n"
            "A is the current best result; B is a new candidate.\n"
            f"User goal: {goal!r}.\n"
            "Choose B only if B better satisfies the user goal than A WITHOUT "
            "clearly breaking overall fidelity to the REFERENCE.\n"
            "If B overfits the goal while damaging the main visual, prefer A or tie.\n"
            'Respond ONLY with JSON: {"winner": "A" | "B" | "tie", "reason": "one sentence"}'
        )
        client = judge_client or _default_client
        verdicts: list[str] = []
        for tag, first, second in (
            ("fwd", current_render_path, candidate_render_path),
            ("rev", candidate_render_path, current_render_path),
        ):
            panel = _compose_panel(
                [("REFERENCE", Path(reference_path)), ("A", Path(first)), ("B", Path(second))],
                Path(work_dir) / f"directed_panel_{tag}.png",
            )
            data = _parse_json(
                client(system_prompt, "Which render better satisfies the user goal without breaking fidelity?", [str(panel)])
            )
            if data is None:
                return None
            verdicts.append(str(data.get("winner", "tie")).strip().upper())
        fwd, rev = verdicts
        rev_mapped = {"A": "B", "B": "A"}.get(rev, "tie")  # rev call had panels swapped
        result = fwd if fwd == rev_mapped and fwd in ("A", "B") else "tie"
        _CACHE[cache_key] = result
        return result
    except Exception:
        logger.warning("VLM directed pairwise judge failed", exc_info=True)
        return None

"""Serverless CLI entrypoint for the PNG-to-Shader pipeline.

Runs one PNG -> GLSL conversion without FastAPI or any web-layer dependency.
The pipeline defaults to deterministic / AI-off mode (llm_enabled=False,
vlm_judge_enabled=0) so no LLM or VLM keys are required.

Usage::

    python3 -m p2s_agent.cli --image path/to/image.png [--seed-glsl path/to/seed.glsl] [--out ./cli_out]

Imports ONLY from p2s_agent.* — never from app.* or fastapi.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from p2s_agent.core.pipeline.graph import run_png_shader_pipeline
from p2s_agent.core.pipeline.input_spec import build_input_spec

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the pipeline, write outputs, and return an exit code.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]`` when *None*).

    Returns
    -------
    int
        0 on success, non-zero on failure.
    """
    parser = argparse.ArgumentParser(
        prog="p2s_agent.cli",
        description="Convert a PNG image to a GLSL shader (no web server required).",
    )
    parser.add_argument(
        "--image",
        required=True,
        metavar="PATH",
        help="Path to the source PNG image.",
    )
    parser.add_argument(
        "--seed-glsl",
        default=None,
        metavar="PATH",
        help="Optional path to a seed GLSL file. When given, the pipeline refines "
             "this shader rather than generating new candidates.",
    )
    parser.add_argument(
        "--out",
        default="cli_out",
        metavar="DIR",
        help="Output directory for the selected GLSL and metrics JSON "
             "(default: ./cli_out).",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING).",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s %(message)s",
    )

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"error: image not found: {image_path}", file=sys.stderr)
        return 1

    seed_glsl: str | None = None
    if args.seed_glsl is not None:
        seed_path = Path(args.seed_glsl)
        if not seed_path.exists():
            print(f"error: seed GLSL not found: {seed_path}", file=sys.stderr)
            return 1
        seed_glsl = seed_path.read_text(encoding="utf-8")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a deterministic (AI-off) input spec — no LLM or VLM keys required.
    input_spec = build_input_spec(
        image_path,
        candidates={
            "llm_enabled": False,
            "cv_enabled": True,
            "glsl_render_enabled": False,
        },
        quality={
            "vlm_judge_enabled": 0,
            "refinement_mode": "off",
            "max_refinement_iterations": 0,
        },
    )

    try:
        result = run_png_shader_pipeline(
            image_path,
            input_spec,
            seed_glsl=seed_glsl,
        )
    except Exception as exc:
        print(f"error: pipeline failed: {exc}", file=sys.stderr)
        logger.exception("pipeline failed")
        return 1

    selected_glsl: str | None = result.get("selected_glsl")
    quality_router: dict = result.get("quality_router") or {}
    final_score: float | None = quality_router.get("final_score")
    run_id: str = result.get("run_id", "unknown")

    glsl_path = out_dir / "selected_shader.glsl"
    if selected_glsl:
        glsl_path.write_text(selected_glsl, encoding="utf-8")
    else:
        glsl_path.write_text("", encoding="utf-8")

    metrics_path = out_dir / "metrics.json"
    metrics_payload = {
        "run_id": run_id,
        "final_score": final_score,
        "objective_metrics": result.get("objective_metrics", {}),
        "quality_router": quality_router,
        "selected_candidate_id": result.get("selected_candidate_id"),
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, default=str),
        encoding="utf-8",
    )

    score_str = f"{final_score:.4f}" if final_score is not None else "n/a"
    print(f"run_id:      {run_id}")
    print(f"final_score: {score_str}")
    print(f"glsl:        {glsl_path}")
    print(f"metrics:     {metrics_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

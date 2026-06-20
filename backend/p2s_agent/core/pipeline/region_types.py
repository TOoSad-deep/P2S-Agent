"""Pure-core region dataclasses shared by compute and orchestration modules.

This is a dependency-free leaf: only stdlib/typing/dataclasses are imported.
It exists so ``core`` compute modules (region_metrics, image_composite) can
reference these types without importing orchestration modules
(human_constraints, fusion_plans) that pull in ``p2s_agent.core.pipeline.artifacts``.

Both orchestration modules re-export these classes for back-compat.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegionConstraint:
    """A spatial region that the user wants to modify or protect."""

    id: str
    label: str
    mode: str           # "modify" | "protect"
    instruction: str
    geometry_type: str  # "rect" (V4.2 adds "polygon"|"mask")
    geometry: dict      # rect: {"x","y","w","h"} normalised 0..1
    strength: float = 0.5


@dataclass
class FusionRegion:
    id: str
    label: str
    source_run_id: str
    instruction: str
    geometry_type: str          # "rect" first
    geometry: dict              # rect {"x","y","w","h"} normalized 0..1
    strength: float = 0.5
    blend_mode: str = "soft"    # "soft" | "replace_target" | "protect_base"
    feather: float = 0.08

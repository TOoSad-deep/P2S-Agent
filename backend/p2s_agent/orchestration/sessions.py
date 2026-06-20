"""Session-level orchestration helpers for the PNG-to-Shader pipeline.

Framework-free home for the run/session lifecycle glue that sits *between* the
in-memory store and the persistent orchestration records. Extracted from
``app.routers.png_shader`` so the web layer holds no domain orchestration of its
own.

Dependency direction: this module imports ``p2s_agent.store`` (attribute access)
and ``p2s_agent.orchestration.fusion_plans``; it never imports the worker layer
or any ``app.*`` / ``fastapi`` symbol (enforced by test_agent_web_boundary).

Cycle note: the worker layer (``p2s_agent.workers``) calls
``_finalize_fusion_for_run`` via a function-body (lazy) import at its call site,
so a future module-level ``import p2s_agent.workers`` from this module would not
close an import cycle at load time.
"""

from __future__ import annotations

import logging
from time import time
from typing import Optional

from p2s_agent import store
from p2s_agent.orchestration.fusion_plans import update_plan_status

logger = logging.getLogger(__name__)

# Single source of truth for the on-disk fusions root. The web layer references
# this by ATTRIBUTE (``sessions._FUSIONS_ROOT``) so a test that overrides it is
# seen by both the fusion endpoints and the worker's terminal finalize.
_FUSIONS_ROOT: Optional[str] = None  # tests override to isolate (V4.5 fusion)


def _finalize_fusion_for_run(run_id: str, status: str) -> None:
    """Best-effort: close out the fusion plan a finished run belongs to (Bug 5).

    When a worker reaches a terminal state, if the run's lineage carries a
    ``fusion_id`` we mark the corresponding FusionPlanRecord ``completed`` /
    ``failed`` so ``GET /fusions/{id}`` stops reporting ``running`` forever and
    the frontend poll terminates. Fully wrapped so a failure here can never
    break the worker thread."""
    try:
        with store._run_store_lock:
            stored = store._run_store.get(run_id) or {}
            lineage = stored.get("lineage") or {}
            fusion_id = stored.get("fusion_id") or lineage.get("fusion_id")
        if not fusion_id:
            return
        update_plan_status(
            str(fusion_id), status, updated_at=time(), root=_FUSIONS_ROOT
        )
    except Exception:
        logger.warning("fusion plan finalize failed for run_id=%s", run_id, exc_info=True)

"""Back-compat shim — moved to p2s_agent.orchestration.preferences (L1 agent/web split, T7). Retire in T10."""
from __future__ import annotations

from p2s_agent.orchestration.preferences import *  # noqa: F401,F403

# Re-export every remaining module-level name (incl. private/underscore symbols
# that ``import *`` skips) so old ``from app.pipeline.preferences import _x`` paths still resolve.
import p2s_agent.orchestration.preferences as _src  # noqa: E402

for _name in dir(_src):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_src, _name))
del _src, _name

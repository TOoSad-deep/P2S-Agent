"""Back-compat shim — moved to p2s_agent.core.metrics.compute (L1 agent/web split, T6). Retire in T10."""
from __future__ import annotations

from p2s_agent.core.metrics.compute import *  # noqa: F401,F403

# Re-export every remaining module-level name (incl. private/underscore symbols
# that ``import *`` skips) so old ``from app.metrics.compute import _x`` paths still resolve.
import p2s_agent.core.metrics.compute as _src  # noqa: E402

for _name in dir(_src):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_src, _name))
del _src, _name

"""Agent-domain error hierarchy.

These exceptions are raised by the agent package (p2s_agent/**) and translated
to HTTP responses at the web boundary (app/main.py exception handlers).  The
agent package itself never imports fastapi or starlette.
"""
from __future__ import annotations

from typing import Optional


class AgentError(Exception):
    """Base for agent-domain errors translated to HTTP at the web boundary."""

    def __init__(self, message: str, *, field: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.field = field


class AgentInputError(AgentError):  # -> HTTP 422
    pass


class AgentConflictError(AgentError):  # -> HTTP 409
    pass


class AgentNotFoundError(AgentError):  # -> HTTP 404
    pass

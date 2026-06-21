"""P2S-Agent: PNG-to-Shader Pipeline Agent

FastAPI application entry point for the standalone PNG-to-Shader pipeline.
"""
from __future__ import annotations

import logging
import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from p2s_agent.config import settings as agent_settings
from p2s_agent.core.errors import AgentConflictError, AgentInputError, AgentNotFoundError

from app.routers import models, png_shader, strategy_config
from p2s_agent.core.tracing import configure_langsmith
from p2s_agent.core.logging_config import log_event, logging_context, setup_logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="P2S-Agent",
    description="PNG-to-Shader Pipeline Agent",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[agent_settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Central translation of agent-domain errors to HTTP responses. The agent
# package (p2s_agent/**) raises these instead of HTTPException; this is the ONE
# place they become HTTP responses, keeping the agent layer framework-free.
def _agent_error_handler(status: int):
    async def handler(_: Request, exc: AgentInputError) -> JSONResponse:
        # Body is EXACTLY {"detail": <message>} to stay byte-identical with the
        # pre-refactor HTTPException(status_code=..., detail=<message>) responses
        # these agent errors replaced. ``exc.field`` is retained on the exception
        # for internal use but is deliberately NOT serialized into the body.
        return JSONResponse(status_code=status, content={"detail": exc.message})

    return handler


def register_agent_error_handlers(target_app: FastAPI) -> None:
    """Register the agent-domain → HTTP exception handlers on *target_app*.

    Single source of truth for the AgentInputError→422 / AgentConflictError→409 /
    AgentNotFoundError→404 mapping, so tests building a bare app (and the real
    app) translate these identically.
    """
    target_app.add_exception_handler(AgentInputError, _agent_error_handler(422))
    target_app.add_exception_handler(AgentConflictError, _agent_error_handler(409))
    target_app.add_exception_handler(AgentNotFoundError, _agent_error_handler(404))


register_agent_error_handlers(app)

app.include_router(png_shader.router)
app.include_router(strategy_config.router)
app.include_router(models.router)


def _run_id_from_path(path: str) -> str | None:
    parts = [part for part in path.split("/") if part]
    if len(parts) >= 3 and parts[0] == "png-shader":
        if parts[1] in {"status", "refine"}:
            return parts[2]
        if parts[1] == "runs":
            return parts[2]
    return None


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or f"req_{uuid4().hex[:10]}"
    run_id = request.headers.get("x-run-id") or _run_id_from_path(request.url.path)
    started = time.perf_counter()
    with logging_context(request_id=request_id, run_id=run_id):
        log_event(
            logger,
            "http_request_start",
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else None,
        )
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = int((time.perf_counter() - started) * 1000)
            logger.exception(
                "http request failed: method=%s path=%s duration_ms=%d",
                request.method,
                request.url.path,
                duration_ms,
            )
            raise
        duration_ms = int((time.perf_counter() - started) * 1000)
        response.headers["x-request-id"] = request_id
        log_event(
            logger,
            "http_request_end",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return response


@app.on_event("startup")
async def startup_event():
    setup_logging()
    configure_langsmith()
    # One retention pass per boot (P2S_RETENTION_* env). Never raises — a cleanup
    # failure must not block the server coming up.
    from p2s_agent.orchestration.retention import cleanup_at_startup

    cleanup_at_startup()


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "p2s-agent"}

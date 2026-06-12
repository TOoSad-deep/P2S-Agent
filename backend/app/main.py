"""P2S-Agent: PNG-to-Shader Pipeline Agent

FastAPI application entry point for the standalone PNG-to-Shader pipeline.
"""
from __future__ import annotations


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import png_shader, strategy_config
from app.services.langsmith_tracing import configure_langsmith
from app.services.logging_config import setup_logging

app = FastAPI(
    title="P2S-Agent",
    description="PNG-to-Shader Pipeline Agent",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(png_shader.router)
app.include_router(strategy_config.router)


@app.on_event("startup")
async def startup_event():
    setup_logging()
    configure_langsmith()


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "p2s-agent"}

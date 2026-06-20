"""P2S-Agent web service configuration (host/port only). Agent config lives in p2s_agent.config."""
from __future__ import annotations
from pydantic_settings import BaseSettings

class WebSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8001
    # ``extra="ignore"`` tolerates agent-only .env keys (LLM_*, MODEL_*,
    # LANGSMITH_*, …) that now belong to p2s_agent.config.Settings; without it,
    # WebSettings() would reject every non-host/port key at import (extra_forbidden).
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

settings = WebSettings()

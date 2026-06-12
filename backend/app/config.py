"""P2S-Agent Configuration

Simplified configuration for the PNG-to-Shader standalone agent.
Only includes LLM and service configuration needed for P2S pipeline.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class ModelConfig:
    """LLM model configuration"""
    def __init__(self, api_key: str, base_url: str, model: str, proxy: str | None = None):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.proxy = proxy


class Settings(BaseSettings):
    # LLM Configuration (for candidate generation and refinement)
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o"
    llm_supports_image: bool = False

    # Proxy configuration (global)
    proxy: str = ""

    # Service configuration
    host: str = "0.0.0.0"
    port: int = 8001
    frontend_url: str = "http://localhost:5174"

    # Screenshot configuration
    screenshot_width: int = 512
    screenshot_height: int = 512
    render_timeout_ms: int = 2000

    # LangSmith tracing (disabled by default)
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "p2s-agent"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_tags: str = "p2s-agent"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def llm(self) -> ModelConfig:
        return ModelConfig(
            self.llm_api_key,
            self.llm_base_url,
            self._resolve_llm_model(),
            self.proxy,
        )

    def _resolve_llm_model(self) -> str:
        """Resolve the effective LLM model name."""
        if self.llm_supports_image and self.llm_model == "mimo-v2.5-pro":
            return "mimo-v2.5"
        return self.llm_model


settings = Settings()

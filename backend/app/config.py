"""P2S-Agent Configuration

Simplified configuration for the PNG-to-Shader standalone agent.
Only includes LLM and service configuration needed for P2S pipeline.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

from pydantic_settings import BaseSettings


class ModelConfig:
    """LLM model configuration"""
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        proxy: str | None = None,
        supports_image: bool = False,
        id: str | None = None,
        label: str | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.proxy = proxy
        self.supports_image = supports_image
        self.id = id
        self.label = label


# Active-model override for the current run.
#
# The pipeline reads the global ``settings.llm`` from many places, and each run
# executes in its own background thread (see ``png_shader.py``). A ContextVar is
# default-empty in a freshly spawned thread and set-isolated per thread, so a
# value set inside one run's thread never leaks into another's. This lets a run
# select its model without threading a ModelConfig through every call site.
_active_model: ContextVar[Optional[ModelConfig]] = ContextVar("active_model", default=None)


@contextmanager
def use_active_model(model_config: Optional[ModelConfig]) -> Iterator[None]:
    """Override ``settings.llm`` with ``model_config`` for the duration of the block.

    Passing ``None`` is a no-op (falls back to the .env default model).
    """
    if model_config is None:
        yield
        return
    token = _active_model.set(model_config)
    try:
        yield
    finally:
        _active_model.reset(token)


class Settings(BaseSettings):
    # LLM Configuration (legacy single-model default / fallback)
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o"
    llm_supports_image: bool = False

    # Preset models (frontend-selectable). Reserved slots loaded from .env.
    # An empty api_key marks a preset as "not configured" (placeholder).
    model_1_id: str = ""
    model_1_label: str = ""
    model_1_api_key: str = ""
    model_1_base_url: str = ""
    model_1_model: str = ""
    model_1_supports_image: bool = False

    model_2_id: str = ""
    model_2_label: str = ""
    model_2_api_key: str = ""
    model_2_base_url: str = ""
    model_2_model: str = ""
    model_2_supports_image: bool = False

    model_3_id: str = ""
    model_3_label: str = ""
    model_3_api_key: str = ""
    model_3_base_url: str = ""
    model_3_model: str = ""
    model_3_supports_image: bool = False

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

    # ``protected_namespaces=()`` disables pydantic's "model_" guard so our
    # MODEL_N_* preset fields don't collide with it.
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "protected_namespaces": (),
    }

    @property
    def llm(self) -> ModelConfig:
        """Effective LLM config: the active-run override if set, else the default."""
        override = _active_model.get()
        if override is not None:
            return override
        return self._default_llm()

    def _default_llm(self) -> ModelConfig:
        return ModelConfig(
            self.llm_api_key,
            self.llm_base_url,
            self._resolve_llm_model(),
            self.proxy,
            supports_image=self.llm_supports_image,
        )

    def _resolve_llm_model(self) -> str:
        """Resolve the effective LLM model name."""
        if self.llm_supports_image and self.llm_model == "mimo-v2.5-pro":
            return "mimo-v2.5"
        return self.llm_model

    @property
    def model_presets(self) -> list[ModelConfig]:
        """Assemble the configured preset models from numbered .env slots.

        A slot is included when it has a non-empty model id. An empty api_key is
        kept (placeholder / "not configured"); callers check ``api_key`` to know
        whether the preset is usable.
        """
        presets: list[ModelConfig] = []
        for n in (1, 2, 3):
            model = getattr(self, f"model_{n}_model", "")
            if not model:
                continue
            preset_id = getattr(self, f"model_{n}_id", "") or model
            presets.append(
                ModelConfig(
                    api_key=getattr(self, f"model_{n}_api_key", ""),
                    base_url=getattr(self, f"model_{n}_base_url", ""),
                    model=model,
                    proxy=self.proxy,
                    supports_image=getattr(self, f"model_{n}_supports_image", False),
                    id=preset_id,
                    label=getattr(self, f"model_{n}_label", "") or preset_id,
                )
            )
        return presets


settings = Settings()

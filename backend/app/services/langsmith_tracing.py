"""LangSmith observability helpers.

The project uses both LangGraph and raw OpenAI-compatible SDK calls. LangGraph
can auto-trace from environment variables, while raw SDK calls need wrapping.
This module centralizes those concerns so tracing remains optional.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from p2s_agent.config import settings


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _split_tags(value: str | None) -> list[str]:
    if not value:
        return []
    return [tag.strip() for tag in value.split(",") if tag.strip()]


def is_langsmith_enabled() -> bool:
    """Return whether tracing should be enabled for the current process."""
    raw = (
        os.environ.get("LANGSMITH_TRACING")
        or os.environ.get("LANGCHAIN_TRACING_V2")
        or settings.langsmith_tracing
    )
    return _truthy(raw)


def langsmith_project() -> str:
    return os.environ.get("LANGSMITH_PROJECT") or settings.langsmith_project or "vfx-agent"


def langsmith_tags(*extra_tags: str) -> list[str]:
    tags = _split_tags(os.environ.get("LANGSMITH_TAGS") or settings.langsmith_tags)
    for tag in extra_tags:
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def configure_langsmith() -> dict:
    """Synchronize project settings into environment variables used by SDKs.

    Pydantic reads ``backend/.env`` into ``settings`` but does not automatically
    export those values to ``os.environ``. LangSmith and LangChain integrations
    read the environment directly, so we bridge that here.
    """
    enabled = is_langsmith_enabled()
    if not enabled:
        return {
            "enabled": False,
            "project": langsmith_project(),
            "endpoint": os.environ.get("LANGSMITH_ENDPOINT") or settings.langsmith_endpoint,
        }

    os.environ["LANGSMITH_TRACING"] = "true"
    # Older langchain/langgraph versions still key off LANGCHAIN_*.
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

    if settings.langsmith_api_key and not os.environ.get("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    if settings.langsmith_project and not os.environ.get("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_project and not os.environ.get("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    if settings.langsmith_endpoint and not os.environ.get("LANGSMITH_ENDPOINT"):
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    if settings.langsmith_tags and not os.environ.get("LANGSMITH_TAGS"):
        os.environ["LANGSMITH_TAGS"] = settings.langsmith_tags

    return {
        "enabled": True,
        "project": langsmith_project(),
        "endpoint": os.environ.get("LANGSMITH_ENDPOINT") or settings.langsmith_endpoint,
        "api_key_configured": bool(os.environ.get("LANGSMITH_API_KEY")),
    }


def wrap_openai_client(client: Any) -> Any:
    """Wrap an OpenAI-compatible client so LLM calls become LangSmith child runs."""
    configure_langsmith()
    if not is_langsmith_enabled():
        return client
    try:
        from langsmith.wrappers import wrap_openai
    except Exception:
        return client
    return wrap_openai(client)


@contextmanager
def trace_context(
    name: str,
    run_type: str = "chain",
    *,
    inputs: dict | None = None,
    metadata: dict | None = None,
    tags: list[str] | None = None,
) -> Iterator[Any | None]:
    """Create an optional LangSmith trace context.

    Yields the LangSmith run tree when tracing is active, otherwise ``None``.
    Callers may use ``run.end(outputs=...)`` before leaving the context.
    """
    configure_langsmith()
    if not is_langsmith_enabled():
        yield None
        return

    try:
        import langsmith as ls
    except Exception:
        yield None
        return

    merged_tags = langsmith_tags(*(tags or []))
    with ls.trace(
        name,
        run_type,
        inputs=inputs or {},
        project_name=langsmith_project(),
        tags=merged_tags,
        metadata=metadata or {},
    ) as run_tree:
        yield run_tree

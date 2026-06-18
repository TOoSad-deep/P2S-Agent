"""Agent 基类：统一 LLM 调用封装，自动适配 Gemini 和 OpenAI API，支持代理"""

import base64
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlsplit

import httpx
from openai import OpenAI, BadRequestError
from google import genai
from google.genai import types

from app.config import ModelConfig
from app.services.langsmith_tracing import wrap_openai_client
from app.services.logging_config import log_event

logger = logging.getLogger(__name__)


# 不再接受 `temperature` 参数的模型前缀（OpenAI 推理系列、GPT-5、
# 以及 aihubmix 上的 claude-opus-4-8 等）。匹配为小写前缀。
_NO_TEMPERATURE_PREFIXES = (
    "o1", "o3", "o4",
    "gpt-5",
    "claude-opus-4-8",
)


def _model_supports_temperature(model: str) -> bool:
    name = (model or "").lower()
    return not any(name.startswith(p) for p in _NO_TEMPERATURE_PREFIXES)


def _safe_base_url(base_url: str) -> str:
    try:
        parsed = urlsplit(base_url)
    except Exception:
        return "<invalid>"
    if not parsed.netloc:
        return base_url[:80]
    return f"{parsed.scheme}://{parsed.netloc}"


def _message_content_to_text(raw_content: Any) -> str:
    """Normalize provider-specific message content into plain text.

    Some OpenAI-compatible Claude gateways return Anthropic-style content
    blocks instead of a bare string. Passing Python's list/dict repr upstream
    makes JSON parsing fail even when the model returned valid JSON inside a
    text block.
    """
    if raw_content is None:
        return ""
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts: list[str] = []
        for block in raw_content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, list):
                    text = _message_content_to_text(text)
                if text:
                    parts.append(str(text))
                continue
            text = getattr(block, "text", None) or getattr(block, "content", None)
            if text:
                parts.append(str(text))
        return "\n".join(parts)
    return str(raw_content)


class BaseAgent:
    """所有 Agent 的基类，封装 LLM 调用逻辑，自动检测 API 类型，支持代理"""

    def __init__(self, model_config: ModelConfig):
        """
        根据传入的 ModelConfig 初始化 LLM 客户端。
        自动检测 base_url 判断使用 Gemini SDK 还是 OpenAI SDK。
        支持代理配置（http/https/socks5）。
        """
        self.model = model_config.model
        self.model_config = model_config
        self._is_gemini = self._detect_gemini_api(model_config.base_url)

        # 创建 httpx client（带代理和超时）
        self._http_client = self._create_http_client(model_config.proxy)

        if self._is_gemini:
            # Gemini 原生 API（暂不使用，因为 SDK 代理配置复杂）
            # 改用 OpenAI-compatible URL
            self._is_gemini = False
            self._openai_client = wrap_openai_client(
                OpenAI(
                    api_key=model_config.api_key,
                    base_url=model_config.base_url.replace("/v1beta", "/v1beta/openai"),
                    http_client=self._http_client,
                )
            )
        else:
            # OpenAI-compatible API (OpenAI, Moonshot, GLM, DeepSeek, Gemini OpenAI-compatible 等)
            self._openai_client = wrap_openai_client(
                OpenAI(
                    api_key=model_config.api_key,
                    base_url=model_config.base_url,
                    http_client=self._http_client,
                )
            )

    def close(self) -> None:
        """Close the underlying httpx.Client, releasing pooled sockets/FDs.

        Also evicts this agent from the module-level cache so a subsequent
        ``get_agent`` for the same config rebuilds a fresh, open client.
        """
        try:
            self._http_client.close()
        except Exception:  # pragma: no cover - best effort cleanup
            pass
        _evict_agent(self)

    def __enter__(self) -> "BaseAgent":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _create_http_client(self, proxy: Optional[str]) -> httpx.Client:
        """创建 httpx client，支持代理和超时配置"""
        # Increase timeout for LLM API calls (may need longer for code generation)
        timeout = httpx.Timeout(120.0, connect=10.0, read=120.0, write=120.0)
        
        # 获取代理配置
        proxy_url = proxy
        if not proxy_url:
            # 检查环境变量中的代理
            import os
            proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or os.environ.get("ALL_PROXY")
        
        if proxy_url:
            return httpx.Client(timeout=timeout, proxy=proxy_url)
        else:
            return httpx.Client(timeout=timeout)

    def _detect_gemini_api(self, base_url: str) -> bool:
        """检测是否为 Gemini 原生 API（非 OpenAI-compatible）"""
        if not base_url:
            return False
        
        # OpenAI-compatible URL 特征（优先检测）
        openai_compatible_patterns = [
            "/openai",
            "api.openai.com",
            "api.moonshot.cn",
            "open.bigmodel.cn",
            "api.deepseek.com",
        ]
        for pattern in openai_compatible_patterns:
            if pattern in base_url:
                return False
        
        # Gemini 原生 API URL 特征
        gemini_native_patterns = [
            "generativelanguage.googleapis.com/v1beta",
            "generativelanguage.googleapis.com/v1",
        ]
        for pattern in gemini_native_patterns:
            if pattern in base_url:
                return True
        
        return False

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: Optional[list[str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        return_raw: bool = False,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> Union[str, dict]:
        """
        调用 LLM，支持可选的图片输入（多模态）

        Args:
            return_raw: 如果 True，返回包含原始响应的 dict，用于显示 reasoning
            enable_thinking: 开启思考模式（仅支持 Qwen3/DeepSeek 等推理模型）
            thinking_budget: 思考预算（tokens），建议 1024-4096
            response_format: OpenAI-compatible response_format，例如
                ``{"type": "json_object"}`` 用于强制 JSON 输出。
        """
        return self._chat_openai(
            system_prompt, user_prompt, image_paths, temperature, max_tokens,
            return_raw, enable_thinking, thinking_budget, response_format
        )

    def _chat_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: Optional[list[str]],
        temperature: float,
        max_tokens: int,
        return_raw: bool = False,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> Union[str, dict]:
        """OpenAI-compatible API 调用

        V2.0: 支持 thinking mode（Qwen3/DeepSeek）
        V2.1: 支持 response_format（强制 JSON 输出）
        """
        content: list[Any] = [{"type": "text", "text": user_prompt}]

        if image_paths:
            for path in image_paths:
                image_data = base64.b64encode(Path(path).read_bytes()).decode()
                ext = Path(path).suffix.lower()
                mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
                mime = mime_map.get(ext, "image/png")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{image_data}"},
                })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content if image_paths else user_prompt},
        ]

        # 构建 extra_body（thinking mode 参数）
        extra_body = {}
        if enable_thinking:
            extra_body["enable_thinking"] = True
            if thinking_budget:
                extra_body["thinking_budget"] = thinking_budget

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if _model_supports_temperature(self.model):
            create_kwargs["temperature"] = temperature
        if extra_body:
            create_kwargs["extra_body"] = extra_body
        if response_format is not None:
            create_kwargs["response_format"] = response_format

        log_event(
            logger,
            "llm_chat_request",
            model=self.model,
            base_url=_safe_base_url(self.model_config.base_url),
            max_tokens=max_tokens,
            has_images=bool(image_paths),
            image_count=len(image_paths or []),
            response_format=response_format,
            temperature=create_kwargs.get("temperature"),
            enable_thinking=enable_thinking,
        )

        try:
            response = self._openai_client.chat.completions.create(**create_kwargs)
        except BadRequestError as e:
            message = str(e).lower()
            # 兜底：未知模型若返回 temperature 已弃用错误，剥离参数后重试一次
            if "temperature" in message and "temperature" in create_kwargs:
                create_kwargs.pop("temperature", None)
                log_event(
                    logger,
                    "llm_chat_retry",
                    level=logging.WARNING,
                    reason="bad_request_temperature",
                    model=self.model,
                    response_format=create_kwargs.get("response_format"),
                )
                try:
                    response = self._openai_client.chat.completions.create(**create_kwargs)
                except BadRequestError as retry_error:
                    if "response_format" in create_kwargs:
                        create_kwargs.pop("response_format", None)
                        log_event(
                            logger,
                            "llm_chat_retry",
                            level=logging.WARNING,
                            reason="bad_request_response_format_after_temperature",
                            model=self.model,
                        )
                        response = self._openai_client.chat.completions.create(**create_kwargs)
                    else:
                        raise retry_error
            elif "response_format" in create_kwargs:
                create_kwargs.pop("response_format", None)
                log_event(
                    logger,
                    "llm_chat_retry",
                    level=logging.WARNING,
                    reason="bad_request_response_format",
                    model=self.model,
                )
                response = self._openai_client.chat.completions.create(**create_kwargs)
            else:
                raise

        raw_content = response.choices[0].message.content
        response_content = _message_content_to_text(raw_content)
        reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None) or ""
        finish_reason = response.choices[0].finish_reason

        usage_payload = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
            "completion_tokens": response.usage.completion_tokens if response.usage else None,
            "total_tokens": response.usage.total_tokens if response.usage else None,
            "reasoning_tokens": getattr(response.usage.completion_tokens_details, 'reasoning_tokens', None)
                if response.usage and hasattr(response.usage, 'completion_tokens_details') else None,
        }
        log_event(
            logger,
            "llm_chat_response",
            model=self.model,
            finish_reason=finish_reason,
            raw_content_type=type(raw_content).__name__,
            content_len=len(response_content),
            reasoning_len=len(reasoning_content),
            usage=usage_payload,
        )

        # 截断 / 空 content / content_filter 应当被显式区分，否则上层只会看到 ""
        if finish_reason and finish_reason != "stop":
            log_event(
                logger,
                "llm_chat_non_stop_finish",
                level=logging.WARNING,
                finish_reason=finish_reason,
                content_len=len(response_content),
                reasoning_len=len(reasoning_content),
            )
        if raw_content is None and not return_raw:
            log_event(
                logger,
                "llm_chat_empty_content",
                level=logging.WARNING,
                finish_reason=finish_reason,
                reasoning_len=len(reasoning_content),
            )
        
        # 打印 reasoning 信息（如果启用）
        if enable_thinking and reasoning_content:
            reasoning_tokens = 0
            if response.usage and hasattr(response.usage, 'completion_tokens_details'):
                details = response.usage.completion_tokens_details
                reasoning_tokens = getattr(details, 'reasoning_tokens', 0) if details else 0
            log_event(
                logger,
                "llm_chat_reasoning",
                reasoning_tokens=reasoning_tokens,
                reasoning_len=len(reasoning_content),
            )
        
        if return_raw:
            return {
                "content": response_content,
                "reasoning_content": reasoning_content,
                "model": self.model,
                "finish_reason": finish_reason,
                "usage": usage_payload,
            }

        return response_content


# --- BaseAgent cache (Bug 1: avoid leaking one httpx.Client per construction) ---
#
# Each BaseAgent owns an httpx.Client with a connection pool. Reconstructing an
# agent per judge/generation call left those clients (and their sockets/FDs)
# unclosed, growing without bound across a run. We instead reuse one agent (and
# its keep-alive client) per ModelConfig identity, bounded with simple LRU
# eviction so the cache itself cannot grow without limit.

_AGENT_CACHE_MAX = 16
_AGENT_CACHE: "OrderedDict[tuple, BaseAgent]" = OrderedDict()
_AGENT_CACHE_LOCK = threading.Lock()


def _agent_cache_key(model_config: ModelConfig) -> tuple:
    """Identity tuple of the fields that change httpx/OpenAI client behaviour."""
    return (
        model_config.base_url,
        model_config.model,
        model_config.api_key,
        model_config.proxy,
        bool(model_config.supports_image),
    )


def get_agent(model_config: ModelConfig) -> BaseAgent:
    """Return a cached BaseAgent for ``model_config`` identity, building one lazily.

    Repeated calls for an equivalent ModelConfig reuse the same agent and its
    httpx.Client (keep-alive), instead of leaking a fresh client each time.
    """
    key = _agent_cache_key(model_config)
    with _AGENT_CACHE_LOCK:
        agent = _AGENT_CACHE.get(key)
        if agent is not None and not agent._http_client.is_closed:
            _AGENT_CACHE.move_to_end(key)
            return agent
        # Build outside-lock would race; constructing here is cheap (no network).
        agent = BaseAgent(model_config)
        _AGENT_CACHE[key] = agent
        _AGENT_CACHE.move_to_end(key)
        evicted: list[BaseAgent] = []
        while len(_AGENT_CACHE) > _AGENT_CACHE_MAX:
            _, old = _AGENT_CACHE.popitem(last=False)
            evicted.append(old)
    for old in evicted:
        try:
            old._http_client.close()
        except Exception:  # pragma: no cover - best effort cleanup
            pass
    return agent


def _evict_agent(agent: BaseAgent) -> None:
    """Remove ``agent`` from the cache (used by ``BaseAgent.close``)."""
    with _AGENT_CACHE_LOCK:
        for key, cached in list(_AGENT_CACHE.items()):
            if cached is agent:
                del _AGENT_CACHE[key]


def clear_agent_cache() -> None:
    """Close and drop all cached agents (test hygiene / shutdown)."""
    with _AGENT_CACHE_LOCK:
        agents = list(_AGENT_CACHE.values())
        _AGENT_CACHE.clear()
    for agent in agents:
        try:
            agent._http_client.close()
        except Exception:  # pragma: no cover - best effort cleanup
            pass

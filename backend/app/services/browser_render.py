"""Playwright 浏览器自动化截图服务：将 shader 注入 WebGL 预览页并截图"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import os
import tempfile

from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright

from app.config import settings


def _new_screenshot_path(prefix: str) -> str:
    """Allocate a unique screenshot temp path without leaving a dangling fd.

    Uses ``NamedTemporaryFile(delete=False)`` so the name is reserved on disk
    (closing the handle immediately) and the file can be reopened by Playwright
    for writing. The caller is responsible for unlinking it — see
    ``_unlink_quietly`` / ``_cleanup_paths``.
    """
    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=".png", delete=False) as fh:
        return fh.name


def _unlink_quietly(path: str | os.PathLike) -> None:
    """Delete ``path`` if it exists, swallowing all OS errors."""
    with contextlib.suppress(FileNotFoundError, OSError):
        os.unlink(path)


def _cleanup_paths(paths) -> None:
    """Unlink every path in ``paths`` best-effort (used on the error path)."""
    for p in paths:
        _unlink_quietly(p)


async def render_and_screenshot(
    shader_code: str,
    time_seconds: float = 1.0,
    width: int | None = None,
    height: int | None = None,
) -> str:
    """
    在浏览器中渲染 shader 并截图，返回截图文件路径。

    Args:
        shader_code: Shadertoy 格式 GLSL 代码
        time_seconds: 渲染到第几秒时截图（用于动画效果）
        width: 截图宽度
        height: 截图高度

    Returns:
        截图 PNG 文件路径
    """
    width = width or settings.screenshot_width
    height = height or settings.screenshot_height

    # 将 shader 代码编码为 URL-safe base64，通过 URL 参数传给前端
    shader_b64 = base64.urlsafe_b64encode(shader_code.encode()).decode()

    preview_url = f"{settings.frontend_url}?shader={shader_b64}&t={time_seconds}"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": width, "height": height})
            await page.goto(preview_url, wait_until="networkidle")

            # 等待 shader 编译和渲染
            await page.wait_for_timeout(500)

            # 等待渲染器标记就绪（超时也会进入 finally 关闭浏览器）
            await page.wait_for_function(
                "() => window.__shaderReady === true",
                timeout=settings.render_timeout_ms,
            )
            shader_error = await page.evaluate("() => window.__shaderError || null")
            if shader_error:
                raise RuntimeError(f"shader preview failed: {shader_error}")

            # 截图
            screenshot_path = _new_screenshot_path("vfx_screenshot_")
            await page.screenshot(path=screenshot_path, type="png")
        finally:
            # 保证浏览器在超时/异常下也被关闭
            with contextlib.suppress(Exception):
                await browser.close()

    return screenshot_path


def render_multiple_frames(
    shader_code: str,
    times: list[float] | None = None,
    width: int | None = None,
    height: int | None = None,
) -> list[str]:
    """
    渲染 shader 在多个时间点的截图，用于动画对比。

    Args:
        shader_code: Shadertoy 格式 GLSL 代码
        times: 截图时间点列表（秒），默认 [0, 0.5, 1.0, 1.5, 2.0]
        width: 截图宽度
        height: 截图高度

    Returns:
        截图文件路径列表
    """
    times = times or [0.0, 0.5, 1.0, 1.5, 2.0]
    screenshots: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(
                viewport={"width": width or settings.screenshot_width, "height": height or settings.screenshot_height}
            )

            shader_b64 = base64.urlsafe_b64encode(shader_code.encode()).decode()
            page.goto(f"{settings.frontend_url}?shader={shader_b64}", wait_until="networkidle")
            page.wait_for_timeout(500)
            page.wait_for_function(
                "() => window.__shaderReady === true",
                timeout=settings.render_timeout_ms,
            )
            shader_error = page.evaluate("() => window.__shaderError || null")
            if shader_error:
                raise RuntimeError(f"shader preview failed: {shader_error}")

            for t in times:
                # 通过 JS 设置渲染器时间并等待一帧
                page.evaluate(f"window.__setShaderTime({t})")
                page.wait_for_timeout(100)

                path = _new_screenshot_path(f"vfx_t{t}_")
                page.screenshot(path=path, type="png")
                screenshots.append(path)
        except Exception:
            # 失败时清理已生成的临时截图，避免泄漏
            _cleanup_paths(screenshots)
            raise
        finally:
            # 保证浏览器在超时/异常下也被关闭
            with contextlib.suppress(Exception):
                browser.close()

    return screenshots

from __future__ import annotations

import asyncio
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest
import uvicorn

from interface_ai_challenge.demo_app.app import create_app
from interface_ai_challenge.demo_app.scenarios import Scenario


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _serve(scenario: Scenario) -> Iterator[str]:
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(scenario), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("demo app did not start")
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


@pytest.fixture(scope="session")
def chromium_available() -> None:
    async def probe() -> None:
        from playwright.async_api import async_playwright

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            await browser.close()

    try:
        asyncio.run(probe())
    except Exception as error:
        pytest.skip(
            f"Chromium cannot start ({type(error).__name__}); run `sudo uv run playwright install-deps chromium`"
        )


@pytest.fixture
def demo_app(chromium_available: None) -> Iterator[Callable[[str], str]]:
    stack: list[Iterator[str]] = []

    def start(scenario: str) -> str:
        context = _serve(Scenario(scenario))
        stack.append(context)
        return context.__enter__()

    yield start
    for context in reversed(stack):
        context.__exit__(None, None, None)


@pytest.fixture(scope="session")
def demo_app_session(chromium_available: None) -> Iterator[Callable[[str], str]]:
    contexts: dict[str, tuple[Iterator[str], str]] = {}

    def start(scenario: str) -> str:
        if scenario not in contexts:
            context = _serve(Scenario(scenario))
            contexts[scenario] = (context, context.__enter__())
        return contexts[scenario][1]

    yield start
    for context, _ in contexts.values():
        context.__exit__(None, None, None)

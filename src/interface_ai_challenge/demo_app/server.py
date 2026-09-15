from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import uvicorn

from interface_ai_challenge.demo_app.app import create_app
from interface_ai_challenge.demo_app.scenarios import Scenario


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def serve_in_thread(scenario: Scenario, port: int | None = None) -> Iterator[str]:
    """Runs the stand-in application on loopback for scripted demos and tests; yields its base URL."""

    chosen = port or free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(scenario), host="127.0.0.1", port=chosen, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("demo app did not start")
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{chosen}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)

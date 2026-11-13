"""Run an ASGI app with uvicorn in a background thread (used to host the A2A reviewer locally)."""

from __future__ import annotations

import socket
import threading
import time

import uvicorn


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class BackgroundServer:
    def __init__(self, app, port: int, host: str = "127.0.0.1") -> None:
        self.server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> BackgroundServer:
        self.thread.start()
        deadline = time.monotonic() + 15
        while not self.server.started:
            if time.monotonic() > deadline or not self.thread.is_alive():
                raise RuntimeError("background server failed to start")
            time.sleep(0.05)
        return self

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


def reviewer_in_background() -> tuple[BackgroundServer, str]:
    from .reviewer import build_app

    port = free_port()
    url = f"http://127.0.0.1:{port}"
    return BackgroundServer(build_app(url), port), url

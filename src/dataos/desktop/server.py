"""Server lifecycle for the Windows desktop wrapper.

Separated from `__main__.py` so it can be unit-tested without a display:
`pywebview`'s `webview.start()` needs a real native window and cannot
run in this project's test environment, but "does the backend actually
come up and answer health checks" can be verified with nothing more
than a background thread and an HTTP request - exactly what the desktop
wrapper itself depends on before it ever opens a window.
"""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request

import uvicorn

from dataos.api.app import create_app

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def start_server_in_background(*, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> threading.Thread:
    """Runs the same `create_app()` backend the browser-based web app
    uses - one codebase, no separate desktop backend to maintain."""
    app = create_app()
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return thread


def wait_for_health(*, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 15.0) -> None:
    """Polls `/api/health` until the server answers or `timeout` elapses.

    Raises `TimeoutError` rather than letting the desktop window open
    against a backend that never started - a blank/broken window with
    no explanation is worse than a clear startup failure.
    """
    deadline = time.monotonic() + timeout
    url = f"http://{host}:{port}/api/health"
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last_error = exc
            time.sleep(0.25)
    raise TimeoutError(f"backend server at {url} did not become healthy within {timeout}s") from last_error

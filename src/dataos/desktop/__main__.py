"""Windows desktop wrapper: `python -m dataos.desktop`.

Runs the exact same backend the browser-based web app uses
(`dataos.api.app.create_app`) in a background thread, then opens it in
a native window via `pywebview` - one codebase, direct local file
access, nothing to keep in sync between "the web app" and "the desktop
app" (see the Windows-app scoping decision this module implements).

Requires the `desktop` optional dependency group (`pip install -e
".[desktop]"`): `pywebview` is not a core dependency, since the web app
and API do not need it.
"""

from __future__ import annotations

from dataos.desktop.server import DEFAULT_HOST, DEFAULT_PORT, start_server_in_background, wait_for_health


def main() -> None:
    try:
        import webview
    except ImportError as exc:
        raise SystemExit(
            "pywebview is not installed - run `pip install -e \".[desktop]\"` to use the desktop app"
        ) from exc

    start_server_in_background(host=DEFAULT_HOST, port=DEFAULT_PORT)
    wait_for_health(host=DEFAULT_HOST, port=DEFAULT_PORT)

    webview.create_window(
        "DataOS Analyst Workbench",
        f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/",
        width=1280,
        height=860,
        min_size=(900, 600),
    )
    webview.start()


if __name__ == "__main__":
    main()

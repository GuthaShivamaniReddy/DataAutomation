"""Run the DataOS Analyst Workbench API/web server: `python -m dataos.api`.

Reads ANTHROPIC_API_KEY / DATAOS_STORE_ROOT / DATAOS_SEMANTIC_DICTIONARY_PATH
from the environment (see `app.py`'s `create_app` defaults) rather than
taking them as flags, so the same environment configuration works
whether this is launched directly, from the desktop wrapper, or from a
packaged executable.
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("DATAOS_HOST", "127.0.0.1")
    port = int(os.environ.get("DATAOS_PORT", "8765"))
    uvicorn.run("dataos.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()

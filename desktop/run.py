"""PyInstaller entry point for the Windows desktop app.

Build (from the repo root, with the `desktop` optional dependency group
installed - `pip install -e ".[desktop]"`):

    pyinstaller desktop/dataos_desktop.spec

The built executable is written to `dist/DataOS Analyst Workbench/`.
This file is intentionally a thin shim, not the real entry point logic
- see `dataos.desktop.__main__` (also runnable directly as
`python -m dataos.desktop` without building an executable at all).
"""

from dataos.desktop.__main__ import main

if __name__ == "__main__":
    main()

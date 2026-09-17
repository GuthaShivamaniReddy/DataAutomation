# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for the Windows desktop app.

Build from the repo root:  pyinstaller desktop/dataos_desktop.spec
Output:                    dist/DataOS Analyst Workbench/

Bundles `src/dataos/api/static` as data (the frontend `create_app()`
serves) and the dynamic-import modules `uvicorn`/`pywebview` load by
name at runtime, which PyInstaller's static analysis cannot see on its
own.
"""

from pathlib import Path

repo_root = Path(SPECPATH).resolve().parent
src_dir = repo_root / "src"

a = Analysis(
    ["run.py"],
    pathex=[str(src_dir)],
    binaries=[],
    datas=[(str(src_dir / "dataos" / "api" / "static"), "dataos/api/static")],
    hiddenimports=[
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "webview.platforms.winforms",
        "webview.platforms.edgechromium",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DataOS Analyst Workbench",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="DataOS Analyst Workbench",
)

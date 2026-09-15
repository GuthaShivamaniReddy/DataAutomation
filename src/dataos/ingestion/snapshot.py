"""Immutable dataset snapshots.

Source of truth: Reliability-First Master Blueprint Section 5 ("Snapshot:
immutable raw object plus metadata and ingestion timestamp") and Section
16's `dataset_version` entity ("immutable snapshot hash, schema, profile").

A file is never operated on in place. `ingest_file` copies the source
bytes into content-addressed storage (keyed by SHA-256) before any parsing
or transformation happens, so the raw input can never be silently mutated
by a later step (Blueprint 1.2 "No hidden mutations").
"""

from __future__ import annotations

import mimetypes
import shutil
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from dataos.ingestion.hashing import file_size, sha256_file

DEFAULT_STORAGE_ROOT = Path(".dataos_store")


class DatasetVersion(BaseModel):
    snapshot_id: str
    """The SHA-256 hash of the raw bytes - the content-addressed identity."""

    original_filename: str
    sha256: str
    size_bytes: int
    mime_type: str | None
    storage_path: str
    ingested_at: str
    """ISO-8601 UTC timestamp."""


def ingest_file(source_path: str | Path, storage_root: str | Path = DEFAULT_STORAGE_ROOT) -> DatasetVersion:
    """Create (or reuse) an immutable snapshot of `source_path`.

    Same bytes always resolve to the same snapshot_id (content-addressed),
    so re-ingesting an identical file is a safe no-op rather than a
    duplicate mutation.
    """
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"source file does not exist: {source_path}")

    storage_root = Path(storage_root)
    digest = sha256_file(source_path)
    size = file_size(source_path)
    mime_type, _ = mimetypes.guess_type(source_path.name)

    snapshot_dir = storage_root / "raw" / digest
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    dest_path = snapshot_dir / source_path.name

    if not dest_path.exists():
        shutil.copy2(source_path, dest_path)
    else:
        # Content-addressed dedup: verify the existing copy really is the
        # same content rather than trusting the path alone.
        if sha256_file(dest_path) != digest:
            raise RuntimeError(
                f"storage integrity violation: {dest_path} does not match expected hash {digest}"
            )

    return DatasetVersion(
        snapshot_id=digest,
        original_filename=source_path.name,
        sha256=digest,
        size_bytes=size,
        mime_type=mime_type,
        storage_path=str(dest_path),
        ingested_at=datetime.now(timezone.utc).isoformat(),
    )

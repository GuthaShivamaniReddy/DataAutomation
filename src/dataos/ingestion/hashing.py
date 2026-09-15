"""File integrity hashing.

Source of truth: Reliability-First Master Blueprint, Section 5 "Data
Ingestion and Source-of-Truth Controls" - File integrity control:
"SHA-256 hash, size, MIME/type verification, decompression limits,
malware scanning hook".

Malware scanning and decompression-bomb limits are out of scope for Phase 1
(no untrusted network upload path exists yet) but hashing and size capture
are foundational and used by every downstream snapshot.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file's exact bytes.

    Used as the immutable identity of every raw source file (Blueprint 5,
    16 dataset_version.hash). Two uploads with identical bytes resolve to
    the same content-addressed snapshot.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def file_size(path: str | Path) -> int:
    return Path(path).stat().st_size

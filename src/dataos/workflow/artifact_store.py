"""Run-scoped immutable artifact storage.

Source of truth: Blueprint Section 9 exit focus "immutable artifacts" and
Section 1.2 "No hidden mutations": every transformation creates a derived
artifact, never mutating a prior one. Once a (run_id, node_id) parquet
file is written by a COMPLETED step, `WorkflowOrchestrator` never asks
this store to overwrite it - a retry of a non-completed step is the only
thing that writes to a given path again.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dataos.evidence.models import ArtifactRef
from dataos.ingestion.hashing import sha256_file

DEFAULT_ARTIFACT_ROOT = Path(".dataos_store") / "run_artifacts"


class ArtifactStore:
    def __init__(self, root: str | Path = DEFAULT_ARTIFACT_ROOT) -> None:
        self._root = Path(root)

    def save(self, run_id: str, node_id: str, df: pl.DataFrame) -> ArtifactRef:
        safe_node_id = node_id.replace(":", "_")
        run_dir = self._root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"{safe_node_id}.parquet"

        df.write_parquet(path)
        checksum = sha256_file(path)

        return ArtifactRef(
            artifact_id=f"{run_id}/{node_id}",
            checksum=checksum,
            storage_path=str(path),
            source_snapshot_ids=[],
            row_count=df.height,
            column_types={name: str(dtype) for name, dtype in zip(df.columns, df.dtypes)},
        )

    def load(self, artifact_ref: ArtifactRef) -> pl.DataFrame:
        return pl.read_parquet(artifact_ref.storage_path)

    def path_for(self, run_id: str, node_id: str) -> Path:
        safe_node_id = node_id.replace(":", "_")
        return self._root / run_id / f"{safe_node_id}.parquet"

    def load_by_ids(self, run_id: str, node_id: str) -> pl.DataFrame:
        """Reload a previously-saved artifact purely from (run_id, node_id) -
        the deterministic path convention means a resumed run never needs
        the original in-memory ArtifactRef, only these two ids (which
        StepRunRecord.output_artifact_id already encodes as
        "<run_id>/<node_id>")."""
        return pl.read_parquet(self.path_for(run_id, node_id))

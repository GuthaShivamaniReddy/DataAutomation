"""Minimal evidence/lineage types carried through Phase 1 operations.

Source of truth: Reliability-First Master Blueprint Section 16 "Data Model
and Storage" (artifact, lineage_edge, validation_result entities) and
AI Prompt Library Appendix B "Standard Agent Envelope" / Appendix D
"Release Evidence Bundle".

These are the minimal subset needed so every operation in the registry
produces real, inspectable evidence instead of prose claims. Full audit /
lineage-graph persistence arrives in the governance phase (Blueprint
roadmap Phase 9); for now these are plain in-memory/serializable records
attached to each OperationResult.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RowImpact(BaseModel):
    """Blueprint 11 "Dropped records" prevention: every transformation must
    be able to account for exactly what happened to every row.
    """

    rows_in: int
    rows_out: int
    rows_rejected: int = 0
    rejected_sample: list[dict] = Field(default_factory=list)
    """A bounded sample of rejected rows for inspection - never silently discarded."""

    @property
    def is_lossy(self) -> bool:
        return self.rows_out < self.rows_in or self.rows_rejected > 0


class ValidationResult(BaseModel):
    check_id: str
    observed: float | int | str | bool | None
    expected: float | int | str | bool | None
    tolerance: float | None = None
    passed: bool
    severity: str = "BLOCKING"
    """BLOCKING | WARNING, per AI Prompt Library Section 19 (Validation Rule Generator)."""


class ArtifactRef(BaseModel):
    """A pointer to an immutable derived (or raw) artifact.

    Blueprint 1.2 "No hidden mutations": raw inputs are immutable, every
    transformation creates a derived artifact with lineage back to its
    source snapshot(s).
    """

    artifact_id: str
    checksum: str
    storage_path: str
    source_snapshot_ids: list[str] = Field(default_factory=list)
    row_count: int | None = None
    column_types: dict[str, str] = Field(default_factory=dict)
    """column name -> dtype string, captured at artifact creation time."""

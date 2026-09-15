"""Error taxonomy.

Source of truth: AI Prompt Library, Appendix C - Error Taxonomy.

Every failure in the platform must resolve to one of these codes. A generic
exception is not an acceptable failure mode for anything that reaches a user
or a release decision - the whole point of the reliability-first design is
that failures are typed, explainable, and never silent.
"""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    REQ_AMBIGUOUS = "REQ_AMBIGUOUS"
    """Material requirement has multiple plausible meanings."""

    REQ_MISSING = "REQ_MISSING"
    """Material requirement information is absent."""

    SEMANTIC_UNGOVERNED = "SEMANTIC_UNGOVERNED"
    """Metric/domain term lacks an approved definition."""

    SCHEMA_MISSING = "SCHEMA_MISSING"
    """Required field/source is unavailable."""

    SCHEMA_DRIFT = "SCHEMA_DRIFT"
    """Source schema changed beyond approved compatibility."""

    JOIN_UNSAFE = "JOIN_UNSAFE"
    """Join cardinality/key cannot be proven safe."""

    DATA_QUALITY_BLOCK = "DATA_QUALITY_BLOCK"
    """Source quality fails a blocking rule."""

    VALIDATION_FAIL = "VALIDATION_FAIL"
    """Deterministic assertion failed."""

    RECONCILIATION_FAIL = "RECONCILIATION_FAIL"
    """Independent totals/control checks failed."""

    VERIFICATION_FAIL = "VERIFICATION_FAIL"
    """Final outputs do not satisfy requirement contract."""

    POLICY_DENIED = "POLICY_DENIED"
    """Security/privacy/authorization policy blocks action."""

    EXTERNAL_WRITE_UNVERIFIED = "EXTERNAL_WRITE_UNVERIFIED"
    """Side effect could not be confirmed safely."""

    MODEL_OUTPUT_INVALID = "MODEL_OUTPUT_INVALID"
    """Agent response failed schema/type validation."""

    NO_SAFE_OPERATION = "NO_SAFE_OPERATION"
    """Required action lacks an approved deterministic primitive."""

    PREDICTION_UNSUPPORTED = "PREDICTION_UNSUPPORTED"
    """Data/method cannot support requested forecast/prediction."""

    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
    """Final claim lacks required evidence."""


class PlatformError(Exception):
    """Base exception for all reliability-relevant platform failures.

    Every raise site must supply a code from the error taxonomy and a
    concrete, evidence-bearing reason - never a bare message that hides
    which non-negotiable rule was violated.
    """

    def __init__(self, code: ErrorCode, reason: str, *, evidence: dict | None = None) -> None:
        self.code = code
        self.reason = reason
        self.evidence = evidence or {}
        super().__init__(f"[{code.value}] {reason}")

    def to_dict(self) -> dict:
        return {"code": self.code.value, "reason": self.reason, "evidence": self.evidence}

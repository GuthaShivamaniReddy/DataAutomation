"""Operation base class - the deterministic Operation Registry contract.

Source of truth: Reliability-First Master Blueprint Section 7
"Deterministic Operation Registry": "The planner is not allowed to
generate arbitrary production code as its primary execution path. It
selects from a registry of versioned, tested operations. Each operation
has typed inputs, preconditions, effects, invariants, output schema
behavior, resource limits, and tests."

Also: AI Prompt Library Section 10 "Operation Registry Selector" and
Section 14 "Deterministic Transformation Compiler" - operations must
produce lineage and validation evidence, and an unsupported request must
resolve to NO_SAFE_OPERATION rather than improvised code.

Every concrete Operation:
  - declares a typed Params model (no untyped dict of arguments reaching
    execution code),
  - checks explicit preconditions before touching data,
  - returns an OperationResult carrying the operation-specific mandatory
    evidence from Blueprint Section 7's table, plus a RowImpact so no
    row-count change can happen unaccounted for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import polars as pl
from pydantic import BaseModel

from dataos.errors import ErrorCode, PlatformError
from dataos.evidence.models import RowImpact


@dataclass
class OperationResult:
    output: pl.DataFrame
    evidence: dict = field(default_factory=dict)
    row_impact: RowImpact | None = None


class Operation(ABC):
    operation_id: ClassVar[str]
    version: ClassVar[str]

    class Params(BaseModel):
        """Override in each subclass with the operation's typed arguments."""

    @classmethod
    def full_id(cls) -> str:
        return f"{cls.operation_id}@{cls.version}"

    @abstractmethod
    def check_preconditions(self, df: pl.DataFrame, params: "Operation.Params") -> list[str]:
        """Return a list of human-readable violation reasons.

        An empty list means preconditions are satisfied. This must never
        raise directly - `execute` turns any non-empty list into a
        PlatformError so every caller gets a uniform failure shape.
        """

    @abstractmethod
    def run(self, df: pl.DataFrame, params: "Operation.Params") -> OperationResult:
        """Perform the operation. Only called after preconditions pass."""

    def execute(self, df: pl.DataFrame, params: "Operation.Params | dict") -> OperationResult:
        parsed_params = params if isinstance(params, self.Params) else self.Params.model_validate(params)

        violations = self.check_preconditions(df, parsed_params)
        if violations:
            raise PlatformError(
                ErrorCode.VALIDATION_FAIL,
                f"precondition failed for {self.full_id()}: {'; '.join(violations)}",
                evidence={"violations": violations, "operation": self.full_id()},
            )

        result = self.run(df, parsed_params)

        if result.row_impact is None:
            raise PlatformError(
                ErrorCode.EVIDENCE_INSUFFICIENT,
                f"{self.full_id()} did not report a RowImpact - every operation must account for row effects",
            )

        return result


class BinaryOperation(ABC):
    """A second, binary Operation base for the first two-input primitive
    (`join`). Deliberately separate from `Operation` rather than a
    breaking generalization of it: every operation built so far
    (select_filter, cast, deduplicate, derive, aggregate, export) is
    genuinely single-input, and rewriting their shared contract to
    accommodate a hypothetical N-input case they don't need would violate
    "don't design for hypothetical future requirements." `join` is the
    first operation that fundamentally needs two dataframes, so it gets
    its own small, parallel contract with the same
    preconditions/evidence/RowImpact discipline as `Operation.execute`.
    """

    operation_id: ClassVar[str]
    version: ClassVar[str]

    class Params(BaseModel):
        """Override in each subclass with the operation's typed arguments."""

    @classmethod
    def full_id(cls) -> str:
        return f"{cls.operation_id}@{cls.version}"

    @abstractmethod
    def check_preconditions_join(
        self, left: pl.DataFrame, right: pl.DataFrame, params: "BinaryOperation.Params"
    ) -> list[str]:
        """Same contract as Operation.check_preconditions: empty list means OK."""

    @abstractmethod
    def run_join(self, left: pl.DataFrame, right: pl.DataFrame, params: "BinaryOperation.Params") -> OperationResult:
        """Perform the join. Only called after preconditions pass."""

    def execute_join(self, left: pl.DataFrame, right: pl.DataFrame, params: "BinaryOperation.Params | dict") -> OperationResult:
        parsed_params = params if isinstance(params, self.Params) else self.Params.model_validate(params)

        violations = self.check_preconditions_join(left, right, parsed_params)
        if violations:
            raise PlatformError(
                ErrorCode.VALIDATION_FAIL,
                f"precondition failed for {self.full_id()}: {'; '.join(violations)}",
                evidence={"violations": violations, "operation": self.full_id()},
            )

        result = self.run_join(left, right, parsed_params)

        if result.row_impact is None:
            raise PlatformError(
                ErrorCode.EVIDENCE_INSUFFICIENT,
                f"{self.full_id()} did not report a RowImpact - every operation must account for row effects",
            )

        return result

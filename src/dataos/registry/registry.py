"""The Operation Registry itself.

Source of truth: Reliability-First Master Blueprint Section 7 and AI
Prompt Library Section 10 "Operation Registry Selector": "You are not
allowed to invent an unregistered operation... If no registered operation
satisfies the step, return NO_SAFE_OPERATION."

No caller can execute an operation by class reference alone - everything
goes through `get()` with an explicit "operation_id@version" string, so
the set of operations that can ever run is exactly the set registered
here. This is what later phases' Planner/Operation Registry Selector
agents are constrained to select from.
"""

from __future__ import annotations

from dataos.errors import ErrorCode, PlatformError
from dataos.registry.base import BinaryOperation, Operation
from dataos.registry.operations import (
    AggregateOperation,
    CastOperation,
    DeduplicateOperation,
    DeriveOperation,
    ExportOperation,
    JoinOperation,
    SelectFilterOperation,
    ValidateSchemaOperation,
)


class OperationRegistry:
    def __init__(self) -> None:
        self._operations: dict[str, Operation | BinaryOperation] = {}

    def register(self, operation: Operation | BinaryOperation) -> None:
        full_id = operation.full_id()
        if full_id in self._operations:
            raise ValueError(f"operation already registered: {full_id}")
        self._operations[full_id] = operation

    def get(self, operation_id: str, version: str | None = None) -> Operation | BinaryOperation:
        """Resolve an operation by id (+ optional version).

        Raises PlatformError(NO_SAFE_OPERATION) rather than KeyError so
        every caller in the platform gets the same typed failure for an
        unsupported step, per Prompt Library Section 10.
        """
        if version is not None:
            full_id = f"{operation_id}@{version}"
            op = self._operations.get(full_id)
            if op is None:
                raise PlatformError(
                    ErrorCode.NO_SAFE_OPERATION,
                    f"no registered operation for '{full_id}'",
                    evidence={"requested": full_id, "available": sorted(self._operations)},
                )
            return op

        candidates = [op for fid, op in self._operations.items() if fid.split("@")[0] == operation_id]
        if not candidates:
            raise PlatformError(
                ErrorCode.NO_SAFE_OPERATION,
                f"no registered operation with id '{operation_id}'",
                evidence={"requested": operation_id, "available": sorted(self._operations)},
            )
        # Highest version wins when unspecified.
        return max(candidates, key=lambda op: op.version)

    def list_operations(self) -> list[str]:
        return sorted(self._operations)


def _build_default_registry() -> OperationRegistry:
    registry = OperationRegistry()
    for op_cls in (
        SelectFilterOperation,
        CastOperation,
        DeduplicateOperation,
        DeriveOperation,
        AggregateOperation,
        ExportOperation,
        JoinOperation,
        ValidateSchemaOperation,
    ):
        registry.register(op_cls())
    return registry


default_registry = _build_default_registry()

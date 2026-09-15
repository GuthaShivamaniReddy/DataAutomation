"""validate_schema operation.

Blueprint Section 9.1's workflow DSL example runs `validate_schema@2.1`
with `args: {required: [...]}` as the first step of a pipeline. Section
10.1 "Structural" validation family: "schema, type, required columns,
cardinality, uniqueness, referential integrity".

Column existence and declared dtype are cheap structural preconditions.
Null-rate and uniqueness are data-driven, so they run the exact checks in
`validation/checks/data_quality.py` (`check_null_rate`,
`check_no_duplicate_rows`) - the engine and this operation must never
silently disagree about what "null" or "duplicate" means.

Unlike the other Phase 1 operations, validate_schema never transforms
rows - it is a gate: pass the input through unchanged once every BLOCKING
check passes, otherwise raise VALIDATION_FAIL with every check result as
evidence.
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl
from pydantic import Field

from dataos.errors import ErrorCode, PlatformError
from dataos.evidence.models import RowImpact, ValidationResult
from dataos.registry.base import Operation, OperationResult
from dataos.validation.checks.data_quality import check_no_duplicate_rows, check_null_rate


class ValidateSchemaOperation(Operation):
    operation_id: ClassVar[str] = "validate_schema"
    version: ClassVar[str] = "1.0"

    class Params(Operation.Params):
        required_columns: list[str] = Field(default_factory=list)
        """Columns that must exist."""
        dtypes: dict[str, str] = Field(default_factory=dict)
        """Column -> expected polars dtype name (e.g. "Int64"), checked for existing columns."""
        null_policy: dict[str, float] = Field(default_factory=dict)
        """Column -> maximum tolerated null rate in [0,1]."""
        unique_keys: list[str] = Field(default_factory=list)
        """Columns that together must have no duplicate combinations. Empty means no check."""

    def check_preconditions(self, df: pl.DataFrame, params: "ValidateSchemaOperation.Params") -> list[str]:
        violations = []
        existing = set(df.columns)

        missing_required = [c for c in params.required_columns if c not in existing]
        if missing_required:
            violations.append(f"required column(s) do not exist: {missing_required}")

        missing_dtype_columns = [c for c in params.dtypes if c not in existing]
        if missing_dtype_columns:
            violations.append(f"dtypes column(s) do not exist: {missing_dtype_columns}")

        for column, max_rate in params.null_policy.items():
            if column not in existing:
                violations.append(f"null_policy column does not exist: {column}")
            elif not (0.0 <= max_rate <= 1.0):
                violations.append(f"null_policy max rate for '{column}' must be in [0,1]")

        missing_key_columns = [c for c in params.unique_keys if c not in existing]
        if missing_key_columns:
            violations.append(f"unique_keys column(s) do not exist: {missing_key_columns}")

        return violations

    def run(self, df: pl.DataFrame, params: "ValidateSchemaOperation.Params") -> OperationResult:
        rows_in = df.height
        checks: list[ValidationResult] = []

        for column, expected_dtype in params.dtypes.items():
            actual_dtype = str(df[column].dtype)
            checks.append(
                ValidationResult(
                    check_id=f"dtype:{column}",
                    observed=actual_dtype,
                    expected=expected_dtype,
                    passed=actual_dtype == expected_dtype,
                    severity="BLOCKING",
                )
            )

        for column, max_rate in params.null_policy.items():
            checks.append(check_null_rate(df, column, max_rate, severity="BLOCKING"))

        if params.unique_keys:
            checks.append(check_no_duplicate_rows(df, subset=params.unique_keys, severity="BLOCKING"))

        failed = [c for c in checks if c.severity == "BLOCKING" and not c.passed]
        if failed:
            raise PlatformError(
                ErrorCode.VALIDATION_FAIL,
                f"schema validation failed: {[c.check_id for c in failed]}",
                evidence={
                    "failed_checks": [c.model_dump() for c in failed],
                    "all_checks": [c.model_dump() for c in checks],
                },
            )

        evidence = {
            "required_columns_present": params.required_columns,
            "checks": [c.model_dump() for c in checks],
        }
        row_impact = RowImpact(rows_in=rows_in, rows_out=rows_in, rows_rejected=0)
        return OperationResult(output=df, evidence=evidence, row_impact=row_impact)

"""cast operation.

Blueprint Section 7: preconditions "parseability threshold, target type";
mandatory evidence "type changes; failed rows captured".

Blueprint Section 11 "Null mishandling" prevention and Section 1.2
"No hidden mutations": a value that fails to parse into the target type
must never be silently turned into null and forgotten. It is always
captured in evidence; if the failure rate exceeds the configured
threshold, the operation blocks instead of proceeding.
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl
from pydantic import BaseModel, Field, field_validator

from dataos.errors import ErrorCode, PlatformError
from dataos.evidence.models import RowImpact
from dataos.registry.base import Operation, OperationResult

_DTYPE_MAP: dict[str, pl.DataType] = {
    "Int64": pl.Int64,
    "Int32": pl.Int32,
    "Float64": pl.Float64,
    "Float32": pl.Float32,
    "Utf8": pl.Utf8,
    "Boolean": pl.Boolean,
    "Date": pl.Date,
    "Datetime": pl.Datetime,
}

_REJECTED_SAMPLE_CAP = 20


class CastSpec(BaseModel):
    column: str
    target_type: str
    parseability_threshold: float = 1.0
    """Minimum fraction of non-null source values that must successfully
    parse. 1.0 (default) means zero tolerated parse failures."""

    @field_validator("target_type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        if v not in _DTYPE_MAP:
            raise ValueError(f"unsupported target_type '{v}'; supported: {sorted(_DTYPE_MAP)}")
        return v


class CastOperation(Operation):
    operation_id: ClassVar[str] = "cast"
    version: ClassVar[str] = "1.0"

    class Params(Operation.Params):
        casts: list[CastSpec] = Field(default_factory=list)

    def check_preconditions(self, df: pl.DataFrame, params: "CastOperation.Params") -> list[str]:
        violations = []
        existing = set(df.columns)
        for spec in params.casts:
            if spec.column not in existing:
                violations.append(f"cast column does not exist: {spec.column}")
            if not (0.0 <= spec.parseability_threshold <= 1.0):
                violations.append(f"parseability_threshold for '{spec.column}' must be in [0,1]")
        return violations

    def run(self, df: pl.DataFrame, params: "CastOperation.Params") -> OperationResult:
        rows_in = df.height
        result = df
        type_changes = []

        for spec in params.casts:
            source_series = result[spec.column]
            source_dtype = str(source_series.dtype)
            target_dtype = _DTYPE_MAP[spec.target_type]

            non_null_count = rows_in - int(source_series.null_count())

            if target_dtype in (pl.Date, pl.Datetime) and source_series.dtype == pl.Utf8:
                if target_dtype == pl.Date:
                    casted = source_series.str.to_date(strict=False)
                else:
                    casted = source_series.str.to_datetime(strict=False)
            else:
                casted = source_series.cast(target_dtype, strict=False)

            originally_non_null = source_series.is_not_null()
            newly_null = casted.is_null()
            failed_mask = originally_non_null & newly_null
            failed_count = int(failed_mask.sum())

            if non_null_count > 0:
                success_rate = 1.0 - (failed_count / non_null_count)
            else:
                success_rate = 1.0

            failed_rows_sample = []
            if failed_count:
                failed_rows_df = result.filter(failed_mask).select(spec.column).head(_REJECTED_SAMPLE_CAP)
                failed_rows_sample = failed_rows_df.to_dicts()

            if success_rate < spec.parseability_threshold:
                raise PlatformError(
                    ErrorCode.DATA_QUALITY_BLOCK,
                    (
                        f"cast of '{spec.column}' to {spec.target_type} succeeded for only "
                        f"{success_rate:.4f} of non-null values, below required threshold "
                        f"{spec.parseability_threshold}"
                    ),
                    evidence={
                        "column": spec.column,
                        "target_type": spec.target_type,
                        "failed_count": failed_count,
                        "non_null_count": non_null_count,
                        "failed_rows_sample": failed_rows_sample,
                    },
                )

            result = result.with_columns(casted.alias(spec.column))
            type_changes.append(
                {
                    "column": spec.column,
                    "from_type": source_dtype,
                    "to_type": spec.target_type,
                    "failed_count": failed_count,
                    "failed_rows_sample": failed_rows_sample,
                }
            )

        rows_out = result.height
        total_failed = sum(tc["failed_count"] for tc in type_changes)
        evidence = {"type_changes": type_changes}
        row_impact = RowImpact(
            rows_in=rows_in,
            rows_out=rows_out,
            rows_rejected=0,
            rejected_sample=[r for tc in type_changes for r in tc["failed_rows_sample"]][:_REJECTED_SAMPLE_CAP] if total_failed else [],
        )
        return OperationResult(output=result, evidence=evidence, row_impact=row_impact)

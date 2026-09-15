"""derive operation.

Blueprint Section 7: preconditions "typed expression"; mandatory evidence
"null propagation and range checks".

Derived columns come from a small allow-listed set of operations, not
arbitrary code/eval (Blueprint 7's "escape hatch for custom code" is
explicitly a separate, sandboxed, later concern - not this primitive).
Blueprint Section 11 "Null mishandling" prevention: null_policy="error"
blocks the operation if any required input is null rather than silently
propagating a null (or a wrong non-null value) through the derivation.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import polars as pl
from pydantic import BaseModel, Field, model_validator

from dataos.errors import ErrorCode, PlatformError
from dataos.evidence.models import RowImpact
from dataos.registry.base import Operation, OperationResult

_ARITHMETIC_OPS = {"add", "subtract", "multiply", "divide"}
_SAMPLE_CAP = 20


class DeriveSpec(BaseModel):
    output_column: str
    op: Literal["add", "subtract", "multiply", "divide", "concat", "coalesce"]
    inputs: list[str]
    null_policy: Literal["propagate", "error"] = "error"

    @model_validator(mode="after")
    def _check_input_arity(self) -> "DeriveSpec":
        if self.op in _ARITHMETIC_OPS and len(self.inputs) != 2:
            raise ValueError(f"op '{self.op}' requires exactly 2 input columns, got {len(self.inputs)}")
        if self.op in ("concat", "coalesce") and len(self.inputs) < 1:
            raise ValueError(f"op '{self.op}' requires at least 1 input column")
        return self


class DeriveOperation(Operation):
    operation_id: ClassVar[str] = "derive"
    version: ClassVar[str] = "1.0"

    class Params(Operation.Params):
        derivations: list[DeriveSpec] = Field(default_factory=list)

    def check_preconditions(self, df: pl.DataFrame, params: "DeriveOperation.Params") -> list[str]:
        violations = []
        existing = set(df.columns)
        for spec in params.derivations:
            missing = [c for c in spec.inputs if c not in existing]
            if missing:
                violations.append(f"derive '{spec.output_column}' references missing columns: {missing}")
            if spec.output_column in existing:
                violations.append(
                    f"derive output_column '{spec.output_column}' collides with an existing column"
                )
        return violations

    def run(self, df: pl.DataFrame, params: "DeriveOperation.Params") -> OperationResult:
        rows_in = df.height
        result = df
        derivation_evidence = []

        for spec in params.derivations:
            if spec.null_policy == "error":
                null_mask = pl.lit(False)
                for c in spec.inputs:
                    null_mask = null_mask | pl.col(c).is_null()
                blocked_rows = result.filter(null_mask)
                blocked_count = blocked_rows.height
                if blocked_count:
                    raise PlatformError(
                        ErrorCode.VALIDATION_FAIL,
                        (
                            f"derive '{spec.output_column}' has null_policy=error but "
                            f"{blocked_count} row(s) have null input values in {spec.inputs}"
                        ),
                        evidence={
                            "output_column": spec.output_column,
                            "inputs": spec.inputs,
                            "blocked_row_count": blocked_count,
                            "blocked_rows_sample": blocked_rows.select(list(dict.fromkeys(spec.inputs))).head(_SAMPLE_CAP).to_dicts(),
                        },
                    )

            expr = _build_expr(spec)
            result = result.with_columns(expr.alias(spec.output_column))

            out_series = result[spec.output_column]
            output_null_count = int(out_series.null_count())
            min_v = max_v = None
            if output_null_count < rows_in:
                try:
                    mn, mx = out_series.min(), out_series.max()
                    min_v, max_v = (str(mn) if mn is not None else None, str(mx) if mx is not None else None)
                except Exception:
                    pass

            derivation_evidence.append(
                {
                    "output_column": spec.output_column,
                    "op": spec.op,
                    "inputs": spec.inputs,
                    "null_policy": spec.null_policy,
                    "output_null_count": output_null_count,
                    "output_min": min_v,
                    "output_max": max_v,
                }
            )

        row_impact = RowImpact(rows_in=rows_in, rows_out=result.height, rows_rejected=0)
        return OperationResult(output=result, evidence={"derivations": derivation_evidence}, row_impact=row_impact)


def _build_expr(spec: DeriveSpec) -> pl.Expr:
    if spec.op == "add":
        return pl.col(spec.inputs[0]) + pl.col(spec.inputs[1])
    if spec.op == "subtract":
        return pl.col(spec.inputs[0]) - pl.col(spec.inputs[1])
    if spec.op == "multiply":
        return pl.col(spec.inputs[0]) * pl.col(spec.inputs[1])
    if spec.op == "divide":
        return pl.col(spec.inputs[0]) / pl.col(spec.inputs[1])
    if spec.op == "concat":
        return pl.concat_str([pl.col(c) for c in spec.inputs], separator="")
    if spec.op == "coalesce":
        return pl.coalesce([pl.col(c) for c in spec.inputs])
    raise AssertionError(f"unhandled derive op: {spec.op}")

"""select/filter operation.

Blueprint Section 7: preconditions "column existence, predicate type";
mandatory evidence "row subset; no mutation of values".

Filters are a structured, allow-listed condition list - never an arbitrary
expression string - so the planner (present or future) cannot smuggle
arbitrary code through a "predicate".
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import polars as pl
from pydantic import BaseModel

from dataos.evidence.models import RowImpact
from dataos.registry.base import Operation, OperationResult

_NULLARY_OPS = {"is_null", "is_not_null"}


class FilterCondition(BaseModel):
    column: str
    op: Literal["==", "!=", "<", "<=", ">", ">=", "in", "not_in", "is_null", "is_not_null"]
    value: Any = None


class SelectFilterOperation(Operation):
    operation_id: ClassVar[str] = "select_filter"
    version: ClassVar[str] = "1.0"

    class Params(Operation.Params):
        columns: list[str] | None = None
        """Columns to keep. None keeps all existing columns."""
        filters: list[FilterCondition] = []
        """AND-combined filter conditions."""

    def check_preconditions(self, df: pl.DataFrame, params: "SelectFilterOperation.Params") -> list[str]:
        violations = []
        existing = set(df.columns)

        if params.columns is not None:
            missing = [c for c in params.columns if c not in existing]
            if missing:
                violations.append(f"columns to select do not exist: {missing}")

        for f in params.filters:
            if f.column not in existing:
                violations.append(f"filter column does not exist: {f.column}")
            if f.op not in _NULLARY_OPS and f.value is None:
                violations.append(f"filter on '{f.column}' with op '{f.op}' requires a value")

        return violations

    def run(self, df: pl.DataFrame, params: "SelectFilterOperation.Params") -> OperationResult:
        rows_in = df.height
        result = df

        for f in params.filters:
            col = pl.col(f.column)
            if f.op == "==":
                expr = col == f.value
            elif f.op == "!=":
                expr = col != f.value
            elif f.op == "<":
                expr = col < f.value
            elif f.op == "<=":
                expr = col <= f.value
            elif f.op == ">":
                expr = col > f.value
            elif f.op == ">=":
                expr = col >= f.value
            elif f.op == "in":
                expr = col.is_in(f.value)
            elif f.op == "not_in":
                expr = ~col.is_in(f.value)
            elif f.op == "is_null":
                expr = col.is_null()
            else:  # is_not_null
                expr = col.is_not_null()
            result = result.filter(expr)

        if params.columns is not None:
            result = result.select(params.columns)

        rows_out = result.height
        evidence = {
            "filters_applied": [f.model_dump() for f in params.filters],
            "columns_selected": params.columns,
            "rows_filtered_out": rows_in - rows_out,
            "value_mutation": False,
        }
        row_impact = RowImpact(rows_in=rows_in, rows_out=rows_out, rows_rejected=0)
        return OperationResult(output=result, evidence=evidence, row_impact=row_impact)

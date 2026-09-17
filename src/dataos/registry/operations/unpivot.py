"""unpivot operation (wide -> long reshape).

The reverse of `pivot.py`: turns a set of value columns into two
columns (a variable name and its value), one row per original
(index, variable) pair. Unlike pivot, this never has an aggregation
ambiguity to resolve - every input cell maps to exactly one output row,
so there is no "collapse without an explicit rule" case to guard
against.
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl
from pydantic import Field

from dataos.evidence.models import RowImpact
from dataos.registry.base import Operation, OperationResult


class UnpivotOperation(Operation):
    operation_id: ClassVar[str] = "unpivot"
    version: ClassVar[str] = "1.0"

    class Params(Operation.Params):
        index: list[str] = Field(min_length=1)
        """Column(s) to keep as-is; repeated once per unpivoted value column."""
        on: list[str] = Field(min_length=1)
        """The value column(s) to unpivot."""
        variable_name: str = "variable"
        value_name: str = "value"

    def check_preconditions(self, df: pl.DataFrame, params: "UnpivotOperation.Params") -> list[str]:
        violations = []
        existing = set(df.columns)
        missing = [c for c in (*params.index, *params.on) if c not in existing]
        if missing:
            violations.append(f"column(s) do not exist: {missing}")
        if params.variable_name in existing or params.value_name in existing:
            clashing = [n for n in (params.variable_name, params.value_name) if n in existing]
            violations.append(f"variable_name/value_name would collide with existing column(s): {clashing}")
        return violations

    def run(self, df: pl.DataFrame, params: "UnpivotOperation.Params") -> OperationResult:
        rows_in = df.height
        result = df.unpivot(
            params.on, index=params.index, variable_name=params.variable_name, value_name=params.value_name
        )
        evidence = {
            "index": params.index,
            "on": params.on,
            "variable_name": params.variable_name,
            "value_name": params.value_name,
        }
        row_impact = RowImpact(rows_in=rows_in, rows_out=result.height, rows_rejected=0)
        return OperationResult(output=result, evidence=evidence, row_impact=row_impact)

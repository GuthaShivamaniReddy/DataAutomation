"""pivot operation (long -> wide reshape).

Blueprint Section 7's operation-registry discipline extended to a
reshape primitive the registry did not have yet: "arrange the data"
requests (a long table of one row per (entity, category) turned into
one row per entity with a column per category) need a typed, audited
primitive of their own rather than a planner improvising it with
arbitrary code.

Polars' own `DataFrame.pivot` silently raises a bare exception when a
single (index, on) combination has more than one value for `values`
and no `aggregate_function` was given - "will raise error if multiple
values are in group" per its own docs. That failure is exactly the
same shape as `join.py`'s unexpected-duplicate-key case, so it gets the
same treatment here: checked first with a rich, sample-bearing
`PlatformError` rather than trusted to whatever polars' bare exception
happens to say, per Blueprint 1.2 "no hidden mutations" (an implicit
aggregation choice a caller never asked for) and Section 11 "never
merge/collapse values without an explicit, auditable rule."
"""

from __future__ import annotations

from typing import ClassVar, Literal

import polars as pl
from pydantic import Field

from dataos.errors import ErrorCode, PlatformError
from dataos.evidence.models import RowImpact
from dataos.registry.base import Operation, OperationResult

_SAMPLE_CAP = 20

AggregateFunction = Literal["min", "max", "first", "last", "sum", "mean", "median", "len"]


class PivotOperation(Operation):
    operation_id: ClassVar[str] = "pivot"
    version: ClassVar[str] = "1.0"

    class Params(Operation.Params):
        index: list[str] = Field(min_length=1)
        """Column(s) that remain from the input; one output row per unique combination."""
        on: str
        """The column whose distinct values become new output columns."""
        values: str
        """The existing column moved under the new columns from `on`."""
        aggregate_function: AggregateFunction | None = None
        """Required whenever an (index, on) combination could have more than
        one `values` row - never defaulted, since picking one silently would
        be exactly the "collapse without an explicit rule" Section 11
        forbids."""

    def check_preconditions(self, df: pl.DataFrame, params: "PivotOperation.Params") -> list[str]:
        violations = []
        existing = set(df.columns)
        missing = [c for c in (*params.index, params.on, params.values) if c not in existing]
        if missing:
            violations.append(f"column(s) do not exist: {missing}")
        return violations

    def run(self, df: pl.DataFrame, params: "PivotOperation.Params") -> OperationResult:
        rows_in = df.height

        if params.aggregate_function is None:
            group_keys = [*params.index, params.on]
            duplicate_groups = df.group_by(group_keys).len(name="count").filter(pl.col("count") > 1)
            if duplicate_groups.height > 0:
                raise PlatformError(
                    ErrorCode.DATA_QUALITY_BLOCK,
                    (
                        f"{duplicate_groups.height} (index, on) combination(s) have more than one '{params.values}' "
                        "value but no aggregate_function was declared - declare one explicitly rather than let "
                        "the pivot silently pick or error on a value"
                    ),
                    evidence={
                        "index": params.index,
                        "on": params.on,
                        "duplicate_group_count": duplicate_groups.height,
                        "duplicate_groups_sample": duplicate_groups.head(_SAMPLE_CAP).to_dicts(),
                    },
                )

        result = df.pivot(params.on, index=params.index, values=params.values, aggregate_function=params.aggregate_function)

        evidence = {
            "index": params.index,
            "on": params.on,
            "values": params.values,
            "aggregate_function": params.aggregate_function,
            "new_columns": [c for c in result.columns if c not in params.index],
        }
        row_impact = RowImpact(rows_in=rows_in, rows_out=result.height, rows_rejected=0)
        return OperationResult(output=result, evidence=evidence, row_impact=row_impact)

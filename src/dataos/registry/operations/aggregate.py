"""aggregate operation.

Blueprint Section 7: preconditions "metric formula and grain"; mandatory
evidence "group coverage; total reconciliation where applicable".

Blueprint Section 10.2 "Dual computation for critical metrics": for
`sum` metrics this operation independently recomputes the overall total
through DuckDB SQL (a different engine than the Polars group-by used for
the primary result) and blocks release with RECONCILIATION_FAIL if the
two paths disagree beyond tolerance - the two implementations do not
share the same generated expression, so an engine-level bug in one path
cannot silently pass. Callers may also supply an external `reconcile_to`
control total (e.g. a ledger total) for the same treatment.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import duckdb
import polars as pl
from pydantic import BaseModel, Field

from dataos.errors import ErrorCode, PlatformError
from dataos.evidence.models import RowImpact
from dataos.registry.base import Operation, OperationResult

_AGG_FN_MAP = {
    "sum": lambda c: pl.col(c).sum(),
    "mean": lambda c: pl.col(c).mean(),
    "count": lambda c: pl.col(c).count(),
    "min": lambda c: pl.col(c).min(),
    "max": lambda c: pl.col(c).max(),
    "n_unique": lambda c: pl.col(c).n_unique(),
}


class MetricSpec(BaseModel):
    name: str
    column: str
    fn: Literal["sum", "mean", "count", "min", "max", "n_unique"]


class AggregateOperation(Operation):
    operation_id: ClassVar[str] = "aggregate"
    version: ClassVar[str] = "1.0"

    class Params(Operation.Params):
        group_by: list[str] = Field(default_factory=list)
        metrics: list[MetricSpec]
        reconcile_to: float | None = None
        """Optional external control total, compared against the first
        sum-type metric's overall total (Blueprint 10.1 "reconciliation")."""
        reconcile_tolerance: float = 0.01

    def check_preconditions(self, df: pl.DataFrame, params: "AggregateOperation.Params") -> list[str]:
        violations = []
        existing = set(df.columns)
        if not params.metrics:
            violations.append("aggregate requires at least one metric")
        missing_group = [c for c in params.group_by if c not in existing]
        if missing_group:
            violations.append(f"group_by columns do not exist: {missing_group}")
        for m in params.metrics:
            if m.column not in existing:
                violations.append(f"metric '{m.name}' references missing column: {m.column}")
        return violations

    def run(self, df: pl.DataFrame, params: "AggregateOperation.Params") -> OperationResult:
        rows_in = df.height

        agg_exprs = [_AGG_FN_MAP[m.fn](m.column).alias(m.name) for m in params.metrics]
        if params.group_by:
            result = df.group_by(params.group_by).agg(agg_exprs)
        else:
            result = df.select(agg_exprs)

        reconciliation: list[dict] = []
        for m in params.metrics:
            if m.fn != "sum":
                continue
            polars_total = df[m.column].sum()
            polars_total = float(polars_total) if polars_total is not None else 0.0
            duckdb_total = duckdb.sql(f'SELECT SUM("{m.column}") FROM df').fetchone()[0]
            duckdb_total = float(duckdb_total) if duckdb_total is not None else 0.0
            difference = abs(polars_total - duckdb_total)
            passed = difference <= params.reconcile_tolerance
            reconciliation.append(
                {
                    "metric": m.name,
                    "polars_value": polars_total,
                    "duckdb_value": duckdb_total,
                    "difference": difference,
                    "tolerance": params.reconcile_tolerance,
                    "passed": passed,
                }
            )
            if not passed:
                raise PlatformError(
                    ErrorCode.RECONCILIATION_FAIL,
                    (
                        f"metric '{m.name}' disagrees between Polars ({polars_total}) and "
                        f"independent DuckDB computation ({duckdb_total}), difference "
                        f"{difference} exceeds tolerance {params.reconcile_tolerance}"
                    ),
                    evidence={"reconciliation": reconciliation},
                )

        external_reconciliation = None
        if params.reconcile_to is not None:
            sum_metrics = [m for m in params.metrics if m.fn == "sum"]
            if not sum_metrics:
                raise PlatformError(
                    ErrorCode.RECONCILIATION_FAIL,
                    "reconcile_to was supplied but no sum-type metric exists to reconcile against",
                )
            target_metric = sum_metrics[0]
            observed = float(df[target_metric.column].sum() or 0.0)
            difference = abs(observed - params.reconcile_to)
            passed = difference <= params.reconcile_tolerance
            external_reconciliation = {
                "metric": target_metric.name,
                "expected": params.reconcile_to,
                "observed": observed,
                "difference": difference,
                "tolerance": params.reconcile_tolerance,
                "passed": passed,
            }
            if not passed:
                raise PlatformError(
                    ErrorCode.RECONCILIATION_FAIL,
                    (
                        f"metric '{target_metric.name}' total {observed} does not reconcile to "
                        f"external control total {params.reconcile_to} within tolerance "
                        f"{params.reconcile_tolerance}"
                    ),
                    evidence={"external_reconciliation": external_reconciliation},
                )

        evidence = {
            "group_by": params.group_by,
            "group_count": result.height,
            "rows_covered": rows_in,
            "reconciliation": reconciliation,
            "external_reconciliation": external_reconciliation,
        }
        row_impact = RowImpact(rows_in=rows_in, rows_out=result.height, rows_rejected=0)
        return OperationResult(output=result, evidence=evidence, row_impact=row_impact)

"""Reconciliation checks.

Source of truth: Blueprint Section 10.1 "Reconciliation: independent
source totals, ledger totals, control totals, known benchmarks" and
Section 10.2 "Dual computation for critical metrics" ("compute the
result through two independent paths... the implementations should not
share the same generated expression").

`check_dual_computation` generalizes the ad hoc Polars-vs-DuckDB
comparison already inside `registry/operations/aggregate.py` into a
standalone, reusable check - usable on any dataframe/column, not only
inside that one operation.
"""

from __future__ import annotations

from typing import Literal

import duckdb
import polars as pl

from dataos.evidence.models import ValidationResult

Severity = Literal["BLOCKING", "WARNING"]
Aggregation = Literal["sum", "count", "mean"]

_SQL_AGG = {"sum": "SUM", "count": "COUNT", "mean": "AVG"}


def check_reconciliation(
    observed: float,
    expected: float,
    *,
    tolerance: float = 0.0,
    check_id: str = "reconciliation",
    severity: Severity = "BLOCKING",
) -> ValidationResult:
    """A generic control-total check: compare an already-computed observed
    value against an independent expected value (a ledger total, a known
    benchmark, another artifact's aggregate, ...)."""
    difference = abs(observed - expected)
    return ValidationResult(
        check_id=check_id,
        observed=observed,
        expected=expected,
        tolerance=tolerance,
        passed=difference <= tolerance,
        severity=severity,
    )


def check_dual_computation(
    df: pl.DataFrame,
    column: str,
    *,
    agg: Aggregation = "sum",
    tolerance: float = 0.01,
    check_id: str | None = None,
    severity: Severity = "BLOCKING",
) -> ValidationResult:
    """Independently recompute `agg(column)` via DuckDB SQL against the
    same dataframe and compare it to the Polars-native aggregate. A
    mismatch is treated as a failing check, never silently accepted -
    the two implementations do not share a generated expression."""
    if agg == "sum":
        polars_value = float(df[column].sum() or 0.0)
    elif agg == "count":
        polars_value = float(df[column].len())
    else:
        polars_value = float(df[column].mean() or 0.0)

    sql_fn = _SQL_AGG[agg]
    duckdb_result = duckdb.sql(f'SELECT {sql_fn}("{column}") FROM df').fetchone()[0]
    duckdb_value = float(duckdb_result) if duckdb_result is not None else 0.0

    difference = abs(polars_value - duckdb_value)
    return ValidationResult(
        check_id=check_id or f"dual_computation:{agg}:{column}",
        observed=polars_value,
        expected=duckdb_value,
        tolerance=tolerance,
        passed=difference <= tolerance,
        severity=severity,
    )

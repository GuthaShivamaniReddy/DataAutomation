"""Conservation checks.

Source of truth: Blueprint Section 10.1 "Conservation: sum/count/balance
before vs after operations that should preserve them" and Section 10.3's
metamorphic checks ("A left join should not reduce the left-side record
set unless explicitly filtered later").
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from dataos.evidence.models import ValidationResult

Severity = Literal["BLOCKING", "WARNING"]
Aggregation = Literal["sum", "count", "mean"]


def _aggregate(series: pl.Series, agg: Aggregation) -> float:
    if agg == "sum":
        return float(series.sum() or 0.0)
    if agg == "count":
        return float(series.len())
    if agg == "mean":
        return float(series.mean() or 0.0)
    raise ValueError(f"unsupported aggregation: {agg}")


def check_conservation(
    before: pl.DataFrame,
    after: pl.DataFrame,
    column: str,
    *,
    agg: Aggregation = "sum",
    tolerance: float = 0.0,
    check_id: str | None = None,
    severity: Severity = "BLOCKING",
) -> ValidationResult:
    before_value = _aggregate(before[column], agg)
    after_value = _aggregate(after[column], agg)
    difference = abs(after_value - before_value)
    return ValidationResult(
        check_id=check_id or f"conservation:{agg}:{column}",
        observed=after_value,
        expected=before_value,
        tolerance=tolerance,
        passed=difference <= tolerance,
        severity=severity,
    )


def check_row_count_conservation(
    before: pl.DataFrame,
    after: pl.DataFrame,
    *,
    allow_decrease: bool = False,
    check_id: str = "row_count_conservation",
    severity: Severity = "BLOCKING",
) -> ValidationResult:
    passed = True if allow_decrease else after.height >= before.height
    return ValidationResult(
        check_id=check_id,
        observed=after.height,
        expected=before.height,
        passed=passed,
        severity=severity,
    )

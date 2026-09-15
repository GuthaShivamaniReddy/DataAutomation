"""Data-quality checks.

Source of truth: Blueprint Section 10.1 "Data-quality: nulls, ranges,
categories, formats, duplicates, outliers, freshness" and AI Prompt
Library Section 19 "Validation Rule Generator": "Every validation must
include severity, exact predicate, tolerance, and action on failure."

These are also the exact rules `registry/operations/validate_schema.py`
runs for its `null_policy` and `unique_keys` parameters - reused rather
than duplicated, so the engine and the operation can never silently
disagree about what "null" or "duplicate" means.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from dataos.evidence.models import ValidationResult

Severity = Literal["BLOCKING", "WARNING"]


def check_null_rate(
    df: pl.DataFrame, column: str, max_rate: float, *, check_id: str | None = None, severity: Severity = "BLOCKING"
) -> ValidationResult:
    null_count = int(df[column].null_count())
    rate = (null_count / df.height) if df.height else 0.0
    return ValidationResult(
        check_id=check_id or f"null_rate:{column}",
        observed=rate,
        expected=max_rate,
        tolerance=None,
        passed=rate <= max_rate,
        severity=severity,
    )


def check_range(
    df: pl.DataFrame,
    column: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    check_id: str | None = None,
    severity: Severity = "BLOCKING",
) -> ValidationResult:
    series = df[column].drop_nulls()
    out_of_range = 0
    if min_value is not None:
        out_of_range += int((series < min_value).sum())
    if max_value is not None:
        out_of_range += int((series > max_value).sum())
    return ValidationResult(
        check_id=check_id or f"range:{column}",
        observed=out_of_range,
        expected=0,
        passed=out_of_range == 0,
        severity=severity,
    )


def check_allowed_values(
    df: pl.DataFrame, column: str, allowed: list, *, check_id: str | None = None, severity: Severity = "BLOCKING"
) -> ValidationResult:
    series = df[column].drop_nulls()
    invalid_count = int(series.filter(~series.is_in(allowed)).len())
    return ValidationResult(
        check_id=check_id or f"allowed_values:{column}",
        observed=invalid_count,
        expected=0,
        passed=invalid_count == 0,
        severity=severity,
    )


def check_no_duplicate_rows(
    df: pl.DataFrame,
    *,
    subset: list[str] | None = None,
    check_id: str | None = None,
    severity: Severity = "BLOCKING",
) -> ValidationResult:
    target = df.select(subset) if subset else df
    duplicate_count = int(target.is_duplicated().sum())
    default_id = f"unique:{','.join(subset)}" if subset else "unique_rows"
    return ValidationResult(
        check_id=check_id or default_id,
        observed=duplicate_count,
        expected=0,
        passed=duplicate_count == 0,
        severity=severity,
    )

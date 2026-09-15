"""Join-safety checks.

Source of truth: Blueprint Section 10.1 "Join safety: pre/post row count,
key uniqueness, match rate, unmatched sides, multiplication factor" and
Section 7's join row: "mandatory evidence: match-rate and
row-multiplication report."

These generalize the match-rate/multiplication-factor math already inside
`registry/operations/join.py`'s `run_join` into standalone, reusable
checks - usable against any two dataframes and key list, not only from
inside a `join` step, matching the precedent set by
`reconciliation.check_dual_computation`.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from dataos.evidence.models import ValidationResult

Severity = Literal["BLOCKING", "WARNING"]


def check_join_match_rate(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    left_keys: list[str],
    right_keys: list[str],
    min_match_rate: float,
    check_id: str = "join_match_rate",
    severity: Severity = "BLOCKING",
) -> ValidationResult:
    """Fraction of left rows that find at least one match on the right."""
    if left.height == 0:
        rate = 1.0
    else:
        unmatched = left.join(right, left_on=left_keys, right_on=right_keys, how="anti")
        rate = (left.height - unmatched.height) / left.height
    return ValidationResult(
        check_id=check_id,
        observed=rate,
        expected=min_match_rate,
        passed=rate >= min_match_rate,
        severity=severity,
    )


def check_join_multiplication_factor(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    left_keys: list[str],
    right_keys: list[str],
    max_factor: float,
    how: Literal["inner", "left", "right", "outer"] = "inner",
    check_id: str = "join_multiplication_factor",
    severity: Severity = "BLOCKING",
) -> ValidationResult:
    """Ratio of output rows to left input rows - catches accidental
    many-to-many row fan-out that inflates the row count."""
    if left.height == 0:
        factor = 0.0
    else:
        result = left.join(right, left_on=left_keys, right_on=right_keys, how=how)
        factor = result.height / left.height
    return ValidationResult(
        check_id=check_id,
        observed=factor,
        expected=max_factor,
        passed=factor <= max_factor,
        severity=severity,
    )

"""text_clean operation.

AI Prompt Library Section 11 "Cleaning Strategy Agent": "NEVER
automatically ... coerce failed parses to null and continue silently
... normalize identifiers in ways that can change identity." This
operation is the registered, audited primitive a cleaning rule (once
approved - see `cleaning_strategy_agent.py`) actually executes against;
it never decides on its own that a column *should* be cleaned.

Three kinds of transformation, in increasing order of risk:
  - `normalize`: trim / collapse internal whitespace / case-fold a
    column in place. Reversible in principle, but the *values* change,
    so identity collapse is checked explicitly (see below) rather than
    assumed safe.
  - `regex_extract`: pulls a capture group into a *new* column,
    leaving the source column untouched - non-destructive, like
    `derive.py`'s style.
  - `regex_replace`: find/replace against an explicit, bounded regex -
    never arbitrary code/eval, the same "allow-listed operation, not a
    code escape hatch" discipline `derive.py` already applies.

`normalize` on a column requires `allow_identity_collapse=True` the
moment it would actually reduce that column's distinct-value count
(computed before and after, never guessed) - two previously-distinct
values folding into one is exactly Section 11's "normalize identifiers
in ways that can change identity," and the fix must be an explicit
decision, not a side effect nobody asked for. This deliberately never
attempts fuzzy/near-duplicate *entity* matching (e.g. "Acme Inc" vs
"ACME, Inc."); Section 11 requires entity-resolution evidence for that,
which is a governed decision this primitive has no authority to make.
"""

from __future__ import annotations

import re
from typing import ClassVar, Literal

import polars as pl
from pydantic import BaseModel, Field

from dataos.errors import ErrorCode, PlatformError
from dataos.evidence.models import RowImpact
from dataos.registry.base import Operation, OperationResult

_SAMPLE_CAP = 20
NormalizeStep = Literal["trim", "collapse_whitespace", "lower", "upper", "title"]


class NormalizeSpec(BaseModel):
    column: str
    steps: list[NormalizeStep] = Field(min_length=1)
    """Applied in the given order, e.g. ["trim", "lower"]."""


class RegexExtractSpec(BaseModel):
    column: str
    pattern: str
    output_column: str
    group: int = 1


class RegexReplaceSpec(BaseModel):
    column: str
    pattern: str
    replacement: str


class TextCleanOperation(Operation):
    operation_id: ClassVar[str] = "text_clean"
    version: ClassVar[str] = "1.0"

    class Params(Operation.Params):
        normalize: list[NormalizeSpec] = Field(default_factory=list)
        regex_extract: list[RegexExtractSpec] = Field(default_factory=list)
        regex_replace: list[RegexReplaceSpec] = Field(default_factory=list)
        allow_identity_collapse: bool = False

    def check_preconditions(self, df: pl.DataFrame, params: "TextCleanOperation.Params") -> list[str]:
        violations = []
        existing = set(df.columns)

        for spec in params.normalize:
            if spec.column not in existing:
                violations.append(f"normalize column does not exist: {spec.column}")
        for spec in params.regex_extract:
            if spec.column not in existing:
                violations.append(f"regex_extract source column does not exist: {spec.column}")
            elif spec.output_column in existing:
                violations.append(f"regex_extract output_column already exists: {spec.output_column}")
            self._check_pattern(spec.pattern, violations)
        for spec in params.regex_replace:
            if spec.column not in existing:
                violations.append(f"regex_replace column does not exist: {spec.column}")
            self._check_pattern(spec.pattern, violations)

        return violations

    def _check_pattern(self, pattern: str, violations: list[str]) -> None:
        try:
            re.compile(pattern)
        except re.error as exc:
            violations.append(f"invalid regex pattern {pattern!r}: {exc}")

    def run(self, df: pl.DataFrame, params: "TextCleanOperation.Params") -> OperationResult:
        rows_in = df.height
        result = df
        evidence: dict = {"normalized_columns": [], "extracted_columns": [], "replaced_columns": []}

        for spec in params.normalize:
            before_distinct = result[spec.column].n_unique()
            before_sample = result[spec.column].drop_nulls().head(_SAMPLE_CAP).to_list()

            expr = pl.col(spec.column)
            for step in spec.steps:
                if step == "trim":
                    expr = expr.str.strip_chars()
                elif step == "collapse_whitespace":
                    expr = expr.str.replace_all(r"\s+", " ").str.strip_chars()
                elif step == "lower":
                    expr = expr.str.to_lowercase()
                elif step == "upper":
                    expr = expr.str.to_uppercase()
                elif step == "title":
                    expr = expr.str.to_titlecase()
            result = result.with_columns(expr.alias(spec.column))

            after_distinct = result[spec.column].n_unique()
            if after_distinct < before_distinct and not params.allow_identity_collapse:
                raise PlatformError(
                    ErrorCode.DATA_QUALITY_BLOCK,
                    (
                        f"normalizing '{spec.column}' would collapse {before_distinct - after_distinct} "
                        f"distinct value(s) into an existing one - this changes identity and requires "
                        "allow_identity_collapse=True as an explicit decision, not a side effect"
                    ),
                    evidence={
                        "column": spec.column,
                        "distinct_before": before_distinct,
                        "distinct_after": after_distinct,
                        "sample_before": before_sample,
                    },
                )

            evidence["normalized_columns"].append(
                {"column": spec.column, "steps": spec.steps, "distinct_before": before_distinct, "distinct_after": after_distinct}
            )

        for spec in params.regex_extract:
            result = result.with_columns(
                pl.col(spec.column).str.extract(spec.pattern, spec.group).alias(spec.output_column)
            )
            match_count = int(result[spec.output_column].is_not_null().sum())
            evidence["extracted_columns"].append(
                {"column": spec.column, "output_column": spec.output_column, "pattern": spec.pattern, "match_count": match_count}
            )

        for spec in params.regex_replace:
            before_column = result[spec.column]
            result = result.with_columns(
                pl.col(spec.column).str.replace_all(spec.pattern, spec.replacement).alias(spec.column)
            )
            changed_count = int((before_column != result[spec.column]).sum())
            evidence["replaced_columns"].append(
                {
                    "column": spec.column,
                    "pattern": spec.pattern,
                    "replacement": spec.replacement,
                    "changed_count": changed_count,
                }
            )

        row_impact = RowImpact(rows_in=rows_in, rows_out=result.height, rows_rejected=0)
        return OperationResult(output=result, evidence=evidence, row_impact=row_impact)

"""join operation.

Blueprint Section 7: preconditions "key types, uniqueness/cardinality
expectation"; mandatory evidence "match-rate and row-multiplication
report".

AI Prompt Library Section 12 "Join Safety Reviewer": "Reject accidental
many-to-many joins. Reject joins based only on similar column names."
Blueprint 3.2: "If join keys are not unique where uniqueness is required,
stop the join and show duplicate-key evidence."

Cheap structural checks (columns exist, key-list lengths match, dtype
compatibility, the many-to-many opt-in flag) are ordinary preconditions.
The uniqueness check is data-driven and needs a rich, sample-bearing
evidence payload, so - matching the precedent set by cast.py's
parseability-threshold check - it is evaluated first thing inside
`run_join` and raises directly with structured evidence rather than being
squeezed into a plain string violation.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import polars as pl
from pydantic import model_validator

from dataos.errors import ErrorCode, PlatformError
from dataos.evidence.models import RowImpact
from dataos.registry.base import BinaryOperation, OperationResult

_SAMPLE_CAP = 20


class JoinOperation(BinaryOperation):
    operation_id: ClassVar[str] = "join"
    version: ClassVar[str] = "1.0"

    class Params(BinaryOperation.Params):
        left_keys: list[str]
        right_keys: list[str]
        how: Literal["inner", "left", "right", "outer"] = "inner"
        expected_cardinality: Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many"]
        allow_many_to_many: bool = False

        @model_validator(mode="after")
        def _key_lists_same_length(self) -> "JoinOperation.Params":
            if len(self.left_keys) != len(self.right_keys):
                raise ValueError(
                    f"left_keys and right_keys must be the same length, got "
                    f"{len(self.left_keys)} and {len(self.right_keys)}"
                )
            return self

    def check_preconditions_join(self, left: pl.DataFrame, right: pl.DataFrame, params: "JoinOperation.Params") -> list[str]:
        violations = []
        left_missing = [k for k in params.left_keys if k not in left.columns]
        right_missing = [k for k in params.right_keys if k not in right.columns]
        if left_missing:
            violations.append(f"left join key(s) do not exist: {left_missing}")
        if right_missing:
            violations.append(f"right join key(s) do not exist: {right_missing}")

        if not left_missing and not right_missing:
            for lk, rk in zip(params.left_keys, params.right_keys):
                left_dtype = str(left[lk].dtype)
                right_dtype = str(right[rk].dtype)
                if left_dtype != right_dtype:
                    violations.append(
                        f"key dtype mismatch: left.{lk} ({left_dtype}) vs right.{rk} ({right_dtype})"
                    )

        if params.expected_cardinality == "many_to_many" and not params.allow_many_to_many:
            violations.append(
                "expected_cardinality is many_to_many but allow_many_to_many is False - "
                "accidental many-to-many joins are rejected; set allow_many_to_many=True to proceed"
            )

        return violations

    def run_join(self, left: pl.DataFrame, right: pl.DataFrame, params: "JoinOperation.Params") -> OperationResult:
        _block_on_unexpected_duplicates(left, right, params)

        matched = left.join(right, left_on=params.left_keys, right_on=params.right_keys, how="inner")
        unmatched_left = left.join(right, left_on=params.left_keys, right_on=params.right_keys, how="anti")
        unmatched_right = right.join(left, left_on=params.right_keys, right_on=params.left_keys, how="anti")

        result = left.join(right, left_on=params.left_keys, right_on=params.right_keys, how=params.how)

        rows_in = left.height
        rows_out = result.height
        match_rate_left = (rows_in - unmatched_left.height) / rows_in if rows_in else 1.0
        match_rate_right = (right.height - unmatched_right.height) / right.height if right.height else 1.0
        multiplication_factor = (rows_out / rows_in) if rows_in else (0.0 if rows_out == 0 else float("inf"))

        evidence = {
            "left_keys": params.left_keys,
            "right_keys": params.right_keys,
            "how": params.how,
            "expected_cardinality": params.expected_cardinality,
            "left_rows_in": rows_in,
            "right_rows_in": right.height,
            "matched_row_count": matched.height,
            "unmatched_left_row_count": unmatched_left.height,
            "unmatched_right_row_count": unmatched_right.height,
            "match_rate_left": match_rate_left,
            "match_rate_right": match_rate_right,
            "row_multiplication_factor": multiplication_factor,
        }
        row_impact = RowImpact(rows_in=rows_in, rows_out=rows_out, rows_rejected=0)
        return OperationResult(output=result, evidence=evidence, row_impact=row_impact)


def _block_on_unexpected_duplicates(left: pl.DataFrame, right: pl.DataFrame, params: "JoinOperation.Params") -> None:
    left_must_be_unique = params.expected_cardinality in ("one_to_one", "one_to_many")
    right_must_be_unique = params.expected_cardinality in ("one_to_one", "many_to_one")

    if left_must_be_unique:
        _raise_if_duplicated(left, params.left_keys, side="left")
    if right_must_be_unique:
        _raise_if_duplicated(right, params.right_keys, side="right")


def _raise_if_duplicated(df: pl.DataFrame, keys: list[str], *, side: str) -> None:
    groups = df.group_by(keys).len(name="count").filter(pl.col("count") > 1)
    if groups.height == 0:
        return
    raise PlatformError(
        ErrorCode.DATA_QUALITY_BLOCK,
        (
            f"join declares the {side} side as unique on {keys}, but {groups.height} duplicate "
            f"key group(s) were found - stopping before the join to avoid silent row multiplication"
        ),
        evidence={
            "side": side,
            "keys": keys,
            "duplicate_group_count": groups.height,
            "duplicate_groups_sample": groups.head(_SAMPLE_CAP).to_dicts(),
        },
    )

"""deduplicate operation.

Blueprint Section 7: preconditions "key definition, keep rule"; mandatory
evidence "duplicate groups + removed-row evidence".

Blueprint Section 11 "Duplicate counting" prevention: entity keys + dedup
rule + duplicate-group artifact. A row is never silently dropped - the
exact groups and the removed rows themselves are captured as evidence,
and Appendix C's "Duplicate join explosion"-adjacent golden scenario for
Phase 1 (plain dedup, no join yet) checks this directly.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import polars as pl

from dataos.evidence.models import RowImpact
from dataos.registry.base import Operation, OperationResult

_ROW_INDEX_COL = "_dataos_row_idx"
_SAMPLE_CAP = 50


class DeduplicateOperation(Operation):
    operation_id: ClassVar[str] = "deduplicate"
    version: ClassVar[str] = "1.0"

    class Params(Operation.Params):
        keys: list[str]
        keep: Literal["first", "last"] = "first"
        order_by: str | None = None
        """Optional column to sort by before applying keep=first/last,
        e.g. keep the latest_updated_at row (Blueprint 9.1 example)."""

    def check_preconditions(self, df: pl.DataFrame, params: "DeduplicateOperation.Params") -> list[str]:
        violations = []
        existing = set(df.columns)
        if not params.keys:
            violations.append("deduplicate requires at least one key column")
        missing_keys = [k for k in params.keys if k not in existing]
        if missing_keys:
            violations.append(f"key columns do not exist: {missing_keys}")
        if params.order_by is not None and params.order_by not in existing:
            violations.append(f"order_by column does not exist: {params.order_by}")
        return violations

    def run(self, df: pl.DataFrame, params: "DeduplicateOperation.Params") -> OperationResult:
        rows_in = df.height

        group_counts = df.group_by(params.keys).len(name="count").filter(pl.col("count") > 1)
        duplicate_group_count = group_counts.height
        duplicate_groups_sample = group_counts.head(_SAMPLE_CAP).to_dicts()

        indexed = df.with_row_index(_ROW_INDEX_COL)
        working = indexed.sort(params.order_by) if params.order_by else indexed
        kept = working.unique(subset=params.keys, keep=params.keep, maintain_order=True)

        kept_indices = set(kept[_ROW_INDEX_COL].to_list())
        removed = indexed.filter(~pl.col(_ROW_INDEX_COL).is_in(list(kept_indices)))

        result = kept.sort(_ROW_INDEX_COL).drop(_ROW_INDEX_COL)
        removed_rows_sample = removed.drop(_ROW_INDEX_COL).head(_SAMPLE_CAP).to_dicts()

        rows_out = result.height
        evidence = {
            "keys": params.keys,
            "keep": params.keep,
            "order_by": params.order_by,
            "duplicate_group_count": duplicate_group_count,
            "duplicate_groups_sample": duplicate_groups_sample,
            "removed_row_count": removed.height,
            "removed_rows_sample": removed_rows_sample,
        }
        row_impact = RowImpact(rows_in=rows_in, rows_out=rows_out, rows_rejected=0)
        return OperationResult(output=result, evidence=evidence, row_impact=row_impact)

"""Golden test scenarios.

Source of truth: Reliability-First Master Blueprint, Appendix C "Initial
Golden Test Scenarios". Only the scenarios in scope for Phase 1 (no join
or currency-conversion operations exist yet) are implemented here; the
rest (duplicate join explosion, currency mix, partial payment
reconciliation, prompt injection, destination retry, forecast
insufficient data, narrative fabrication) are deferred to the phases that
introduce join/currency/forecast/agent operations.

Per Blueprint Section 19 "Regression tests": every one of these is meant
to become a permanent, non-deletable test as the corresponding failure
mode is discovered in the real system.
"""

from __future__ import annotations

import polars as pl
import pytest

from dataos.errors import ErrorCode, PlatformError
from dataos.registry.operations.deduplicate import DeduplicateOperation
from dataos.registry.operations.derive import DeriveOperation
from dataos.registry.operations.select_filter import SelectFilterOperation
from dataos.ingestion.parsers import parse_csv


class TestNullDeletionScenario:
    """Appendix C "Null deletion": cleaning agent must not drop/impute a
    field without an explicit rule."""

    def test_derive_refuses_to_silently_propagate_null_without_explicit_opt_in(self, fixtures_dir):
        df = pl.read_csv(fixtures_dir / "orders_nulls.csv")
        op = DeriveOperation()

        # Default null_policy is "error" - the operation must block rather
        # than silently computing a wrong/null downstream value.
        with pytest.raises(PlatformError) as excinfo:
            op.execute(
                df,
                {"derivations": [{"output_column": "doubled", "op": "multiply", "inputs": ["net_amount", "net_amount"]}]},
            )
        assert excinfo.value.code == ErrorCode.VALIDATION_FAIL


class TestSilentParseLossScenario:
    """Appendix C "Silent parse loss": malformed CSV rows must expose
    rejected count; cannot claim full-data result."""

    def test_permissive_parse_never_claims_full_data(self, fixtures_dir):
        df, report = parse_csv(fixtures_dir / "orders_malformed.csv", mode="permissive")
        assert report.rows_rejected > 0
        assert df.height < report.rows_seen
        assert len(report.rejected_sample) == report.rows_rejected

    def test_strict_parse_blocks_entirely_rather_than_silently_dropping(self, fixtures_dir):
        with pytest.raises(PlatformError) as excinfo:
            parse_csv(fixtures_dir / "orders_malformed.csv", mode="strict")
        assert excinfo.value.code == ErrorCode.DATA_QUALITY_BLOCK


class TestMissingRequiredFieldScenario:
    """Appendix C "Missing required field": automation preflight must
    stop rather than silently proceed without a field a downstream step
    depends on."""

    def test_operation_blocks_when_required_column_absent(self, orders_basic_df):
        df_without_region = orders_basic_df.drop("region")
        op = SelectFilterOperation()

        with pytest.raises(PlatformError) as excinfo:
            op.execute(df_without_region, {"filters": [{"column": "region", "op": "==", "value": "East"}]})
        assert excinfo.value.code == ErrorCode.VALIDATION_FAIL
        assert "region" in excinfo.value.reason


class TestDuplicateDetectionScenario:
    """Duplicate-key detection golden scenario (Phase 1 scope; the
    Appendix C "Duplicate join explosion" scenario itself requires the
    `join` operation, deferred to a later phase)."""

    def test_duplicate_groups_and_removed_rows_are_both_evidenced(self, fixtures_dir):
        df = pl.read_csv(fixtures_dir / "orders_duplicates.csv")
        op = DeduplicateOperation()

        result = op.execute(df, {"keys": ["order_id"], "keep": "first"})

        assert result.row_impact.rows_in > result.row_impact.rows_out
        assert result.evidence["duplicate_group_count"] > 0
        assert result.evidence["removed_row_count"] == result.row_impact.rows_in - result.row_impact.rows_out
        # The removed rows themselves, not just a count, must be inspectable.
        assert len(result.evidence["removed_rows_sample"]) == result.evidence["removed_row_count"]

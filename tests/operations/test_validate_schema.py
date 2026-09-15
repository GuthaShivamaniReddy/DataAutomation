import polars as pl
import pytest

from dataos.errors import ErrorCode, PlatformError
from dataos.registry.operations.validate_schema import ValidateSchemaOperation


def test_validate_schema_passes_and_reports_checks(orders_basic_df):
    op = ValidateSchemaOperation()
    result = op.execute(
        orders_basic_df,
        {
            "required_columns": ["order_id", "order_date", "region", "net_amount"],
            "null_policy": {"net_amount": 0.0},
            "unique_keys": ["order_id"],
        },
    )

    assert result.output.equals(orders_basic_df)
    assert result.evidence["required_columns_present"] == ["order_id", "order_date", "region", "net_amount"]
    check_ids = [c["check_id"] for c in result.evidence["checks"]]
    assert "null_rate:net_amount" in check_ids
    assert "unique:order_id" in check_ids


def test_missing_required_column_fails_preconditions(orders_basic_df):
    op = ValidateSchemaOperation()
    with pytest.raises(PlatformError) as excinfo:
        op.execute(orders_basic_df, {"required_columns": ["order_id", "does_not_exist"]})

    assert excinfo.value.code == ErrorCode.VALIDATION_FAIL
    assert "does_not_exist" in str(excinfo.value.evidence["violations"])


def test_null_policy_violation_blocks_with_evidence(fixtures_dir):
    df = pl.read_csv(fixtures_dir / "orders_nulls.csv")
    op = ValidateSchemaOperation()
    with pytest.raises(PlatformError) as excinfo:
        op.execute(df, {"null_policy": {"net_amount": 0.0}})

    assert excinfo.value.code == ErrorCode.VALIDATION_FAIL
    failed_ids = [c["check_id"] for c in excinfo.value.evidence["failed_checks"]]
    assert "null_rate:net_amount" in failed_ids


def test_unique_keys_violation_blocks_with_evidence(fixtures_dir):
    df = pl.read_csv(fixtures_dir / "orders_duplicates.csv")
    op = ValidateSchemaOperation()
    with pytest.raises(PlatformError) as excinfo:
        op.execute(df, {"unique_keys": ["order_id"]})

    assert excinfo.value.code == ErrorCode.VALIDATION_FAIL
    failed_ids = [c["check_id"] for c in excinfo.value.evidence["failed_checks"]]
    assert "unique:order_id" in failed_ids


def test_dtype_mismatch_blocks_with_evidence(orders_basic_df):
    op = ValidateSchemaOperation()
    with pytest.raises(PlatformError) as excinfo:
        op.execute(orders_basic_df, {"dtypes": {"order_id": "Utf8"}})

    assert excinfo.value.code == ErrorCode.VALIDATION_FAIL
    failed = excinfo.value.evidence["failed_checks"][0]
    assert failed["check_id"] == "dtype:order_id"
    assert failed["expected"] == "Utf8"

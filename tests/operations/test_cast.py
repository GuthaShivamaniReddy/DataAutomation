import pytest

from dataos.errors import ErrorCode, PlatformError
from dataos.registry.operations.cast import CastOperation


def test_cast_succeeds_and_reports_type_change(orders_basic_df):
    op = CastOperation()
    result = op.execute(orders_basic_df, {"casts": [{"column": "order_date", "target_type": "Date"}]})

    assert str(result.output["order_date"].dtype) == "Date"
    tc = result.evidence["type_changes"][0]
    assert tc["from_type"] == "String"
    assert tc["to_type"] == "Date"
    assert tc["failed_count"] == 0


def test_cast_failure_is_captured_not_silently_nulled(orders_basic_df):
    # region is not numeric-parseable, so casting to Int64 must fail for every row.
    op = CastOperation()
    with pytest.raises(PlatformError) as excinfo:
        op.execute(orders_basic_df, {"casts": [{"column": "region", "target_type": "Int64"}]})

    assert excinfo.value.code == ErrorCode.DATA_QUALITY_BLOCK
    assert excinfo.value.evidence["failed_count"] == 5
    assert len(excinfo.value.evidence["failed_rows_sample"]) == 5


def test_cast_below_threshold_is_tolerated_and_evidenced():
    import polars as pl

    df = pl.DataFrame({"mixed": ["1", "2", "not_a_number", "4"]})
    op = CastOperation()
    result = op.execute(
        df,
        {"casts": [{"column": "mixed", "target_type": "Int64", "parseability_threshold": 0.5}]},
    )
    tc = result.evidence["type_changes"][0]
    assert tc["failed_count"] == 1
    assert tc["failed_rows_sample"] == [{"mixed": "not_a_number"}]

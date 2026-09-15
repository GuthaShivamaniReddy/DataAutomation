import pytest

from dataos.errors import ErrorCode, PlatformError
from dataos.registry.operations.select_filter import FilterCondition, SelectFilterOperation


def test_filter_by_equality(orders_basic_df):
    op = SelectFilterOperation()
    result = op.execute(orders_basic_df, {"filters": [{"column": "region", "op": "==", "value": "East"}]})

    assert result.row_impact.rows_in == 5
    assert result.row_impact.rows_out == 2
    assert set(result.output["region"].to_list()) == {"East"}
    assert result.evidence["value_mutation"] is False


def test_select_columns_subset(orders_basic_df):
    op = SelectFilterOperation()
    result = op.execute(orders_basic_df, {"columns": ["order_id", "region"]})
    assert result.output.columns == ["order_id", "region"]
    assert result.row_impact.rows_out == result.row_impact.rows_in


def test_missing_column_blocks_before_execution(orders_basic_df):
    op = SelectFilterOperation()
    with pytest.raises(PlatformError) as excinfo:
        op.execute(orders_basic_df, {"columns": ["does_not_exist"]})
    assert excinfo.value.code == ErrorCode.VALIDATION_FAIL


def test_filter_missing_value_for_comparison_op_blocks(orders_basic_df):
    op = SelectFilterOperation()
    with pytest.raises(PlatformError):
        op.execute(orders_basic_df, {"filters": [FilterCondition(column="region", op="==", value=None)]})

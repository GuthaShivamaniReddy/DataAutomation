import pytest

from dataos.errors import ErrorCode, PlatformError
from dataos.registry.operations.aggregate import AggregateOperation


def test_aggregate_sum_by_group_with_dual_computation_reconciliation(orders_basic_df):
    op = AggregateOperation()
    result = op.execute(
        orders_basic_df,
        {"group_by": ["region"], "metrics": [{"name": "total_amount", "column": "net_amount", "fn": "sum"}]},
    )
    assert result.evidence["group_count"] == 4  # East, West, North, South
    assert result.evidence["reconciliation"][0]["passed"] is True
    assert result.evidence["reconciliation"][0]["metric"] == "total_amount"


def test_aggregate_external_reconciliation_passes_within_tolerance(orders_basic_df):
    expected_total = float(orders_basic_df["net_amount"].sum())
    op = AggregateOperation()
    result = op.execute(
        orders_basic_df,
        {
            "metrics": [{"name": "total", "column": "net_amount", "fn": "sum"}],
            "reconcile_to": expected_total,
        },
    )
    assert result.evidence["external_reconciliation"]["passed"] is True


def test_aggregate_external_reconciliation_fails_outside_tolerance_blocks_release(orders_basic_df):
    op = AggregateOperation()
    with pytest.raises(PlatformError) as excinfo:
        op.execute(
            orders_basic_df,
            {
                "metrics": [{"name": "total", "column": "net_amount", "fn": "sum"}],
                "reconcile_to": 999999.0,
                "reconcile_tolerance": 0.01,
            },
        )
    assert excinfo.value.code == ErrorCode.RECONCILIATION_FAIL


def test_aggregate_missing_metric_column_blocks(orders_basic_df):
    op = AggregateOperation()
    with pytest.raises(PlatformError) as excinfo:
        op.execute(
            orders_basic_df,
            {"metrics": [{"name": "x", "column": "does_not_exist", "fn": "sum"}]},
        )
    assert excinfo.value.code == ErrorCode.VALIDATION_FAIL

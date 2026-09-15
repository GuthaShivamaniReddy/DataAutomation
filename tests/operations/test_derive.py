import polars as pl
import pytest

from dataos.errors import ErrorCode, PlatformError
from dataos.registry.operations.derive import DeriveOperation


def test_derive_arithmetic(orders_basic_df):
    op = DeriveOperation()
    result = op.execute(
        orders_basic_df,
        {
            "derivations": [
                {"output_column": "amount_x2", "op": "multiply", "inputs": ["net_amount", "net_amount"]}
            ]
        },
    )
    assert result.output["amount_x2"].to_list() == [v * v for v in orders_basic_df["net_amount"].to_list()]
    assert result.row_impact.rows_in == result.row_impact.rows_out == 5


def test_derive_null_policy_error_blocks_on_null_input(fixtures_dir):
    df = pl.read_csv(fixtures_dir / "orders_nulls.csv")
    op = DeriveOperation()

    with pytest.raises(PlatformError) as excinfo:
        op.execute(
            df,
            {
                "derivations": [
                    {
                        "output_column": "amount_doubled",
                        "op": "multiply",
                        "inputs": ["net_amount", "net_amount"],
                        "null_policy": "error",
                    }
                ]
            },
        )
    assert excinfo.value.code == ErrorCode.VALIDATION_FAIL
    assert excinfo.value.evidence["blocked_row_count"] == 1


def test_derive_null_policy_propagate_allows_nulls(fixtures_dir):
    df = pl.read_csv(fixtures_dir / "orders_nulls.csv")
    op = DeriveOperation()

    result = op.execute(
        df,
        {
            "derivations": [
                {
                    "output_column": "amount_doubled",
                    "op": "multiply",
                    "inputs": ["net_amount", "net_amount"],
                    "null_policy": "propagate",
                }
            ]
        },
    )
    d = result.evidence["derivations"][0]
    assert d["output_null_count"] == 1  # the null net_amount row propagates to a null output

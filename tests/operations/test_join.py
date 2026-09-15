import polars as pl
import pytest

from dataos.errors import ErrorCode, PlatformError
from dataos.registry.operations.join import JoinOperation


def test_one_to_one_join_reports_match_rate_and_multiplication():
    left = pl.DataFrame({"order_id": [1, 2, 3], "region": ["East", "West", "East"]})
    right = pl.DataFrame({"order_id": [1, 2], "customer_name": ["Alice", "Bob"]})
    op = JoinOperation()

    result = op.execute_join(
        left,
        right,
        {
            "left_keys": ["order_id"],
            "right_keys": ["order_id"],
            "how": "left",
            "expected_cardinality": "one_to_one",
        },
    )

    assert result.row_impact.rows_in == 3
    assert result.row_impact.rows_out == 3
    assert result.evidence["matched_row_count"] == 2
    assert result.evidence["unmatched_left_row_count"] == 1
    assert result.evidence["match_rate_left"] == pytest.approx(2 / 3)
    assert result.evidence["row_multiplication_factor"] == pytest.approx(1.0)


def test_many_to_many_rejected_without_opt_in():
    left = pl.DataFrame({"id": [1, 1, 2]})
    right = pl.DataFrame({"id": [1, 1, 2]})
    op = JoinOperation()

    with pytest.raises(PlatformError) as excinfo:
        op.execute_join(
            left,
            right,
            {
                "left_keys": ["id"],
                "right_keys": ["id"],
                "expected_cardinality": "many_to_many",
            },
        )

    assert excinfo.value.code == ErrorCode.VALIDATION_FAIL
    assert any("many_to_many" in v for v in excinfo.value.evidence["violations"])


def test_many_to_many_allowed_with_opt_in():
    left = pl.DataFrame({"id": [1, 1, 2]})
    right = pl.DataFrame({"id": [1, 1, 2]})
    op = JoinOperation()

    result = op.execute_join(
        left,
        right,
        {
            "left_keys": ["id"],
            "right_keys": ["id"],
            "expected_cardinality": "many_to_many",
            "allow_many_to_many": True,
        },
    )

    # id=1 fans out 2x2=4 rows, id=2 contributes 1 row -> 5 rows out.
    assert result.row_impact.rows_out == 5


def test_unexpected_duplicate_key_blocks_before_join():
    left = pl.DataFrame({"order_id": [1, 1, 2], "region": ["East", "East", "West"]})
    right = pl.DataFrame({"order_id": [1, 2], "customer_name": ["Alice", "Bob"]})
    op = JoinOperation()

    with pytest.raises(PlatformError) as excinfo:
        op.execute_join(
            left,
            right,
            {
                "left_keys": ["order_id"],
                "right_keys": ["order_id"],
                "expected_cardinality": "one_to_one",
            },
        )

    assert excinfo.value.code == ErrorCode.DATA_QUALITY_BLOCK
    assert excinfo.value.evidence["side"] == "left"
    assert excinfo.value.evidence["duplicate_group_count"] == 1


def test_missing_join_key_column_fails_preconditions():
    left = pl.DataFrame({"order_id": [1, 2]})
    right = pl.DataFrame({"customer_id": [1, 2]})
    op = JoinOperation()

    with pytest.raises(PlatformError) as excinfo:
        op.execute_join(
            left,
            right,
            {
                "left_keys": ["order_id"],
                "right_keys": ["order_id"],
                "expected_cardinality": "one_to_one",
            },
        )

    assert excinfo.value.code == ErrorCode.VALIDATION_FAIL
    assert any("right join key" in v for v in excinfo.value.evidence["violations"])


def test_key_dtype_mismatch_fails_preconditions():
    left = pl.DataFrame({"order_id": [1, 2]})
    right = pl.DataFrame({"order_id": ["1", "2"]})
    op = JoinOperation()

    with pytest.raises(PlatformError) as excinfo:
        op.execute_join(
            left,
            right,
            {
                "left_keys": ["order_id"],
                "right_keys": ["order_id"],
                "expected_cardinality": "one_to_one",
            },
        )

    assert excinfo.value.code == ErrorCode.VALIDATION_FAIL
    assert any("dtype mismatch" in v for v in excinfo.value.evidence["violations"])


def test_left_right_key_length_mismatch_is_a_validation_error():
    with pytest.raises(Exception):
        JoinOperation.Params(
            left_keys=["order_id", "region"],
            right_keys=["order_id"],
            expected_cardinality="one_to_one",
        )

import polars as pl
import pytest

from dataos.compiler.validation_rule_generator import CheckSpec
from dataos.errors import ErrorCode, PlatformError
from dataos.validation.executor import ValidationRuleExecutor


def test_null_rate_check_passes_and_fails_correctly():
    df = pl.DataFrame({"net_amount": [1.0, 2.0, None]})
    rules = [
        CheckSpec(
            check_id="null_rate:net_amount",
            stage="s1",
            predicate="null_rate(net_amount) <= 0.0",
            check_function="check_null_rate",
            check_args={"column": "net_amount", "max_rate": 0.0},
            severity="BLOCKING",
        )
    ]

    report = ValidationRuleExecutor().execute(rules, {"s1": df})

    assert report.passed is False
    assert report.results[0].check_id == "null_rate:net_amount"
    assert report.results[0].passed is False


def test_no_duplicate_rows_check():
    df = pl.DataFrame({"order_id": [1, 1, 2]})
    rules = [
        CheckSpec(
            check_id="unique:s1:order_id",
            stage="s1",
            predicate="no duplicate rows on ['order_id']",
            check_function="check_no_duplicate_rows",
            check_args={"subset": ["order_id"]},
        )
    ]

    report = ValidationRuleExecutor().execute(rules, {"s1": df})

    assert report.passed is False
    assert report.blocking_failures[0].observed == 2  # both rows of the order_id=1 group


def test_dual_computation_check_passes_for_a_correct_sum():
    df = pl.DataFrame({"net_amount": [10.0, 20.0, 30.0]})
    rules = [
        CheckSpec(
            check_id="reconciliation:revenue",
            stage="s1",
            predicate="dual_computation(sum, net_amount) within 0.01",
            check_function="check_dual_computation",
            check_args={"column": "net_amount", "agg": "sum", "tolerance": 0.01},
            tolerance=0.01,
        )
    ]

    report = ValidationRuleExecutor().execute(rules, {"s1": df})

    assert report.passed is True
    assert report.results[0].observed == 60.0


def test_structural_check_is_recorded_as_passing_without_a_dataframe():
    rules = [
        CheckSpec(
            check_id="join_cardinality:j1",
            stage="j1",
            predicate="join keys satisfy declared cardinality 'many_to_one'",
            check_function=None,
        )
    ]

    report = ValidationRuleExecutor().execute(rules, {})

    assert report.passed is True
    assert report.results[0].passed is True


def test_missing_dataframe_for_a_stage_fails_the_check():
    rules = [
        CheckSpec(
            check_id="null_rate:net_amount",
            stage="s1",
            predicate="null_rate(net_amount) <= 0.0",
            check_function="check_null_rate",
            check_args={"column": "net_amount", "max_rate": 0.0},
        )
    ]

    report = ValidationRuleExecutor().execute(rules, {})  # no "s1" dataframe supplied

    assert report.passed is False
    assert report.results[0].observed is None


def test_unsupported_multi_frame_check_function_raises():
    rules = [
        CheckSpec(
            check_id="conservation:net_amount",
            stage="s1",
            predicate="sum(net_amount) conserved",
            check_function="check_conservation",
            check_args={"column": "net_amount"},
        )
    ]

    with pytest.raises(PlatformError) as excinfo:
        ValidationRuleExecutor().execute(rules, {"s1": pl.DataFrame({"net_amount": [1.0]})})

    assert excinfo.value.code == ErrorCode.NO_SAFE_OPERATION

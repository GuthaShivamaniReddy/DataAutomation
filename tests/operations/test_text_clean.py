import polars as pl
import pytest

from dataos.errors import ErrorCode, PlatformError
from dataos.registry.operations.text_clean import TextCleanOperation


def test_trim_and_lower_normalize_without_collapsing_identity():
    df = pl.DataFrame({"email": ["  Alice@Example.com ", "Bob@example.com"]})
    op = TextCleanOperation()

    result = op.execute(df, {"normalize": [{"column": "email", "steps": ["trim", "lower"]}]})

    assert result.output["email"].to_list() == ["alice@example.com", "bob@example.com"]
    assert result.evidence["normalized_columns"][0]["distinct_before"] == 2
    assert result.evidence["normalized_columns"][0]["distinct_after"] == 2


def test_normalize_that_collapses_identity_is_blocked_without_explicit_opt_in():
    df = pl.DataFrame({"code": ["ABC", "abc"]})  # two distinct values today
    op = TextCleanOperation()

    with pytest.raises(PlatformError) as excinfo:
        op.execute(df, {"normalize": [{"column": "code", "steps": ["lower"]}]})

    assert excinfo.value.code == ErrorCode.DATA_QUALITY_BLOCK
    assert excinfo.value.evidence["distinct_before"] == 2
    assert excinfo.value.evidence["distinct_after"] == 1


def test_normalize_that_collapses_identity_succeeds_with_explicit_opt_in():
    df = pl.DataFrame({"code": ["ABC", "abc"]})
    op = TextCleanOperation()

    result = op.execute(
        df, {"normalize": [{"column": "code", "steps": ["lower"]}], "allow_identity_collapse": True}
    )

    assert result.output["code"].to_list() == ["abc", "abc"]


def test_collapse_whitespace_step():
    df = pl.DataFrame({"name": ["John   Smith"]})
    op = TextCleanOperation()

    result = op.execute(df, {"normalize": [{"column": "name", "steps": ["collapse_whitespace"]}]})

    assert result.output["name"].to_list() == ["John Smith"]


def test_title_case_step():
    df = pl.DataFrame({"name": ["john smith"]})
    op = TextCleanOperation()

    result = op.execute(df, {"normalize": [{"column": "name", "steps": ["title"]}]})

    assert result.output["name"].to_list() == ["John Smith"]


def test_regex_extract_adds_a_new_column_without_touching_the_source():
    df = pl.DataFrame({"sku": ["ITEM-1234-A", "ITEM-5678-B"]})
    op = TextCleanOperation()

    result = op.execute(
        df,
        {
            "regex_extract": [
                {"column": "sku", "pattern": r"ITEM-(\d+)-", "output_column": "sku_number", "group": 1}
            ]
        },
    )

    assert result.output["sku"].to_list() == ["ITEM-1234-A", "ITEM-5678-B"]  # untouched
    assert result.output["sku_number"].to_list() == ["1234", "5678"]
    assert result.evidence["extracted_columns"][0]["match_count"] == 2


def test_regex_extract_output_column_collision_is_a_precondition_violation():
    df = pl.DataFrame({"sku": ["ITEM-1234"]})
    op = TextCleanOperation()

    with pytest.raises(PlatformError) as excinfo:
        op.execute(
            df, {"regex_extract": [{"column": "sku", "pattern": r"(\d+)", "output_column": "sku", "group": 1}]}
        )

    assert excinfo.value.code == ErrorCode.VALIDATION_FAIL


def test_regex_replace_reports_changed_count():
    df = pl.DataFrame({"phone": ["(555) 123-4567", "555.987.6543"]})
    op = TextCleanOperation()

    result = op.execute(
        df, {"regex_replace": [{"column": "phone", "pattern": r"[^\d]", "replacement": ""}]}
    )

    assert result.output["phone"].to_list() == ["5551234567", "5559876543"]
    assert result.evidence["replaced_columns"][0]["changed_count"] == 2


def test_invalid_regex_pattern_is_a_precondition_violation():
    df = pl.DataFrame({"x": ["a"]})
    op = TextCleanOperation()

    with pytest.raises(PlatformError) as excinfo:
        op.execute(df, {"regex_replace": [{"column": "x", "pattern": "(unclosed", "replacement": ""}]})

    assert excinfo.value.code == ErrorCode.VALIDATION_FAIL


def test_missing_column_is_a_precondition_violation():
    df = pl.DataFrame({"x": ["a"]})
    op = TextCleanOperation()

    with pytest.raises(PlatformError) as excinfo:
        op.execute(df, {"normalize": [{"column": "does_not_exist", "steps": ["trim"]}]})

    assert excinfo.value.code == ErrorCode.VALIDATION_FAIL

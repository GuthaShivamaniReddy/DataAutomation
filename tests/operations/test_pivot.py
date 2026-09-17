import polars as pl
import pytest

from dataos.errors import ErrorCode, PlatformError
from dataos.registry.operations.pivot import PivotOperation


def test_pivot_reshapes_long_to_wide():
    df = pl.DataFrame(
        {
            "student": ["Cady", "Cady", "Karen", "Karen"],
            "subject": ["maths", "physics", "maths", "physics"],
            "score": [98, 99, 61, 58],
        }
    )
    op = PivotOperation()

    result = op.execute(df, {"index": ["student"], "on": "subject", "values": "score"})

    assert result.row_impact.rows_in == 4
    assert result.row_impact.rows_out == 2
    assert set(result.output.columns) == {"student", "maths", "physics"}
    row = result.output.filter(pl.col("student") == "Cady").to_dicts()[0]
    assert row["maths"] == 98
    assert row["physics"] == 99
    assert result.evidence["new_columns"] == ["maths", "physics"]


def test_pivot_with_duplicate_combination_and_no_aggregate_function_is_blocked():
    df = pl.DataFrame(
        {
            "student": ["Cady", "Cady"],
            "subject": ["maths", "maths"],  # same (student, subject) twice
            "score": [98, 100],
        }
    )
    op = PivotOperation()

    with pytest.raises(PlatformError) as excinfo:
        op.execute(df, {"index": ["student"], "on": "subject", "values": "score"})

    assert excinfo.value.code == ErrorCode.DATA_QUALITY_BLOCK
    assert excinfo.value.evidence["duplicate_group_count"] == 1


def test_pivot_with_duplicate_combination_and_explicit_aggregate_function_succeeds():
    df = pl.DataFrame(
        {
            "student": ["Cady", "Cady"],
            "subject": ["maths", "maths"],
            "score": [98, 100],
        }
    )
    op = PivotOperation()

    result = op.execute(
        df, {"index": ["student"], "on": "subject", "values": "score", "aggregate_function": "sum"}
    )

    assert result.output["maths"].to_list() == [198]
    assert result.evidence["aggregate_function"] == "sum"


def test_pivot_missing_column_is_a_precondition_violation():
    df = pl.DataFrame({"student": ["Cady"], "subject": ["maths"], "score": [98]})
    op = PivotOperation()

    with pytest.raises(PlatformError) as excinfo:
        op.execute(df, {"index": ["student"], "on": "subject", "values": "does_not_exist"})

    assert excinfo.value.code == ErrorCode.VALIDATION_FAIL

import polars as pl
import pytest

from dataos.errors import ErrorCode, PlatformError
from dataos.registry.operations.unpivot import UnpivotOperation


def test_unpivot_reshapes_wide_to_long():
    df = pl.DataFrame({"student": ["Cady", "Karen"], "maths": [98, 61], "physics": [99, 58]})
    op = UnpivotOperation()

    result = op.execute(df, {"index": ["student"], "on": ["maths", "physics"]})

    assert result.row_impact.rows_in == 2
    assert result.row_impact.rows_out == 4
    assert set(result.output.columns) == {"student", "variable", "value"}
    cady_maths = result.output.filter((pl.col("student") == "Cady") & (pl.col("variable") == "maths"))
    assert cady_maths["value"].to_list() == [98]


def test_unpivot_with_custom_names():
    df = pl.DataFrame({"student": ["Cady"], "maths": [98]})
    op = UnpivotOperation()

    result = op.execute(
        df, {"index": ["student"], "on": ["maths"], "variable_name": "subject", "value_name": "score"}
    )

    assert set(result.output.columns) == {"student", "subject", "score"}


def test_unpivot_missing_column_is_a_precondition_violation():
    df = pl.DataFrame({"student": ["Cady"], "maths": [98]})
    op = UnpivotOperation()

    with pytest.raises(PlatformError) as excinfo:
        op.execute(df, {"index": ["student"], "on": ["does_not_exist"]})

    assert excinfo.value.code == ErrorCode.VALIDATION_FAIL


def test_unpivot_name_collision_is_a_precondition_violation():
    df = pl.DataFrame({"student": ["Cady"], "maths": [98], "value": ["x"]})
    op = UnpivotOperation()

    with pytest.raises(PlatformError) as excinfo:
        op.execute(df, {"index": ["student"], "on": ["maths"]})

    assert excinfo.value.code == ErrorCode.VALIDATION_FAIL

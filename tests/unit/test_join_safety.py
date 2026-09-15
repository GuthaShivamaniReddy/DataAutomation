import polars as pl
import pytest

from dataos.validation.checks.join_safety import check_join_match_rate, check_join_multiplication_factor


def test_match_rate_reflects_unmatched_left_rows():
    left = pl.DataFrame({"id": [1, 2, 3, 4]})
    right = pl.DataFrame({"id": [1, 2]})

    result = check_join_match_rate(left, right, left_keys=["id"], right_keys=["id"], min_match_rate=0.9)

    assert result.observed == 0.5
    assert result.passed is False
    assert result.severity == "BLOCKING"


def test_match_rate_passes_when_threshold_met():
    left = pl.DataFrame({"id": [1, 2]})
    right = pl.DataFrame({"id": [1, 2, 3]})

    result = check_join_match_rate(left, right, left_keys=["id"], right_keys=["id"], min_match_rate=1.0)

    assert result.observed == 1.0
    assert result.passed is True


def test_multiplication_factor_flags_many_to_many_fanout():
    left = pl.DataFrame({"id": [1, 1, 2]})
    right = pl.DataFrame({"id": [1, 1, 2]})

    result = check_join_multiplication_factor(
        left, right, left_keys=["id"], right_keys=["id"], max_factor=1.5
    )

    # id=1 matches 2x2=4 rows, id=2 matches 1 row -> 5 rows out / 3 rows in
    assert result.observed == pytest.approx(5 / 3)
    assert result.passed is False

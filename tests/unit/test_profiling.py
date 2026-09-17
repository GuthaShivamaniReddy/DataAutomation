import polars as pl

from dataos.ingestion.profiling import profile_dataset


def test_profile_basic_orders(orders_basic_df):
    profile = profile_dataset(orders_basic_df)

    assert profile.row_count == 5
    assert profile.column_count == 5
    assert profile.duplicate_row_count == 0
    assert "order_id" in profile.candidate_key_columns

    order_id_profile = profile.column("order_id")
    assert order_id_profile.null_count == 0
    assert order_id_profile.is_candidate_key is True
    assert order_id_profile.distinct_count == 5


def test_profile_detects_nulls(fixtures_dir):
    df = pl.read_csv(fixtures_dir / "orders_nulls.csv")
    profile = profile_dataset(df)

    amount_profile = profile.column("net_amount")
    assert amount_profile.null_count == 1
    assert amount_profile.null_rate == 1 / 4

    customer_profile = profile.column("customer_id")
    assert customer_profile.null_count == 1
    assert customer_profile.is_candidate_key is False  # nulls disqualify candidate-key status


def test_profile_detects_duplicate_rows(fixtures_dir):
    df = pl.read_csv(fixtures_dir / "orders_duplicates.csv")
    profile = profile_dataset(df)

    # order_id 3001 appears twice with fully identical rows -> exactly 1 duplicate row.
    assert profile.duplicate_row_count == 1
    order_id_profile = profile.column("order_id")
    assert order_id_profile.is_candidate_key is False  # order_id repeats, not unique at this grain


def test_profile_detects_case_and_whitespace_inconsistency():
    df = pl.DataFrame({"region": ["East", "east ", "West"]})
    profile = profile_dataset(df)

    region_profile = profile.column("region")
    assert region_profile.distinct_count == 3
    assert region_profile.trimmed_lowercase_distinct_count == 2  # "East" and "east " fold together


def test_profile_reports_no_inconsistency_for_already_clean_text():
    df = pl.DataFrame({"region": ["East", "West"]})
    profile = profile_dataset(df)

    region_profile = profile.column("region")
    assert region_profile.distinct_count == region_profile.trimmed_lowercase_distinct_count == 2


def test_profile_trimmed_lowercase_distinct_count_is_none_for_non_string_columns():
    df = pl.DataFrame({"amount": [1.0, 2.0]})
    profile = profile_dataset(df)

    assert profile.column("amount").trimmed_lowercase_distinct_count is None


def test_profile_case_inconsistency_check_counts_nulls_consistently():
    df = pl.DataFrame({"region": ["East", None, "east "]})
    profile = profile_dataset(df)

    region_profile = profile.column("region")
    # 3 distinct including null ("East", null, "east "); trimmed/lowered folds
    # "East"/"east " together but null still counts as its own category on
    # both sides, so the comparison is apples-to-apples, not null-inflated.
    assert region_profile.distinct_count == 3
    assert region_profile.trimmed_lowercase_distinct_count == 2

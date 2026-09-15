import polars as pl

from dataos.registry.operations.deduplicate import DeduplicateOperation


def test_deduplicate_reports_groups_and_removed_rows(fixtures_dir):
    df = pl.read_csv(fixtures_dir / "orders_duplicates.csv")
    op = DeduplicateOperation()

    result = op.execute(df, {"keys": ["order_id"], "keep": "first"})

    # 5 rows in, 3 distinct order_ids (3001, 3002, 3003) -> 3 rows out.
    assert result.row_impact.rows_in == 5
    assert result.row_impact.rows_out == 3
    assert result.evidence["duplicate_group_count"] == 2  # order_id 3001 and 3003 each repeat
    assert result.evidence["removed_row_count"] == 2
    assert len(result.evidence["removed_rows_sample"]) == 2

    kept_ids = result.output["order_id"].to_list()
    assert sorted(kept_ids) == [3001, 3002, 3003]


def test_deduplicate_keep_last_with_order_by(fixtures_dir):
    df = pl.read_csv(fixtures_dir / "orders_duplicates.csv")
    op = DeduplicateOperation()

    result = op.execute(df, {"keys": ["order_id"], "keep": "last", "order_by": "order_date"})

    kept_3003 = result.output.filter(pl.col("order_id") == 3003)
    # order_date for 3003 rows are 2024-03-03 and 2024-03-04; keep="last" after sorting by
    # order_date must keep the 2024-03-04 row (net_amount 35.00), not silently pick either.
    assert kept_3003["order_date"].to_list() == ["2024-03-04"]
    assert kept_3003["net_amount"].to_list() == [35.00]

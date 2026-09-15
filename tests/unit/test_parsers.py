import pytest

from dataos.errors import ErrorCode, PlatformError
from dataos.ingestion.parsers import parse_csv


def test_parse_csv_well_formed_reports_no_rejections(fixtures_dir):
    df, report = parse_csv(fixtures_dir / "orders_basic.csv", mode="strict")
    assert df.height == 5
    assert report.rows_rejected == 0
    assert report.is_partial is False


def test_parse_csv_strict_mode_blocks_on_malformed_rows(fixtures_dir):
    with pytest.raises(PlatformError) as excinfo:
        parse_csv(fixtures_dir / "orders_malformed.csv", mode="strict")
    assert excinfo.value.code == ErrorCode.DATA_QUALITY_BLOCK
    assert excinfo.value.evidence["rows_rejected"] == 2


def test_parse_csv_permissive_mode_excludes_but_reports_rejections(fixtures_dir):
    df, report = parse_csv(fixtures_dir / "orders_malformed.csv", mode="permissive")
    # 4 total data rows, 2 well-formed (2001, 2004), 2 malformed (2002 extra field, 2003 missing field)
    assert report.rows_seen == 4
    assert report.rows_rejected == 2
    assert report.rows_parsed == 2
    assert df.height == 2
    assert report.is_partial is True
    # Never silently discarded - a sample of the exact rejected rows must be present.
    assert len(report.rejected_sample) == 2
    order_ids_kept = set(df["order_id"].to_list())
    assert order_ids_kept == {2001, 2004}

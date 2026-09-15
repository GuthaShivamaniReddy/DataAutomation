from dataos.ingestion.hashing import sha256_file
from dataos.registry.operations.export import ExportOperation


def test_export_csv_produces_checksum_and_schema_manifest(orders_basic_df, tmp_path):
    op = ExportOperation()
    dest = tmp_path / "out" / "orders.csv"

    result = op.execute(orders_basic_df, {"format": "csv", "destination_path": str(dest)})

    assert dest.exists()
    assert result.evidence["row_count"] == orders_basic_df.height
    assert result.evidence["checksum"] == sha256_file(dest)
    assert set(result.evidence["schema_manifest"]) == set(orders_basic_df.columns)


def test_export_parquet_roundtrip(orders_basic_df, tmp_path):
    import polars as pl

    op = ExportOperation()
    dest = tmp_path / "orders.parquet"
    op.execute(orders_basic_df, {"format": "parquet", "destination_path": str(dest)})

    reloaded = pl.read_parquet(dest)
    assert reloaded.height == orders_basic_df.height
    assert reloaded.columns == orders_basic_df.columns

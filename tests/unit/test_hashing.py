from dataos.ingestion.hashing import file_size, sha256_file
from dataos.ingestion.snapshot import ingest_file


def test_sha256_is_stable_for_identical_content(fixtures_dir):
    path = fixtures_dir / "orders_basic.csv"
    assert sha256_file(path) == sha256_file(path)


def test_sha256_differs_for_different_content(fixtures_dir):
    a = sha256_file(fixtures_dir / "orders_basic.csv")
    b = sha256_file(fixtures_dir / "orders_duplicates.csv")
    assert a != b


def test_file_size_matches_stat(fixtures_dir):
    path = fixtures_dir / "orders_basic.csv"
    assert file_size(path) == path.stat().st_size


def test_ingest_file_creates_content_addressed_immutable_snapshot(fixtures_dir, tmp_storage_root):
    source = fixtures_dir / "orders_basic.csv"
    original_bytes = source.read_bytes()

    version = ingest_file(source, storage_root=tmp_storage_root)

    assert version.snapshot_id == sha256_file(source)
    assert version.sha256 == version.snapshot_id

    stored_path = tmp_storage_root / "raw" / version.snapshot_id / source.name
    assert stored_path.exists()
    assert stored_path.read_bytes() == original_bytes

    # Source file itself must be completely untouched.
    assert source.read_bytes() == original_bytes


def test_ingest_file_is_idempotent_for_identical_content(fixtures_dir, tmp_storage_root):
    source = fixtures_dir / "orders_basic.csv"
    v1 = ingest_file(source, storage_root=tmp_storage_root)
    v2 = ingest_file(source, storage_root=tmp_storage_root)
    assert v1.snapshot_id == v2.snapshot_id
    assert v1.storage_path == v2.storage_path

import polars as pl
import pytest

from dataos.api.store import SessionStore
from dataos.contracts.requirement_contract import RequirementContract, Source
from dataos.errors import PlatformError
from dataos.ingestion.snapshot import ingest_file


def test_add_and_reload_dataset_round_trips_the_frame(tmp_path, fixtures_dir):
    version = ingest_file(fixtures_dir / "orders_basic.csv", storage_root=tmp_path / "raw")
    df = pl.read_csv(fixtures_dir / "orders_basic.csv")
    store = SessionStore(root=tmp_path / "store")

    record = store.add_dataset(name="orders", df=df, version=version)

    assert record.dataset_id == version.snapshot_id
    reloaded = store.load_dataset_frame(record.dataset_id)
    assert reloaded.equals(df)


def test_dataset_index_survives_a_fresh_store_instance(tmp_path, fixtures_dir):
    version = ingest_file(fixtures_dir / "orders_basic.csv", storage_root=tmp_path / "raw")
    df = pl.read_csv(fixtures_dir / "orders_basic.csv")
    store_root = tmp_path / "store"
    SessionStore(root=store_root).add_dataset(name="orders", df=df, version=version)

    reopened = SessionStore(root=store_root)

    assert len(reopened.list_datasets()) == 1
    assert reopened.get_dataset(version.snapshot_id).name == "orders"


def test_require_dataset_raises_for_an_unknown_id(tmp_path):
    store = SessionStore(root=tmp_path / "store")

    with pytest.raises(PlatformError):
        store.require_dataset("does-not-exist")


def test_contract_lifecycle_add_update_link_run(tmp_path):
    store = SessionStore(root=tmp_path / "store")
    contract = RequirementContract(objective="x", sources=[Source(name="orders")])

    record = store.add_contract(contract=contract, dataset_ids_by_source_name={"orders": "abc123"})
    assert store.get_contract(record.contract_id) is not None

    updated_contract = contract.model_copy(update={"objective": "y"})
    store.update_contract(record.contract_id, contract=updated_contract, run_id="run-1")

    linked = store.get_contract_for_run("run-1")
    assert linked.contract_id == record.contract_id
    assert linked.contract.objective == "y"
    assert linked.run_id == "run-1"


def test_get_contract_for_run_raises_when_unlinked(tmp_path):
    store = SessionStore(root=tmp_path / "store")

    with pytest.raises(PlatformError):
        store.get_contract_for_run("no-such-run")

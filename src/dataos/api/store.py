"""Local session store for the web/desktop API layer.

Not part of the AI Prompt Library or Blueprint spec - this is plain
plumbing so the API has somewhere to keep the two things the compiler
pipeline itself never persists: uploaded datasets (a `DatasetVersion` +
`DatasetProfile` + the parsed dataframe, cached to parquet) and
in-progress `RequirementContract`s (which `RunStore`/`ArtifactStore`
only start tracking once a run actually begins). Everything downstream
of a started run still goes through the real, already-tested
`RunStore`/`ArtifactStore` - this store never duplicates that.

Single local user, no concurrency control beyond "last write wins" -
matches this app's current scope. The on-disk index is plain JSON, not
a database, since it holds a handful of records, not rows.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import polars as pl
from pydantic import BaseModel, Field

from dataos.contracts.requirement_contract import RequirementContract
from dataos.errors import ErrorCode, PlatformError
from dataos.ingestion.profiling import DatasetProfile, profile_dataset
from dataos.ingestion.snapshot import DatasetVersion
from dataos.workflow.dsl import Workflow
from dataos.workflow.store import now_iso

DEFAULT_STORE_ROOT = Path(".dataos_store") / "api"


class DatasetRecord(BaseModel):
    dataset_id: str
    name: str
    version: DatasetVersion
    profile: DatasetProfile
    parquet_path: str
    created_at: str


class ContractRecord(BaseModel):
    contract_id: str
    contract: RequirementContract
    dataset_ids_by_source_name: dict[str, str] = Field(default_factory=dict)
    workflow: Workflow | None = None
    run_id: str | None = None
    created_at: str
    updated_at: str


class _Index(BaseModel):
    datasets: dict[str, DatasetRecord] = Field(default_factory=dict)
    contracts: dict[str, ContractRecord] = Field(default_factory=dict)
    run_to_contract: dict[str, str] = Field(default_factory=dict)


class SessionStore:
    def __init__(self, root: str | Path = DEFAULT_STORE_ROOT) -> None:
        self._root = Path(root)
        self._datasets_dir = self._root / "datasets"
        self._datasets_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._root / "index.json"
        self._index = self._load_index()

    def _load_index(self) -> _Index:
        if self._index_path.exists():
            return _Index.model_validate_json(self._index_path.read_text(encoding="utf-8"))
        return _Index()

    def _save(self) -> None:
        self._index_path.write_text(self._index.model_dump_json(indent=2), encoding="utf-8")

    # ---- datasets ----

    def add_dataset(self, *, name: str, df: pl.DataFrame, version: DatasetVersion) -> DatasetRecord:
        """`version.snapshot_id` (content-addressed) is reused as the
        dataset_id, so re-uploading identical bytes updates the same
        record rather than creating a duplicate."""
        dataset_id = version.snapshot_id
        parquet_path = self._datasets_dir / f"{dataset_id}.parquet"
        df.write_parquet(parquet_path)

        record = DatasetRecord(
            dataset_id=dataset_id,
            name=name,
            version=version,
            profile=profile_dataset(df),
            parquet_path=str(parquet_path),
            created_at=now_iso(),
        )
        self._index.datasets[dataset_id] = record
        self._save()
        return record

    def get_dataset(self, dataset_id: str) -> DatasetRecord | None:
        return self._index.datasets.get(dataset_id)

    def require_dataset(self, dataset_id: str) -> DatasetRecord:
        record = self.get_dataset(dataset_id)
        if record is None:
            raise PlatformError(ErrorCode.SCHEMA_MISSING, f"no dataset with id '{dataset_id}'")
        return record

    def list_datasets(self) -> list[DatasetRecord]:
        return list(self._index.datasets.values())

    def load_dataset_frame(self, dataset_id: str) -> pl.DataFrame:
        record = self.require_dataset(dataset_id)
        return pl.read_parquet(record.parquet_path)

    # ---- requirement contracts ----

    def add_contract(
        self, *, contract: RequirementContract, dataset_ids_by_source_name: dict[str, str]
    ) -> ContractRecord:
        contract_id = str(uuid.uuid4())
        record = ContractRecord(
            contract_id=contract_id,
            contract=contract,
            dataset_ids_by_source_name=dataset_ids_by_source_name,
            created_at=now_iso(),
            updated_at=now_iso(),
        )
        self._index.contracts[contract_id] = record
        self._save()
        return record

    def get_contract(self, contract_id: str) -> ContractRecord | None:
        return self._index.contracts.get(contract_id)

    def require_contract(self, contract_id: str) -> ContractRecord:
        record = self.get_contract(contract_id)
        if record is None:
            raise PlatformError(ErrorCode.SCHEMA_MISSING, f"no requirement contract with id '{contract_id}'")
        return record

    def list_contracts(self) -> list[ContractRecord]:
        return list(self._index.contracts.values())

    def update_contract(
        self,
        contract_id: str,
        *,
        contract: RequirementContract | None = None,
        workflow: Workflow | None = None,
        run_id: str | None = None,
    ) -> ContractRecord:
        record = self.require_contract(contract_id)
        updates: dict = {"updated_at": now_iso()}
        if contract is not None:
            updates["contract"] = contract
        if workflow is not None:
            updates["workflow"] = workflow
        if run_id is not None:
            updates["run_id"] = run_id
        updated = record.model_copy(update=updates)
        self._index.contracts[contract_id] = updated
        if run_id is not None:
            self._index.run_to_contract[run_id] = contract_id
        self._save()
        return updated

    def get_contract_for_run(self, run_id: str) -> ContractRecord:
        contract_id = self._index.run_to_contract.get(run_id)
        if contract_id is None:
            raise PlatformError(ErrorCode.SCHEMA_MISSING, f"no requirement contract is linked to run '{run_id}'")
        return self.require_contract(contract_id)

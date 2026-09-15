import polars as pl
import pytest

from dataos.ingestion.snapshot import ingest_file
from dataos.registry.operations.aggregate import AggregateOperation
from dataos.registry.operations.deduplicate import DeduplicateOperation
from dataos.registry.operations.select_filter import SelectFilterOperation
from dataos.registry.registry import OperationRegistry
from dataos.workflow.artifact_store import ArtifactStore
from dataos.workflow.dsl import Workflow, WorkflowStep
from dataos.workflow.orchestrator import WorkflowOrchestrator
from dataos.workflow.state_machine import RunState
from dataos.workflow.store import RunStore


def _step_status(store: RunStore, run_id: str, step_id: str) -> str:
    step = store.get_step_run(run_id, step_id)
    assert step is not None
    return step.status


def _pipeline_workflow(dedup_params: dict | None = None) -> Workflow:
    return Workflow(
        workflow_version=1,
        requirement_contract_id="rc_test",
        sources=["orders"],
        steps=[
            WorkflowStep(id="s1", operation_id="select_filter", operation_version="1.0", inputs=["source:orders"]),
            WorkflowStep(
                id="s2",
                operation_id="deduplicate",
                operation_version="1.0",
                inputs=["s1"],
                params=dedup_params if dedup_params is not None else {"keys": ["order_id"], "keep": "first"},
            ),
            WorkflowStep(
                id="s3",
                operation_id="aggregate",
                operation_version="1.0",
                inputs=["s2"],
                params={
                    "group_by": ["region"],
                    "metrics": [{"name": "total_amount", "column": "net_amount", "fn": "sum"}],
                },
            ),
        ],
    )


class CountingSelectFilter(SelectFilterOperation):
    def __init__(self) -> None:
        self.call_count = 0

    def run(self, df, params):
        self.call_count += 1
        return super().run(df, params)


class CrashOnceOperation(DeduplicateOperation):
    def __init__(self) -> None:
        self._raised = False

    def run(self, df, params):
        if not self._raised:
            self._raised = True
            raise RuntimeError("simulated worker crash mid-step")
        return super().run(df, params)


def test_end_to_end_pipeline_reaches_verifying(fixtures_dir, tmp_path):
    df = pl.read_csv(fixtures_dir / "orders_duplicates.csv")
    version = ingest_file(fixtures_dir / "orders_duplicates.csv", storage_root=tmp_path / "dataos_store")

    registry = OperationRegistry()
    for op_cls in (SelectFilterOperation, DeduplicateOperation, AggregateOperation):
        registry.register(op_cls())

    run_store = RunStore(tmp_path / "runs.db")
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    orchestrator = WorkflowOrchestrator(registry, run_store, artifact_store)

    workflow = _pipeline_workflow()
    run_id = orchestrator.start_run(workflow, source_frames={"orders": df}, source_versions={"orders": version})
    run = orchestrator.run(run_id, workflow)

    assert run.state == RunState.VERIFYING
    assert run.source_snapshot_ids["orders"] == version.snapshot_id

    for step_id in ("s1", "s2", "s3"):
        step = run_store.get_step_run(run_id, step_id)
        assert step is not None
        assert step.status == "COMPLETED"
        assert step.evidence  # non-empty: real evidence, not a bare pass/fail

    final_df = artifact_store.load_by_ids(run_id, "s3")
    result = dict(zip(final_df["region"].to_list(), final_df["total_amount"].to_list()))
    # orders_duplicates.csv after dedup(keep=first): 3001 East 10.00, 3002 West 20.00, 3003 East 30.00
    assert result == {"East": 40.0, "West": 20.0}


def test_step_failure_quarantines_run_and_skips_downstream(fixtures_dir, tmp_path):
    df = pl.read_csv(fixtures_dir / "orders_duplicates.csv")
    version = ingest_file(fixtures_dir / "orders_duplicates.csv", storage_root=tmp_path / "dataos_store")

    registry = OperationRegistry()
    for op_cls in (SelectFilterOperation, DeduplicateOperation, AggregateOperation):
        registry.register(op_cls())

    run_store = RunStore(tmp_path / "runs.db")
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    orchestrator = WorkflowOrchestrator(registry, run_store, artifact_store)

    # s2 references a key column that does not exist -> its precondition check fails.
    workflow = _pipeline_workflow(dedup_params={"keys": ["does_not_exist"]})
    run_id = orchestrator.start_run(workflow, source_frames={"orders": df}, source_versions={"orders": version})
    run = orchestrator.run(run_id, workflow)

    assert run.state == RunState.QUARANTINED
    assert _step_status(run_store, run_id, "s1") == "COMPLETED"
    assert _step_status(run_store, run_id, "s2") == "FAILED"
    assert run_store.get_step_run(run_id, "s3") is None  # never attempted


def test_crash_mid_step_then_resume_does_not_reexecute_completed_steps(fixtures_dir, tmp_path):
    df = pl.read_csv(fixtures_dir / "orders_duplicates.csv")
    version = ingest_file(fixtures_dir / "orders_duplicates.csv", storage_root=tmp_path / "dataos_store")

    counting_filter = CountingSelectFilter()
    crash_once_dedup = CrashOnceOperation()

    registry = OperationRegistry()
    registry.register(counting_filter)
    registry.register(crash_once_dedup)
    registry.register(AggregateOperation())

    db_path = tmp_path / "runs.db"
    artifacts_path = tmp_path / "artifacts"

    run_store = RunStore(db_path)
    artifact_store = ArtifactStore(artifacts_path)
    orchestrator = WorkflowOrchestrator(registry, run_store, artifact_store)

    workflow = _pipeline_workflow()
    run_id = orchestrator.start_run(workflow, source_frames={"orders": df}, source_versions={"orders": version})

    with pytest.raises(RuntimeError):
        orchestrator.run(run_id, workflow)  # "crashes" partway through s2

    assert counting_filter.call_count == 1
    mid_crash_run = run_store.get_run(run_id)
    assert mid_crash_run is not None
    assert mid_crash_run.state == RunState.RUNNING  # a real crash never reaches QUARANTINED
    assert _step_status(run_store, run_id, "s1") == "COMPLETED"
    assert _step_status(run_store, run_id, "s2") == "RUNNING"  # interrupted mid-flight
    assert run_store.get_step_run(run_id, "s3") is None

    # Simulate a process restart: brand new store instances over the same on-disk paths.
    resumed_run_store = RunStore(db_path)
    resumed_artifact_store = ArtifactStore(artifacts_path)
    resumed_orchestrator = WorkflowOrchestrator(registry, resumed_run_store, resumed_artifact_store)

    final_run = resumed_orchestrator.run(run_id, workflow)

    assert final_run.state == RunState.VERIFYING
    assert counting_filter.call_count == 1  # s1 was NOT re-executed on resume
    assert _step_status(resumed_run_store, run_id, "s2") == "COMPLETED"
    assert _step_status(resumed_run_store, run_id, "s3") == "COMPLETED"


def test_run_is_a_no_op_once_already_verifying(fixtures_dir, tmp_path):
    df = pl.read_csv(fixtures_dir / "orders_duplicates.csv")
    version = ingest_file(fixtures_dir / "orders_duplicates.csv", storage_root=tmp_path / "dataos_store")

    counting_filter = CountingSelectFilter()
    registry = OperationRegistry()
    registry.register(counting_filter)
    registry.register(DeduplicateOperation())
    registry.register(AggregateOperation())

    run_store = RunStore(tmp_path / "runs.db")
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    orchestrator = WorkflowOrchestrator(registry, run_store, artifact_store)

    workflow = _pipeline_workflow()
    run_id = orchestrator.start_run(workflow, source_frames={"orders": df}, source_versions={"orders": version})
    orchestrator.run(run_id, workflow)
    assert counting_filter.call_count == 1

    again = orchestrator.run(run_id, workflow)
    assert again.state == RunState.VERIFYING
    assert counting_filter.call_count == 1  # calling run() again on a finished run touches nothing

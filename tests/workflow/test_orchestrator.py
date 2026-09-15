import polars as pl
import pytest

from dataos.compiler.explanation_agent import ExplanationAgent
from dataos.compiler.workflow_planner import WorkflowPlanner
from dataos.contracts.requirement_contract import (
    DefinitionStatus,
    Metric,
    RequirementContract,
    RequirementStatus,
    Source,
)
from dataos.errors import ErrorCode, PlatformError
from dataos.ingestion.snapshot import ingest_file
from dataos.llm.deterministic import DeterministicLLMClient
from dataos.registry.operations.aggregate import AggregateOperation
from dataos.registry.operations.deduplicate import DeduplicateOperation
from dataos.registry.operations.export import ExportOperation
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


def test_start_run_from_contract_plans_and_starts_a_run(fixtures_dir, tmp_path):
    df = pl.read_csv(fixtures_dir / "orders_basic.csv")
    version = ingest_file(fixtures_dir / "orders_basic.csv", storage_root=tmp_path / "dataos_store")

    registry = OperationRegistry()
    registry.register(AggregateOperation())

    run_store = RunStore(tmp_path / "runs.db")
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    planner = WorkflowPlanner(DeterministicLLMClient(), registry)
    orchestrator = WorkflowOrchestrator(registry, run_store, artifact_store, planner)

    contract = RequirementContract(
        objective="show total net amount by region",
        sources=[Source(name="orders")],
        metrics=[
            Metric(
                name="total_net_amount",
                formula="sum(orders.net_amount)",
                definition_status=DefinitionStatus.GOVERNED,
            )
        ],
        group_by=["region"],
        status=RequirementStatus.APPROVED,
    )

    planned = orchestrator.start_run_from_contract(
        contract=contract,
        contract_id="rc_orchestrator_test",
        source_frames={"orders": df},
        source_versions={"orders": version},
    )

    assert planned.planner_output.workflow.requirement_contract_id == "rc_orchestrator_test"
    assert planned.planner_output.envelope.status == "OK"
    assert planned.policy_decision.decision == "PROCEED"

    run = orchestrator.run(planned.run_id, planned.planner_output.workflow)
    assert run.state == RunState.VERIFYING

    step_id = planned.planner_output.workflow.steps[0].id
    final_df = artifact_store.load_by_ids(planned.run_id, step_id)
    result = dict(zip(final_df["region"].to_list(), final_df["total_net_amount"].to_list()))
    assert result == {"East": 250.75, "West": 200.00, "North": 75.00, "South": 300.10}


def test_start_run_from_contract_without_planner_raises(tmp_path):
    registry = OperationRegistry()
    registry.register(AggregateOperation())

    run_store = RunStore(tmp_path / "runs.db")
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    orchestrator = WorkflowOrchestrator(registry, run_store, artifact_store)  # no planner

    contract = RequirementContract(
        objective="show total net amount",
        sources=[Source(name="orders")],
        metrics=[
            Metric(name="m", formula="sum(orders.net_amount)", definition_status=DefinitionStatus.GOVERNED)
        ],
        status=RequirementStatus.APPROVED,
    )

    with pytest.raises(PlatformError) as excinfo:
        orchestrator.start_run_from_contract(
            contract=contract,
            contract_id="rc_no_planner",
            source_frames={},
            source_versions={},
        )

    assert excinfo.value.code == ErrorCode.NO_SAFE_OPERATION


def _export_workflow(dest_path: str) -> Workflow:
    return Workflow(
        workflow_version=1,
        requirement_contract_id="rc_export_test",
        sources=["orders"],
        steps=[
            WorkflowStep(
                id="s1",
                operation_id="export",
                operation_version="1.0",
                inputs=["source:orders"],
                params={"format": "csv", "destination_path": dest_path},
            )
        ],
    )


def test_start_run_blocks_export_step_without_approval(fixtures_dir, tmp_path):
    # Policy/Approval Gate: `export` writes to an external destination path
    # (Constitution rule 7 / Section 29), so it always needs an explicit
    # approval token - even when the caller builds the Workflow directly
    # and skips start_run_from_contract entirely.
    df = pl.read_csv(fixtures_dir / "orders_basic.csv")
    version = ingest_file(fixtures_dir / "orders_basic.csv", storage_root=tmp_path / "dataos_store")

    registry = OperationRegistry()
    registry.register(ExportOperation())

    run_store = RunStore(tmp_path / "runs.db")
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    orchestrator = WorkflowOrchestrator(registry, run_store, artifact_store)

    dest = tmp_path / "out.csv"
    workflow = _export_workflow(str(dest))

    with pytest.raises(PlatformError) as excinfo:
        orchestrator.start_run(workflow, source_frames={"orders": df}, source_versions={"orders": version})

    assert excinfo.value.code == ErrorCode.POLICY_DENIED
    assert excinfo.value.evidence["required_approvals"] == ["export"]
    assert not dest.exists()  # blocked before the run - and its export step - ever ran


def test_start_run_with_export_step_proceeds_when_approved(fixtures_dir, tmp_path):
    df = pl.read_csv(fixtures_dir / "orders_basic.csv")
    version = ingest_file(fixtures_dir / "orders_basic.csv", storage_root=tmp_path / "dataos_store")

    registry = OperationRegistry()
    registry.register(ExportOperation())

    run_store = RunStore(tmp_path / "runs.db")
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    orchestrator = WorkflowOrchestrator(registry, run_store, artifact_store)

    dest = tmp_path / "out.csv"
    workflow = _export_workflow(str(dest))

    run_id = orchestrator.start_run(
        workflow,
        source_frames={"orders": df},
        source_versions={"orders": version},
        granted_approvals=frozenset({"export"}),
    )
    run = orchestrator.run(run_id, workflow)

    assert run.state == RunState.VERIFYING
    assert dest.exists()


def test_release_after_full_pipeline_reaches_released(fixtures_dir, tmp_path):
    df = pl.read_csv(fixtures_dir / "orders_basic.csv")
    version = ingest_file(fixtures_dir / "orders_basic.csv", storage_root=tmp_path / "dataos_store")

    registry = OperationRegistry()
    registry.register(AggregateOperation())

    run_store = RunStore(tmp_path / "runs.db")
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    planner = WorkflowPlanner(DeterministicLLMClient(), registry)
    orchestrator = WorkflowOrchestrator(registry, run_store, artifact_store, planner)

    contract = RequirementContract(
        objective="show order count",
        sources=[Source(name="orders")],
        metrics=[
            Metric(name="order_count", formula="count(orders.order_id)", definition_status=DefinitionStatus.GOVERNED)
        ],
        status=RequirementStatus.APPROVED,
    )

    planned = orchestrator.start_run_from_contract(
        contract=contract,
        contract_id="rc_release_test",
        source_frames={"orders": df},
        source_versions={"orders": version},
    )
    orchestrator.run(planned.run_id, planned.planner_output.workflow)

    result = orchestrator.release(planned.run_id, planned.planner_output.workflow, contract)

    assert result.run.state == RunState.RELEASED
    assert result.verifier_result.verdict == "PASS"
    assert result.release_decision.decision == "RELEASE"

    # Calling release() again is a safe no-op: terminal state, no second
    # transition attempt (which state_machine.transition would reject).
    again = orchestrator.release(planned.run_id, planned.planner_output.workflow, contract)
    assert again.run.state == RunState.RELEASED


def test_release_quarantines_when_a_declared_metric_is_never_computed(fixtures_dir, tmp_path):
    df = pl.read_csv(fixtures_dir / "orders_basic.csv")
    version = ingest_file(fixtures_dir / "orders_basic.csv", storage_root=tmp_path / "dataos_store")

    registry = OperationRegistry()
    registry.register(AggregateOperation())

    run_store = RunStore(tmp_path / "runs.db")
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    orchestrator = WorkflowOrchestrator(registry, run_store, artifact_store)

    contract = RequirementContract(
        objective="show order count and total amount",
        sources=[Source(name="orders")],
        metrics=[
            Metric(name="order_count", formula="count(orders.order_id)", definition_status=DefinitionStatus.GOVERNED),
            Metric(name="total_amount", formula="sum(orders.net_amount)", definition_status=DefinitionStatus.GOVERNED),
        ],
        status=RequirementStatus.APPROVED,
    )

    # The plan only ever computes order_count - total_amount is silently dropped.
    workflow = Workflow(
        workflow_version=1,
        requirement_contract_id="rc_quarantine_test",
        sources=["orders"],
        steps=[
            WorkflowStep(
                id="s1",
                operation_id="aggregate",
                operation_version="1.0",
                inputs=["source:orders"],
                params={"metrics": [{"name": "order_count", "column": "order_id", "fn": "count"}]},
                requirement_refs=["order_count"],
            )
        ],
    )

    run_id = orchestrator.start_run(workflow, source_frames={"orders": df}, source_versions={"orders": version})
    orchestrator.run(run_id, workflow)

    result = orchestrator.release(run_id, workflow, contract)

    assert result.run.state == RunState.QUARANTINED
    assert result.verifier_result.verdict == "FAIL"
    assert any("total_amount" in d for d in result.verifier_result.defects)
    assert result.release_decision.decision == "QUARANTINE"


def _released_run(fixtures_dir, tmp_path):
    df = pl.read_csv(fixtures_dir / "orders_basic.csv")
    version = ingest_file(fixtures_dir / "orders_basic.csv", storage_root=tmp_path / "dataos_store")

    registry = OperationRegistry()
    registry.register(AggregateOperation())

    run_store = RunStore(tmp_path / "runs.db")
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    planner = WorkflowPlanner(DeterministicLLMClient(), registry)
    explainer = ExplanationAgent(DeterministicLLMClient())
    orchestrator = WorkflowOrchestrator(registry, run_store, artifact_store, planner, explainer=explainer)

    contract = RequirementContract(
        objective="show order count",
        sources=[Source(name="orders")],
        metrics=[
            Metric(name="order_count", formula="count(orders.order_id)", definition_status=DefinitionStatus.GOVERNED)
        ],
        status=RequirementStatus.APPROVED,
    )

    planned = orchestrator.start_run_from_contract(
        contract=contract,
        contract_id="rc_explain_test",
        source_frames={"orders": df},
        source_versions={"orders": version},
    )
    orchestrator.run(planned.run_id, planned.planner_output.workflow)
    orchestrator.release(planned.run_id, planned.planner_output.workflow, contract)

    return orchestrator, planned, contract


def test_explain_after_release_produces_a_grounded_finding(fixtures_dir, tmp_path):
    orchestrator, planned, contract = _released_run(fixtures_dir, tmp_path)

    output = orchestrator.explain(planned.run_id, planned.planner_output.workflow, contract)

    assert output.explanation.findings
    finding = output.explanation.findings[0]
    assert finding.statement == "order_count is 5."
    assert "order_count" in finding.evidence_refs
    assert output.envelope.status == "OK"


def test_explain_refuses_a_run_that_is_not_released(fixtures_dir, tmp_path):
    df = pl.read_csv(fixtures_dir / "orders_basic.csv")
    version = ingest_file(fixtures_dir / "orders_basic.csv", storage_root=tmp_path / "dataos_store")

    registry = OperationRegistry()
    registry.register(AggregateOperation())

    run_store = RunStore(tmp_path / "runs.db")
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    planner = WorkflowPlanner(DeterministicLLMClient(), registry)
    explainer = ExplanationAgent(DeterministicLLMClient())
    orchestrator = WorkflowOrchestrator(registry, run_store, artifact_store, planner, explainer=explainer)

    contract = RequirementContract(
        objective="show order count",
        sources=[Source(name="orders")],
        metrics=[
            Metric(name="order_count", formula="count(orders.order_id)", definition_status=DefinitionStatus.GOVERNED)
        ],
        status=RequirementStatus.APPROVED,
    )

    planned = orchestrator.start_run_from_contract(
        contract=contract,
        contract_id="rc_not_released_test",
        source_frames={"orders": df},
        source_versions={"orders": version},
    )
    orchestrator.run(planned.run_id, planned.planner_output.workflow)  # reaches VERIFYING, never released

    with pytest.raises(PlatformError) as excinfo:
        orchestrator.explain(planned.run_id, planned.planner_output.workflow, contract)

    assert excinfo.value.code == ErrorCode.VALIDATION_FAIL


def test_explain_without_explainer_raises(fixtures_dir, tmp_path):
    _orchestrator, planned, contract = _released_run(fixtures_dir, tmp_path)

    # A second orchestrator over the same on-disk run/artifact stores, but
    # built without an ExplanationAgent - e.g. a caller with no LLM
    # configured for this stage.
    run_store = RunStore(tmp_path / "runs.db")
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    registry = OperationRegistry()
    registry.register(AggregateOperation())
    orchestrator_without_explainer = WorkflowOrchestrator(registry, run_store, artifact_store)

    with pytest.raises(PlatformError) as excinfo:
        orchestrator_without_explainer.explain(planned.run_id, planned.planner_output.workflow, contract)

    assert excinfo.value.code == ErrorCode.NO_SAFE_OPERATION


def test_generate_validation_checks_for_a_planned_sum_metric_workflow(tmp_path):
    registry = OperationRegistry()
    registry.register(AggregateOperation())

    run_store = RunStore(tmp_path / "runs.db")
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    planner = WorkflowPlanner(DeterministicLLMClient(), registry)
    orchestrator = WorkflowOrchestrator(registry, run_store, artifact_store, planner)

    contract = RequirementContract(
        objective="show total net amount",
        sources=[Source(name="orders")],
        metrics=[
            Metric(name="total_amount", formula="sum(orders.net_amount)", definition_status=DefinitionStatus.GOVERNED)
        ],
        tolerances={"total_amount": 2.5},
        status=RequirementStatus.APPROVED,
    )

    planned_workflow = planner.plan(contract=contract, contract_id="rc_checks_test").workflow
    checks = orchestrator.generate_validation_checks(planned_workflow, contract)

    assert len(checks) == 1
    check = checks[0]
    assert check.check_id == "reconciliation:total_amount"
    assert check.stage == "source:orders"  # pre-aggregation input, not the group-by step's own output
    assert check.check_function == "check_dual_computation"
    assert check.check_args == {"column": "net_amount", "agg": "sum", "tolerance": 2.5}
    assert check.tolerance == 2.5


def test_release_runs_the_generated_reconciliation_check_and_releases_on_pass(fixtures_dir, tmp_path):
    df = pl.read_csv(fixtures_dir / "orders_basic.csv")
    version = ingest_file(fixtures_dir / "orders_basic.csv", storage_root=tmp_path / "dataos_store")

    registry = OperationRegistry()
    registry.register(AggregateOperation())

    run_store = RunStore(tmp_path / "runs.db")
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    planner = WorkflowPlanner(DeterministicLLMClient(), registry)
    orchestrator = WorkflowOrchestrator(registry, run_store, artifact_store, planner)

    contract = RequirementContract(
        objective="show total net amount",
        sources=[Source(name="orders")],
        metrics=[
            Metric(name="total_amount", formula="sum(orders.net_amount)", definition_status=DefinitionStatus.GOVERNED)
        ],
        status=RequirementStatus.APPROVED,
    )

    planned = orchestrator.start_run_from_contract(
        contract=contract,
        contract_id="rc_reconciliation_test",
        source_frames={"orders": df},
        source_versions={"orders": version},
    )
    orchestrator.run(planned.run_id, planned.planner_output.workflow)

    result = orchestrator.release(planned.run_id, planned.planner_output.workflow, contract)

    assert result.run.state == RunState.RELEASED
    reconciliation_result = next(r for r in result.validation_report.results if r.check_id == "reconciliation:total_amount")
    assert reconciliation_result.passed is True
    assert reconciliation_result.observed == 825.85  # sum(net_amount) in orders_basic.csv


def test_release_quarantines_when_a_generated_null_policy_check_fails(fixtures_dir, tmp_path):
    # orders_nulls.csv has a null net_amount - a contract declaring
    # null_policy="error" for it must actually block release, not just
    # exist as an unexecuted manifest entry.
    df = pl.read_csv(fixtures_dir / "orders_nulls.csv")
    version = ingest_file(fixtures_dir / "orders_nulls.csv", storage_root=tmp_path / "dataos_store")

    registry = OperationRegistry()
    registry.register(SelectFilterOperation())

    run_store = RunStore(tmp_path / "runs.db")
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    orchestrator = WorkflowOrchestrator(registry, run_store, artifact_store)

    contract = RequirementContract(
        objective="show orders",
        sources=[Source(name="orders")],
        null_policy={"net_amount": "error"},
        status=RequirementStatus.APPROVED,
    )
    workflow = Workflow(
        workflow_version=1,
        requirement_contract_id="rc_null_check_test",
        sources=["orders"],
        steps=[WorkflowStep(id="s1", operation_id="select_filter", operation_version="1.0", inputs=["source:orders"])],
    )

    run_id = orchestrator.start_run(workflow, source_frames={"orders": df}, source_versions={"orders": version})
    orchestrator.run(run_id, workflow)

    result = orchestrator.release(run_id, workflow, contract)

    assert result.validation_report.passed is False
    null_check = next(r for r in result.validation_report.results if r.check_id == "null_rate:net_amount")
    assert null_check.passed is False
    assert result.run.state == RunState.QUARANTINED
    assert result.release_decision.decision == "QUARANTINE"
    assert any("blocking validation" in r for r in result.release_decision.reason_codes)
    assert "null_rate:net_amount" in result.release_decision.required_remediation

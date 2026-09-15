from dataos.compiler.connector_planner import ConnectorWritePlanner
from dataos.ingestion.hashing import sha256_file
from dataos.llm.deterministic import DeterministicLLMClient
from dataos.workflow.dsl import Workflow, WorkflowStep
from dataos.workflow.store import StepRunRecord


def _planner() -> ConnectorWritePlanner:
    return ConnectorWritePlanner(DeterministicLLMClient())


def _workflow(steps: list[WorkflowStep]) -> Workflow:
    return Workflow(workflow_version=1, requirement_contract_id="rc_1", sources=["orders"], steps=steps)


def test_pure_transform_workflow_has_no_external_action_plan():
    workflow = _workflow(
        [WorkflowStep(id="s1", operation_id="aggregate", operation_version="1.0", inputs=["source:orders"])]
    )
    plans = _planner().plan(workflow)
    assert plans == []


def test_export_step_produces_a_write_plan(tmp_path):
    dest = tmp_path / "out.csv"
    workflow = _workflow(
        [
            WorkflowStep(
                id="s1",
                operation_id="export",
                operation_version="1.0",
                inputs=["source:orders"],
                params={"format": "csv", "destination_path": str(dest)},
            )
        ]
    )
    plans = _planner().plan(workflow)

    assert len(plans) == 1
    plan = plans[0]
    assert plan.step_id == "s1"
    assert plan.provider == "local_filesystem"
    assert plan.action == "WRITE"
    assert plan.resource == str(dest)
    assert plan.approval_required is True
    assert plan.approval_token == "export"
    assert plan.risk_level == "HIGH"
    assert plan.prechecks
    assert plan.postchecks
    assert str(dest) in plan.scope


def test_model_narrative_cannot_remove_a_hardcoded_precheck():
    from dataos.compiler.raw_connector_plan import RawConnectorPlan, RawExternalAction
    from dataos.llm.client import LLMClient

    class _EmptyPrechecksClient(LLMClient):
        model_id = "fake/tries-to-drop-safety-checks"

        def complete_structured(self, *, system_prompt, user_prompt, response_model):
            return RawConnectorPlan(
                actions=[RawExternalAction(step_id="s1", scope="a made-up scope", prechecks=[], postchecks=[])]
            )

    dest = "/tmp/out.csv"
    workflow = _workflow(
        [
            WorkflowStep(
                id="s1",
                operation_id="export",
                operation_version="1.0",
                inputs=["source:orders"],
                params={"format": "csv", "destination_path": dest},
            )
        ]
    )
    plan = ConnectorWritePlanner(_EmptyPrechecksClient()).plan(workflow)[0]

    # Even though the model proposed empty prechecks/postchecks, the
    # hardcoded safety floor from _CONNECTOR_SPECS still appears.
    assert any("traversal" in p for p in plan.prechecks)
    assert any("checksum" in p for p in plan.postchecks)
    # But the model's own scope text was still used - it just could not weaken safety.
    assert plan.scope == "a made-up scope"


def test_verify_passes_when_written_file_matches_recorded_checksum(tmp_path):
    dest = tmp_path / "out.csv"
    dest.write_text("a,b\n1,2\n", encoding="utf-8")
    checksum = sha256_file(dest)

    workflow = _workflow(
        [
            WorkflowStep(
                id="s1",
                operation_id="export",
                operation_version="1.0",
                inputs=["source:orders"],
                params={"format": "csv", "destination_path": str(dest)},
            )
        ]
    )
    planner = _planner()
    plans = planner.plan(workflow)

    step_runs = {
        "s1": StepRunRecord(
            run_id="run_1",
            step_id="s1",
            status="COMPLETED",
            evidence={
                "destination_path": str(dest),
                "checksum": checksum,
                "row_count": 1,
                "schema_manifest": {"a": "Int64", "b": "Int64"},
            },
        )
    }

    results = planner.verify(plans, step_runs)

    assert len(results) == 1
    assert results[0].verified is True
    assert results[0].issues == []


def test_verify_fails_when_file_was_modified_after_export(tmp_path):
    dest = tmp_path / "out.csv"
    dest.write_text("a,b\n1,2\n", encoding="utf-8")
    stale_checksum = sha256_file(dest)
    dest.write_text("a,b\n1,999\n", encoding="utf-8")  # tampered after the fact

    workflow = _workflow(
        [
            WorkflowStep(
                id="s1",
                operation_id="export",
                operation_version="1.0",
                inputs=["source:orders"],
                params={"format": "csv", "destination_path": str(dest)},
            )
        ]
    )
    planner = _planner()
    plans = planner.plan(workflow)

    step_runs = {
        "s1": StepRunRecord(
            run_id="run_1",
            step_id="s1",
            status="COMPLETED",
            evidence={
                "destination_path": str(dest),
                "checksum": stale_checksum,
                "row_count": 1,
                "schema_manifest": {"a": "Int64", "b": "Int64"},
            },
        )
    }

    results = planner.verify(plans, step_runs)

    assert results[0].verified is False
    assert any("does not match" in issue for issue in results[0].issues)


def test_verify_fails_when_file_is_missing(tmp_path):
    dest = tmp_path / "never_written.csv"
    workflow = _workflow(
        [
            WorkflowStep(
                id="s1",
                operation_id="export",
                operation_version="1.0",
                inputs=["source:orders"],
                params={"format": "csv", "destination_path": str(dest)},
            )
        ]
    )
    planner = _planner()
    plans = planner.plan(workflow)

    step_runs = {
        "s1": StepRunRecord(
            run_id="run_1",
            step_id="s1",
            status="COMPLETED",
            evidence={"destination_path": str(dest), "checksum": "deadbeef", "row_count": 1, "schema_manifest": {}},
        )
    }

    results = planner.verify(plans, step_runs)

    assert results[0].verified is False
    assert any("does not exist on disk" in issue for issue in results[0].issues)


def test_verify_fails_when_step_did_not_complete(tmp_path):
    workflow = _workflow(
        [
            WorkflowStep(
                id="s1",
                operation_id="export",
                operation_version="1.0",
                inputs=["source:orders"],
                params={"format": "csv", "destination_path": str(tmp_path / "out.csv")},
            )
        ]
    )
    planner = _planner()
    plans = planner.plan(workflow)

    results = planner.verify(plans, {})  # no step run recorded at all

    assert results[0].verified is False
    assert any("did not complete" in issue for issue in results[0].issues)

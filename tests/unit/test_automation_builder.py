from dataos.compiler.automation_builder import AutomationWorkflowBuilder
from dataos.llm.deterministic import DeterministicLLMClient
from dataos.workflow.dsl import ReleasePolicy, Workflow, WorkflowStep


def _workflow(**kwargs) -> Workflow:
    defaults = dict(
        workflow_version=3,
        requirement_contract_id="rc_48291",
        sources=["orders"],
        steps=[
            WorkflowStep(id="s1", operation_id="select_filter", operation_version="1.0", inputs=["source:orders"]),
            WorkflowStep(id="s2", operation_id="aggregate", operation_version="2.0", inputs=["s1"]),
        ],
    )
    defaults.update(kwargs)
    return Workflow(**defaults)


def test_version_pins_are_read_from_the_real_workflow():
    spec = AutomationWorkflowBuilder().build(
        automation_id="auto_1", workflow=_workflow(), trigger={"type": "schedule", "cron": "0 6 * * *"}
    )

    assert spec.version_pins.workflow_version == 3
    assert spec.version_pins.requirement_contract_id == "rc_48291"
    assert spec.version_pins.operation_versions == {"select_filter": "1.0", "aggregate": "2.0"}


def test_trigger_is_never_invented_only_echoed_from_the_caller():
    trigger = {"type": "schedule", "cron": "0 6 * * *", "timezone": "America/New_York"}
    spec = AutomationWorkflowBuilder().build(automation_id="auto_1", workflow=_workflow(), trigger=trigger)
    assert spec.trigger == trigger


def test_idempotency_key_is_stable_for_the_same_automation_and_version_pins():
    spec1 = AutomationWorkflowBuilder().build(automation_id="auto_1", workflow=_workflow(), trigger={})
    spec2 = AutomationWorkflowBuilder().build(automation_id="auto_1", workflow=_workflow(), trigger={})
    assert spec1.idempotency_key == spec2.idempotency_key


def test_idempotency_key_changes_when_workflow_version_changes():
    spec1 = AutomationWorkflowBuilder().build(automation_id="auto_1", workflow=_workflow(), trigger={})
    spec2 = AutomationWorkflowBuilder().build(automation_id="auto_1", workflow=_workflow(workflow_version=4), trigger={})
    assert spec1.idempotency_key != spec2.idempotency_key


def test_approval_gates_include_write_capable_step_operations():
    workflow = _workflow(
        steps=[WorkflowStep(id="s1", operation_id="export", operation_version="1.0", inputs=["source:orders"])],
        approval_gates=["finance_sign_off"],
    )
    spec = AutomationWorkflowBuilder().build(automation_id="auto_1", workflow=workflow, trigger={})

    assert "export" in spec.approval_gates
    assert "finance_sign_off" in spec.approval_gates


def test_validation_and_failure_policy_reflect_the_workflows_own_release_policy():
    workflow = _workflow(release_policy=ReleasePolicy(require_all_acceptance_tests=False, on_failure="stop"))
    spec = AutomationWorkflowBuilder().build(automation_id="auto_1", workflow=workflow, trigger={})

    assert "on_failure=stop" in spec.validation_policy
    assert "require_all_acceptance_tests=False" in spec.validation_policy
    assert "QUARANTINE" in spec.failure_policy


def test_notifications_retry_and_rollback_are_caller_supplied():
    spec = AutomationWorkflowBuilder().build(
        automation_id="auto_1",
        workflow=_workflow(),
        trigger={},
        notifications=["slack:#data-alerts"],
        retry_policy="retry once after 5 minutes",
        rollback_policy="none - append-only export",
        sla="99% of runs complete within 30 minutes",
    )

    assert spec.notifications == ["slack:#data-alerts"]
    assert spec.retry_policy == "retry once after 5 minutes"
    assert spec.rollback_policy == "none - append-only export"
    assert spec.sla == "99% of runs complete within 30 minutes"


def test_narrative_is_none_without_an_llm_client():
    spec = AutomationWorkflowBuilder().build(automation_id="auto_1", workflow=_workflow(), trigger={})
    assert spec.narrative is None


def test_narrative_is_populated_without_changing_version_pins_when_llm_client_supplied():
    builder = AutomationWorkflowBuilder(DeterministicLLMClient())
    spec = builder.build(automation_id="auto_1", workflow=_workflow(), trigger={"type": "schedule"})

    assert spec.narrative is not None
    assert spec.version_pins.workflow_version == 3  # the narrative never changes the pins

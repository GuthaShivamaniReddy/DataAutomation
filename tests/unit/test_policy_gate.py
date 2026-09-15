from dataos.compiler.policy_gate import PolicyGate
from dataos.contracts.requirement_contract import RequirementContract, Source
from dataos.workflow.dsl import Workflow, WorkflowStep


def _workflow(steps: list[WorkflowStep], **kwargs) -> Workflow:
    return Workflow(
        workflow_version=1,
        requirement_contract_id="rc_1",
        sources=["orders"],
        steps=steps,
        **kwargs,
    )


def test_pure_transform_workflow_proceeds_with_no_approvals():
    contract = RequirementContract(objective="x", sources=[Source(name="orders")])
    workflow = _workflow(
        [WorkflowStep(id="s1", operation_id="aggregate", operation_version="1.0", inputs=["source:orders"])]
    )

    decision = PolicyGate().evaluate(contract=contract, workflow=workflow)

    assert decision.decision == "PROCEED"
    assert decision.risk_level == "LOW"
    assert decision.required_approvals == []


def test_export_step_requires_approval_and_blocks_without_it():
    contract = RequirementContract(objective="x", sources=[Source(name="orders")])
    workflow = _workflow(
        [WorkflowStep(id="s1", operation_id="export", operation_version="1.0", inputs=["source:orders"])]
    )

    decision = PolicyGate().evaluate(contract=contract, workflow=workflow)

    assert decision.decision == "BLOCKED"
    assert decision.risk_level == "HIGH"
    assert decision.required_approvals == ["export"]
    assert any("export" in r for r in decision.blocking_reasons)


def test_export_step_proceeds_once_approval_is_granted():
    contract = RequirementContract(objective="x", sources=[Source(name="orders")])
    workflow = _workflow(
        [WorkflowStep(id="s1", operation_id="export", operation_version="1.0", inputs=["source:orders"])]
    )

    decision = PolicyGate().evaluate(contract=contract, workflow=workflow, granted_approvals=frozenset({"export"}))

    assert decision.decision == "PROCEED"
    assert decision.required_approvals == ["export"]
    assert decision.blocking_reasons == []


def test_declared_side_effect_requires_its_own_approval_token():
    contract = RequirementContract(
        objective="x", sources=[Source(name="orders")], side_effects=["send_email_to_customer"]
    )
    workflow = _workflow(
        [WorkflowStep(id="s1", operation_id="aggregate", operation_version="1.0", inputs=["source:orders"])]
    )

    decision = PolicyGate().evaluate(contract=contract, workflow=workflow)

    assert decision.decision == "BLOCKED"
    assert decision.risk_level == "HIGH"
    assert "send_email_to_customer" in decision.required_approvals


def test_prohibited_action_blocks_even_with_matching_approval_granted():
    contract = RequirementContract(
        objective="x",
        sources=[Source(name="orders")],
        prohibited_actions=["export"],
    )
    workflow = _workflow(
        [WorkflowStep(id="s1", operation_id="export", operation_version="1.0", inputs=["source:orders"])]
    )

    decision = PolicyGate().evaluate(contract=contract, workflow=workflow, granted_approvals=frozenset({"export"}))

    assert decision.decision == "BLOCKED"
    assert decision.risk_level == "CRITICAL"
    assert any("prohibited_action" in r for r in decision.blocking_reasons)


def test_workflow_declared_approval_gate_is_required():
    contract = RequirementContract(objective="x", sources=[Source(name="orders")])
    workflow = _workflow(
        [WorkflowStep(id="s1", operation_id="aggregate", operation_version="1.0", inputs=["source:orders"])],
        approval_gates=["finance_sign_off"],
    )

    decision = PolicyGate().evaluate(contract=contract, workflow=workflow)

    assert decision.decision == "BLOCKED"
    assert "finance_sign_off" in decision.required_approvals

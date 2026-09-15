import pytest

from dataos.compiler.raw_plan import RawPlan, RawPlanStep
from dataos.compiler.workflow_planner import WorkflowPlanner
from dataos.contracts.requirement_contract import (
    DefinitionStatus,
    Metric,
    RequirementContract,
    RequirementStatus,
    Source,
)
from dataos.errors import ErrorCode, PlatformError
from dataos.llm.client import LLMClient
from dataos.llm.deterministic import DeterministicLLMClient
from dataos.registry.registry import default_registry


def _approved_contract() -> RequirementContract:
    return RequirementContract(
        objective="show order count",
        sources=[Source(name="orders")],
        metrics=[
            Metric(
                name="order_count",
                formula="count(orders.order_id)",
                definition_status=DefinitionStatus.GOVERNED,
            )
        ],
        status=RequirementStatus.APPROVED,
    )


def test_planner_builds_single_aggregate_step_from_governed_metric():
    contract = _approved_contract()
    assert contract.ready_for_planning is True

    planner = WorkflowPlanner(DeterministicLLMClient(), default_registry)
    output = planner.plan(contract=contract, contract_id="rc_1")

    assert output.workflow.requirement_contract_id == "rc_1"
    assert output.workflow.sources == ["orders"]
    assert len(output.workflow.steps) == 1

    step = output.workflow.steps[0]
    assert step.operation_id == "aggregate"
    assert step.operation_version == "1.0"
    assert step.inputs == ["source:orders"]
    assert step.params["metrics"] == [{"name": "order_count", "column": "order_id", "fn": "count"}]
    assert output.envelope.status == "OK"
    assert output.envelope.result["plan_id"]


def test_planner_refuses_a_contract_that_is_not_ready_for_planning():
    contract = RequirementContract(objective="show revenue", sources=[])  # DRAFT, no sources -> blocked
    assert contract.ready_for_planning is False

    planner = WorkflowPlanner(DeterministicLLMClient(), default_registry)
    with pytest.raises(PlatformError) as excinfo:
        planner.plan(contract=contract, contract_id="rc_2")

    assert excinfo.value.code == ErrorCode.VALIDATION_FAIL


class _FakeLLMClient(LLMClient):
    """Returns a fixed RawPlan referencing an operation that does not exist,
    to exercise the registry-validation path independently of the
    DeterministicLLMClient's own (narrower) planning heuristic."""

    model_id = "fake/unregistered-op-planner"

    def complete_structured(self, *, system_prompt, user_prompt, response_model):
        return RawPlan(
            plan_id="p1",
            steps=[
                RawPlanStep(
                    id="s1",
                    type="AGGREGATE",
                    operation_id="does_not_exist",
                    operation_version="1.0",
                    inputs=["source:orders"],
                )
            ],
        )


def test_planner_blocks_on_unregistered_operation():
    contract = _approved_contract()
    planner = WorkflowPlanner(_FakeLLMClient(), default_registry)

    with pytest.raises(PlatformError) as excinfo:
        planner.plan(contract=contract, contract_id="rc_3")

    assert excinfo.value.code == ErrorCode.NO_SAFE_OPERATION
    assert excinfo.value.evidence["unresolved_steps"][0]["operation_id"] == "does_not_exist"

import pytest

from dataos.errors import ErrorCode, PlatformError
from dataos.workflow.dsl import Workflow, WorkflowStep


def _step(id_, operation_id, inputs) -> WorkflowStep:
    return WorkflowStep(id=id_, operation_id=operation_id, operation_version="1.0", inputs=inputs)


def test_valid_linear_workflow_execution_order():
    workflow = Workflow(
        workflow_version=1,
        requirement_contract_id="rc_1",
        sources=["orders"],
        steps=[
            _step("s1", "select_filter", ["source:orders"]),
            _step("s2", "deduplicate", ["s1"]),
            _step("s3", "aggregate", ["s2"]),
        ],
    )
    workflow.validate_dag()
    assert workflow.execution_order() == ["s1", "s2", "s3"]


def test_duplicate_step_ids_rejected():
    workflow = Workflow(
        workflow_version=1,
        requirement_contract_id="rc_1",
        sources=["orders"],
        steps=[
            _step("s1", "select_filter", ["source:orders"]),
            _step("s1", "deduplicate", ["source:orders"]),
        ],
    )
    with pytest.raises(PlatformError) as excinfo:
        workflow.validate_dag()
    assert excinfo.value.code == ErrorCode.VALIDATION_FAIL


def test_unknown_input_reference_rejected():
    workflow = Workflow(
        workflow_version=1,
        requirement_contract_id="rc_1",
        sources=["orders"],
        steps=[_step("s1", "select_filter", ["source:does_not_exist"])],
    )
    with pytest.raises(PlatformError) as excinfo:
        workflow.validate_dag()
    assert excinfo.value.code == ErrorCode.SCHEMA_MISSING


def test_cycle_rejected():
    workflow = Workflow(
        workflow_version=1,
        requirement_contract_id="rc_1",
        sources=["orders"],
        steps=[
            _step("s1", "select_filter", ["s2"]),
            _step("s2", "deduplicate", ["s1"]),
        ],
    )
    with pytest.raises(PlatformError) as excinfo:
        workflow.validate_dag()
    assert excinfo.value.code == ErrorCode.VALIDATION_FAIL
    assert "cycle" in excinfo.value.reason.lower()


def test_step_by_id_lookup():
    workflow = Workflow(
        workflow_version=1,
        requirement_contract_id="rc_1",
        sources=["orders"],
        steps=[_step("s1", "select_filter", ["source:orders"])],
    )
    assert workflow.step_by_id("s1").operation_id == "select_filter"
    with pytest.raises(PlatformError):
        workflow.step_by_id("does_not_exist")


def test_step_id_cannot_use_reserved_source_prefix():
    with pytest.raises(ValueError):
        _step("source:sneaky", "select_filter", ["source:orders"])

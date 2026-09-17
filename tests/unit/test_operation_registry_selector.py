from dataos.compiler.operation_registry_selector import OperationRegistrySelector, StepRequest
from dataos.registry.operations.aggregate import AggregateOperation
from dataos.registry.operations.export import ExportOperation
from dataos.registry.operations.join import JoinOperation
from dataos.registry.operations.select_filter import SelectFilterOperation
from dataos.registry.registry import OperationRegistry


def _registry_with(*op_classes) -> OperationRegistry:
    registry = OperationRegistry()
    for cls in op_classes:
        registry.register(cls())
    return registry


def test_step_type_resolves_to_the_registered_operation():
    registry = _registry_with(SelectFilterOperation)
    result = OperationRegistrySelector(registry).select([StepRequest(step_id="s1", step_type="FILTER")])

    assert result.unresolved == []
    selection = result.selections[0]
    assert selection.step_id == "s1"
    assert selection.operation_id == "select_filter"
    assert selection.version == "1.0"
    assert selection.certification == "registered:select_filter@1.0"
    assert "columns" in selection.required_params


def test_required_params_reflects_which_fields_have_no_default():
    registry = _registry_with(ExportOperation)
    result = OperationRegistrySelector(registry).select([StepRequest(step_id="s1", step_type="EXPORT")])

    selection = result.selections[0]
    # ExportOperation.Params.destination_path has no default -> required.
    assert selection.required_params.get("destination_path") is True


def test_step_type_with_no_mapped_operation_is_unresolved():
    registry = _registry_with(SelectFilterOperation)
    result = OperationRegistrySelector(registry).select([StepRequest(step_id="s1", step_type="PROFILE")])

    assert result.selections == []
    assert result.unresolved[0].step_id == "s1"
    assert result.unresolved[0].step_type == "PROFILE"


def test_mapped_operation_not_registered_is_unresolved():
    registry = OperationRegistry()  # nothing registered at all
    result = OperationRegistrySelector(registry).select([StepRequest(step_id="s1", step_type="AGGREGATE")])

    assert result.selections == []
    assert result.unresolved[0].step_id == "s1"
    assert "no registered operation" in result.unresolved[0].reason


def test_explicit_operation_id_override_is_honored():
    registry = _registry_with(JoinOperation)
    result = OperationRegistrySelector(registry).select(
        [StepRequest(step_id="s1", step_type="JOIN", operation_id="join", operation_version="1.0")]
    )

    assert result.selections[0].operation_id == "join"
    assert "amplify/duplicate" in result.selections[0].reason


def test_multiple_steps_mix_resolved_and_unresolved():
    registry = _registry_with(SelectFilterOperation, AggregateOperation)
    result = OperationRegistrySelector(registry).select(
        [
            StepRequest(step_id="s1", step_type="FILTER"),
            StepRequest(step_id="s2", step_type="AGGREGATE"),
            StepRequest(step_id="s3", step_type="JOIN"),  # not registered
        ]
    )

    assert {s.step_id for s in result.selections} == {"s1", "s2"}
    assert {u.step_id for u in result.unresolved} == {"s3"}


def test_narrative_is_none_without_an_llm_client():
    registry = _registry_with(SelectFilterOperation)
    result = OperationRegistrySelector(registry).select([StepRequest(step_id="s1", step_type="FILTER")])

    assert result.narrative is None


def test_narrative_is_populated_without_changing_selections_when_llm_client_supplied():
    from dataos.llm.deterministic import DeterministicLLMClient

    registry = _registry_with(SelectFilterOperation)
    selector = OperationRegistrySelector(registry, llm_client=DeterministicLLMClient())
    result = selector.select([StepRequest(step_id="s1", step_type="FILTER")])

    assert result.narrative is not None
    assert result.selections[0].operation_id == "select_filter"

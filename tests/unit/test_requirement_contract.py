from dataos.contracts.requirement_contract import (
    Clarification,
    DefinitionStatus,
    Metric,
    RequirementContract,
    RequirementStatus,
    Source,
)


def _base_kwargs():
    return dict(
        objective="monthly_net_revenue_by_region",
        sources=[Source(name="orders"), Source(name="refunds")],
    )


def test_fully_resolved_contract_can_be_approved():
    contract = RequirementContract(
        **_base_kwargs(),
        metrics=[
            Metric(
                name="revenue",
                formula="sum(orders.net_amount) - sum(refunds.amount)",
                definition_status=DefinitionStatus.GOVERNED,
            )
        ],
        status=RequirementStatus.APPROVED,
    )
    assert contract.status == RequirementStatus.APPROVED
    assert contract.ready_for_planning is True
    assert contract.blocking_reasons == []


def test_ambiguous_metric_blocks_regardless_of_requested_status():
    contract = RequirementContract(
        **_base_kwargs(),
        metrics=[
            Metric(name="revenue", definition_status=DefinitionStatus.AMBIGUOUS),
        ],
        status=RequirementStatus.APPROVED,  # caller tries to force approval
    )
    # The "no guessing" rule overrides the caller's requested status.
    assert contract.status == RequirementStatus.BLOCKED
    assert contract.ready_for_planning is False
    assert any("revenue" in r for r in contract.blocking_reasons)


def test_missing_metric_definition_blocks():
    contract = RequirementContract(
        **_base_kwargs(),
        metrics=[Metric(name="churn_rate")],  # definition_status defaults to MISSING
    )
    assert contract.status == RequirementStatus.BLOCKED
    assert contract.ready_for_planning is False


def test_unresolved_clarification_blocks_even_with_no_ambiguous_metrics():
    contract = RequirementContract(
        **_base_kwargs(),
        status=RequirementStatus.APPROVED,
        clarifications=[
            Clarification(
                issue="ambiguous revenue field",
                why_material="gross vs net changes the result",
                question="Which revenue field should be used?",
            )
        ],
    )
    assert contract.status == RequirementStatus.BLOCKED
    assert contract.ready_for_planning is False


def test_no_sources_blocks():
    contract = RequirementContract(objective="anything", status=RequirementStatus.APPROVED)
    assert contract.status == RequirementStatus.BLOCKED
    assert contract.ready_for_planning is False


def test_draft_status_never_ready_even_if_fully_resolved():
    contract = RequirementContract(
        **_base_kwargs(),
        metrics=[Metric(name="revenue", definition_status=DefinitionStatus.EXPLICIT)],
        status=RequirementStatus.DRAFT,
    )
    assert contract.ready_for_planning is False

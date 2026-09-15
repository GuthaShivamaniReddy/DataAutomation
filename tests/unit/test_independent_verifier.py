from dataos.compiler.independent_verifier import IndependentVerifier
from dataos.contracts.requirement_contract import DefinitionStatus, Metric, RequirementContract, Source
from dataos.workflow.dsl import Workflow, WorkflowStep
from dataos.workflow.state_machine import RunState
from dataos.workflow.store import RunRecord, StepRunRecord


def _contract(**kwargs) -> RequirementContract:
    defaults = dict(
        objective="x",
        sources=[Source(name="orders")],
        metrics=[
            Metric(name="order_count", formula="count(orders.order_id)", definition_status=DefinitionStatus.GOVERNED)
        ],
    )
    defaults.update(kwargs)
    return RequirementContract(**defaults)


def _workflow(**kwargs) -> Workflow:
    defaults = dict(
        workflow_version=1,
        requirement_contract_id="rc_1",
        sources=["orders"],
        steps=[
            WorkflowStep(
                id="s1",
                operation_id="aggregate",
                operation_version="1.0",
                inputs=["source:orders"],
                requirement_refs=["order_count"],
            )
        ],
    )
    defaults.update(kwargs)
    return Workflow(**defaults)


def _run(**kwargs) -> RunRecord:
    defaults = dict(
        run_id="run_1",
        workflow_version=1,
        requirement_contract_id="rc_1",
        state=RunState.VERIFYING,
        source_snapshot_ids={"orders": "snap_1"},
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
    )
    defaults.update(kwargs)
    return RunRecord(**defaults)


def _completed_step(step_id: str = "s1", evidence: dict | None = None) -> StepRunRecord:
    return StepRunRecord(
        run_id="run_1",
        step_id=step_id,
        status="COMPLETED",
        operation_full_id="aggregate@1.0",
        evidence=evidence if evidence is not None else {"group_count": 1},
    )


def test_fully_covered_workflow_passes():
    verifier = IndependentVerifier()
    result = verifier.verify(contract=_contract(), workflow=_workflow(), run=_run(), step_runs=[_completed_step()])

    assert result.verdict == "PASS"
    assert result.release_recommendation == "RELEASE"
    assert result.defects == []
    assert result.requirement_coverage[0].status == "PASS"
    assert result.requirement_coverage[0].evidence_refs == ["s1"]


def test_missing_step_record_fails():
    verifier = IndependentVerifier()
    result = verifier.verify(contract=_contract(), workflow=_workflow(), run=_run(), step_runs=[])

    assert result.verdict == "FAIL"
    assert result.release_recommendation == "QUARANTINE"
    assert any("no execution record" in d for d in result.defects)
    assert any("no completed step covering it" in d for d in result.defects)


def test_step_with_no_requirement_ref_leaves_metric_uncovered():
    verifier = IndependentVerifier()
    workflow = _workflow(
        steps=[
            WorkflowStep(
                id="s1", operation_id="aggregate", operation_version="1.0", inputs=["source:orders"]
            )  # no requirement_refs
        ]
    )
    result = verifier.verify(contract=_contract(), workflow=workflow, run=_run(), step_runs=[_completed_step()])

    assert result.verdict == "FAIL"
    assert result.requirement_coverage[0].status == "FAIL"
    assert any("order_count" in d for d in result.defects)


def test_missing_source_snapshot_is_a_defect():
    verifier = IndependentVerifier()
    result = verifier.verify(
        contract=_contract(),
        workflow=_workflow(),
        run=_run(source_snapshot_ids={}),
        step_runs=[_completed_step()],
    )

    assert result.verdict == "FAIL"
    assert any("no locked snapshot id" in d for d in result.defects)


def test_dropped_acceptance_test_is_a_defect():
    verifier = IndependentVerifier()
    contract = _contract(acceptance_tests=["row_count > 0"])
    workflow = _workflow(final_acceptance_tests=[])  # dropped during planning

    result = verifier.verify(contract=contract, workflow=workflow, run=_run(), step_runs=[_completed_step()])

    assert result.verdict == "FAIL"
    assert any("acceptance test dropped" in d for d in result.defects)


def test_sum_metric_without_reconciliation_evidence_needs_review():
    verifier = IndependentVerifier()
    contract = _contract(
        metrics=[Metric(name="revenue", formula="sum(orders.net_amount)", definition_status=DefinitionStatus.GOVERNED)]
    )
    workflow = _workflow(
        steps=[
            WorkflowStep(
                id="s1",
                operation_id="aggregate",
                operation_version="1.0",
                inputs=["source:orders"],
                requirement_refs=["revenue"],
            )
        ]
    )
    # evidence has no "reconciliation"/"external_reconciliation" key at all.
    result = verifier.verify(
        contract=contract, workflow=workflow, run=_run(), step_runs=[_completed_step(evidence={"group_count": 1})]
    )

    assert result.verdict == "NEEDS_REVIEW"
    assert result.release_recommendation == "QUARANTINE"
    assert any("reconciliation" in c for c in result.unverified_claims)


def test_sum_metric_with_reconciliation_evidence_passes():
    verifier = IndependentVerifier()
    contract = _contract(
        metrics=[Metric(name="revenue", formula="sum(orders.net_amount)", definition_status=DefinitionStatus.GOVERNED)]
    )
    workflow = _workflow(
        steps=[
            WorkflowStep(
                id="s1",
                operation_id="aggregate",
                operation_version="1.0",
                inputs=["source:orders"],
                requirement_refs=["revenue"],
            )
        ]
    )
    evidence = {"reconciliation": [{"metric": "revenue", "passed": True}]}
    result = verifier.verify(
        contract=contract, workflow=workflow, run=_run(), step_runs=[_completed_step(evidence=evidence)]
    )

    assert result.verdict == "PASS"
    assert result.unverified_claims == []


def test_prohibited_operation_used_is_a_defect():
    verifier = IndependentVerifier()
    contract = _contract(prohibited_actions=["aggregate"])
    result = verifier.verify(contract=contract, workflow=_workflow(), run=_run(), step_runs=[_completed_step()])

    assert result.verdict == "FAIL"
    assert any("prohibited operation" in d for d in result.defects)


def test_narrative_is_none_without_an_llm_client():
    result = IndependentVerifier().verify(
        contract=_contract(), workflow=_workflow(), run=_run(), step_runs=[_completed_step()]
    )
    assert result.narrative is None


def test_narrative_is_populated_without_changing_the_verdict_when_llm_client_supplied():
    from dataos.llm.deterministic import DeterministicLLMClient

    verifier = IndependentVerifier(DeterministicLLMClient())
    result = verifier.verify(contract=_contract(), workflow=_workflow(), run=_run(), step_runs=[_completed_step()])

    assert result.narrative is not None
    assert result.verdict == "PASS"  # the narrative never changes the decision

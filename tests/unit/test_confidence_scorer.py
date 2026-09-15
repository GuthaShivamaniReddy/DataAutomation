from dataos.compiler.confidence_scorer import ConfidenceScorer
from dataos.compiler.independent_verifier import RequirementCoverage, VerifierResult
from dataos.compiler.reconciliation_agent import ReconciliationReport, ReconciliationTest
from dataos.compiler.release_gate import ReleaseDecision
from dataos.contracts.requirement_contract import (
    DefinitionStatus,
    Metric,
    RequirementContract,
    RequirementStatus,
    Source,
)
from dataos.evidence.models import ValidationResult
from dataos.validation.engine import ValidationReport
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
        status=RequirementStatus.APPROVED,
    )
    defaults.update(kwargs)
    return RequirementContract(**defaults)


def _workflow(steps=None) -> Workflow:
    if steps is None:
        steps = [
            WorkflowStep(
                id="s1",
                operation_id="aggregate",
                operation_version="1.0",
                inputs=["source:orders"],
                requirement_refs=["order_count"],
            )
        ]
    return Workflow(workflow_version=1, requirement_contract_id="rc_1", sources=["orders"], steps=steps)


def _run(state: RunState = RunState.VERIFYING) -> RunRecord:
    return RunRecord(
        run_id="run_1",
        workflow_version=1,
        requirement_contract_id="rc_1",
        state=state,
        source_snapshot_ids={"orders": "snap_1"},
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
    )


def _completed_step_runs() -> list[StepRunRecord]:
    return [StepRunRecord(run_id="run_1", step_id="s1", status="COMPLETED", evidence={"group_count": 1})]


def _clean_verifier_result() -> VerifierResult:
    return VerifierResult(
        verdict="PASS",
        requirement_coverage=[RequirementCoverage(requirement_ref="order_count", status="PASS", evidence_refs=["s1"])],
        release_recommendation="RELEASE",
    )


def test_fully_clean_run_scores_all_dimensions_at_1_and_is_releasable():
    scorer = ConfidenceScorer()
    report = scorer.score(
        contract=_contract(),
        run=_run(),
        workflow=_workflow(),
        step_runs=_completed_step_runs(),
        validation_report=ValidationReport(results=[]),
        reconciliation_report=ReconciliationReport(status="PASS", tests=[], blocking_reasons=[]),
        verifier_result=_clean_verifier_result(),
        release_decision=ReleaseDecision(decision="RELEASE", reason_codes=[], required_remediation=[]),
    )

    assert report.overall_status == "RELEASABLE"
    assert report.scores.requirements == 1.0
    assert report.scores.semantics == 1.0
    assert report.scores.data_quality == 1.0
    assert report.scores.execution == 1.0
    assert report.scores.validation == 1.0
    assert report.scores.reconciliation == 1.0
    assert report.scores.verification == 1.0
    assert report.critical_failures == []
    assert "run:run_1" in report.evidence_refs
    assert "step:s1" in report.evidence_refs


def test_overall_status_never_more_lenient_than_release_decision():
    # Even with every per-dimension score computed as clean/high, a
    # QUARANTINE release_decision must force NOT_RELEASABLE - Section 22:
    # "Do not average away a critical failure."
    scorer = ConfidenceScorer()
    report = scorer.score(
        contract=_contract(),
        run=_run(),
        workflow=_workflow(),
        step_runs=_completed_step_runs(),
        validation_report=ValidationReport(results=[]),
        reconciliation_report=ReconciliationReport(status="PASS", tests=[], blocking_reasons=[]),
        verifier_result=_clean_verifier_result(),
        release_decision=ReleaseDecision(
            decision="QUARANTINE", reason_codes=["some unrelated critical policy flag"], required_remediation=[]
        ),
    )

    assert report.overall_status == "NOT_RELEASABLE"
    assert report.critical_failures == ["some unrelated critical policy flag"]


def test_unresolved_clarification_zeroes_requirements_score():
    from dataos.contracts.requirement_contract import Clarification

    contract = _contract(
        clarifications=[Clarification(issue="x", why_material="y", question="z")], status=RequirementStatus.DRAFT
    )
    scorer = ConfidenceScorer()
    report = scorer.score(
        contract=contract,
        run=_run(),
        workflow=_workflow(),
        step_runs=_completed_step_runs(),
        validation_report=ValidationReport(results=[]),
        reconciliation_report=ReconciliationReport(status="PASS", tests=[], blocking_reasons=[]),
        verifier_result=_clean_verifier_result(),
        release_decision=ReleaseDecision(decision="QUARANTINE", reason_codes=["blocked"], required_remediation=[]),
    )

    assert report.scores.requirements == 0.0


def test_semantics_score_averages_definition_status_weights():
    contract = _contract(
        metrics=[
            Metric(name="a", definition_status=DefinitionStatus.GOVERNED),
            Metric(name="b", definition_status=DefinitionStatus.EXPLICIT),
        ]
    )
    scorer = ConfidenceScorer()
    report = scorer.score(
        contract=contract,
        run=_run(),
        workflow=_workflow(steps=[]),
        step_runs=[],
        validation_report=ValidationReport(results=[]),
        reconciliation_report=ReconciliationReport(status="PASS", tests=[], blocking_reasons=[]),
        verifier_result=_clean_verifier_result(),
        release_decision=ReleaseDecision(decision="RELEASE", reason_codes=[], required_remediation=[]),
    )

    assert report.scores.semantics == (1.0 + 0.85) / 2


def test_data_quality_score_only_counts_data_quality_prefixed_checks():
    validation_report = ValidationReport(
        results=[
            ValidationResult(check_id="null_rate:net_amount", observed=0.5, expected=0.0, passed=False, severity="BLOCKING"),
            ValidationResult(check_id="unique:s1:order_id", observed=0, expected=0, passed=True, severity="BLOCKING"),
            ValidationResult(check_id="reconciliation:revenue", observed=1.0, expected=1.0, passed=True, severity="BLOCKING"),
        ]
    )
    scorer = ConfidenceScorer()
    report = scorer.score(
        contract=_contract(),
        run=_run(),
        workflow=_workflow(),
        step_runs=_completed_step_runs(),
        validation_report=validation_report,
        reconciliation_report=ReconciliationReport(status="PASS", tests=[], blocking_reasons=[]),
        verifier_result=_clean_verifier_result(),
        release_decision=ReleaseDecision(decision="QUARANTINE", reason_codes=["x"], required_remediation=[]),
    )

    # Only null_rate + unique are data-quality-classed: 1 pass out of 2.
    assert report.scores.data_quality == 0.5
    # validation counts all three checks: 2 pass out of 3.
    assert report.scores.validation == 2 / 3


def test_execution_score_reflects_incomplete_steps():
    workflow = _workflow(
        steps=[
            WorkflowStep(id="s1", operation_id="select_filter", operation_version="1.0", inputs=["source:orders"]),
            WorkflowStep(id="s2", operation_id="aggregate", operation_version="1.0", inputs=["s1"]),
        ]
    )
    step_runs = [StepRunRecord(run_id="run_1", step_id="s1", status="COMPLETED", evidence={"x": 1})]
    scorer = ConfidenceScorer()
    report = scorer.score(
        contract=_contract(),
        run=_run(state=RunState.QUARANTINED),
        workflow=workflow,
        step_runs=step_runs,
        validation_report=ValidationReport(results=[]),
        reconciliation_report=ReconciliationReport(status="PASS", tests=[], blocking_reasons=[]),
        verifier_result=_clean_verifier_result(),
        release_decision=ReleaseDecision(decision="QUARANTINE", reason_codes=["step failed"], required_remediation=[]),
    )

    assert report.scores.execution == 0.5


def test_reconciliation_score_penalizes_unavailable_and_failed():
    reconciliation_report = ReconciliationReport(
        status="FAIL",
        tests=[
            ReconciliationTest(name="a", passed=True),
            ReconciliationTest(name="b", passed=False),
        ],
        blocking_reasons=["metric 'c' unavailable"],
    )
    scorer = ConfidenceScorer()
    report = scorer.score(
        contract=_contract(),
        run=_run(),
        workflow=_workflow(),
        step_runs=_completed_step_runs(),
        validation_report=ValidationReport(results=[]),
        reconciliation_report=reconciliation_report,
        verifier_result=_clean_verifier_result(),
        release_decision=ReleaseDecision(decision="QUARANTINE", reason_codes=["x"], required_remediation=[]),
    )

    # 1 passed out of (2 tests + 1 unavailable) = 1/3.
    assert report.scores.reconciliation == 1 / 3


def test_verifier_defects_zero_the_verification_score():
    verifier_result = VerifierResult(
        verdict="FAIL",
        requirement_coverage=[RequirementCoverage(requirement_ref="order_count", status="PASS", evidence_refs=["s1"])],
        defects=["some independent check could not be verified"],
        release_recommendation="QUARANTINE",
    )
    scorer = ConfidenceScorer()
    report = scorer.score(
        contract=_contract(),
        run=_run(),
        workflow=_workflow(),
        step_runs=_completed_step_runs(),
        validation_report=ValidationReport(results=[]),
        reconciliation_report=ReconciliationReport(status="PASS", tests=[], blocking_reasons=[]),
        verifier_result=verifier_result,
        release_decision=ReleaseDecision(decision="QUARANTINE", reason_codes=["x"], required_remediation=[]),
    )

    assert report.scores.verification == 0.0

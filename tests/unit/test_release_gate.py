from dataos.compiler.independent_verifier import RequirementCoverage, VerifierResult
from dataos.compiler.reconciliation_agent import ReconciliationReport, ReconciliationTest
from dataos.compiler.release_gate import ReleaseGate
from dataos.contracts.requirement_contract import RequirementContract, RequirementStatus, Source
from dataos.workflow.state_machine import RunState
from dataos.workflow.store import RunRecord


def _approved_contract() -> RequirementContract:
    return RequirementContract(
        objective="x", sources=[Source(name="orders")], status=RequirementStatus.APPROVED
    )


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


def _passing_verifier_result() -> VerifierResult:
    return VerifierResult(
        verdict="PASS",
        requirement_coverage=[RequirementCoverage(requirement_ref="m", status="PASS", evidence_refs=["s1"])],
        defects=[],
        unverified_claims=[],
        release_recommendation="RELEASE",
    )


def test_releases_when_everything_checks_out():
    decision = ReleaseGate().decide(contract=_approved_contract(), run=_run(), verifier_result=_passing_verifier_result())

    assert decision.decision == "RELEASE"
    assert decision.reason_codes == []
    assert decision.required_remediation == []


def test_quarantines_when_verifier_fails():
    result = VerifierResult(
        verdict="FAIL",
        defects=["requirement 'x' has no completed step covering it"],
        release_recommendation="QUARANTINE",
    )
    decision = ReleaseGate().decide(contract=_approved_contract(), run=_run(), verifier_result=result)

    assert decision.decision == "QUARANTINE"
    assert any("verdict is 'FAIL'" in r for r in decision.reason_codes)
    assert decision.required_remediation == result.defects


def test_quarantines_when_run_not_verifying():
    decision = ReleaseGate().decide(
        contract=_approved_contract(), run=_run(state=RunState.RUNNING), verifier_result=_passing_verifier_result()
    )

    assert decision.decision == "QUARANTINE"
    assert any("not VERIFYING" in r for r in decision.reason_codes)


def test_quarantines_when_contract_not_approved():
    contract = RequirementContract(objective="x", sources=[Source(name="orders")])  # DRAFT
    decision = ReleaseGate().decide(contract=contract, run=_run(), verifier_result=_passing_verifier_result())

    assert decision.decision == "QUARANTINE"
    assert any("not APPROVED" in r for r in decision.reason_codes)


def test_needs_review_verdict_never_releases():
    result = VerifierResult(
        verdict="NEEDS_REVIEW",
        unverified_claims=["contract declares sum-type metric(s) but no step reported reconciliation evidence"],
        release_recommendation="QUARANTINE",
    )
    decision = ReleaseGate().decide(contract=_approved_contract(), run=_run(), verifier_result=result)

    assert decision.decision == "QUARANTINE"


def test_quarantines_on_failed_reconciliation():
    reconciliation_report = ReconciliationReport(
        status="FAIL",
        tests=[ReconciliationTest(name="row_count_conservation:s1", passed=False)],
    )
    decision = ReleaseGate().decide(
        contract=_approved_contract(),
        run=_run(),
        verifier_result=_passing_verifier_result(),
        reconciliation_report=reconciliation_report,
    )

    assert decision.decision == "QUARANTINE"
    assert any("reconciliation status is 'FAIL'" in r for r in decision.reason_codes)
    assert "row_count_conservation:s1" in decision.required_remediation


def test_quarantines_on_unavailable_reconciliation():
    reconciliation_report = ReconciliationReport(
        status="UNAVAILABLE",
        blocking_reasons=["metric 'revenue' has no independent reconciliation anchor available"],
    )
    decision = ReleaseGate().decide(
        contract=_approved_contract(),
        run=_run(),
        verifier_result=_passing_verifier_result(),
        reconciliation_report=reconciliation_report,
    )

    assert decision.decision == "QUARANTINE"
    assert any("reconciliation status is 'UNAVAILABLE'" in r for r in decision.reason_codes)


def test_releases_when_reconciliation_passes():
    reconciliation_report = ReconciliationReport(status="PASS", tests=[ReconciliationTest(name="ok", passed=True)])
    decision = ReleaseGate().decide(
        contract=_approved_contract(),
        run=_run(),
        verifier_result=_passing_verifier_result(),
        reconciliation_report=reconciliation_report,
    )

    assert decision.decision == "RELEASE"
